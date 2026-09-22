#!/usr/bin/env python3
"""
Clusters symptom-level result maps with UMAP + HDBSCAN, once per regression
modality (network / vlsm / fiber), and searches for the BrainUmap parameters
that put motor, cognitive and emotional maps into three distinct clusters --
the same parameters, as far as possible, for all three modalities.

Notebook this is derived from:
    notebooks/neuroimaging_notebooks/convergent_causal_mapping/
        11_cluster_symptom_networks.ipynb

This is not a CLI. Change the values in the CONFIG section, then run.
Set SCA_ROOT in the environment to relocate the whole results tree.

Outputs, under OUT_ROOT/<modality>_clusters/:
    parameter_sweep.csv     every parameter combination and how it scored
    best_parameters.json    the winning combination (shared across modalities)
    cluster_results.csv     BrainUmap's own report for the winning run
    cluster_assignments.csv cluster_results plus symptom name and domain
    cluster_report.md       human-readable summary of who landed where
    umap_*.html/.svg/.png   BrainUmap's embedding figures
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from itertools import product
from pathlib import Path

TMPDIR = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(TMPDIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMPDIR / "xdg_cache"))

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

CIRCUIT_PYPER_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (CIRCUIT_PYPER_DIR, SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from calvin_utils.file_utils.import_functions import GiiNiiFileImport
from calvin_utils.ml_utils.brain_umap import BrainUmap


# =============================================================================
# CONFIG
# =============================================================================

# Input/output paths. SCA_ROOT lets the same script run against a mounted copy.
SCA_ROOT = os.environ.get(
    "SCA_ROOT",
    "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimization/symptom_on_lhs",
)
OUT_ROOT = os.environ.get("SCA_OUT_ROOT", SCA_ROOT)
# Only needed when SCA_ROOT points at a mounted copy: the paths written into
# cluster_regression_input.csv are rewritten back to this prefix so the
# regression pipeline can find the maps on the real filesystem.
DEVICE_SCA_ROOT = os.environ.get("SCA_DEVICE_ROOT", "")
# Masks, per modality. Like regression_pipeline.py, this is an opaque path --
# GiiNiiFileImport picks the backend from the file it is asked to read, so
# nothing here knows or cares what a fiber is.
VOLUME_MASK = str(CIRCUIT_PYPER_DIR / "resources" / "MNI152_T1_2mm_brain_mask.nii")
FIBER_ATLAS = os.environ.get(
    "FIBER_ATLAS", "/Volumes/OneTouch/resources/Atlas_tck_MNI/Atlas_all30_MNI.npz")

# Which result trees to cluster, and the image each analysis contributes.
MODALITIES = [m for m in os.environ.get("ONLY_MODALITIES", "network,vlsm,fiber").split(",") if m]
TREE_DIRNAME = {                        # <SCA_ROOT>/<dirname>/<var>-on-<x>/regression/
    "network": os.environ.get("SCA_TREE_NETWORK", "network_regressions"),
    "vlsm": os.environ.get("SCA_TREE_VLSM", "vlsm_regressions"),
    "fiber": os.environ.get("SCA_TREE_FIBER", "fiber_regressions"),
}
# The map each analysis contributes, and the mask to read it against. One entry
# per modality; both are passed straight through to the importer.
FILE_TARGET = {
    "network": os.environ.get("FILE_TARGET_NETWORK", "contrast_tval_0.nii.gz"),
    "vlsm": os.environ.get("FILE_TARGET_VLSM", "contrast_tval_0.nii.gz"),
    "fiber": os.environ.get("FILE_TARGET_FIBER", "contrast_tval_0.fib.npy"),
}
MASK_PATH = {
    "network": os.environ.get("MASK_NETWORK", VOLUME_MASK),
    "vlsm": os.environ.get("MASK_VLSM", VOLUME_MASK),
    "fiber": os.environ.get("MASK_FIBER", FIBER_ATLAS),
}

# Column name for the map path in cluster_regression_input.csv. Whatever the
# regression config names in VOXELWISE_VARS; it holds paths, not a file type.
PATH_COLUMN = os.environ.get("PATH_COLUMN", "Image_File_Path")

# Suffix appended to the tree name to name its output folder.
CLUSTER_SUFFIX = os.environ.get("CLUSTER_SUFFIX", "_clusters")

# Symptom -> domain. Anything not named here falls through to the CNRS
# neuropsychiatric battery, which is the emotional domain.
MOTOR_VARS = [                          # BARS
    "Gait", "HeelToShinTestLeft", "HeelToShinTestRight",
    "FingerToNoseTestLeft", "FingerToNoseTestRight",
    "LimbAtaxia", "Speech", "Oculomotor", "TotalBarsScore",
]
COGNITIVE_VARS = [                      # CCAS
    "SematicFluencyRawS", "PhonemicFluencyRawS", "CategorySwitchRawS", "VerbalRegSum",
    "DigitSpanForwardRawS", "DigitSpanBackwardRawS", "CubeDrawRawS",
    "VerbalRecallRawS", "SimiliarityRawS", "GoNoGoRawS",
]
DEFAULT_DOMAIN = "emotional"            # CNRS: Sec*, ScoreCol*, TotalSection*, CNRSTot*
DOMAIN_ORDER = ["motor", "cognitive", "emotional"]

# Higher = better on the CCAS raw scores, so their t-maps point the opposite way
# from the BARS/CNRS scales where higher = worse. Flip them onto a common sign.
# Matched as substrings by BrainUmap, so "Raw" catches every *RawS column.
#
# This flip is load-bearing, not cosmetic. Measured with everything else held at
# FINAL_PARAMS, domain accuracy without it is 0.68 (network), 2 clusters (vlsm),
# 0.79 (fiber); with it, 1.00 everywhere. The flipped set is exactly the
# cognitive domain, so the cognitive cluster in particular depends on this sign
# convention being the right one.
# On the HigherIsWorse spreadsheet every score already runs the same way, so the
# flip must be turned OFF (COLS_TO_FLIP="") or it re-inverts the cognitive maps.
COLS_TO_FLIP = [c for c in os.environ.get("COLS_TO_FLIP", "Raw,VerbalRegSum").split(",") if c]

# Aggregate scores are sums of items already in the set. Keeping them makes the
# emotional domain 53 of 72 maps and it fragments: measured at FINAL_PARAMS,
# keeping them yields 5 clusters and no 3-way domain match for network and vlsm.
# Dropping them also makes the three trees comparable, since the fiber tree
# never had the CNRS subtotals to begin with.
DROP_AGGREGATE_SCORES = True
AGGREGATE_PATTERNS = [
    "TotalBarsScore", "CNRSTotScore", "CNRSTotColAScore", "CNRSTotColBScore",
    "TotalSection1Score", "TotalSection2Score", "TotalSection3Score",
    "TotalSection4Score", "TotalSection5Score",
    "ScoreCol1A", "ScoreCol1B", "ScoreCol2A", "ScoreCol2B", "ScoreCol3A",
    "ScoreCol3B", "ScoreCol4A", "ScoreCol4B", "ScoreCol5A", "ScoreCol5B",
]

# What to do: "sweep" searches, "final" runs FINAL_PARAMS, "both" does each in turn.
MODE = os.environ.get("CLUSTER_MODE", "final")

# Parameter grid. UMAP is refit once per (metric, n_components, n_neighbors,
# min_dist) and every min_cluster_size is then scored off that one embedding,
# so min_cluster_size is nearly free -- widen it before widening the others.
# Each of these can be narrowed from the environment so a long sweep can be run
# in chunks (one metric per invocation, say) instead of one unbounded job.
SWEEP_METRIC = [str(x) for x in os.environ.get("SWEEP_METRIC", "cosine,correlation,euclidean").split(",") if x]
SWEEP_N_COMPONENTS = [int(x) for x in os.environ.get("SWEEP_N_COMPONENTS", "3").split(",") if x]
SWEEP_N_NEIGHBORS = [int(x) for x in os.environ.get("SWEEP_N_NEIGHBORS", "5,8,10,12,15,20,25,30").split(",") if x]
SWEEP_MIN_DIST = [float(x) for x in os.environ.get("SWEEP_MIN_DIST", "0.0,0.01,0.05,0.1,0.25").split(",") if x]
SWEEP_MIN_CLUSTER_SIZE = [int(x) for x in os.environ.get("SWEEP_MIN_CLUSTER_SIZE", "3,4,5,6,8,10").split(",") if x]

REQUIRED_N_CLUSTERS = 3                 # exactly three, noise excluded
MAX_NOISE_FRACTION = 0.50               # reject runs that call most maps noise

# Used when MODE == "final" and no best_parameters.json exists yet. These are
# the notebook-11 values, and they already give a perfectly diagonal
# cluster-by-domain table (accuracy 1.00, zero noise) in all three modalities,
# so the sweep exists to map how wide that plateau is, not to rescue a bad fit.
FINAL_PARAMS = dict(
    n_components=int(os.environ.get("FINAL_N_COMPONENTS", 3)),
    n_neighbors=int(os.environ.get("FINAL_N_NEIGHBORS", 12)),
    min_dist=float(os.environ.get("FINAL_MIN_DIST", 0.0)),
    metric=os.environ.get("FINAL_METRIC", "cosine"),
    min_cluster_size=int(os.environ.get("FINAL_MIN_CLUSTER_SIZE", 5)),
)

# Fixed BrainUmap arguments, straight from the notebook.
BRAINUMAP_FIXED = dict(
    mask=None, projection=None, cluster_voxels=False, visualize_failed_clusters=True,
)


# =============================================================================
# LOADING
# =============================================================================


def symptom_from_column(column):
    """'/.../Sec1BWorries-on-connectivity_t_path/regression/x.nii.gz' -> 'Sec1BWorries'."""
    parts = Path(str(column)).parts
    for part in reversed(parts):
        if "-on-" in part:
            return part.split("-on-")[0]
    return str(column)


def domain_of(symptom):
    if symptom in MOTOR_VARS:
        return "motor"
    if symptom in COGNITIVE_VARS:
        return "cognitive"
    return DEFAULT_DOMAIN


def load_modality(modality):
    """Return (nimg_df, symptoms, domains) for one regression tree."""
    import_path = str(Path(SCA_ROOT) / TREE_DIRNAME[modality] / "*" / "regression")
    giinii = GiiNiiFileImport(
        import_path=import_path,
        file_column=None,
        file_pattern=FILE_TARGET[modality],
        mask_path=MASK_PATH[modality],
    )
    nimg_df = giinii.run()

    if DROP_AGGREGATE_SCORES:
        keep = [
            col for col in nimg_df.columns
            if symptom_from_column(col) not in AGGREGATE_PATTERNS
        ]
        nimg_df = nimg_df[keep]

    # Stable, name-sorted column order so a rerun reproduces the same embedding.
    order = sorted(nimg_df.columns, key=symptom_from_column)
    nimg_df = nimg_df[order]

    symptoms = [symptom_from_column(c) for c in nimg_df.columns]
    domains = [domain_of(s) for s in symptoms]
    return nimg_df, symptoms, domains


# =============================================================================
# SCORING
# =============================================================================


def score_labels(labels, domains):
    """How well do HDBSCAN labels recover the three symptom domains?

    Clusters are matched to domains one-to-one (Hungarian on the contingency
    table) rather than by per-cluster majority, so two clusters cannot both
    claim the same domain and call it a success.

    Returns a dict; `valid` is False when the run does not meet the structural
    requirement of exactly REQUIRED_N_CLUSTERS clusters covering all domains.
    """
    labels = np.asarray(labels)
    domains = np.asarray(domains)
    clusters = sorted(set(labels.tolist()) - {-1})

    n_total = labels.size
    noise_fraction = float((labels == -1).mean())
    out = {
        "n_clusters": len(clusters),
        "noise_fraction": noise_fraction,
        "accuracy": 0.0,
        "min_domain_recall": 0.0,
        "balance": 0.0,
        "valid": False,
        "mapping": {},
    }
    if len(clusters) != REQUIRED_N_CLUSTERS or noise_fraction > MAX_NOISE_FRACTION:
        return out

    # contingency: clusters x domains
    table = np.zeros((len(clusters), len(DOMAIN_ORDER)), dtype=int)
    for ci, cluster in enumerate(clusters):
        members = domains[labels == cluster]
        for di, domain in enumerate(DOMAIN_ORDER):
            table[ci, di] = int((members == domain).sum())

    rows, cols = linear_sum_assignment(-table)          # maximise the diagonal
    mapping = {int(clusters[r]): DOMAIN_ORDER[c] for r, c in zip(rows, cols)}
    if len(set(mapping.values())) != len(DOMAIN_ORDER):
        return out                                       # a domain went unclaimed

    correct = int(table[rows, cols].sum())
    out["accuracy"] = correct / n_total                  # noise counts against you
    out["mapping"] = mapping

    # Every domain must actually be represented, not just nominally assigned.
    recalls = []
    for r, c in zip(rows, cols):
        domain_total = int((domains == DOMAIN_ORDER[c]).sum())
        recalls.append(table[r, c] / domain_total if domain_total else 0.0)
    out["min_domain_recall"] = float(min(recalls))

    sizes = table.sum(axis=1)
    out["balance"] = float(sizes.min() / sizes.max()) if sizes.max() else 0.0
    out["valid"] = out["min_domain_recall"] > 0.0
    return out


# =============================================================================
# SWEEP
# =============================================================================


def sweep_modality(modality, nimg_df, domains):
    """Score every parameter combination for one modality."""
    rows = []
    embeddings_done = 0
    total_fits = (len(SWEEP_METRIC) * len(SWEEP_N_COMPONENTS)
                  * len(SWEEP_N_NEIGHBORS) * len(SWEEP_MIN_DIST))

    for metric, n_components, n_neighbors, min_dist in product(
        SWEEP_METRIC, SWEEP_N_COMPONENTS, SWEEP_N_NEIGHBORS, SWEEP_MIN_DIST
    ):
        try:
            umapper = BrainUmap(
                data=nimg_df, n_components=n_components, n_neighbors=n_neighbors,
                min_dist=min_dist, metric=metric,
                min_cluster_size=min(SWEEP_MIN_CLUSTER_SIZE),
                cols_to_flip=COLS_TO_FLIP, verbose=False, **BRAINUMAP_FIXED,
            )
        except Exception as exc:                         # noqa: BLE001
            print(f"  [{modality}] UMAP failed "
                  f"metric={metric} nn={n_neighbors} md={min_dist}: {exc}", flush=True)
            continue

        # One embedding, many min_cluster_size values. _run_hdbscan is private
        # but reusing it is what keeps this sweep tractable -- refitting UMAP
        # per min_cluster_size would multiply the runtime by len(the list).
        for min_cluster_size in SWEEP_MIN_CLUSTER_SIZE:
            labels, _, _ = umapper._run_hdbscan(min_cluster_size, False)
            result = score_labels(labels, domains)
            rows.append({
                "modality": modality, "metric": metric, "n_components": n_components,
                "n_neighbors": n_neighbors, "min_dist": min_dist,
                "min_cluster_size": min_cluster_size,
                "n_clusters": result["n_clusters"],
                "noise_fraction": round(result["noise_fraction"], 4),
                "accuracy": round(result["accuracy"], 4),
                "min_domain_recall": round(result["min_domain_recall"], 4),
                "balance": round(result["balance"], 4),
                "valid": result["valid"],
                "mapping": json.dumps(result["mapping"]),
            })

        embeddings_done += 1
        if embeddings_done % 10 == 0 or embeddings_done == total_fits:
            best = max((r["accuracy"] for r in rows if r["valid"]), default=0.0)
            print(f"  [{modality}] {embeddings_done}/{total_fits} embeddings, "
                  f"best valid accuracy so far {best:.3f}", flush=True)

    return pd.DataFrame(rows)


PARAM_KEYS = ["metric", "n_components", "n_neighbors", "min_dist", "min_cluster_size"]


def choose_shared_parameters(sweep_df):
    """Pick one parameter set that works for every modality.

    Ranked by the WORST modality's accuracy, so a combination that is excellent
    on two trees and broken on the third loses to one that is decent on all
    three. That is what "as similar as possible" has to mean when the
    parameters are required to be identical.
    """
    valid = sweep_df[sweep_df["valid"]]
    if valid.empty:
        return None, pd.DataFrame()

    per_combo = valid.groupby(PARAM_KEYS)
    records = []
    for params, group in per_combo:
        if set(group["modality"]) != set(MODALITIES):
            continue                                     # not valid everywhere
        record = dict(zip(PARAM_KEYS, params))
        record["min_accuracy"] = group["accuracy"].min()
        record["mean_accuracy"] = group["accuracy"].mean()
        record["min_balance"] = group["balance"].min()
        record["max_noise"] = group["noise_fraction"].max()
        for _, row in group.iterrows():
            record[f"accuracy_{row['modality']}"] = row["accuracy"]
        records.append(record)

    if not records:
        return None, pd.DataFrame()

    ranked = pd.DataFrame(records).sort_values(
        ["min_accuracy", "mean_accuracy", "min_balance"], ascending=False
    ).reset_index(drop=True)
    best = {key: ranked.loc[0, key] for key in PARAM_KEYS}
    best["n_components"] = int(best["n_components"])
    best["n_neighbors"] = int(best["n_neighbors"])
    best["min_cluster_size"] = int(best["min_cluster_size"])
    best["min_dist"] = float(best["min_dist"])
    return best, ranked


# =============================================================================
# FINAL RUN + REPORT
# =============================================================================


def out_dir_for(modality):
    return Path(OUT_ROOT) / f"{TREE_DIRNAME[modality]}{CLUSTER_SUFFIX}"


def run_final(modality, nimg_df, symptoms, domains, params):
    """Run BrainUmap with the chosen parameters and write the full output set."""
    out_dir = out_dir_for(modality)
    out_dir.mkdir(parents=True, exist_ok=True)

    umapper = BrainUmap(
        data=nimg_df, cols_to_flip=COLS_TO_FLIP, verbose=True,
        **params, **BRAINUMAP_FIXED,
    )
    umapper.run(str(out_dir))                            # writes cluster_results.csv + figures

    labels = np.asarray(umapper.cluster_labels)
    result = score_labels(labels, domains)

    assignments = pd.DataFrame({
        "symptom": symptoms,
        "domain": domains,
        "cluster_label": labels,
        "cluster_domain": [result["mapping"].get(int(l), "noise") for l in labels],
        "cluster_probability": umapper.cluster_probabilities,
        "map_path": [str(c) for c in nimg_df.columns],
    })
    assignments["agrees_with_domain"] = (
        assignments["domain"] == assignments["cluster_domain"]
    )
    assignments.to_csv(out_dir / "cluster_assignments.csv", index=False)
    write_regression_input(out_dir, assignments)

    write_report(out_dir / "cluster_report.md", modality, params, assignments, result)
    return assignments, result


def write_regression_input(out_dir, assignments):
    """Write the CSV that regression_pipeline-clusters.py consumes.

    The design matrix is the three cluster indicators, one column each, with no
    intercept -- so the identity contrast matrix yields one t-map per cluster,
    which is the map of what characterises that cluster. The dummies are written
    out pre-encoded rather than relying on ONE_HOT, because
    CalvinStatsmodelsPalm.define_design_matrix takes a fast path
    (`x = data_df[rhs]`) that never expands a categorical through patsy.

    Noise points (cluster -1) are dropped: they belong to no cluster, so they
    have no indicator to load.
    """
    table = assignments[assignments["cluster_label"] >= 0].copy()

    rows = pd.DataFrame({
        PATH_COLUMN: [
            str(p).replace(str(SCA_ROOT), DEVICE_SCA_ROOT) if DEVICE_SCA_ROOT else str(p)
            for p in table["map_path"]
        ],
        "symptom": table["symptom"].to_numpy(),
        "domain": table["domain"].to_numpy(),
        "cluster_label": table["cluster_label"].to_numpy(),
        "cluster_domain": table["cluster_domain"].to_numpy(),
    })
    for domain in DOMAIN_ORDER:
        rows[f"cluster_{domain}"] = (rows["cluster_domain"] == domain).astype(int)

    path = Path(out_dir) / "cluster_regression_input.csv"
    rows.to_csv(path, index=False)
    print(f"  regression input -> {path} ({len(rows)} maps, "
          + ", ".join(f"{d}={int(rows[f'cluster_{d}'].sum())}" for d in DOMAIN_ORDER) + ")")
    return rows


def write_report(path, modality, params, assignments, result):
    lines = [
        f"# Cluster report: {modality} regressions",
        "",
        "## Parameters",
        "",
        "```python",
        "BrainUmap(data=nimg_df, "
        + ", ".join(f"{k}={v!r}" for k, v in params.items())
        + ", mask=None, projection=None, cluster_voxels=False,",
        f"          visualize_failed_clusters=True, cols_to_flip={COLS_TO_FLIP!r})",
        "```",
        "",
        "## Outcome",
        "",
        f"- maps clustered: **{len(assignments)}**",
        f"- clusters found: **{result['n_clusters']}**",
        f"- noise: **{result['noise_fraction']:.1%}**",
        f"- domain accuracy: **{result['accuracy']:.1%}** "
        f"(noise counted as a miss)",
        f"- smallest domain recall: **{result['min_domain_recall']:.1%}**",
        "",
        "## Cluster composition",
        "",
        "| cluster | assigned domain | n | motor | cognitive | emotional |",
        "|---|---|---|---|---|---|",
    ]
    for cluster in sorted(assignments["cluster_label"].unique()):
        block = assignments[assignments["cluster_label"] == cluster]
        counts = Counter(block["domain"])
        name = "noise" if cluster == -1 else result["mapping"].get(int(cluster), "?")
        lines.append(
            f"| {cluster} | {name} | {len(block)} | "
            f"{counts.get('motor', 0)} | {counts.get('cognitive', 0)} | "
            f"{counts.get('emotional', 0)} |"
        )

    lines += ["", "## Members", ""]
    for cluster in sorted(assignments["cluster_label"].unique()):
        block = assignments[assignments["cluster_label"] == cluster].sort_values("symptom")
        name = "noise" if cluster == -1 else result["mapping"].get(int(cluster), "?")
        lines.append(f"### Cluster {cluster} — {name}")
        lines.append("")
        for _, row in block.iterrows():
            flag = "" if row["agrees_with_domain"] or cluster == -1 else "  *(off-domain)*"
            lines.append(f"- `{row['symptom']}` [{row['domain']}]{flag}")
        lines.append("")

    Path(path).write_text("\n".join(lines))


# =============================================================================
# MAIN
# =============================================================================


def main():
    print(f"SCA_ROOT: {SCA_ROOT}")
    data = {}
    for modality in MODALITIES:
        nimg_df, symptoms, domains = load_modality(modality)
        counts = Counter(domains)
        print(f"[{modality}] {nimg_df.shape[1]} maps x {nimg_df.shape[0]} voxels | "
              + ", ".join(f"{d}={counts.get(d, 0)}" for d in DOMAIN_ORDER), flush=True)
        data[modality] = (nimg_df, symptoms, domains)

    params = dict(FINAL_PARAMS)

    if MODE in ("sweep", "both"):
        frames = []
        for modality in MODALITIES:
            nimg_df, _, domains = data[modality]
            print(f"Sweeping {modality} ...", flush=True)
            frames.append(sweep_modality(modality, nimg_df, domains))
        sweep_df = pd.concat(frames, ignore_index=True)

        for modality in MODALITIES:
            out_dir = out_dir_for(modality)
            out_dir.mkdir(parents=True, exist_ok=True)
            sweep_df[sweep_df["modality"] == modality].to_csv(
                out_dir / "parameter_sweep.csv", index=False)

        best, ranked = choose_shared_parameters(sweep_df)
        if best is None:
            print("\nNo single parameter set gave three domain-aligned clusters in "
                  "all three modalities. Best per modality:")
            for modality in MODALITIES:
                sub = sweep_df[(sweep_df["modality"] == modality) & sweep_df["valid"]]
                if sub.empty:
                    print(f"  {modality}: none valid")
                else:
                    top = sub.sort_values("accuracy", ascending=False).iloc[0]
                    print(f"  {modality}: {dict(top[PARAM_KEYS])} "
                          f"accuracy={top['accuracy']:.3f}")
            return
        params = best
        print(f"\nShared parameters: {params}")
        print(ranked.head(10).to_string(index=False))
        for modality in MODALITIES:
            out_dir = out_dir_for(modality)
            (out_dir / "best_parameters.json").write_text(json.dumps(params, indent=2))
            ranked.head(50).to_csv(out_dir / "shared_parameter_ranking.csv", index=False)

    if MODE in ("final", "both"):
        for modality in MODALITIES:
            nimg_df, symptoms, domains = data[modality]
            print(f"\nFinal run: {modality} with {params}", flush=True)
            assignments, result = run_final(modality, nimg_df, symptoms, domains, params)
            print(f"  clusters={result['n_clusters']} "
                  f"accuracy={result['accuracy']:.3f} "
                  f"noise={result['noise_fraction']:.1%} -> {out_dir_for(modality)}")


if __name__ == "__main__":
    main()
