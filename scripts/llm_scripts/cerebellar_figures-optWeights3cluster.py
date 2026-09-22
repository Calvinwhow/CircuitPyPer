#!/usr/bin/env python3
"""
SUIT cerebellar flatmaps of cluster-driver regression maps.

For each modality and each cluster: take that cluster's contrast map, apply the
sign policy, summarise it within the SUIT anatomical parcels, and render a
flatmap coloured by the cluster's domain.

Contrast N is cluster N. Which domain that is comes from the tree's own
cluster_membership.csv, never from a hardcoded order -- HDBSCAN numbers clusters
arbitrarily and the mapping differs per tree.

This is not a CLI. Change the values in the CONFIG section, then run.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

CIRCUIT_PYPER_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (str(CIRCUIT_PYPER_DIR.parent), str(CIRCUIT_PYPER_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from calvin_utils.permutation_analysis_utils.parcelwise_regression_similarity import (  # noqa: E402
    ParcelwiseDamageMap, load_parcel_df,
)
from calvin_utils.neuroimaging_utils.nifti_utils.cerebellum_plot import (  # noqa: E402
    SUITCerebellumPlotter,
)


# =============================================================================
# CONFIG
# =============================================================================

# Where the cluster folders live, and where figures go.
SCA_ROOT = "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimization/symptom_on_lhs"
FIGURE_DIR = os.path.join(SCA_ROOT, "cerebellar_figures_optWeights3cluster")

# The parcellation the flatmap is drawn from, and the space everything is put
# into before parcellation. Any map not already on this grid is resampled to it.
PARCEL_GLOB = "/Users/cu135/hires_backdrops/suit/atl-Anatom_space-MNI_dseg_rois/*.nii.gz"
REFERENCE_SPACE = str(CIRCUIT_PYPER_DIR / "resources" / "MNI152_T1_2mm_brain_mask.nii")

# Weighted regressions of every pool map on the cluster indicators (weights = signed
# in-sample optimized weights). Contrast rows: 0-2 each cluster vs 0, 3-5 each vs others.
_ANALYSIS = "regression_identity_and_vs_others/*-on-domain_*"
_CONTRASTS = {0: ("motor", "Motor vs 0"), 1: ("cognitive", "Cognitive vs 0"), 2: ("emotional", "Emotional vs 0"),
              3: ("motor", "Motor vs others"), 4: ("cognitive", "Cognitive vs others"), 5: ("emotional", "Emotional vs others")}
# The optimizer's overall sign is arbitrary (RMS rho ignores sign). Each map is oriented
# so its own-domain mean rho with the HigherIsWorse sheet is positive (native-data
# barplots, specificity_native/regressions-HigherIsBetter-optWeights3cluster). Patient
# images are higher = more damage, so after orienting, t > 0 = damage that worsens the domain.
MODALITIES = [
    dict(name="network", cluster_dir="network_regressions-HigherIsBetter-optWeights3cluster",
         analysis=_ANALYSIS, map_pattern="contrast_tval_FWE_{index}.nii.gz", contrasts=_CONTRASTS,
         orient={0: 1, 1: -1, 2: 1, 3: 1, 4: -1, 5: 1}),
    dict(name="vlsm", cluster_dir="vlsm_regressions-HigherIsBetter-optWeights3cluster",
         analysis=_ANALYSIS, map_pattern="contrast_tval_FWE_{index}.nii.gz", contrasts=_CONTRASTS,
         orient={0: 1, 1: -1, 2: -1, 3: 1, 4: -1, 5: -1}),
    dict(name="fiber", cluster_dir="fiber_regressions-HigherIsBetter-optWeights3cluster",
         analysis=_ANALYSIS, map_pattern="contrast_tval_FWE_{index}.nii.gz", contrasts=_CONTRASTS,
         orient={0: 1, 1: -1, 2: 1, 3: 1, 4: -1, 5: 1}),
]
ONLY = []                      # e.g. ["vlsm"]; empty runs every modality above

# Which tail carries the effect of interest.
#   "positive"  t > 0 only   (higher score = worse, so atrophy drives symptoms)
#   "negative"  t < 0 only, sign-flipped to positive
#   "signed"    keep both tails
SIGN_MODE = "positive"

# How each parcel is summarised. Any DamageScorer metric:
#   avg_over_roi   mean across the whole parcel, zeros included  (size-free)
#   frac_in_roi    fraction of the parcel that is non-zero       (size- and scale-free)
#   avg_in_target  mean over surviving voxels only
#   max_in_roi, sum, num_in_roi, dice, spatial_correlation, cosine
# sum and num_in_roi correlate ~0.7-0.9 with parcel volume; prefer the size-free
# pair when comparing across modalities.
PARCEL_METRIC = "avg_over_roi"

# Exactly one colour-scale policy.
#   "top_k"      each panel from its TOP_K-th strongest drawn parcel (floor, and
#                threshold) to its strongest (ceiling). Every panel shows the
#                same number of parcels, so the modalities' very different units
#                (t for network/vlsm, fiber density for fiber) stop mattering.
#   "fixed"      every panel on SCALE_LIMITS
#   "per_panel"  each panel on its own peak
#   "per_group"  one scale per modality, shared by its clusters
#   "rank"       each parcel replaced by its rank within the panel, 0-100
SCALE_POLICY = "top_k"
TOP_K = 5                      # used by "top_k" only
# Parcels the flatmap does not draw (deep nuclei). They are left out when
# ranking for "top_k"; otherwise they would take slots and show nothing.
NOT_DRAWN = ("Dentate", "Interposed", "Fastigial")
SCALE_LIMITS = (0.0, 5.0)      # used by "fixed" only
SCALE_PERCENTILE = 100         # peak for "per_panel"/"per_group"; lower to tame an outlier
THRESHOLD = None               # colour floor; None = show every non-zero parcel

DOMAIN_COLOURS = {"motor": "#c15656", "cognitive": "#5071a0", "emotional": "#9a8ed1"}
DPI = 300
SAVE_SVG = True


# =============================================================================
# SPACE
# =============================================================================


def resample_to(source_img, reference_img):
    """Trilinear-sample a map onto the reference grid. Identity if already there."""
    data = np.nan_to_num(np.asanyarray(source_img.dataobj, dtype=np.float32))
    if (source_img.shape[:3] == reference_img.shape[:3]
            and np.allclose(source_img.affine, reference_img.affine)):
        return data
    matrix = np.linalg.inv(source_img.affine) @ reference_img.affine
    grid = np.indices(reference_img.shape[:3], dtype=np.float32).reshape(3, -1)
    grid = np.vstack([grid, np.ones((1, grid.shape[1]), np.float32)])
    sampled = map_coordinates(data, (matrix @ grid)[:3], order=1, mode="constant", cval=0.0)
    return sampled.reshape(reference_img.shape[:3])


def apply_sign(data):
    if SIGN_MODE == "positive":
        return np.where(data > 0, data, 0.0)
    if SIGN_MODE == "negative":
        return np.where(data < 0, -data, 0.0)
    if SIGN_MODE == "signed":
        return data
    raise ValueError(f"SIGN_MODE must be positive, negative or signed; got {SIGN_MODE!r}")


# =============================================================================
# INPUTS
# =============================================================================


def contrasts_for(cluster_dir):
    """{contrast index: domain}, read from the tree. Raises if unavailable."""
    path = Path(cluster_dir) / "cluster_membership.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is required: it is the only record of which cluster each "
            "contrast is, and guessing would mislabel every panel.")
    table = pd.read_csv(path)
    mapping = {int(label): group["cluster_domain"].iloc[0]
               for label, group in table.groupby("cluster_label") if label >= 0}
    counts = {int(label): len(group)
              for label, group in table.groupby("cluster_label") if label >= 0}
    return dict(sorted(mapping.items())), counts


def find_analysis(spec):
    base = Path(SCA_ROOT) / spec["cluster_dir"]
    hits = sorted(p for p in base.glob(spec["analysis"]) if p.is_dir())
    if not hits:
        raise FileNotFoundError(f"no analysis matching {spec['analysis']!r} under {base}")
    if len(hits) > 1:
        print(f"  {spec['name']}: {len(hits)} analyses match; using {hits[0].name}")
    return hits[0] / "regression"


# =============================================================================
# SCALE
# =============================================================================


def parcel_values(values, masks):
    """{parcel: value} for the drawn parcels of a parcellated volume."""
    return {name: float(np.median(values[mask])) for name, mask in masks.items()}


def top_k_limits(values, masks):
    """(k-th strongest, strongest) drawn parcel value; floor 0 if fewer than k are non-zero."""
    ranked = sorted((v for v in parcel_values(values, masks).values() if v > 0), reverse=True)
    if not ranked:
        return 0.0, 0.0
    return (ranked[TOP_K - 1] if len(ranked) >= TOP_K else 0.0), ranked[0]


def limits_for(peak, group_peak):
    """(vmin, vmax) under the configured policy."""
    if SCALE_POLICY == "fixed":
        return float(SCALE_LIMITS[0]), float(SCALE_LIMITS[1])
    if SCALE_POLICY == "rank":
        return 0.0, 100.0
    top = peak if SCALE_POLICY == "per_panel" else group_peak
    return (-top, top) if SIGN_MODE == "signed" else (0.0, top)


def to_rank(values):
    """Replace each distinct parcel value by its rank among them, 0-100."""
    out = np.zeros_like(values)
    levels = np.unique(values[values != 0])
    for position, level in enumerate(levels):
        out[values == level] = 100.0 * (position + 1) / levels.size
    return out


def colour_for(domain):
    """A white-centred diverging ramp for signed maps, else white -> the hex."""
    hex_colour = DOMAIN_COLOURS[domain]
    if SIGN_MODE != "signed":
        return hex_colour
    from matplotlib.colors import LinearSegmentedColormap

    name = f"signed_{hex_colour.lstrip('#')}"
    if name not in matplotlib.colormaps:
        matplotlib.colormaps.register(
            LinearSegmentedColormap.from_list(
                name, ["#2f4858", "#8fa3ad", "#ffffff", hex_colour + "80", hex_colour], N=256),
            name=name)
    return name


# =============================================================================
# MAIN
# =============================================================================


def main():
    reference = nib.load(REFERENCE_SPACE)
    parcel_files = sorted(Path(PARCEL_GLOB).parent.glob(Path(PARCEL_GLOB).name))
    if not parcel_files:
        sys.exit(f"No parcels matched {PARCEL_GLOB}")
    parcel_df = load_parcel_df(parcel_path=PARCEL_GLOB, mask_path=REFERENCE_SPACE)
    drawn_masks = {
        path.name.split(".")[0]: resample_to(nib.load(str(path)), reference) > 0.5
        for path in parcel_files if path.name.split(".")[0] not in NOT_DRAWN}
    drawn_masks = {name: mask for name, mask in drawn_masks.items() if mask.any()}
    print(f"{len(parcel_files)} parcels on {parcel_df.shape[0]} voxels in "
          f"{Path(REFERENCE_SPACE).name}\n", flush=True)

    made, skipped = [], []
    for spec in MODALITIES:
        if ONLY and spec["name"] not in ONLY:
            continue
        name = spec["name"]
        try:
            regression = find_analysis(spec)
            if "contrasts" in spec:
                contrasts = {i: d for i, (d, _) in spec["contrasts"].items()}
                counts = {}
            else:
                contrasts, counts = contrasts_for(Path(SCA_ROOT) / spec["cluster_dir"])
        except FileNotFoundError as error:
            print(f"skip {name}: {error}")
            skipped.append(f"{name}: {error}")
            continue

        work = regression.parent / "parcellated_files"
        work.mkdir(parents=True, exist_ok=True)
        panels = {}

        for index, domain in contrasts.items():
            source = regression / spec["map_pattern"].format(index=index)
            if not source.is_file():
                skipped.append(f"{name}/{domain}: no {source.name}")
                continue

            data = resample_to(nib.load(str(source)), reference) * spec.get("orient", {}).get(index, 1)
            data = apply_sign(data)
            kept = int(np.count_nonzero(data))
            orient_tag = "_flipped" if spec.get("orient", {}).get(index, 1) < 0 else ""
            staged = work / f"{source.name.split('.')[0]}{orient_tag}_{SIGN_MODE}.nii.gz"
            nib.save(nib.Nifti1Image(data, affine=reference.affine), str(staged))

            parcellated = ParcelwiseDamageMap(
                target_map=str(staged), parcel_df=parcel_df, mask_path=REFERENCE_SPACE,
                out_dir=str(work), selected_damage=PARCEL_METRIC,
                # The class defaults to dropping zero voxels from each parcel, which
                # silently turns avg_over_roi into avg_in_target. Whole-parcel
                # metrics must see the zeros.
                score_nonzero_only=PARCEL_METRIC not in ("avg_over_roi", "frac_in_roi"),
                output_name=f"{staged.stem.replace('.nii','')}_{PARCEL_METRIC}",
            ).run()

            values = np.nan_to_num(
                np.asanyarray(nib.load(str(parcellated)).dataobj, dtype=np.float32))
            if SCALE_POLICY == "rank":
                values = to_rank(values)
                nib.save(nib.Nifti1Image(values, affine=reference.affine), str(parcellated))
            magnitude = np.abs(values[values != 0])
            peak = float(np.percentile(magnitude, SCALE_PERCENTILE)) if magnitude.size else 0.0
            limits = top_k_limits(values, drawn_masks) if SCALE_POLICY == "top_k" else None
            panels[index] = (domain, Path(parcellated), peak, limits)
            print(f"  {name:8} {domain:10} {kept:8,} voxels, "
                  f"{int(np.count_nonzero(np.unique(values))):2} parcels, peak {peak:.2f}",
                  flush=True)

        if not panels:
            continue
        group_peak = max(p[2] for p in panels.values())
        out_dir = Path(FIGURE_DIR) / name
        out_dir.mkdir(parents=True, exist_ok=True)

        for index, (domain, parcellated, peak, limits) in sorted(panels.items()):
            vmin, vmax = limits if limits else limits_for(peak, group_peak)
            # top_k draws only its k parcels, so the threshold sits just under the
            # floor. With fewer than k non-zero parcels the floor is 0, and the
            # threshold must still be > 0 or every zero is painted over the anatomy.
            threshold = (max(vmin, 1e-6 * vmax) * 0.999) if limits else THRESHOLD
            print(f"  {name:8} {domain:10} scale [{vmin:.4g}, {vmax:.4g}]", flush=True)
            if vmax <= vmin:
                skipped.append(f"{name}/{domain}: nothing to draw")
                continue
            plotter = SUITCerebellumPlotter(str(parcellated))
            plotter.run(space="MNI", out_file=None, cmap=colour_for(domain), colorbar=True,
                        cscale=[vmin, vmax], threshold=threshold)

            figure = plt.gcf()
            title = spec["contrasts"][index][1] if "contrasts" in spec else f"{domain.capitalize()} cluster"
            figure.text(0.5, 1.045, title, ha="center",
                        va="bottom", fontsize=15, fontweight="bold",
                        color=DOMAIN_COLOURS[domain])
            figure.text(0.5, 1.005, "  ·  ".join([
                name, f"contrast {index}" if "contrasts" in spec else f"cluster {index}",
                *([] if "contrasts" in spec else [f"n = {counts.get(index, '?')} maps"]),
                *(["map sign flipped"] if spec.get("orient", {}).get(index, 1) < 0 else []),
                PARCEL_METRIC, SIGN_MODE] + ([f"top {TOP_K} parcels"] if limits else [])),
                ha="center", va="bottom", fontsize=9.5, color="#4b5359")

            base = out_dir / (f"{name}_c{index}_" + title.replace(" ", "_") if "contrasts" in spec
                              else f"{name}_{domain}_cluster")
            figure.savefig(base.with_suffix(".png"), dpi=DPI, bbox_inches="tight",
                           transparent=True)
            if SAVE_SVG:
                figure.savefig(base.with_suffix(".svg"), dpi=DPI, bbox_inches="tight")
            plt.close("all")
            made.append(base.with_suffix(".png"))

    print(f"\n{len(made)} figure(s) -> {FIGURE_DIR}")
    for note in skipped:
        print(f"  skipped: {note}")


if __name__ == "__main__":
    main()
