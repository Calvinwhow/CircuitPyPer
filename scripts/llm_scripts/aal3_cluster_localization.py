#!/usr/bin/env python3
"""
Intersects the FWE-corrected cluster-driver maps with the AAL3 atlas and reports
where each cluster's signature localises, per modality and per contrast-matrix
variant.

The driver maps are signed: a cluster's regressor can push a region up or down
relative to the others, and motor/emotional in particular are strongly
anti-correlated. So every region is scored three ways -- the fraction of it
surviving FWE with t > 0, the fraction with t < 0, and the net of the two.
Coverage rather than raw voxel counts, because counts mostly track region size.

Handles either atlas layout: a 3-D integer label volume, or the 4-D stack of
binary masks that ROI_MNI_V7_resampled_grouped*.nii.gz uses (one volume per
region, in the order of the accompanying .txt).

Outputs land in OUTPUT_DIR/<variant>/:
    aal3_localization.csv           one row per modality x cluster x region
    aal3_localization_net.csv       regions x (modality x cluster) net coverage
    aal3_localization_abs.csv       the same grid, total coverage
    aal3_localization_heatmap.png   the heatmap (also .svg)
    aal3_localization_report.md     top regions per modality x cluster
plus OUTPUT_DIR/aal3_localization_all_variants.csv holding every row.

This is not a CLI. Change the values in the CONFIG section, then run.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates


# =============================================================================
# CONFIG
# =============================================================================

SCA_ROOT = os.environ.get(
    "SCA_ROOT",
    "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimization/symptom_on_lhs",
)

OUTPUT_DIR = os.environ.get("AAL3_OUT", os.path.join(SCA_ROOT, "aal3_localization"))

# AAL3 and its label table. The "fine" grouping is 88 bilateral AAL3 structures;
# swap to ROI_MNI_V7_resampled_grouped.nii.gz for the 39-region coarse version.
AAL_DIR = os.environ.get("AAL3_DIR", "/Volumes/HowExp/resources/atlases/mni_space/aal_atlas")
ATLAS_NII = os.environ.get("AAL3_NII", f"{AAL_DIR}/ROI_MNI_V7_resampled_grouped_fine.nii.gz")
ATLAS_LUT = os.environ.get("AAL3_LUT", f"{AAL_DIR}/ROI_MNI_V7_resampled_grouped_fine.txt")

# The two contrast-matrix versions, each its own analysis.
VARIANTS = ["centered_standardized", "identity_standardized"]

ANALYSIS_GLOB = "*-on-cluster_motor*"        # tolerates the Nifti_/Fiber_ prefix difference

MODALITIES = {                                # label -> cluster folder
    "network": "network_regressions_clusters",
    "vlsm": "vlsm_regressions_clusters",
    "fiber": "fiber_regressions_clusters",
}

# contrast index -> cluster, in design-matrix column order.
CONTRASTS = {0: "motor", 1: "cognitive", 2: "emotional"}

MIN_REGION_VOXELS = 10          # ignore regions too small to survive resampling
MIN_COVERAGE_TO_PLOT = 0.05     # a region needs this much coverage somewhere to appear
MAX_ROWS_IN_HEATMAP = 90        # hard cap so the figure stays legible
TOP_N_PER_CELL = 10             # regions listed per modality x cluster in the report


# =============================================================================
# ATLAS
# =============================================================================


def read_lut(path):
    """'<index> <name> ...' per line. Returns {int: name}."""
    if not path or not Path(path).is_file():
        return {}
    lut = {}
    for line in Path(path).read_text(errors="ignore").splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) >= 2 and parts[0].lstrip("-").isdigit():
            lut[int(parts[0])] = parts[1]
        elif len(parts) >= 2 and parts[-1].lstrip("-").isdigit():
            lut[int(parts[-1])] = parts[0]
    return lut


def load_regions(atlas_path, lut):
    """
    Return [(label, name, mask_bool_3d), ...] plus the atlas affine.

    A 4-D atlas is read as one binary mask per volume, numbered from 1 in volume
    order; a 3-D atlas is read as an integer label volume.
    """
    img = nib.load(atlas_path)
    data = np.asanyarray(img.dataobj)
    regions = []
    if data.ndim == 4:
        for index in range(data.shape[3]):
            label = index + 1
            regions.append((label, lut.get(label, f"label_{label}"), data[..., index] > 0))
    else:
        labels = np.rint(data).astype(np.int32)
        for label in np.unique(labels):
            if label == 0:
                continue
            regions.append((int(label), lut.get(int(label), f"label_{int(label)}"),
                            labels == label))
    return regions, img.affine, img.shape[:3]


def resample_mask(mask, atlas_affine, ref_img):
    """Nearest-neighbour a boolean mask onto the reference grid via the affines."""
    if mask.shape == ref_img.shape[:3] and np.allclose(atlas_affine, ref_img.affine):
        return mask
    matrix = np.linalg.inv(atlas_affine) @ ref_img.affine
    grid = np.indices(ref_img.shape[:3], dtype=np.float32).reshape(3, -1)
    grid = np.vstack([grid, np.ones((1, grid.shape[1]), np.float32)])
    out = map_coordinates(mask.astype(np.float32), (matrix @ grid)[:3],
                          order=0, mode="constant", cval=0.0)
    return out.reshape(ref_img.shape[:3]) > 0.5


# =============================================================================
# TABLE
# =============================================================================


def find_regression_dir(cluster_folder, variant):
    base = Path(SCA_ROOT) / cluster_folder / f"cluster_regression_{variant}"
    hits = sorted(p for p in base.glob(ANALYSIS_GLOB) if p.is_dir())
    if not hits:
        raise FileNotFoundError(f"no analysis folder matching {ANALYSIS_GLOB} under {base}")
    return hits[0] / "regression"


def build_table(regions, atlas_affine, variant):
    rows = []
    resampled_cache = {}

    for modality, folder in MODALITIES.items():
        reg = find_regression_dir(folder, variant)
        for index, cluster in CONTRASTS.items():
            path = reg / f"contrast_tval_FWE_{index}.nii.gz"
            if not path.is_file():
                print(f"  missing {modality}/{cluster}: {path}")
                continue

            img = nib.load(str(path))
            tmap = np.nan_to_num(np.asanyarray(img.dataobj, dtype=np.float32))
            positive, negative = tmap > 0, tmap < 0

            key = (img.shape[:3], img.affine.tobytes())
            if key not in resampled_cache:
                resampled_cache[key] = [
                    (label, name, resample_mask(mask, atlas_affine, img))
                    for label, name, mask in regions
                ]

            for label, name, mask in resampled_cache[key]:
                n_region = int(mask.sum())
                if n_region < MIN_REGION_VOXELS:
                    continue
                n_pos = int((mask & positive).sum())
                n_neg = int((mask & negative).sum())
                values = tmap[mask & (positive | negative)]
                rows.append({
                    "variant": variant,
                    "modality": modality,
                    "cluster": cluster,
                    "label": label,
                    "region": name,
                    "region_voxels": n_region,
                    "sig_pos": n_pos,
                    "sig_neg": n_neg,
                    "coverage": round((n_pos + n_neg) / n_region, 4),
                    "coverage_pos": round(n_pos / n_region, 4),
                    "coverage_neg": round(n_neg / n_region, 4),
                    "net_coverage": round((n_pos - n_neg) / n_region, 4),
                    "mean_t": round(float(values.mean()), 3) if values.size else 0.0,
                    "mean_abs_t": round(float(np.abs(values).mean()), 3) if values.size else 0.0,
                    "peak_abs_t": round(float(np.abs(values).max()), 3) if values.size else 0.0,
                })
            print(f"  {modality:8} {cluster:10} "
                  f"+{int(positive.sum()):>8,} / -{int(negative.sum()):>8,} voxels", flush=True)

    return pd.DataFrame(rows)


def column_order():
    return [f"{m}_{c}" for m in MODALITIES for c in CONTRASTS.values()]


def pivot(long, value):
    wide = long.pivot_table(index=["label", "region"], columns=["modality", "cluster"],
                            values=value, fill_value=0.0)
    wide.columns = [f"{m}_{c}" for m, c in wide.columns]
    for col in column_order():
        if col not in wide.columns:
            wide[col] = 0.0
    return wide[column_order()]


# =============================================================================
# FIGURE
# =============================================================================


def draw_heatmap(net, title, path_png):
    keep = net[net.abs().max(axis=1) >= MIN_COVERAGE_TO_PLOT]
    keep = keep.reindex(keep.abs().max(axis=1).sort_values(ascending=False).index)
    keep = keep.head(MAX_ROWS_IN_HEATMAP)
    keep = keep.reindex(keep.apply(lambda r: (int(np.argmax(r.values)), -r.max()), axis=1)
                        .sort_values().index)

    names = [region for _, region in keep.index]
    height = max(4.0, 0.22 * len(keep) + 2.0)
    fig, ax = plt.subplots(figsize=(8.0, height))
    image = ax.imshow(keep.values, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0)

    ax.set_xticks(range(len(keep.columns)))
    ax.set_xticklabels([c.replace("_", "\n") for c in keep.columns], fontsize=8)
    ax.set_yticks(range(len(keep)))
    ax.set_yticklabels(names, fontsize=7)
    for boundary in (len(CONTRASTS) - 0.5, 2 * len(CONTRASTS) - 0.5):
        ax.axvline(boundary, color="black", linewidth=1.2)
    ax.set_title(f"AAL3 localisation -- {title}\n"
                 "colour = net signed fraction of region surviving FWE", fontsize=10)
    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    bar.set_label("net coverage  (positive - negative)", fontsize=8)
    bar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(path_png, dpi=300)
    fig.savefig(str(path_png).replace(".png", ".svg"))
    plt.close(fig)
    return len(keep)


def write_report(long, path_md, variant):
    lines = [
        f"# AAL3 localisation -- {variant}",
        "",
        f"Atlas: `{ATLAS_NII}`",
        "Contrasts in design-matrix order: "
        + ", ".join(f"{i}={c}" for i, c in CONTRASTS.items()),
        "",
        "Coverage is the fraction of an AAL3 region's voxels surviving FWE; pos and "
        "neg split it by the sign of t. Net is pos minus neg.",
        "",
    ]
    for modality in MODALITIES:
        lines.append(f"## {modality}")
        lines.append("")
        for cluster in CONTRASTS.values():
            subset = long[(long.modality == modality) & (long.cluster == cluster)].copy()
            subset["rank"] = subset.net_coverage.abs()
            subset = subset.sort_values("rank", ascending=False).head(TOP_N_PER_CELL)
            lines.append(f"### {cluster}")
            lines.append("")
            lines.append("| region | net | pos | neg | mean t | voxels |")
            lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
            for _, row in subset.iterrows():
                lines.append(f"| {row.region} | {row.net_coverage:+.3f} | "
                             f"{row.coverage_pos:.3f} | {row.coverage_neg:.3f} | "
                             f"{row.mean_t:+.2f} | {int(row.region_voxels):,} |")
            lines.append("")
    Path(path_md).write_text("\n".join(lines))


# =============================================================================
# MAIN
# =============================================================================


def main():
    if not Path(ATLAS_NII).is_file():
        sys.exit(f"Atlas not found: {ATLAS_NII}\n"
                 "Set AAL3_DIR (or AAL3_NII/AAL3_LUT), or mount the drive it lives on.")

    lut = read_lut(ATLAS_LUT)
    regions, atlas_affine, atlas_shape = load_regions(ATLAS_NII, lut)
    print(f"atlas {Path(ATLAS_NII).name} {atlas_shape} | "
          f"{len(regions)} regions | {len(lut)} LUT names\n")

    out_root = Path(OUTPUT_DIR)
    every = []

    for variant in VARIANTS:
        print(f"=== {variant}")
        long = build_table(regions, atlas_affine, variant)
        if long.empty:
            print(f"  nothing found for {variant}; skipping")
            continue
        every.append(long)

        net = pivot(long, "net_coverage")
        total = pivot(long, "coverage")

        out = out_root / variant
        out.mkdir(parents=True, exist_ok=True)
        long.to_csv(out / "aal3_localization.csv", index=False)
        net.reset_index().to_csv(out / "aal3_localization_net.csv", index=False)
        total.reset_index().to_csv(out / "aal3_localization_abs.csv", index=False)
        n_plotted = draw_heatmap(net, variant.replace("_", " "),
                                 out / "aal3_localization_heatmap.png")
        write_report(long, out / "aal3_localization_report.md", variant)
        print(f"  {len(long)} rows over {net.shape[0]} regions; "
              f"{n_plotted} in the heatmap -> {out}\n")

    if every:
        combined = pd.concat(every, ignore_index=True)
        combined.to_csv(out_root / "aal3_localization_all_variants.csv", index=False)
        print(f"wrote {out_root / 'aal3_localization_all_variants.csv'} "
              f"({len(combined)} rows)")


if __name__ == "__main__":
    main()
