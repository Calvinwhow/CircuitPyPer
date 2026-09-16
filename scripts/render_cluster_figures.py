#!/usr/bin/env python3
"""
Renders neuro_plotter figures for the cluster-driver regression maps.

The driver regressions were run with RUN_FIGURES off, and re-running
regression_pipeline-clusters.py will not fill them in: run_voxelwise_regression
returns early once contrast_tval_0 exists, so the plotting call is skipped along
with the regression. This walks the finished maps directly instead.

Each map renders in its own interpreter, for the same reason
FIGURES_IN_SUBPROCESS exists in the regression pipelines: every yabplot view
opens an off-screen OpenGL context, and macOS stops handing them out after a few
hundred. A fresh process per map tears that state down each time.

This is not a CLI. Change the values in the CONFIG section, then run.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

CIRCUIT_PYPER_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (CIRCUIT_PYPER_DIR, SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# =============================================================================
# CONFIG
# =============================================================================

SCA_ROOT = os.environ.get(
    "SCA_ROOT",
    "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimization/symptom_on_lhs",
)
CLUSTER_DIRS = [
    "network_regressions_clusters",
    "vlsm_regressions_clusters",
    "fiber_regressions_clusters",
]
ANALYSIS_DIR = "cluster_driver_regression/Nifti_File_Path-on-cluster_motor"

# contrast_tval_FWE_0/1/2 are motor, cognitive, emotional in design-matrix order.
CONTRASTS = ["0", "1", "2"]
FWE_ONLY = True                 # False also renders the uncorrected contrast_tval_*
SKIP_COMPLETED = True           # a figure folder with an index.html is left alone


CLUSTER_NAMES = {"0": "motor", "1": "cognitive", "2": "emotional"}


def maps_to_render():
    """Yield (label, source_map, output_dir) for every map that needs figures."""
    for cluster_dir in CLUSTER_DIRS:
        regression = Path(SCA_ROOT) / cluster_dir / ANALYSIS_DIR / "regression"
        figures = Path(SCA_ROOT) / cluster_dir / ANALYSIS_DIR / "figures"
        if not regression.is_dir():
            print(f"Skipping {cluster_dir}: no regression output at {regression}")
            continue

        stems = [f"contrast_tval_FWE_{i}" for i in CONTRASTS]
        if not FWE_ONLY:
            stems += [f"contrast_tval_{i}" for i in CONTRASTS]

        for stem in stems:
            source = regression / f"{stem}.nii.gz"
            if not source.is_file():
                print(f"Skipping {cluster_dir}/{stem}: not found")
                continue
            out_dir = figures / stem
            if SKIP_COMPLETED and (out_dir / "index.html").is_file():
                print(f"Skipping {cluster_dir}/{stem}: already rendered")
                continue
            contrast = stem.rsplit("_", 1)[-1]
            label = f"{cluster_dir.split('_')[0]} · {CLUSTER_NAMES.get(contrast, contrast)} · {stem}"
            yield label, source, out_dir


def main():
    plotter = SCRIPT_DIR / "neuro_plotter.py"
    jobs = list(maps_to_render())
    if not jobs:
        print("Nothing to render.")
        return

    print(f"Rendering {len(jobs)} map(s).\n")
    failures = []
    for index, (label, source, out_dir) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {label}", flush=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run([
            sys.executable, str(plotter),
            "--nifti", str(source),
            "--output-dir", str(out_dir),
            "--no-open-html",
        ])
        if completed.returncode != 0:
            failures.append(label)
            print(f"    failed (exit {completed.returncode}); continuing.", flush=True)

    print(f"\nDone. {len(jobs) - len(failures)} rendered, {len(failures)} failed.")
    for label in failures:
        print(f"  failed: {label}")


if __name__ == "__main__":
    main()
