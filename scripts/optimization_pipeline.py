#!/usr/bin/env python3
"""Optionally add basis cohorts, then optimize the selected basis library."""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from calvin_utils.neuroimaging_utils.ccm_utils.optimization.basis_maps import build
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.pipeline import basis_damage
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.run_optimization import RunOptimization
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.pipeline.pre_processing import datasets_from_loader, make_dataset_loader, optional_datasets


# =============================================================================
# CONFIG
# =============================================================================

BASIS_INPUTS = {}  # {"mask_path": ..., "transform": "rank", "cohorts": {...}}
BASIS_DIR = basis_damage.BASIS_DIR
BASIS_COHORTS = None
BASIS_OVERRIDE = False

FITTING_DATASETS = {}
FITTING_MANIFEST_PATH = None
TEST_DATASETS = {}
TEST_MANIFEST_PATH = None
OPTIMIZATION_TARGETS = ()

OUT_DIR = ""
MAP_OUTPUT_TYPE = None
SUBJECT_COL = None

WITHIN_DATASET_FOLDS = None
CROSS_DATASET_VALIDATION = False
DATA_MODE = "memmap"
RANDOM_SEED = 2026
WEIGHT_INIT_MODE = "unweighted"
REGULARIZATION_STRENGTH = 0.1
PROJECT_PREDICTION_VARIANCE = False
WINNER_ONLY = False

RUN_BASIS_MAP_REGRESSION = False
N_PERMUTATIONS = 1000
STORE_ITERS = False
RENDER_GIF = False
GIF_FPS = 5
GIF_MAX_FRAMES = 60
GIF_VIEW = "auto"


def main():
    """Build requested bases, compare algorithms, and save fitting, validation, and test results."""
    if not OUT_DIR:
        raise ValueError("Set OUT_DIR.")
    out_dir = Path(OUT_DIR).expanduser()
    if BASIS_INPUTS:
        build.main(BASIS_INPUTS, BASIS_DIR, overwrite=BASIS_OVERRIDE)
    mask_path = basis_damage.load_library(BASIS_COHORTS, BASIS_DIR)["mask_path"]
    if OPTIMIZATION_TARGETS:
        RunOptimization.run_targets(FITTING_DATASETS, OPTIMIZATION_TARGETS, test_specs=TEST_DATASETS, subject_col=SUBJECT_COL, basis_cohorts=BASIS_COHORTS, basis_dir=BASIS_DIR, weight_mode=WEIGHT_INIT_MODE, regularization_strength=REGULARIZATION_STRENGTH, project_prediction_variance=PROJECT_PREDICTION_VARIANCE, seed=RANDOM_SEED, within_dataset_folds=WITHIN_DATASET_FOLDS, cross_dataset_validation=CROSS_DATASET_VALIDATION, out_dir=out_dir, output_type=MAP_OUTPUT_TYPE, mask_path=mask_path, store_history=STORE_ITERS, render_gif=RENDER_GIF, gif_fps=GIF_FPS, gif_max_frames=GIF_MAX_FRAMES, gif_view=GIF_VIEW, winner_only=WINNER_ONLY, n_permutations=N_PERMUTATIONS if RUN_BASIS_MAP_REGRESSION else 0)
        return
    fitting_loader = make_dataset_loader(out_dir / "fitting_data", FITTING_DATASETS, FITTING_MANIFEST_PATH, mask_path, SUBJECT_COL)
    fitting = datasets_from_loader(fitting_loader, data_mode=DATA_MODE)
    test = optional_datasets(out_dir / "test_data", TEST_DATASETS, TEST_MANIFEST_PATH, mask_path, SUBJECT_COL, DATA_MODE)
    output_type = getattr(fitting_loader, "output_type", None) or MAP_OUTPUT_TYPE
    optimizer = RunOptimization(fitting, cohorts=BASIS_COHORTS, basis_dir=BASIS_DIR, weight_mode=WEIGHT_INIT_MODE, regularization_strength=REGULARIZATION_STRENGTH, project_prediction_variance=PROJECT_PREDICTION_VARIANCE, seed=RANDOM_SEED)
    optimizer.orchestrate_run(within_dataset_folds=WITHIN_DATASET_FOLDS, cross_dataset_validation=CROSS_DATASET_VALIDATION, test_datasets=test)
    optimizer.save(out_dir, output_type, mask_path, STORE_ITERS, RENDER_GIF, GIF_FPS, GIF_MAX_FRAMES, GIF_VIEW, winner_only=WINNER_ONLY)
    if RUN_BASIS_MAP_REGRESSION:
        inference_dir = out_dir if WINNER_ONLY else out_dir / optimizer.method
        optimizer.result["basis_map_regression"] = optimizer.infer(optimizer.result, inference_dir, output_type, mask_path, N_PERMUTATIONS)


if __name__ == "__main__":
    main()
