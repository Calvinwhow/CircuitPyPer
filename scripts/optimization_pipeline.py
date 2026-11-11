#!/usr/bin/env python3
"""Optionally add basis cohorts, then optimize the selected basis library.

BASIS_INPUTS example (the registry passed to build.main; see build.main for every field):
    {"mask_path": "/data/MNI152_T1_2mm_brain_mask.nii", "transform": "rank", "cohorts": {
        "stroke": {"pathology": "Stroke", "input_path": "/data/stroke.csv", "subject_column": "subject", "regressand_list": ["gait", "speech"], "voxelwise_vars": ["lesion", "network"], "covariates_list": ["age"], "drop_rows": [["selected", "equal", 0]]},
        "priors": {"maps": {"tremor": {"path": "/data/tremor.nii.gz", "representation": "networks"}, "gait": {"path": "/data/gait.fib.json", "representation": "fibers", "n": 40}}}}}
"""

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
OPTIMIZATION_TARGETS = None              # None: every symptom in the basis library. A tuple runs only those targets.

OUT_DIR = ""
MAP_OUTPUT_TYPE = None
SUBJECT_COL = None

CROSS_VALIDATION = "lodo"              # "lodo": fit D-1 datasets, predict the omitted one. "loocv" or k: hold out 1/k of every dataset per fold, fit the rest jointly, average each dataset's fold rhos. None: one fit on every patient, the only fit that is tested on TEST_DATASETS and passed to basis-map regression.
DATA_MODE = "memmap"
RANDOM_SEED = 2026
WEIGHT_INIT_MODE = "unweighted"
REGULARIZATION_STRENGTH = 0.1           # The scale of the penalty functions, relative to the target functions. Default = 0.1.
PROJECT_PREDICTION_VARIANCE = False
WINNER_ONLY = False

SPARSE_SEMIDEFINITE = False            # Fit sparse_semidefinite too. Off by default: its SDP is slow with hundreds of bases.
NEURAL_NETWORK = False                 # Also fit the anatomical network on every algorithm's solution and compare it with them.
FULLY_TRAINABLE = False                # Train the network's basis-by-algorithm coefficients too, instead of fixing them at the algorithm solutions.
LEARNING_RATE = 0.01
MAX_EPOCHS = 2000
PATIENCE = 100                          # How many epochs may pass without a change in training loss above some tolerance value (1e-7)
DEVICE = "auto"                         # the device PyTorch trains on (cpu, gpu, cuda, mps, etc)

RUN_BASIS_MAP_REGRESSION = False       # Checks significance of the final topography by permuting the weights; needs CROSS_VALIDATION = None
N_PERMUTATIONS = 1000                   # N permutations for the weight permutation
STORE_ITERS = False                     # Each algorithm saves its fitting iterations to optimization_history.npz.
STORE_4D_NIFTI = False                  # Each model, algorithm or network, saves its map at evenly spaced iterations as a 4D NIfTI.
RENDER_GIF = False                      # Renders the gif of each algorithm's fitted topography along with a dashboard of the weights. Does not work for the neural network. 
GIF_FPS = 5
GIF_MAX_FRAMES = 100                     # Determines how many frames enter either the 4D nifti or the rendering gif at evenly spaced intervals along the process
GIF_VIEW = "auto"


def optimizer_settings():
    return {"cohorts": BASIS_COHORTS, "basis_dir": BASIS_DIR, "sparse_semidefinite": SPARSE_SEMIDEFINITE, "neural_network": NEURAL_NETWORK, "fully_trainable": FULLY_TRAINABLE, "weight_mode": WEIGHT_INIT_MODE, "regularization_strength": REGULARIZATION_STRENGTH, "project_prediction_variance": PROJECT_PREDICTION_VARIANCE, "learning_rate": LEARNING_RATE, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "device": DEVICE, "seed": RANDOM_SEED}


def main():
    """Build requested bases, compare algorithms, and save fitting, validation, and test results."""
    if not OUT_DIR:
        raise ValueError("Set OUT_DIR.")
    out_dir = Path(OUT_DIR).expanduser()
    if BASIS_INPUTS:
        build.main(BASIS_INPUTS, BASIS_DIR, overwrite=BASIS_OVERRIDE)
    mask_path = basis_damage.load_library(BASIS_COHORTS, BASIS_DIR)["mask_path"]
    if FITTING_MANIFEST_PATH is None:
        RunOptimization.run_targets(FITTING_DATASETS, OPTIMIZATION_TARGETS, test_specs=TEST_DATASETS, subject_col=SUBJECT_COL, cross_validation=CROSS_VALIDATION, out_dir=out_dir, output_type=MAP_OUTPUT_TYPE, mask_path=mask_path, store_history=STORE_ITERS, store_map_stack=STORE_4D_NIFTI, render_gif=RENDER_GIF, gif_fps=GIF_FPS, gif_max_frames=GIF_MAX_FRAMES, gif_view=GIF_VIEW, winner_only=WINNER_ONLY, n_permutations=N_PERMUTATIONS if RUN_BASIS_MAP_REGRESSION else 0, **optimizer_settings())
        return
    fitting_loader = make_dataset_loader(out_dir / "fitting_data", FITTING_DATASETS, FITTING_MANIFEST_PATH, mask_path, SUBJECT_COL)
    fitting = datasets_from_loader(fitting_loader, data_mode=DATA_MODE)
    test = optional_datasets(out_dir / "test_data", TEST_DATASETS, TEST_MANIFEST_PATH, mask_path, SUBJECT_COL, DATA_MODE)
    output_type = getattr(fitting_loader, "output_type", None) or MAP_OUTPUT_TYPE
    RunOptimization(fitting, **optimizer_settings()).run(cross_validation=CROSS_VALIDATION, test_datasets=test, out_dir=out_dir, output_type=output_type, mask_path=mask_path, store_history=STORE_ITERS, store_map_stack=STORE_4D_NIFTI, render_gif=RENDER_GIF, gif_fps=GIF_FPS, gif_max_frames=GIF_MAX_FRAMES, gif_view=GIF_VIEW, winner_only=WINNER_ONLY, n_permutations=N_PERMUTATIONS if RUN_BASIS_MAP_REGRESSION else 0)

if __name__ == "__main__":
    main()
