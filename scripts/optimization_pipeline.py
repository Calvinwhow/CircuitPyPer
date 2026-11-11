#!/usr/bin/env python3
"""Get a basis library, then fit, compare, and save optimization models that weight its bases to predict symptoms.

Basis library (shards in BASIS_DIR), three ways to supply it:
    1. Regression cohort in BASIS_INPUTS: symptoms are regressed on each image column, one t-map basis per symptom per column. voxelwise_vars maps local_anatomy / network_anatomy / fiber_anatomy to image columns, and the key sets each basis's anatomy; fiber columns hold .fib.json files, whose descriptor names the atlas.
    2. Map cohort in BASIS_INPUTS: {"maps": {name: {"path", "representation", ...}}} loads existing maps (NIfTI, .npy, or .fib.json) as one shard, without regression.
    3. BASIS_INPUTS = {}: use the shards already in BASIS_DIR as they are.
    Existing shards are reused when built from the same entry and refused when the entry changed; BASIS_OVERRIDE = True deletes every shard first.

Outputs, per model, in its final_model/ folder (written only for a fit on every patient, i.e. when weights exist):
    optimized_weights.csv; the final map when MAP_OUTPUT_TYPE is set; and, when SAVE_ANATOMY_MAPS is True, three voxel-domain NIfTIs that sum to the final map plus canonical anatomy_fibers_{atlas}.fib.json / .fib.npy visualizations.

BASIS_INPUTS example (the registry passed to build.main; see build.main for every field):
    {"mask_path": "/data/MNI152_T1_2mm_brain_mask.nii", "transform": "rank", "cohorts": {
        "stroke": {"pathology": "Stroke", "input_path": "/data/stroke.csv", "subject_column": "subject", "regressand_list": ["gait", "speech"], "voxelwise_vars": {"local_anatomy": "lesion_path", "network_anatomy": "network_path", "fiber_anatomy": "fiber_json_path"}, "covariates_list": ["age"], "drop_rows": [["selected", "equal", 0]]},
        "priors": {"maps": {"tremor": {"path": "/data/tremor.nii.gz", "representation": "networks"}, "gait": {"path": "/data/gait.fib.json", "representation": "fibers", "n": 40}}}}}

OPTIMIZATION_TARGETS syntax:
    ("gait", "speech") runs two independent optimizations and produces two maps.
    (("gait", "speech"),) jointly fits one shared map to both outcomes.
    ("gait", ("speech", "oculomotor")) mixes one independent target with one joint group.
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from calvin_utils.neuroimaging_utils.ccm_utils.optimization.basis_maps import build
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.pipeline.damage_matrix_utils.basis_library import BASIS_DIR as DEFAULT_BASIS_DIR, BasisLibrary
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.run_optimization import RunOptimization


# =============================================================================
# CONFIG
# =============================================================================

BASIS_INPUTS = {}                        # Registry of cohorts to build into BASIS_DIR (see the docstring); {} uses the existing shards.
BASIS_DIR = DEFAULT_BASIS_DIR            # Folder of basis shards: read for optimization, written when BASIS_INPUTS is set.
BASIS_COHORTS = None                     # None uses every shard in BASIS_DIR; a list of cohort names uses only those.
BASIS_OVERRIDE = False                   # True deletes every shard in BASIS_DIR before building BASIS_INPUTS.

FITTING_DATASETS = {}                    # Datasets that fit the weights: {name: {"csv_path", "nifti_column"}}, optional "subject_column" and "sheet"; each CSV needs a column per target.
TEST_DATASETS = {}                       # Independent datasets in FITTING_DATASETS' format, scored only by the fit on every patient.
OPTIMIZATION_TARGETS = None              # None runs each basis symptom independently. Each nested tuple is one joint target fitted with one shared map; see examples above.

OUT_DIR = ""                             # Required output folder.
MAP_OUTPUT_TYPE = None                   # Format of the final map: "nifti", "gii", or "fiber"; None skips it.
SUBJECT_COL = None                       # Patient-ID column for datasets without their own "subject_column".

CROSS_VALIDATION = "lodo"              # "lodo": fit D-1 datasets, predict the omitted one. "loocv" or k: hold out 1/k of every dataset per fold, fit the rest jointly, average each dataset's fold rhos. None: one fit on every patient, the only fit that is tested on TEST_DATASETS and passed to basis-map regression.
RANDOM_SEED = 2026                       # Seed for cross-validation splits and stochastic fits.
WEIGHT_INIT_MODE = "unweighted"          # rms_rho's starting weights: "unweighted", "gaussian", or "weighted" by each basis's n.
REGULARIZATION_STRENGTH = 0.1           # The scale of the penalty functions, relative to the target functions. Default = 0.1.
PROJECT_PREDICTION_VARIANCE = False      # True holds elastic_net_ols, pairwise_proxy, and rayleigh_refinement at unit prediction variance while fitting; every final weight vector is scaled to it either way.
WINNER_ONLY = False                      # True saves only the best model, directly in OUT_DIR (per target).
SAVE_ANATOMY_MAPS = True                # True saves three voxel-domain maps that sum to the final map, plus native fiber visualizations.

SPARSE_SEMIDEFINITE = False            # Fit sparse_semidefinite too. Off by default: its SDP is slow with hundreds of bases.
NEURAL_NETWORK = True                 # Also fit the anatomical network on every algorithm's solution and compare it with them.
FULLY_TRAINABLE = False                # Train the network's basis-by-algorithm coefficients too, instead of fixing them at the algorithm solutions.
LEARNING_RATE = 0.01                     # Network only: Adam learning rate.
MAX_EPOCHS = 2000                        # Network only: maximum training epochs.
PATIENCE = 100                          # How many epochs may pass without a change in training loss above some tolerance value (1e-7)
DEVICE = "auto"                          # Network only: PyTorch device, "auto", "cpu", "cuda", or "mps".

RUN_BASIS_MAP_REGRESSION = False       # Checks significance of the final topography by permuting the weights; needs CROSS_VALIDATION = None
N_PERMUTATIONS = 1000                   # N permutations for the weight permutation
STORE_ITERS = False                     # Each algorithm saves its fitting iterations to optimization_history.npz.
STORE_4D_NIFTI = True                  # Each model, algorithm or network, saves its map at evenly spaced iterations as a 4D NIfTI.
RENDER_GIF = False                      # Renders the gif of each algorithm's fitted topography along with a dashboard of the weights. Does not work for the neural network. 
GIF_FPS = 5                              # Playback speed of rendered GIFs.
GIF_MAX_FRAMES = 100                     # Determines how many frames enter either the 4D nifti or the rendering gif at evenly spaced intervals along the process
GIF_VIEW = "auto"                        # GIF layout: "auto", "spatial", "circuit", or "metrics".


def optimizer_settings():
    return {"sparse_semidefinite": SPARSE_SEMIDEFINITE, "neural_network": NEURAL_NETWORK, "fully_trainable": FULLY_TRAINABLE, "weight_mode": WEIGHT_INIT_MODE, "regularization_strength": REGULARIZATION_STRENGTH, "project_prediction_variance": PROJECT_PREDICTION_VARIANCE, "learning_rate": LEARNING_RATE, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "device": DEVICE, "seed": RANDOM_SEED}


def main():
    """Build requested bases, compare algorithms, and save fitting, validation, and test results."""
    if not OUT_DIR:
        raise ValueError("Set OUT_DIR.")
    out_dir = Path(OUT_DIR).expanduser()
    if BASIS_INPUTS:
        build.main(BASIS_INPUTS, BASIS_DIR, overwrite=BASIS_OVERRIDE)
    mask_path = BasisLibrary.load(BASIS_COHORTS, BASIS_DIR).mask_path
    n_permutations = 0
    if RUN_BASIS_MAP_REGRESSION:
        n_permutations = N_PERMUTATIONS
    RunOptimization.run_targets(FITTING_DATASETS, OPTIMIZATION_TARGETS, test_specs=TEST_DATASETS, subject_col=SUBJECT_COL, cohorts=BASIS_COHORTS, basis_dir=BASIS_DIR, cross_validation=CROSS_VALIDATION, out_dir=out_dir, output_type=MAP_OUTPUT_TYPE, mask_path=mask_path, store_history=STORE_ITERS, store_map_stack=STORE_4D_NIFTI, render_gif=RENDER_GIF, gif_fps=GIF_FPS, gif_max_frames=GIF_MAX_FRAMES, gif_view=GIF_VIEW, winner_only=WINNER_ONLY, save_anatomy_maps=SAVE_ANATOMY_MAPS, n_permutations=n_permutations, **optimizer_settings())

if __name__ == "__main__":
    main()
