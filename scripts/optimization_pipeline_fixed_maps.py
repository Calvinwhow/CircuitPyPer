#!/usr/bin/env python3
"""Outer validation and final weight fitting for fixed component maps.

Configure the map files, CSV scoring dictionary, and output directory below,
then run this script. The existing CCM data preparer creates the private arrays
and DataLoader JSON. An existing DataLoader JSON can be supplied instead.
The component maps must have been built independently of these patients. There
is no inner regression loop because the component maps remain fixed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from calvin_utils.neuroimaging_utils.ccm_utils.optimization.fixed_cv import (
    evaluate_fixed_maps_outer_cv,
    save_fixed_cv,
)
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.fixed_pipeline import (
    export_maps,
    infer_map_sample_sizes,
    load_map_files,
    make_scoring_loader,
    optimize_maps,
)
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.visualization import (
    render_optimization_history,
)


# =============================================================================
# CONFIG
# =============================================================================

MAP_FILES = {}                   # {map_name: '/path/map.nii.gz' or '.fib.npy'}
SCORING_DATASETS = {}            # {name: {'csv_path': ..., 'nifti_column': ..., 'symptom_column': ...}}
SCORING_MANIFEST_PATH = None     # Existing DataLoader JSON instead of the CSV dictionary.
SUBJECT_COL = None               # Optional default; each dataset can set subject_column.
OUT_DIR = ""
MASK_PATH = None                 # NIfTI mask or ordered fiber atlas.
MAP_OUTPUT_TYPE = None           # Needed only when MAP_FILES contains bare .npy maps.

OUTER_FOLDS = 5                 # Or 'loocv'; untouched generalization patients.
RANDOM_SEED = 2026
DATA_MODE = "memmap"             # 'memmap' or 'ram' for scoring arrays.
MAX_ITERS = 500
WEIGHT_INIT_MODE = "unweighted"  # 'unweighted', 'gaussian', or 'weighted'.
STORE_ITERS = False              # Save compact final-fit history for a GIF.
RENDER_GIF = False               # Save history and render after optimization.
GIF_OPTIONS = {                  # Passed to render_optimization_history.
    "fps": 5, "max_frames": 60, "view": "auto",
}


def main():
    if not MAP_FILES or not OUT_DIR:
        raise ValueError("Set MAP_FILES and OUT_DIR in CONFIG.")
    out_dir = Path(OUT_DIR).expanduser()
    maps, output_type = load_map_files(
        MAP_FILES, mask_path=MASK_PATH, map_output_type=MAP_OUTPUT_TYPE
    )
    sample_sizes = (
        infer_map_sample_sizes(MAP_FILES)
        if WEIGHT_INIT_MODE == "weighted" else None
    )
    loader = make_scoring_loader(
        out_dir / "scoring_data",
        scoring_datasets=SCORING_DATASETS,
        scoring_manifest_path=SCORING_MANIFEST_PATH,
        mask_path=MASK_PATH,
        subject_col=SUBJECT_COL,
        # Fixed maps score native patient vectors; no cohort-wide transform is
        # allowed to expose outer-validation patients to training preprocessing.
        data_transform_method=None,
    )
    predictions, summary, outer_fold_weights = evaluate_fixed_maps_outer_cv(
        maps, loader, outer_folds=OUTER_FOLDS, seed=RANDOM_SEED,
        data_mode=DATA_MODE, weight_mode=WEIGHT_INIT_MODE,
        map_sample_sizes=sample_sizes, max_iters=MAX_ITERS,
    )
    validation_dir = out_dir / "outer_validation"
    save_fixed_cv(validation_dir, predictions, summary, outer_fold_weights)

    # Outer CV estimates generalization. This separate all-patient refit creates
    # the production map and is not used for the outer-validation statistics.
    results_dir = out_dir / "final_model"
    capture_history = STORE_ITERS or RENDER_GIF
    history_path = (
        results_dir / "optimization_history.npz" if capture_history else None
    )
    optimized_map, optimizer = optimize_maps(
        maps, loader, output_type,
        data_mode=DATA_MODE, weight_init_mode=WEIGHT_INIT_MODE,
        history_path=history_path, map_sample_sizes=sample_sizes,
        mask_path=MASK_PATH, max_iters=MAX_ITERS,
        random_state=RANDOM_SEED,
        return_optimizer=True,
    )
    export_maps(optimized_map, output_type, results_dir, mask_path=MASK_PATH)
    pd.DataFrame({
        "map": optimizer.corr_map_names,
        "weight": optimizer.engine.best_W.reshape(-1),
    }).to_csv(results_dir / "optimized_weights.csv", index=False)
    if RENDER_GIF:
        gif_path = render_optimization_history(
            history_path, results_dir / "optimization.gif", **GIF_OPTIONS
        )
        print(f"Saved optimization GIF to: {gif_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
