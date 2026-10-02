#!/usr/bin/env python3
"""Outer validation and final weight fitting for fixed component maps.

Configure the map files, CSV fitting dictionary, and output directory below,
then run this script. The existing CCM data preparer creates the private arrays
and DataLoader JSON. An existing DataLoader JSON can be supplied instead.
The component maps must have been built independently of these patients. There
is no inner regression loop because the component maps remain fixed. Optional
test datasets are an independent cohort, opened only after the final weighted
map is frozen and scored exactly once.
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
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.nested_cv import (
    datasets_from_loader,
)
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.test_evaluation import (
    evaluate_test_datasets,
    fitting_ids_from_datasets,
    save_test_evaluation,
)
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.visualization import (
    render_optimization_history,
)


# =============================================================================
# CONFIG
# =============================================================================

MAP_FILES = {}                   # {map_name: '/path/map.nii.gz' or '.fib.npy'}
FITTING_DATASETS = {}            # {name: {'csv_path': ..., 'nifti_column': ..., 'symptom_column': ...}}
FITTING_MANIFEST_PATH = None     # Existing DataLoader JSON instead of the CSV dictionary.
TEST_DATASETS = {}               # Optional, same schema; scored once on the frozen map.
TEST_MANIFEST_PATH = None        # Existing DataLoader JSON instead of TEST_DATASETS.
SUBJECT_COL = None               # Optional default; each dataset can set subject_column.
OUT_DIR = ""
MASK_PATH = None                 # NIfTI mask or ordered fiber atlas.
MAP_OUTPUT_TYPE = None           # Needed only when MAP_FILES contains bare .npy maps.

OUTER_FOLDS = None                 # Or 'loocv'; untouched generalization patients.
RANDOM_SEED = 2026
DATA_MODE = "memmap"             # 'memmap' or 'ram' for scoring arrays.
MAX_ITERS = 500
WEIGHT_INIT_MODE = "gaussian"  # 'unweighted', 'gaussian', or 'weighted'.
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
        out_dir / "fitting_data",
        scoring_datasets=FITTING_DATASETS,
        scoring_manifest_path=FITTING_MANIFEST_PATH,
        mask_path=MASK_PATH,
        subject_col=SUBJECT_COL,
        # Fixed maps score native patient vectors; no cohort-wide transform is
        # allowed to expose outer-validation patients to training preprocessing.
        data_transform_method=None,
    )
    outer_cv_results = evaluate_fixed_maps_outer_cv(
        maps, loader, outer_folds=OUTER_FOLDS, seed=RANDOM_SEED,
        data_mode=DATA_MODE, weight_mode=WEIGHT_INIT_MODE,
        map_sample_sizes=sample_sizes, max_iters=MAX_ITERS,
    )
    if outer_cv_results is not None:
        validation_dir = out_dir / "outer_validation"
        save_fixed_cv(validation_dir, predictions=outer_cv_results[0], summary=outer_cv_results[1], outer_fold_weights=outer_cv_results[2])
        print(outer_cv_results[1].to_string(index=False))

    # Outer CV estimates generalization. This separate all-patient refit creates
    # the final map. Above estimated generalization. 
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
    

    # Apparent fit of the frozen map in its own fitting patients. This is
    # in-sample and is not a performance estimate; it is here because the
    # objective squares rho, so the sign of the fitted relationship is not
    # fixed by construction and has to be read off the training side.
    fitting_datasets = datasets_from_loader(loader)
    apparent_predictions, apparent_summary = evaluate_test_datasets(
        optimized_map, fitting_datasets
    )
    apparent_summary = apparent_summary.rename(columns={
        "test_spearman_rho": "apparent_spearman_rho",
        "test_spearman_p": "apparent_spearman_p",
    })
    apparent_predictions.to_csv(results_dir / "apparent_predictions.csv", index=False)
    apparent_summary.to_csv(results_dir / "apparent_summary.csv", index=False)
    print(apparent_summary.to_string(index=False))

    # The weighted map is frozen above. Test patients are loaded only now and
    # never contributed a component map, a weight, or a calibration score.
    if TEST_DATASETS or TEST_MANIFEST_PATH:
        test_loader = make_scoring_loader(
            out_dir / "test_data",
            scoring_datasets=TEST_DATASETS,
            scoring_manifest_path=TEST_MANIFEST_PATH,
            mask_path=MASK_PATH,
            subject_col=SUBJECT_COL,
            data_transform_method=None,
        )
        test_predictions, test_summary = evaluate_test_datasets(
            optimized_map, datasets_from_loader(test_loader),
            fitting_ids=fitting_ids_from_datasets(fitting_datasets),
        )
        save_test_evaluation(out_dir / "test_evaluation", test_predictions, test_summary)


if __name__ == "__main__":
    main()
