#!/usr/bin/env python3
"""Nested validation and final fitting for regression-derived component maps.

Map-generating, fitting, and test datasets use the same configuration schema.
Supply either a DataLoader JSON or a dictionary whose entries define
``csv_path``, ``nifti_column``, and ``symptom_column``. CSV inputs are packaged
by the existing CCM regression-data preparer; raw array dictionaries are not
accepted. Inner folds select weights from rebuilt regression maps. Outer folds
test the complete map-building and weight-selection procedure on untouched
patients. Optional test datasets are an independent cohort, opened only after
the final map is frozen and scored exactly once.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from calvin_utils.neuroimaging_utils.ccm_utils.optimization.nested_cv import (
    datasets_from_loader,
    evaluate_regression_pipeline_outer_cv,
    fit_weights_with_inner_cv,
    map_inputs_from_loader,
    save_fit,
    save_history,
    save_nested_cv,
)
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.fixed_pipeline import (
    make_dataset_loader,
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

OUT_DIR = ""
MASK_PATH = None                 # NIfTI mask or ordered fiber atlas.
MAP_OUTPUT_TYPE = None           # Required with MAP_MANIFEST_PATH; inferred for CSV dictionaries.
SUBJECT_COL = None               # Optional default; each entry can set subject_column.

MAP_DATASETS = {}                # {map_name: {'csv_path': ..., 'nifti_column': ..., 'symptom_column': ...}}
MAP_MANIFEST_PATH = None         # Existing DataLoader JSON instead of MAP_DATASETS.
FITTING_DATASETS = {}            # Same CSV dictionary schema; selects the weights.
FITTING_MANIFEST_PATH = None     # Existing DataLoader JSON instead of FITTING_DATASETS.
TEST_DATASETS = {}               # Optional, same schema; scored once on the frozen map.
TEST_MANIFEST_PATH = None        # Existing DataLoader JSON instead of TEST_DATASETS.

INNER_FOLDS = "loocv"           # Select weights inside each outer-training set.
OUTER_FOLDS = 5                  # Test the complete fitted pipeline on unseen patients.
DATA_TRANSFORM_METHOD = "rank"   # Regression maps: 'standardize', 'rank', or None.
RANDOM_SEED = 2026
MAX_ITERS = 500
WEIGHT_INIT_MODE = "unweighted"
STORE_ITERS = False              # Save compact final-fit history for a GIF.
RENDER_GIF = False               # Save history and render after optimization.
GIF_OPTIONS = {                  # Passed to render_optimization_history.
    "fps": 5, "max_frames": 60, "view": "auto",
}


def main():
    if not OUT_DIR or not MASK_PATH:
        raise ValueError("Set OUT_DIR and MASK_PATH.")
    out_dir = Path(OUT_DIR).expanduser()
    map_loader = make_dataset_loader(
        out_dir / "prepared_map_data", dataset_specs=MAP_DATASETS,
        manifest_path=MAP_MANIFEST_PATH, mask_path=MASK_PATH,
        subject_col=SUBJECT_COL,
        # Fold-specific RegressionPrep calls transform only the training rows.
        data_transform_method=None,
    )
    fitting_loader = make_dataset_loader(
        out_dir / "prepared_fitting_data", dataset_specs=FITTING_DATASETS,
        manifest_path=FITTING_MANIFEST_PATH, mask_path=MASK_PATH,
        subject_col=SUBJECT_COL,
        # Damage is calculated from native patient vectors. Transformation is
        # confined to regression fitting inside each inner/outer training set.
        data_transform_method=None,
    )
    output_type = getattr(map_loader, "output_type", MAP_OUTPUT_TYPE)
    if not output_type:
        raise ValueError("Set MAP_OUTPUT_TYPE when using MAP_MANIFEST_PATH.")
    map_X, map_ids, map_outcomes = map_inputs_from_loader(map_loader)
    fitting = datasets_from_loader(fitting_loader)

    outer_results = evaluate_regression_pipeline_outer_cv(
        map_X, map_ids, map_outcomes, fitting,
        inner_folds=INNER_FOLDS, outer_folds=OUTER_FOLDS,
        seed=RANDOM_SEED, data_transform_method=DATA_TRANSFORM_METHOD,
        weight_mode=WEIGHT_INIT_MODE, max_iters=MAX_ITERS,
        score_mode="reference_z",
    )
    if outer_results is not None:
        save_nested_cv(out_dir / "outer_validation", *outer_results)

    # After estimating generalization above, fit the production map on all
    # patients. Its weights are still selected exclusively by inner CV.
    capture_history = STORE_ITERS or RENDER_GIF
    result = fit_weights_with_inner_cv(
        map_X, map_ids, map_outcomes,
        fitting, inner_folds=INNER_FOLDS, seed=RANDOM_SEED,
        data_transform_method=DATA_TRANSFORM_METHOD,
        weight_mode=WEIGHT_INIT_MODE, max_iters=MAX_ITERS,
        score_mode="reference_z", store_iters=capture_history,
        return_optimizer=capture_history,
    )
    if capture_history:
        optimized_map, predictions, summary, weights, optimizer = result
    else:
        optimized_map, predictions, summary, weights = result
    final_dir = out_dir / "final_model"
    save_fit(
        final_dir, optimized_map, predictions, summary, weights,
        output_type, mask_path=MASK_PATH, prefix="inner_cv",
    )
    if capture_history:
        history_path = save_history(
            optimizer, final_dir / "optimization_history.npz",
            output_type=output_type, mask_path=MASK_PATH,
        )
        if RENDER_GIF:
            gif_path = render_optimization_history(
                history_path, final_dir / "optimization.gif", **GIF_OPTIONS
            )
            print(f"Saved optimization GIF to: {gif_path}")

    # The map is frozen above. Test patients are loaded only now and are never
    # used to build maps, select weights, or calibrate scores.
    if TEST_DATASETS or TEST_MANIFEST_PATH:
        test_loader = make_dataset_loader(
            out_dir / "prepared_test_data", dataset_specs=TEST_DATASETS,
            manifest_path=TEST_MANIFEST_PATH, mask_path=MASK_PATH,
            subject_col=SUBJECT_COL, data_transform_method=None,
        )
        test = datasets_from_loader(test_loader)
        test_predictions, test_summary = evaluate_test_datasets(
            optimized_map, test,
            fitting_ids=fitting_ids_from_datasets(
                fitting, map_ids
            ),
        )
        save_test_evaluation(out_dir / "test_evaluation", test_predictions, test_summary)


if __name__ == "__main__":
    main()
