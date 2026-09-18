#!/usr/bin/env python3
"""
Runs voxelwise, vertexwise, fiber-wise permuted regressions with cross validations
After each regression, runs visualization software. 
Can run TIME-VARYING (4D) versions of each of the above file types. 

This is  not a CLI. Change the values in the CONFIG section, then run. 
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

TMPDIR = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(TMPDIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMPDIR / "xdg_cache"))

import nibabel as nib
import numpy as np
import pandas as pd
from itertools import product

CIRCUIT_PYPER_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(CIRCUIT_PYPER_DIR) not in sys.path:
    sys.path.insert(0, str(CIRCUIT_PYPER_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from calvin_utils.permutation_analysis_utils.statsmodels_palm import CalvinStatsmodelsPalm
from calvin_utils.permutation_analysis_utils.voxelwise_regression import VoxelwiseRegression
from calvin_utils.permutation_analysis_utils.voxelwise_regression_prep import RegressionPrep
from calvin_utils.neuroimaging_utils.output_functions import NeuroimageFileOutporter
# import circuit_pyper.scripts.circuit_viewer_orchestrator as circuit_viewer_orchestrator


# =============================================================================
# CONFIG
# =============================================================================


# Input/output paths.
INPUT_PATH = "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimzation/optimized_master_list_filtered_HigherIsWorse.csv" # Form: "/path/to/input.csv"
SHEET = None # Specify sheet if using excel (i.e. "Sheet1")
OUT_DIR = "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimization/symptom_on_lhs/fiber_regressions_HigherIsWorse" # Form: "/path/to/output_dir"
MASK_PATH = "/Volumes/OneTouch/resources/Atlas_tck_MNI/Atlas_all30_MNI.npz" # Guerrero fibers use the original TCK-derived MNI geometry; the TRK-derived copy has an incorrect affine offset.

# Model setup.

var_list = [
 'Gait', 'HeelToShinTestLeft', 'HeelToShinTestRight', 'FingerToNoseTestLeft', 'FingerToNoseTestRight', 'LimbAtaxia', 'Speech', 'Oculomotor',
 
 'SematicFluencyRawS', 'PhonemicFluencyRawS', 'CategorySwitchRawS', 'VerbalRegSum', 'DigitSpanForwardRawS', 'DigitSpanBackwardRawS', 'CubeDrawRawS', 'VerbalRecallRawS', 'SimiliarityRawS', 'GoNoGoRawS'

# 'SematicFluencyFailS', 'PhonemicFluencyFailS', 'CategorySwitchFailS', 'DigitSpanForwardFailS', 'DigitSpanBackwardFailS', 'CubeDrawFailS', 'VerbalRecallFailS', 'SimiliarityFailS', 'GoNoGoFailS', 'AffectFailS', 

 'Sec1ADifficultFocus', 'Sec1AEasilyDistracted', 'Sec1AOntheGo', 'Sec1AFeelsCompelled', 'Sec1AFeelsDriven', 'Sec1BWorries', 'Sec1BRepeats', 'Sec1BMentallyStuck', 
 'Sec1BCauseDistress', 'Sec2AActHastily', 'Sec2ARapidChanges', 'Sec2ACryingLaughing', 'Sec2AOverAnxious', 
 'Sec2BLackOfPleasure', 'Sec2BNegativeAttitude', 'Sec2BUneasyWithLife', 'Sec2BSadDepressed', 
 'Sec3ARepetitiveMovements', 'Sec3ASensoryExp', 'Sec3BSensitive', 'Sec3BOverwhelmed', 
 'Sec4ACommunicates', 'Sec4AConcerns', 'Sec4ASeesHearsThings', 
 'Sec4BTroubleUnderstand', 'Sec4BDistant', 'Sec4BIndifferent', 
 'Sec5AAngry', 'Sec5AUpset', 'Sec5AIntolerant', 'Sec5AArgumentative', 
 'Sec5Bimmature', 'Sec5BUnaware', 'Sec5BManner', 'Sec5BTrusting', 

#  'ScoreCol1A', 'ScoreCol1B', 
#  'ScoreCol2A', 'ScoreCol2B', 
#  'ScoreCol3A', 'ScoreCol3B', 
#  'ScoreCol4A', 'ScoreCol4B', 
#  'ScoreCol5A', 'ScoreCol5B', 
 
 #Major Summations
#  'TotalBarsScore',
#  'TotalCCASFailScore', 'TotalCCASRawScore',
#  'CNRSTotScore', 
 ]
regressand_list = var_list               # On left hand=side of the equation. Often is the outcome variable.         Will run an analysis for each value.
regressor_list =  ["fiber_path_guerrera"]   # On right hand-side of the equation. Often is the neuroimaging variable.   Will run an analysis for each value. 
VOXELWISE_VARS = ["fiber_path_guerrera"]    # Name the variables that are stored in neuroimaging files
VOXELWISE_INTERACTIONS = []             # If you want interactions, specify them. 
COVARIATES_LIST = []                    # List of all nuisance variables to adjust for. If you want interactions, make them in your spreadsheet and add them here. Will NOT trigger a new analysis for each value, but will be present in every analysis.
ADD_INTERCEPT = False                  # Three exhaustive cluster indicators require no intercept (avoids perfect collinearity).

# Optional preprocessing.
DROP_ROWS = [("selected", "equal", 0)]                        # Conditions for droppping some rows. Example: DROP_ROWS = [("group", "equal", "control"), ("age", "below", 18)]
ONE_HOT = None                          # columns to one-hot encode
EXCHANGEABILITY_COL = None              # Exchangeability blocks to restrict permutations within
WEIGHTS_COL = None                      # Weights for weighted regression. Defaults to equal. 
DATA_TRANSFORM_METHOD = "rank"   # Standardize continuous design/outcome data across observations.
INVERT_REGRESSAND = False                # Multiply regressand by -1. Default: False

# Each cluster mean contrasted against the mean of the other two clusters.
CONTRAST_MATRIX = [[1]]
# Regression settings.
REGRESSION_TYPE = "linear"              # Default linear
N_PERMUTATIONS = 1000                   # Default 1000
RUN_FIGURES = False                      # Delegate result-map visualization to scripts/neuro_plotter.py.
FIGURES_IN_SUBPROCESS = True            # Render each map in a fresh interpreter (frees the OpenGL context).
SUMMARY_FILENAME = "regression_summary.csv"
CV = "loocv"                            # None disables CV; "all" runs LOOCV, 2-, 5-, 10-fold, and leave-all-in.
CV_INDEPENDENT_VAR = "Nifti_File_Path"  # Image column used to predict each current regressand in var_list.
DROP_NANS = True                        # Apply the same row filtering to regression and cross-validation.


def _load_analysis_dataframe(regressand, regressor, output_dir):
    """Load and filter the rows used by both regression and cross-validation."""
    cal_palm = CalvinStatsmodelsPalm(input_csv_path=INPUT_PATH, output_dir=output_dir, sheet=SHEET)
    data_df = cal_palm.read_and_display_data()

    if INVERT_REGRESSAND:
        print(f"INVERT_REGRESSAND=True, MULTIPLYING {regressand} BY -1")
        data_df[regressand] = data_df[regressand] * -1

    if DROP_NANS:
        drop_nan_list = [regressand, regressor] + list(COVARIATES_LIST)
        if CV and CV_INDEPENDENT_VAR and CV_INDEPENDENT_VAR in data_df.columns:
            drop_nan_list.append(CV_INDEPENDENT_VAR)
        drop_nan_list = list(dict.fromkeys(drop_nan_list))
        try:
            data_df = cal_palm.drop_nans_from_columns(columns_to_drop_from=drop_nan_list)
        except KeyError as exc:
            missing = exc.args[0] if exc.args else None
            missing = list(missing) if isinstance(missing, (list, tuple, np.ndarray, pd.Index)) else [missing]
            rhs_columns = [column.strip() for column in regressor.split("+") if column.strip()]

            # A full additive RHS is a formula expression, not a literal CSV column.
            # Catch only that exact KeyError and retry NaN filtering on its component columns.
            if missing != [regressor] or len(rhs_columns) < 2:
                raise
            absent_rhs_columns = [column for column in rhs_columns if column not in data_df.columns]
            if absent_rhs_columns:
                raise

            expanded_drop_nan_list = [regressand] + rhs_columns + list(COVARIATES_LIST)
            if CV and CV_INDEPENDENT_VAR and CV_INDEPENDENT_VAR in data_df.columns:
                expanded_drop_nan_list.append(CV_INDEPENDENT_VAR)
            expanded_drop_nan_list = list(dict.fromkeys(expanded_drop_nan_list))
            data_df = cal_palm.drop_nans_from_columns(
                columns_to_drop_from=expanded_drop_nan_list
            )

    if DROP_ROWS:
        for column, condition, value in DROP_ROWS:
            data_df, _ = cal_palm.drop_rows_based_on_value(column, condition, value)

    if ONE_HOT:
        for column in ONE_HOT:
            dummies = pd.get_dummies(data_df[column], prefix=column, dtype=int)
            data_df = data_df.join(dummies)

    return cal_palm, data_df


def _configured_cv_schemes(n_obs):
    """Return the requested CV scheme(s), omitting folds larger than the cohort."""
    if not CV:
        return []
    if str(CV).lower() == "all":
        schemes = ["loocv", 2, 5, 10, "leave_all_in"]
    else:
        try:
            schemes = [int(CV)]
        except (TypeError, ValueError):
            schemes = [str(CV).lower()]
    return [scheme for scheme in schemes if not isinstance(scheme, int) or scheme <= n_obs]


def _cv_outputs_exist(regression_dir, cv_scheme, n_contrasts):
    """Return True only when all scatterplots for one CV scheme are present."""
    scatter_dir = Path(regression_dir) / "cross_validations" / "scatterplots"
    label = str(cv_scheme)
    contrast_plots = all(
        any(scatter_dir.glob(f"{label}_contrast_{idx}_correlation*scatterplot.svg"))
        for idx in range(n_contrasts)
    )
    aggregate_plot = any(
        scatter_dir.glob(f"{label}_voxelwisemodel_aggregate-prediction*scatterplot.svg")
    )
    return contrast_plots and aggregate_plot


def _regression_outputs_exist(new_out_dir):
    """Return True when the regression already wrote its first contrast t-map."""
    regression_dir = Path(new_out_dir) / "regression"
    return any(regression_dir.glob("contrast_tval_0*"))


def plot_regression_results(new_out_dir, regression_dir):
    """Build the shared 3D NIfTI viewer and static contrast-map galleries."""
    figure_root = Path(new_out_dir) / "figures"
    nifti_candidates = sorted(
        path
        for path in Path(regression_dir).iterdir()
        if path.is_file()
        and (path.name.endswith(".nii") or path.name.endswith(".nii.gz"))
    )
    volume_files = []
    for path in nifti_candidates:
        try:
            shape = nib.load(str(path)).shape
        except Exception as exc:
            print(f"Skipping unreadable NIfTI result {path}: {exc}")
            continue
        if len(shape) != 3:
            print(f"Skipping volumetric viewer for non-3D result: {path}")
            continue
        volume_files.append(path)

    volume_viewer = None
    if volume_files:
        volume_viewer = circuit_viewer_orchestrator.write_volume_viewer(
            volume_files,
            figure_root / "volumetric_viewer.html",
            title="Regression volumetric results",
        )

    result_files = [
        path for path in volume_files if path.name.startswith("contrast_tval_")
    ]

    for result_file in result_files:
        out_dir = figure_root / circuit_viewer_orchestrator.nifti_stem(result_file)

        if not FIGURES_IN_SUBPROCESS:
            circuit_viewer_orchestrator.dispatch(
                source_nifti=result_file,
                output_dir=out_dir,
                figures=circuit_viewer_orchestrator.FIGURES,
                make_html=circuit_viewer_orchestrator.MAKE_HTML,
                open_html=False,
                volume_viewer=volume_viewer,
            )
            continue

        # Every yabplot render opens an off-screen OpenGL context, and macOS
        # stops handing them out after a few hundred: a long batch dies with
        # pyvista's "Render window is not current" on whichever figure happens
        # to ask next. A fresh interpreter per map tears that GL state down
        # each time, so the batch length stops mattering.
        command = [
            sys.executable,
            str(Path(circuit_viewer_orchestrator.__file__).resolve()),
            "--nifti", str(result_file),
            "--output-dir", str(out_dir),
            "--no-open-html",
        ]
        if volume_viewer is not None:
            command.extend(["--volume-viewer", str(volume_viewer)])
        completed = subprocess.run(command)
        if completed.returncode != 0:
            print(
                f"[warn] figures failed for {result_file.name} "
                f"(exit {completed.returncode}); continuing with the batch."
            )

    if volume_viewer is not None:
        gallery_links = [
            (
                circuit_viewer_orchestrator.nifti_stem(result_file),
                figure_root / circuit_viewer_orchestrator.nifti_stem(result_file) / "index.html",
            )
            for result_file in result_files
        ]
        index_path = circuit_viewer_orchestrator.write_index(
            figure_root,
            volume_files[0],
            cards=[],
            volume_viewer=volume_viewer,
            title="Regression result maps",
            gallery_links=gallery_links,
        )
        print(f"Regression HTML viewer: {index_path}")

    return result_files


def _find_fwe_result_files(regression_dir, output_handler):
    """Find backend-native FWE t-statistic maps, including nested runs."""
    candidates = sorted(
        path
        for path in Path(regression_dir).rglob("contrast_tval_FWE_*")
        if path.is_file()
    )
    return [path for path in candidates if output_handler.is_native_map_file(path)]


def _max_abs_fwe_statistic(regression_dir, output_handler):
    """Return the largest absolute finite value across all native FWE t-maps."""
    result_files = _find_fwe_result_files(regression_dir, output_handler)
    maxima = []
    for result_file in result_files:
        values = output_handler.load_map_values(result_file)
        finite = values[np.isfinite(values)]
        maxima.append(float(np.max(np.abs(finite))) if finite.size else 0.0)
    maximum = max(maxima) if maxima else np.nan
    return maximum, result_files


def _loocv_performance(regression_dir):
    """Read contrast-0 LOOCV Spearman rho from the generated scatterplot."""
    scatter_dir = Path(regression_dir) / "cross_validations" / "scatterplots"
    plots = sorted(scatter_dir.glob("loocv_contrast_0_correlation*scatterplot.svg"))
    if not plots:
        return np.nan, None

    text = " ".join(plots[0].read_text(errors="ignore").split())
    match = re.search(
        r"Rho\s*=\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan)",
        text,
        flags=re.IGNORECASE,
    )
    return (float(match.group(1)) if match else np.nan), plots[0]


def _regression_metadata(dataset_json):
    """Return the saved regression payload, or an empty mapping if unavailable."""
    dataset_json = Path(dataset_json)
    if not dataset_json.is_file():
        return {}
    with dataset_json.open() as stream:
        return json.load(stream).get("neuroimaging_regression", {})


def write_regression_summary(regression_pairs):
    """Write one root-level summary row for every requested DV/IV regression."""
    rows = []
    for regressand, regressor in regression_pairs:
        analysis_dir = Path(OUT_DIR) / f"{regressand}-on-{regressor}"
        regression_dir = analysis_dir / "regression"
        metadata = _regression_metadata(analysis_dir / "dataset_dict.json")
        output_ftype = metadata.get("output_ftype", "")
        output_handler = (
            NeuroimageFileOutporter(
                output_ftype=output_ftype,
                mask_path=metadata.get("mask_path"),
            )
            if output_ftype
            else None
        )

        contrast_path = metadata.get("contrast_matrix", analysis_dir / "contrast_matrix.npy")
        contrast_path = Path(contrast_path)
        if not contrast_path.is_absolute():
            contrast_path = analysis_dir / contrast_path
        contrast_matrix = (
            json.dumps(np.load(contrast_path).tolist(), separators=(",", ":"))
            if contrast_path.is_file()
            else ""
        )

        loocv_value, loocv_plot = _loocv_performance(regression_dir)
        if output_handler is None:
            fwe_max, fwe_files = np.nan, []
        else:
            fwe_max, fwe_files = _max_abs_fwe_statistic(
                regression_dir,
                output_handler=output_handler,
            )
        rows.append(
            {
                "dependent_variable": regressand,
                "independent_variable": regressor,
                "regression_type": REGRESSION_TYPE,
                "data_transform": DATA_TRANSFORM_METHOD or "none",
                "loocv_performance": loocv_value,
                "loocv_metric": "Spearman rho (contrast 0)",
                "contrast_matrix": contrast_matrix,
                "fwe_max_abs_statistic": fwe_max,
                "fwe_has_nonzero_statistic": bool(np.isfinite(fwe_max) and fwe_max > 0),
                "output_ftype": output_ftype,
                "fwe_result_files": json.dumps([str(path) for path in fwe_files]),
                "loocv_plot": "" if loocv_plot is None else str(loocv_plot),
                "analysis_dir": str(analysis_dir),
            }
        )

    summary_path = Path(OUT_DIR) / SUMMARY_FILENAME
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = summary_path.with_suffix(summary_path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary_path, index=False)
    os.replace(temporary_path, summary_path)
    print(f"Regression summary: {summary_path}")
    return summary_path


def run_voxelwise_regression(regressand, regressor):
    """Orchestrates the regression"""
    FORMULA = f"{regressand} ~ {regressor}"
    if COVARIATES_LIST:
        FORMULA += " + " + " + ".join(COVARIATES_LIST)
    print("Running formula: ", FORMULA)
    
    NEW_OUT_DIR = os.path.join(OUT_DIR, f"{regressand}-on-{regressor}")
    if _regression_outputs_exist(NEW_OUT_DIR):    # Skip ones that already produced result maps.
        print(f"Skipping completed regression: {FORMULA}")
        return
    
    # =============================================================================
    # CONFIG 
    # =============================================================================

    RUN_PREDICTION = False
    ALL_OUTPUTS = False # For multi-output regression. Advanced. 

    # =============================================================================
    # PREPARE DATASET
    # ============================================================================
    
    os.makedirs(NEW_OUT_DIR, exist_ok=True)

    cal_palm, data_df = _load_analysis_dataframe(regressand, regressor, NEW_OUT_DIR)

    outcome_df, design_matrix_df = cal_palm.define_design_matrix(
        FORMULA,
        data_df,
        add_intercept=ADD_INTERCEPT,
        voxelwise_variable_list=VOXELWISE_VARS,
        voxelwise_interaction_terms=VOXELWISE_INTERACTIONS,
    )

    if CONTRAST_MATRIX is not None:
        contrast_matrix = CONTRAST_MATRIX
    else:
        contrast_matrix = cal_palm.generate_basic_contrast_matrix(design_matrix_df)

    contrast_matrix_df = cal_palm.finalize_contrast_matrix(
        design_matrix=design_matrix_df,
        contrast_matrix=contrast_matrix,
    )

    exchangeability_block = None
    if EXCHANGEABILITY_COL:
        exchangeability_block = pd.to_numeric(data_df[EXCHANGEABILITY_COL], errors="raise").astype(int).to_numpy()

    weights = None
    if WEIGHTS_COL:
        weights = pd.to_numeric(data_df[WEIGHTS_COL], errors="raise").astype(float).to_numpy()

    preparer = RegressionPrep(
        design_matrix=design_matrix_df,
        contrast_matrix=contrast_matrix_df,
        outcome_df=outcome_df,
        out_dir=NEW_OUT_DIR,
        voxelwise_variables=VOXELWISE_VARS,
        voxelwise_interactions=VOXELWISE_INTERACTIONS,
        mask_path=MASK_PATH,
        exchangeability_block=exchangeability_block,
        data_transform_method=DATA_TRANSFORM_METHOD,
        weights=weights,
        formula=FORMULA,
    )
    _, json_path = preparer.run()

    # =============================================================================
    # RUN REGRESSION
    # =============================================================================
    REGRESSION_DIR = os.path.join(NEW_OUT_DIR, 'regression')
    os.makedirs(REGRESSION_DIR, exist_ok=True)

    regression = VoxelwiseRegression(
        json_path=json_path,
        # Use the backend-resolved mask/atlas stored by RegressionPrep.
        mask_path=None,
        out_dir=REGRESSION_DIR,
        regression_type=REGRESSION_TYPE,
        n_permutations=N_PERMUTATIONS,
    )

    if ALL_OUTPUTS:
        regression.run_all_outputs()
    else:
        regression.run()

    # =============================================================================
    # RUN PREDICTION
    # =============================================================================

    if RUN_PREDICTION:
        PREDICTION_OUT_DIR = os.path.join(NEW_OUT_DIR, 'predictions')
        os.makedirs(PREDICTION_OUT_DIR, exist_ok=True)

        prediction_regression = VoxelwiseRegression(
            json_path,
            mask_path=None,
            out_dir=PREDICTION_OUT_DIR,
            regression_type=REGRESSION_TYPE,
            n_permutations=0,
        )
        predictions = prediction_regression._run_prediction_switch(temp_dir=REGRESSION_DIR)
        prediction_regression.PREDICTIONS = np.asarray(predictions, dtype=np.float32)
        prediction_regression._save_result_maps()
		        
    # =============================================================================
    # RUN PLOTTING
    # =============================================================================

    if RUN_FIGURES:
        plot_regression_results(NEW_OUT_DIR, REGRESSION_DIR)


def run_cross_validation_for_regression(regressand, regressor):
    """Run the configured CV scheme(s) for one completed regression."""
    new_out_dir = os.path.join(OUT_DIR, f"{regressand}-on-{regressor}")
    regression_dir = os.path.join(new_out_dir, "regression")
    json_path = os.path.join(new_out_dir, "dataset_dict.json")

    if not os.path.isfile(json_path):
        raise FileNotFoundError(
            f"Cannot cross-validate {regressand} ~ {regressor}: "
            f"regression dataset not found at {json_path}"
        )

    _, data_df = _load_analysis_dataframe(regressand, regressor, new_out_dir)
    cv_independent_var = CV_INDEPENDENT_VAR or regressor
    if cv_independent_var not in data_df.columns:
        raise KeyError(
            f"Cross-validation image column {cv_independent_var!r} is not present in {INPUT_PATH}"
        )

    y_true = pd.to_numeric(data_df[regressand], errors="raise")
    subject_files = data_df[cv_independent_var].astype(str)
    missing_files = [path for path in subject_files if not os.path.isfile(path)]
    if missing_files:
        preview = "\n".join(missing_files[:5])
        raise FileNotFoundError(
            f"Cross-validation for {regressand} has {len(missing_files)} missing image file(s). "
            f"First missing path(s):\n{preview}"
        )

    regression = VoxelwiseRegression(
        json_path=json_path,
        mask_path=None,
        out_dir=regression_dir,
        regression_type=REGRESSION_TYPE,
        n_permutations=0,
    )
    if len(data_df) != regression.n_obs:
        raise ValueError(
            f"Cross-validation row mismatch for {regressand}: reconstructed {len(data_df)} rows, "
            f"but the saved regression contains {regression.n_obs}."
        )
    if regression.n_obs < 2:
        raise ValueError(f"Cross-validation for {regressand} requires at least two observations.")

    for cv_scheme in _configured_cv_schemes(regression.n_obs):
        if _cv_outputs_exist(regression_dir, cv_scheme, regression.n_contrasts):
            print(f"Skipping completed {cv_scheme} CV: {regressand} ~ {cv_independent_var}")
            continue
        print(f"Running {cv_scheme} CV: {regressand} ~ {cv_independent_var}")
        regression.run_cross_validation(
            y_true=y_true,
            subject_files=subject_files,
            cv=cv_scheme,
        )


# =============================================================================
# LOOP OVER REGRESSION PAIRS
# =============================================================================

def main():
    regression_pairs = list(product(regressand_list, regressor_list))

    for regressand, regressor in regression_pairs:
        run_voxelwise_regression(regressand, regressor)

    if CV:
        print("Running cross-validation for every dependent variable.")
        for regressand, regressor in regression_pairs:
            run_cross_validation_for_regression(regressand, regressor)

    write_regression_summary(regression_pairs)


if __name__ == "__main__":
    main()
