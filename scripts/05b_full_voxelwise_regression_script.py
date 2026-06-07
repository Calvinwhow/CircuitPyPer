#!/usr/bin/env python3
"""
Plain, edit-the-variables-at-the-top runner for full voxelwise regression.

This is intentionally not a CLI. Change the values in the CONFIG section, then
run this file directly.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

TMPDIR = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(TMPDIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMPDIR / "xdg_cache"))

import numpy as np
import pandas as pd
from itertools import product

CIRCUIT_PYPER_DIR = Path(__file__).resolve().parents[1]
if str(CIRCUIT_PYPER_DIR) not in sys.path:
    sys.path.insert(0, str(CIRCUIT_PYPER_DIR))

from calvin_utils.permutation_analysis_utils.statsmodels_palm import CalvinStatsmodelsPalm
from calvin_utils.permutation_analysis_utils.voxelwise_regression import VoxelwiseRegression
from calvin_utils.permutation_analysis_utils.voxelwise_regression_prep import RegressionPrep
from calvin_utils.neuroimaging_utils.nifti_utils.cerebellum_plot import SUITCerebellumPlotter
from calvin_utils.plotting_utils.parcelwise_plot import ParcelwisePlot
from calvin_utils.permutation_analysis_utils import ParcelwiseDamageMap


# =============================================================================
# CONFIG
# =============================================================================


# Input/output paths.
INPUT_PATH = "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimzation/optimized_master_list.csv" # Form: "/path/to/input.csv"
SHEET = None # Specify sheet if using excel (i.e. "Sheet1")
OUT_DIR = "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimzation/symptom_on_lhs/proper_regressions" # Form: "/path/to/output_dir"
MASK_PATH = "/Users/cu135/Software_Local/calvin_utils_project/circuit_pyper/resources/MNI152_T1_2mm_brain_mask.nii"

# Model setup.

# var_list = [ 'SematicFluencyRawS', 'PhonemicFluencyRawS', 'CategorySwitchRawS', 'VerbalRegSum', 
#             'DigitSpanForwardRawS', 'DigitSpanBackwardRawS', 'CubeDrawRawS', 'VerbalRecallRawS', 'SimiliarityRawS', 'GoNoGoRawS', ]
var_list = ['Gait', 'HeelToShinTestLeft', 'HeelToShinTestRight', 'FingerToNoseTestLeft', 'FingerToNoseTestRight', 'LimbAtaxia', 'Speech', 'Oculomotor', 'TotalBarsScore',
 
 'SematicFluencyFailS', 'PhonemicFluencyFailS', 'CategorySwitchFailS', 'DigitSpanForwardFailS', 'DigitSpanBackwardFailS', 'CubeDrawFailS', 'VerbalRecallFailS', 'SimiliarityFailS', 'GoNoGoFailS', 'AffectFailS', 'TotalCCASFailScore', 


 'Sec1ADifficultFocus', 'Sec1AEasilyDistracted', 'Sec1AOntheGo', 'Sec1AFeelsCompelled', 'Sec1AFeelsDriven', 'Sec1BWorries', 'Sec1BRepeats', 'Sec1BMentallyStuck', 
 'Sec1BCauseDistress', 'Sec2AActHastily', 'Sec2ARapidChanges', 'Sec2ACryingLaughing', 'Sec2AOverAnxious', 
 'Sec2BLackOfPleasure', 'Sec2BNegativeAttitude', 'Sec2BUneasyWithLife', 'Sec2BSadDepressed', 
 'Sec3ARepetitiveMovements', 'Sec3ASensoryExp', 'Sec3BSensitive', 'Sec3BOverwhelmed', 
 'Sec4ACommunicates', 'Sec4AConcerns', 'Sec4ASeesHearsThings', 
 'Sec4BTroubleUnderstand', 'Sec4BDistant', 'Sec4BIndifferent', 
 'Sec5AAngry', 'Sec5AUpset', 'Sec5AIntolerant', 'Sec5AArgumentative', 
 'Sec5Bimmature', 'Sec5BUnaware', 'Sec5BManner', 'Sec5BTrusting', 
 'ScoreCol1A', 'ScoreCol1B', 
 'ScoreCol2A', 'ScoreCol2B', 
 'ScoreCol3A', 'ScoreCol3B', 
 'ScoreCol4A', 'ScoreCol4B', 
 'ScoreCol5A', 'ScoreCol5B', 
 'TotalSection1Score', 'TotalSection2Score', 'TotalSection3Score', 'TotalSection4Score', 'TotalSection5Score', 'CNRSTotColAScore', 'CNRSTotColBScore', 'CNRSTotScore', 
 ]
regressand_list = var_list              # On left hand=side of the equation. Often is the outcome variable.         Will run an analysis for each value.
regressor_list = ["Nifti_File_Path"]    # On right hand-side of the equation. Often is the neuroimaging variable.   Will run an analysis for each value. 
VOXELWISE_VARS = ["Nifti_File_Path"]    # Name the variables that are stored in neuroimaging files
VOXELWISE_INTERACTIONS = []             # If you want interactions, specify them. 
COVARIATES_LIST = []                    # List of all nuisance variables to adjust for. If you want interactions, make them in your spreadsheet and add them here. Will NOT trigger a new analysis for each value, but will be present in every analysis.
ADD_INTERCEPT = False                   # Not needed if DATA_TRANSFORM_METHOD = "standardize"

# Optional preprocessing.
DROP_ROWS = [("selected", "equal", 0)]                        # Conditions for droppping some rows. Example: DROP_ROWS = [("group", "equal", "control"), ("age", "below", 18)]
ONE_HOT = None                          # columns to one-hot encode
EXCHANGEABILITY_COL = None              # Exchangeability blocks to restrict permutations within
WEIGHTS_COL = None                      # Weights for weighted regression. Defaults to equal. 
DATA_TRANSFORM_METHOD = "standardize"   # Generally default to standardize with no intercept. Options: 'standardize' | 'rank' | None
INVERT_REGRESSAND = False                # Multiply regressand by -1. Default: False

# Leave empty to use the default basic contrast matrix.
CONTRAST_MATRIX = None                  # Example: [[ 0, 1], [-1, 1]]

# Regression settings.
REGRESSION_TYPE = "linear"              # Default linear
N_PERMUTATIONS = 1000                   # Default 1000
CV = "loocv"                            # Default loocv
CV_DEPENDENT_VAR = None                 # Must specify. This is your outcome variable that ana anlysis should be related to. i.e. gait outcomes. 
CV_INDEPENDENT_VAR = "Nifti_File_Path"  # Must specify. This is your neuroimaging variable. For example, if you made a regresison relating gait to connectivity, this should be the connectivity column. 

# Plot Options
PLOT_CMAP = "Blues"                      # Matplotlib cmaps
PLOT_CSCALE = [0,1]                     # Defines the bounds of the cmap
ATLAS_THRESHOLD = 0                     # NaN threshold applied before automatic parcel/tract scoring calls ``DamageScorer._calculate_metrics``  threshold : float | tuple[float, float] | None | Optional.  
CEREBELLUM_ATLAS = "SUIT"
CORTEX_ATLAS = "aal3"                   # Defined within yabplot documentation. 
SUBCORTEX_ATLAS = "aal3"                # Defined within yabplot documentation.
TRACT_ATLAS = "xtract_large"            # Defined within yabplot documentation.
VIEWS = ["right_lateral", "left_lateral", "anterior", "right_medial", "left_medial", "posterior"]
RUN_TRACT_FIGURES = True


def run_voxelwise_regression(regressand, regressor):
    """Orchestrates the regression"""
    FORMULA = f"{regressand} ~ {regressor}"
    if COVARIATES_LIST:
        FORMULA += " + " + " + ".join(COVARIATES_LIST)
    print("Running formula: ", FORMULA)
    
    NEW_OUT_DIR = os.path.join(OUT_DIR, f"{regressand}-on-{regressor}")
    if os.path.isdir(NEW_OUT_DIR):    # Skip ones that have already run. 
        return
    
    # =============================================================================
    # CONFIG 
    # =============================================================================

    RUN_PREDICTION = False
    RUN_PARCELLATION = True
    RUN_FIGURES = True
    DROP_NANS = True
    ALL_OUTPUTS = False # For multi-output regression. Advanced. 

    # =============================================================================
    # PREPARE DATASET
    # ============================================================================
    
    os.makedirs(NEW_OUT_DIR, exist_ok=True)

    cal_palm = CalvinStatsmodelsPalm(input_csv_path=INPUT_PATH, output_dir=NEW_OUT_DIR, sheet=SHEET)
    data_df = cal_palm.read_and_display_data()

    if INVERT_REGRESSAND:
        print(f"INVERT_REGRESSAND=True, MULTIPLYING {regressand} BY -1")
        data_df[regressand] = data_df[regressand]*-1
    
    if DROP_NANS:
        drop_nan_list = [regressand, regressor] + list(COVARIATES_LIST)
        data_df = cal_palm.drop_nans_from_columns(columns_to_drop_from=drop_nan_list)

    if DROP_ROWS:
        for column, condition, value in DROP_ROWS:
            data_df, _ = cal_palm.drop_rows_based_on_value(column, condition, value)

    if ONE_HOT:
        for column in ONE_HOT:
            dummies = pd.get_dummies(data_df[column], prefix=column, dtype=int)
            data_df = data_df.join(dummies)

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
        mask_path=MASK_PATH,
        out_dir=REGRESSION_DIR,
        regression_type=REGRESSION_TYPE,
        n_permutations=N_PERMUTATIONS,
    )

    if ALL_OUTPUTS:
        regression.run_all_outputs()
    else:
        regression.run()

    if CV:
        cv_dependent_var = CV_DEPENDENT_VAR or regressand
        cv_independent_var = CV_INDEPENDENT_VAR or regressor
        if str(CV).lower() == "all":
            regression.run_cross_validation(
                y_true=data_df[cv_dependent_var],
                subject_files=data_df[cv_independent_var],
            )
        else:
            try:
                cv_arg = int(CV)
            except Exception:
                cv_arg = str(CV).lower()
            regression._evaluate_map(
                y_true=data_df[cv_dependent_var],
                subject_files=data_df[cv_independent_var],
                cv=cv_arg,
            )

    # =============================================================================
    # RUN PREDICTION
    # =============================================================================

    if RUN_PREDICTION:
        PREDICTION_OUT_DIR = os.path.join(NEW_OUT_DIR, 'predictions')
        os.makedirs(PREDICTION_OUT_DIR, exist_ok=True)

        prediction_regression = VoxelwiseRegression(
            json_path,
            mask_path=MASK_PATH,
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

    def _file_yielder():
        for file in os.listdir(REGRESSION_DIR):
            if "contrast_tval_" in file:        # only interested in automatically plotting the contrasts
                fname = os.path.basename(file).split('.nii')[0]
                yield os.path.join(REGRESSION_DIR, file), fname
    
    if RUN_FIGURES:
        FIGURE_DIR = os.path.join(NEW_OUT_DIR, 'figures')
        os.makedirs(FIGURE_DIR, exist_ok=True)  
        for file, fname in _file_yielder():
            # Cerebellum Plots
            plotter = SUITCerebellumPlotter(file)
            plotter.run(
                space="MNI",
                out_file=os.path.join(FIGURE_DIR, fname + '-cerebellum.svg'),
                cmap=PLOT_CMAP,
                colorbar=True,
                cscale = PLOT_CSCALE,
                threshold = None,
            )
		            
            # Cortex Plots
            plotter = ParcelwisePlot(map_path=file, out_file=os.path.join(FIGURE_DIR, fname + f'-cortex-{CORTEX_ATLAS}.svg'))
            ax = plotter.run(
                project='vol2surf',
                bmesh="midthickness",
                plot="cortical",
                atlas=CORTEX_ATLAS,
	                plot_kwargs={
	                    "views": VIEWS,
	                    "style": "default",
	                    "display_type": "matplotlib",
	                    "cmap": PLOT_CMAP,
	                    "vminmax": (PLOT_CSCALE[0], PLOT_CSCALE[1])
	                },
                threshold=ATLAS_THRESHOLD,
                score_nonzero_only=True,
            )
		            
            # Subcortex Plots
            plotter = ParcelwisePlot(map_path=file, out_file=os.path.join(FIGURE_DIR, fname + f'-subcortex-{SUBCORTEX_ATLAS}.svg'))
            ax = plotter.run(
                project=None,
                bmesh="midthickness",
                plot="subcortical",
                atlas=SUBCORTEX_ATLAS,
	                plot_kwargs={
	                    "views": VIEWS,
	                    "style": "default",
	                    "display_type": "matplotlib",
	                    "cmap": PLOT_CMAP,
	                    "vminmax": (PLOT_CSCALE[0], PLOT_CSCALE[1])
	                },
                threshold=ATLAS_THRESHOLD,
                score_nonzero_only=True,
            )
		            
            if RUN_TRACT_FIGURES:
                # Tract Plots
                plotter = ParcelwisePlot(map_path=file, out_file=os.path.join(FIGURE_DIR, fname + f'-tracts-{TRACT_ATLAS}.svg'))
                ax = plotter.run(
                    project="vol2tract",
                    bmesh="midthickness",
                    plot="tracts",
                    atlas=TRACT_ATLAS,
                    plot_kwargs={
                        "views": VIEWS,
                        "style": "default",
                        "display_type": "matplotlib",
                        "cmap": PLOT_CMAP,
                        "vminmax": (PLOT_CSCALE[0], PLOT_CSCALE[1])
                    },
                    threshold=ATLAS_THRESHOLD,
                    score_nonzero_only=True,
                )
		            
            # Embedded loop to use parcellation to clarify plots
            if RUN_PARCELLATION:
                PARCEL_DIR = os.path.join(NEW_OUT_DIR, "parcellated_files")
                os.makedirs(PARCEL_DIR, exist_ok=True)
                suit_dir = "/Volumes/HowExp/resources/atlases/mni_space/SUIT_cerebellar_atlases/Diedrichsen_2009/atl-Anatom_space-SUIT_dseg_rois"
                outname = f"{fname}_diedrichsen-wholebrain"
                if os.path.isdir(suit_dir):
                    ParcelwiseDamageMap(
                        target_map=file,
                        parcel_path=os.path.join(suit_dir,'*.nii.gz'),
                        mask_path=MASK_PATH,
                        out_dir=PARCEL_DIR,
                        output_name=outname,
                        selected_damage="avg_in_target",
                        score_nonzero_only=True,
                    ).run()
		                    
                    plotter = SUITCerebellumPlotter(os.path.join(PARCEL_DIR, outname+".nii.gz"))
                    plotter.run(
                        space="MNI",
                        out_file=os.path.join(FIGURE_DIR, fname + '-cerebellum-SUIT.svg'),
                        cmap=PLOT_CMAP,
                        colorbar=True,
                        cscale = PLOT_CSCALE,
                        threshold = None
                    )
	            
	        
# =============================================================================
# LOOP OVER REGRESSION PAIRS
# =============================================================================

def main():
    for regressand, regressor in product(regressand_list, regressor_list):
        run_voxelwise_regression(regressand, regressor)


if __name__ == "__main__":
    main()
