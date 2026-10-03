#!/usr/bin/env python3
"""Edit-at-the-top launcher for network specificity analysis."""

from __future__ import annotations

import os
import sys
from pathlib import Path


# Redirect plotting caches before importing the analysis utility.
TMPDIR = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(TMPDIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMPDIR / "xdg_cache"))

# Ensure circuit_pyper is importable when this file is run directly.
CIRCUIT_PYPER_DIR = Path(__file__).resolve().parents[1]
if str(CIRCUIT_PYPER_DIR) not in sys.path:
    sys.path.insert(0, str(CIRCUIT_PYPER_DIR))

from calvin_utils.neuroimaging_utils.ccm_utils.symptom_specificity import (
    NetworkSpecificityAnalysis,
)


# =============================================================================
# CONFIG
# =============================================================================


# Input/output paths.
OUT_DIR = "/Users/cu135/Partners HealthCare Dropbox/Calvin Howard/studies/raynor_network_mapping/results/schmahmann_discovery_clustering_mapping/individual_symptom_specificity/TotalBarsScore"
INPUT_PATH = "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimzation/optimized_master_list.csv"
SHEET = None  # Specify a sheet name for Excel input; use None for CSV.
FILE_COLUMN = "selected_Nifti_File_Path"
MASK_PATH = str(
    CIRCUIT_PYPER_DIR / "resources" / "MNI152_T1_2mm_brain_mask.nii"
)

# Maps whose subject-level damage/similarity scores are tested for specificity.
TARGET_MAPS_DIRECTORY = "/Users/cu135/Partners HealthCare Dropbox/Calvin Howard/studies/raynor_network_mapping/results/schmahmann_discovery_clustering_mapping/all_columns_no_perm_palm/regression_4"
TARGET_MAP_PATTERN = "contrast_tval_0.nii.gz"

# Spreadsheet preparation. These reproduce the notebook defaults.
DROP_NAN_COLUMNS = [
    "TotalBarsScore",
    "TotalCCASFailScore",
    "AffectTotal",
    FILE_COLUMN,
]
KEEP_ROWS = []  # Example: [("focal_cerebellum", 1)]
DROP_ROWS = []  # Example: [("focal_cerebellum", "not", 1)]
PATH_REPLACEMENTS = []  # Example: [("/old/root", "/new/root")]
FILLNA_VALUE = 0

# Each outcome used by the analysis must map to its specificity domain.
LABEL_DICT = {
    # Motor
    "Gait": "Motor",
    "HeelToShinTestLeft": "Motor",
    "HeelToShinTestRight": "Motor",
    "FingerToNoseTestLeft": "Motor",
    "FingerToNoseTestRight": "Motor",
    "LimbAtaxia": "Motor",
    "Speech": "Motor",
    "Oculomotor": "Motor",

    # Cognitive
    "SematicFluencyFailS": "Cognitive",
    "PhonemicFluencyFailS": "Cognitive",
    "CategorySwitchFailS": "Cognitive",
    "DigitSpanForwardFailS": "Cognitive",
    "DigitSpanBackwardFailS": "Cognitive",
    "CubeDrawFailS": "Cognitive",
    "VerbalRecallFailS": "Cognitive",
    "SimiliarityFailS": "Cognitive",
    "GoNoGoFailS": "Cognitive",

    # Emotional
    "Sec1ADifficultFocus": "Emotional",
    "Sec1AEasilyDistracted": "Emotional",
    "Sec1AOntheGo": "Emotional",
    "Sec1AFeelsCompelled": "Emotional",
    "Sec1AFeelsDriven": "Emotional",
    "Sec1BWorries": "Emotional",
    "Sec1BRepeats": "Emotional",
    "Sec1BMentallyStuck": "Emotional",
    "Sec1BCauseDistress": "Emotional",
    "Sec3ARepetitiveMovements": "Emotional",
    "Sec3ASensoryExp": "Emotional",
    "Sec3BSensitive": "Emotional",
    "Sec3BOverwhelmed": "Emotional",
    "Sec4ACommunicates": "Emotional",
    "Sec4AConcerns": "Emotional",
    "Sec4ASeesHearsThings": "Emotional",
    "Sec4BTroubleUnderstand": "Emotional",
    "Sec4BDistant": "Emotional",
    "Sec4BIndifferent": "Emotional",
    "Sec2AActHastily": "Emotional",
    "Sec2ARapidChanges": "Emotional",
    "Sec2ACryingLaughing": "Emotional",
    "Sec2AOverAnxious": "Emotional",
    "Sec2BLackOfPleasure": "Emotional",
    "Sec2BNegativeAttitude": "Emotional",
    "Sec2BUneasyWithLife": "Emotional",
    "Sec2BSadDepressed": "Emotional",
    "Sec5AAngry": "Emotional",
    "Sec5AUpset": "Emotional",
    "Sec5AIntolerant": "Emotional",
    "Sec5AArgumentative": "Emotional",
    "Sec5Bimmature": "Emotional",
    "Sec5BUnaware": "Emotional",
    "Sec5BManner": "Emotional",
    "Sec5BTrusting": "Emotional",
}

# Specificity settings.
DAMAGE_METRIC = "cosine"
CORRELATION = "spearman"
RESAMPLE_METHOD = "permutation"  # "permutation" or "bootstrap"
N_RESAMPLES = 1000
VECTORIZED_CORRELATIONS = False
ABSOLUTE_CORRELATIONS = False
SORT_WITHIN_LABELS = False
SMOOTH_BY_LABEL = True

# Optional target-domain-vs-other-domains comparison from the notebook.
TARGET_LABELS = "Cognitive"
OTHER_LABELS = None  # None pools every domain not in TARGET_LABELS.
TARGET_NAME = "Cognitive"
OTHER_NAME = "Other"
DRAW_COMPARISON_PLOT = True
COMPARISON_YLABEL = "Correlation (Spearman)"


def main():
    analysis = NetworkSpecificityAnalysis(
        input_path=INPUT_PATH,
        out_dir=OUT_DIR,
        sheet=SHEET,
        file_column=FILE_COLUMN,
        target_maps_directory=TARGET_MAPS_DIRECTORY,
        target_map_pattern=TARGET_MAP_PATTERN,
        mask_path=MASK_PATH,
        label_dict=LABEL_DICT,
        required_columns=DROP_NAN_COLUMNS,
        keep_rows=KEEP_ROWS,
        drop_rows=DROP_ROWS,
        path_replacements=PATH_REPLACEMENTS,
        damage_metric=DAMAGE_METRIC,
        correlation=CORRELATION,
        method=RESAMPLE_METHOD,
        vectorize=VECTORIZED_CORRELATIONS,
        absval=ABSOLUTE_CORRELATIONS,
        n_resamples=N_RESAMPLES,
        sort_within_labels=SORT_WITHIN_LABELS,
        smooth_by_label=SMOOTH_BY_LABEL,
        fillna_value=FILLNA_VALUE,
        target_labels=TARGET_LABELS,
        other_labels=OTHER_LABELS,
        target_name=TARGET_NAME,
        other_name=OTHER_NAME,
        draw_comparison_plot=DRAW_COMPARISON_PLOT,
        comparison_ylabel=COMPARISON_YLABEL,
    )
    return analysis.run()


if __name__ == "__main__":
    main()
