#!/usr/bin/env python3
"""Edit-at-the-top launcher for network specificity analysis."""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd


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
    SpecificityAnalysisResult,
    SpecificityAnalyzer,
)
from calvin_utils.plotting_utils.simple_box_plot import SimpleBoxPlotWrapper


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
    "SematicFluencyRawS": "Cognitive",
    "PhonemicFluencyRawS": "Cognitive",
    "CategorySwitchRawS": "Cognitive",
    "VerbalRegSum": "Cognitive",
    "DigitSpanForwardRawS": "Cognitive",
    "DigitSpanBackwardRawS": "Cognitive",
    "CubeDrawRawS": "Cognitive",
    "VerbalRecallRawS": "Cognitive",
    "SimiliarityRawS": "Cognitive",
    "GoNoGoRawS": "Cognitive",

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
    "AffectRawS": "Emotional",
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


def run_precomputed_specificity(
    *,
    data_df,
    damage_df,
    out_dir,
    target_label,
    label_dict=None,
    subject_column=FILE_COLUMN,
    correlation=CORRELATION,
    method=RESAMPLE_METHOD,
    n_resamples=N_RESAMPLES,
    vectorize=VECTORIZED_CORRELATIONS,
    absval=ABSOLUTE_CORRELATIONS,
    sort_within_labels=SORT_WITHIN_LABELS,
    smooth_by_label=SMOOTH_BY_LABEL,
    random_seed=2026,
):
    """Run the standard specificity workflow on leakage-free supplied scores.

    ``damage_df`` must contain one row per row of ``data_df``. This entry point
    is for predictions such as LOOCV scores that cannot be reconstructed from
    one fixed target map by :class:`NetworkSpecificityAnalysis`.
    """
    data_df = pd.DataFrame(data_df).reset_index(drop=True).copy()
    damage_df = pd.DataFrame(damage_df).reset_index(drop=True).copy()
    labels = dict(LABEL_DICT if label_dict is None else label_dict)
    if len(data_df) != len(damage_df):
        raise ValueError("Precomputed damage scores and clinical rows must align.")
    if damage_df.empty or not all(
        pd.api.types.is_numeric_dtype(damage_df[column])
        for column in damage_df.columns
    ):
        raise ValueError("Precomputed damage-score columns must be numeric.")
    missing_outcomes = set(labels) - set(data_df)
    if missing_outcomes:
        raise ValueError(
            f"Clinical table is missing specificity outcomes: {sorted(missing_outcomes)}"
        )
    if target_label not in set(labels.values()):
        raise ValueError(f"Unknown target specificity label: {target_label!r}.")

    analysis_dir = Path(out_dir).expanduser() / method
    analysis_dir.mkdir(parents=True, exist_ok=True)
    data_df = data_df.fillna(FILLNA_VALUE)
    damage_df = damage_df.fillna(FILLNA_VALUE)
    data_df.to_csv(analysis_dir / "prepared_spreadsheet.csv", index=False)
    damage_output = damage_df.copy()
    if subject_column in data_df:
        damage_output.insert(0, "subject", data_df[subject_column].to_numpy())
    damage_output.to_csv(analysis_dir / "damage_scores.csv", index=False)

    np.random.seed(random_seed)
    analyzer = SpecificityAnalyzer(
        X=damage_df,
        Y=data_df,
        y_labels=labels,
        correlation=correlation,
        method=method,
        vectorize=vectorize,
        out_dir=str(analysis_dir),
        absval=absval,
    )
    analyzer.run(
        n_resamples=n_resamples,
        sort_within_labels=sort_within_labels,
        smooth_by_label=smooth_by_label,
    )
    observed = analyzer.extract_r_values_long()
    observed.to_csv(analysis_dir / "observed_correlations.csv", index=False)
    comparison = analyzer.extract_group_comparison_r_values(
        target_labels=target_label,
        other_labels=None,
        target_name=target_label,
        other_name="Other",
    )
    comparison.to_csv(analysis_dir / "group_comparison_r_values.csv", index=False)
    plot_columns = (
        [(target_label, "Other")]
        if len(analyzer.cols) == 1
        else [
            ((measurement, target_label), (measurement, "Other"))
            for measurement in analyzer.cols
        ]
    )
    SimpleBoxPlotWrapper(comparison).plot(
        columns=plot_columns,
        group_labels=(None if len(analyzer.cols) == 1 else list(analyzer.cols)),
        pair_names=[target_label, "Other"],
        dataset_name=" ",
        xlabel=" ",
        ylabel=f"Correlation ({correlation.title()})",
        figsize=(4, 6),
        out_dir=str(analysis_dir),
    )
    manifest = {
        "analysis": "precomputed leakage-free damage-score specificity",
        "n_subjects": len(data_df),
        "n_measurements": len(damage_df.columns),
        "n_correlations_per_measurement": len(labels),
        "domain_counts": pd.Series(labels).value_counts().to_dict(),
        "outcomes": list(labels),
        "target_label": target_label,
        "n_resamples": n_resamples,
        "random_seed": random_seed,
        "correlation": correlation,
    }
    (analysis_dir / "outcome_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return SpecificityAnalysisResult(
        prepared_data=data_df,
        damage_scores=damage_df,
        observed_correlations=observed,
        group_comparison=comparison,
        analyzer=analyzer,
    )


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
