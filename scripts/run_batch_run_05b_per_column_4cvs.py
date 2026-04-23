#!/usr/bin/env python3
"""
Hard-coded 4-pass runner for batch 05b per column.

Runs `batch_run_05b_per_column.py` four times:
  - loocv
  - 2
  - 5
  - 10

Each run writes to:
  /Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimzation/batch_cv_{cv}
"""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    batch_script = "/Users/cu135/Software_Local/calvin_utils_project/circuit_pyper/scripts/batch_run_05b_per_column.py"
    input_csv = "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/master_list_redo.csv"
    mask_path = "/Users/cu135/Software_Local/calvin_utils_project/circuit_pyper/resources/MNI152_T1_2mm_brain_mask.nii"
    paths_col = "onetouch_path"

    filter_col = "selected"
    filter_value = "1"

    # Keep identical columns list across CV runs.
    columns = [
        "Gait",
        "HeelToShinTestLeft",
        "HeelToShinTestRight",
        "FingerToNoseTestLeft",
        "FingerToNoseTestRight",
        "LimbAtaxia",
        "Speech",
        "Oculomotor",
        "TotalBarsScore",
        "delta_days_bars",
        "delta_days_ccas_x",
        "SematicFluencyRawS",
        "SematicFluencyFailS",
        "PhonemicFluencyRawS",
        "PhonemicFluencyFailS",
        "CategorySwitchRawS",
        "CategorySwitchFailS",
        "VerbalRegSum",
        "DigitSpanForwardRawS",
        "DigitSpanForwardFailS",
        "DigitSpanBackwardRawS",
        "DigitSpanBackwardFailS",
        "CubeDrawRawS",
        "CubeDrawFailS",
        "VerbalRecallRawS",
        "VerbalRecallFailS",
        "SimiliarityPair1RawS",
        "SimiliarityPair2RawS",
        "SimiliarityPair3RawS",
        "SimiliarityPair4RawS",
        "SimiliarityRawS",
        "SimiliarityFailS",
        "GoNoGoRawS",
        "GoNoGoFailS",
        "AFFECTAssessments_Angryoraggress",
        "AFFECTAssessments_Difficultywith",
        "AFFECTAssessments_Emotionallylab",
        "AFFECTAssessments_Expressesillog",
        "AFFECTAssessments_Lacksempathyis",
        "AFFECTAssessments_Showseasysenso",
        "AffectRawS",
        "AffectFailS",
        "TotalCCASRawScore",
        "TotalCCASFailScore",
        "MemoryTotal",
        "ExecutiveTotal",
        "LanguageTotal",
        "VisualTotal",
        "AffectTotal",
        "delta_days_ccas_y",
        "Sec1ADifficultFocus",
        "Sec1AEasilyDistracted",
        "Sec1AOntheGo",
        "Sec1AFeelsCompelled",
        "Sec1AFeelsDriven",
        "Sec1BWorries",
        "Sec1BRepeats",
        "Sec1BMentallyStuck",
        "Sec1BCauseDistress",
        "Sec2AActHastily",
        "Sec2ARapidChanges",
        "Sec2ACryingLaughing",
        "Sec2AOverAnxious",
        "Sec2BLackOfPleasure",
        "Sec2BNegativeAttitude",
        "Sec2BUneasyWithLife",
        "Sec2BSadDepressed",
        "Sec3ARepetitiveMovements",
        "Sec3ASensoryExp",
        "Sec3BSensitive",
        "Sec3BOverwhelmed",
        "Sec4ACommunicates",
        "Sec4AConcerns",
        "Sec4ASeesHearsThings",
        "Sec4BTroubleUnderstand",
        "Sec4BDistant",
        "Sec4BIndifferent",
        "Sec5AAngry",
        "Sec5AUpset",
        "Sec5AIntolerant",
        "Sec5AArgumentative",
        "Sec5Bimmature",
        "Sec5BUnaware",
        "Sec5BManner",
        "Sec5BTrusting",
        "ScoreCol1A",
        "ScoreCol1B",
        "ScoreCol2A",
        "ScoreCol2B",
        "ScoreCol3A",
        "ScoreCol3B",
        "ScoreCol4A",
        "ScoreCol4B",
        "ScoreCol5A",
        "ScoreCol5B",
        "TotalSection1Score",
        "TotalSection2Score",
        "TotalSection3Score",
        "TotalSection4Score",
        "TotalSection5Score",
        "CNRSTotColAScore",
        "CNRSTotColBScore",
        "CNRSTotScore",
    ]

    n_permutations = "2"

    common = [
        sys.executable,
        batch_script,
        "--input-csv",
        input_csv,
        "--mask-path",
        mask_path,
        "--paths-col",
        paths_col,
        "--filter-col",
        filter_col,
        "--filter-value",
        filter_value,
        "--summary-csv",
        "batch_05b_summary.csv",
        "--n-permutations",
        n_permutations,
        "--skip-missing-paths",
        "--overwrite",
        "--continue-on-error",
        "--columns",
        *columns,
        "--verbose",
    ]

    # 1) loocv
    subprocess.run(
        [
            *common,
            "--out-root",
            "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimzation/batch_cv_loocv",
            "--cv",
            "loocv",
        ],
        check=True,
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

