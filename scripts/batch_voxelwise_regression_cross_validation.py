#!/usr/bin/env python3
"""
Simple wrapper around `05b_full_voxelwise_regression.py` for batch screening.

Behavior
--------
For each predictor column:
- Run 05b regression with:
  - `formula = "paths ~ <predictor>"`
  - `add_intercept=True`
  - `contrast = [[0, 1]]`
  - `n_permutations` (default 0)
  - `cv=2` (calls the same cross-validation method as the notebook)
- Save outputs into a per-predictor folder under `out_root`
- Append performance metrics to `out_csv`

The goal is to avoid passing long CLI commands: edit the `selected = {...}` dict
below, then run:
`python circuit_pyper/scripts/batch_voxelwise_regression_cross_validation.py`

You can also import:
`from batch_voxelwise_regression_cross_validation import selected`
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


# Edit these defaults.
selected: dict[str, Any] = {
    "input_csv": "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/best_selection.csv",
    "mask_path": "/Users/cu135/Software_Local/calvin_utils_project/circuit_pyper/resources/MNI152_T1_2mm_brain_mask.nii",
    "paths_col": "onetouch_path",
    # Summary output CSV (one row per predictor).
    "out_csv": "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/batch_voxelwise_regression_cross_validation.csv",
    # Per-predictor output root; if None, defaults to out_csv without ".csv"
    "out_root": "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/xval_results",
    # Choose predictors explicitly or use all_numeric
    "columns": [
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
    "CNRSTotScore"
    ],
    "all_numeric": True,
    "exclude_cols": [
        "selected",
        "subid",
        "session",
        "id",
        "local_id",
        "paths",
        "onetouch_path"
    ],
    # Optional generic filter (keeps rows where df[filter_col] == filter_value)
    "filter_col": "selected",
    "filter_value": 1,
    "skip_missing_paths": True,
    "echo_05b": False,
    # 05b knobs
    "cv": 2,
    "n_permutations": 0,
    "data_transform": None,  # None|standardize|rank
    "overwrite": False,
    "verbose": True,
}


def _default_argv_from_selected() -> list[str]:
    cfg = selected
    out_csv = cfg["out_csv"]
    out_root = cfg.get("out_root") or str(Path(out_csv).with_suffix(""))
    argv = [
        "--input-csv",
        str(cfg["input_csv"]),
        "--mask-path",
        str(cfg["mask_path"]),
        "--paths-col",
        str(cfg["paths_col"]),
        "--out-root",
        str(out_root),
        "--summary-csv",
        str(out_csv),
        "--cv",
        str(cfg.get("cv", 2)),
        "--n-permutations",
        str(int(cfg.get("n_permutations", 0))),
    ]
    if cfg.get("data_transform") is not None:
        argv += ["--data-transform", str(cfg["data_transform"])]
    if cfg.get("skip_missing_paths", False):
        argv.append("--skip-missing-paths")
    if cfg.get("all_numeric", False):
        argv.append("--all-numeric")
        for c in cfg.get("exclude_cols", []):
            argv += ["--exclude-col", str(c)]
    else:
        cols = cfg.get("columns", [])
        if cols:
            argv += ["--columns", *[str(c) for c in cols]]
    if cfg.get("filter_col") is not None:
        argv += ["--filter-col", str(cfg["filter_col"]), "--filter-value", str(cfg.get("filter_value", ""))]
    if cfg.get("overwrite", False):
        argv.append("--overwrite")
    if cfg.get("verbose", True):
        argv.append("--verbose")
    return argv


def main() -> int:
    from batch_run_05b_per_column import main as run_batch

    argv = None if len(sys.argv) > 1 else _default_argv_from_selected()
    if argv is not None and selected.get("verbose", True):
        print(f"[wrapper] out_root={selected.get('out_root')}")
        print(f"[wrapper] out_csv={selected.get('out_csv')}")
    return int(run_batch(argv))


if __name__ == "__main__":
    raise SystemExit(main())
