#!/usr/bin/env python3
"""Run the map-comparison workflow with the Schmahmann draft configuration."""

import os
from pathlib import Path
import runpy
import sys

LOCAL_SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_SCRIPTS_DIR = Path(
    "/Users/cu135/Software_Local/calvin_utils_project/circuit_pyper/scripts"
)
SCRIPTS_DIR = Path(
    os.environ.get("CIRCUIT_PYPER_SCRIPTS_DIR", DEFAULT_SCRIPTS_DIR)
).expanduser()
if (LOCAL_SCRIPTS_DIR / "predict_outcomes_with_map_comparison.py").is_file():
    SCRIPTS_DIR = LOCAL_SCRIPTS_DIR
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import predict_outcomes_with_map_comparison as comparison


SOURCE_SCRIPT = SCRIPTS_DIR / "predict_outcomes_with_map-Schmahmann.py"
CANDIDATE_MAP_NAMES = {
    "Motor Map Optimal",
    "Cognitive Map Optimal",
    "Emotional Map Optimal",
}


def main():
    config = runpy.run_path(str(SOURCE_SCRIPT))
    maps = config["MAPS_TO_PREDICT"]

    comparison.INPUT_PATH = config["INPUT_PATH"]
    comparison.SHEET = config["SHEET"]
    comparison.OUT_DIR = str(Path(config["OUT_DIR"]) / "comparisons")
    comparison.MASK_PATH = config["MASK_PATH"]
    comparison.NIFTI_COL = config["NIFTI_COL"]
    comparison.DROP_ROWS = config["DROP_ROWS"]
    comparison.KEEP_ROWS = config["KEEP_ROWS"]
    comparison.COVARIATES_LIST = config["COVARIATES_LIST"]
    comparison.DATA_TRANSFORM_METHOD = config["DATA_TRANSFORM_METHOD"]
    comparison.INVERT_OUTCOME = config["INVERT_OUTCOME"]
    comparison.SIMILARITY = config["SIMILARITY"]
    comparison.CANDIDATE_MAPS = {
        name: path for name, path in maps if name in CANDIDATE_MAP_NAMES
    }
    comparison.COMPARATOR_MAPS = [
        (name, path) for name, path in maps if name not in CANDIDATE_MAP_NAMES
    ]

    if set(comparison.CANDIDATE_MAPS) != CANDIDATE_MAP_NAMES:
        missing = sorted(CANDIDATE_MAP_NAMES - set(comparison.CANDIDATE_MAPS))
        raise ValueError(f"Missing candidate maps in {SOURCE_SCRIPT.name}: {missing}")

    for symptom_column, y_label in config["SYMPTOM_COLUMN_DICT"].items():
        comparison.SYMPTOM_COLUMN = symptom_column
        comparison.Y_LABEL = y_label
        comparison.main()


if __name__ == "__main__":
    main()
