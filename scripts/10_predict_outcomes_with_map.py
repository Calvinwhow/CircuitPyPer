#!/usr/bin/env python3
"""
Plain, edit-the-variables-at-the-top runner for predicting outcomes with maps.

This is intentionally not a CLI. Change the values in the CONFIG section, then
run this file directly. For each map in MAPS_TO_PREDICT, the script writes a
subject-level prediction CSV and a prediction-vs-outcome scatterplot.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

# Redirect outputs to tmpdir.
TMPDIR = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(TMPDIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMPDIR / "xdg_cache"))
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata
from tqdm import tqdm

# Ensure circuit_pyper is on Python path when this file is run directly.
CIRCUIT_PYPER_DIR = Path(__file__).resolve().parents[1]
if str(CIRCUIT_PYPER_DIR) not in sys.path:
    sys.path.insert(0, str(CIRCUIT_PYPER_DIR))

from calvin_utils.neuroimaging_utils.ccm_utils.stat_utils import CorrelationCalculator
from calvin_utils.permutation_analysis_utils.statsmodels_palm import CalvinStatsmodelsPalm
from calvin_utils.statistical_utils.scatterplot import SimpleScatterPlotWrapper


# =============================================================================
# CONFIG
# =============================================================================


# Input/output paths.
INPUT_PATH = "/Volumes/HowExp/datasets/03h_ADVANCE_Alzheimer_DBS/metadata/Fnm_ADvanceI_ADASComponents_Long_Month24.csv"  # "/path/to/input.csv" or "/path/to/input.xlsx"
SHEET = None  # Specify sheet if using excel, e.g. "Sheet1".
OUT_DIR = "/Users/cu135/hires_backdrops/test/REPRODUCTION_AD_FX_rfz"
MASK_PATH = "/Users/cu135/Software_Local/calvin_utils_project/circuit_pyper/resources/MNI152_T1_2mm_brain_mask.nii"

# Symptom setup.
SYMPTOM_COLUMN = "ADASInstructionsScore"  # Spreadsheet column to predict. Run one symptom at a time.
NIFTI_COL = "tconn"  # Column containing patient neuroimaging files.
Y_LABEL = None  # Defaults to SYMPTOM_COLUMN. Example: "Memory Outcome"

# Maps to predict the single SYMPTOM_COLUMN with. Add as many map paths as you want.
# Names are derived from filenames. Optional explicit names are also supported:
# MAPS_TO_PREDICT = [("Memory map", "/path/to/map.nii.gz")]
MAPS_TO_PREDICT = [
    "/Users/cu135/Partners HealthCare Dropbox/Calvin Howard/studies/ccm_memory/memory_map/convergent_memory_map.nii",
    "/Users/cu135/Partners HealthCare Dropbox/Calvin Howard/studies/ccm_memory/memory_map/convergent_memory_target_map.nii.gz"
]

# Optional preprocessing.
DROP_ROWS = []  # Example: [("group", "equal", "control"), ("age", "below", 18)]
KEEP_ROWS = []  # Example: [("focal_cerebellum", 1)]
COVARIATES_LIST = []  # Loaded for compatibility with the spreadsheet prep; predictions do not model covariates.
DATA_TRANSFORM_METHOD = "None"  # Options: "standardize" | "rank" | None
INVERT_OUTCOME = False  # Multiply outcome by -1.

# Prediction settings.
SIMILARITY = "cosine"  # Options: "cosine" | "spatial_correl" | "avg_in_subject" | "avg_in_target"


@dataclass(frozen=True)
class PreparedPredictionData:
    outcome: str
    outcome_dir: str
    data_loader: "SpreadsheetDataLoader"


@dataclass(frozen=True)
class MapPrediction:
    outcome: str
    map_name: str
    map_out_dir: str
    prediction: np.ndarray
    outcome_values: np.ndarray


class SpreadsheetDataLoader:
    """
    Spreadsheet-backed loader for subject images and outcomes.

    It mirrors the small DataLoader API needed by the CCM helper utilities while
    reading subject images and outcomes from one spreadsheet.
    """

    def __init__(
        self,
        data_df,
        outcome_col,
        nifti_col,
        mask_path=None,
        covariates_list=None,
        data_transform_method="standardize",
        dataset_name=None,
    ):
        self.data_df = data_df.reset_index(drop=True)
        self.outcome_col = outcome_col
        self.nifti_col = nifti_col
        self.mask_path = mask_path
        self.covariates_list = list(covariates_list or [])
        self.data_transform_method = data_transform_method
        self.dataset_name = dataset_name or outcome_col
        self.dataset_paths_dict = {self.dataset_name: {"source": "spreadsheet"}}
        self.dataset_names_list = [self.dataset_name]
        self._dataset_cache = None

    def load_dataset(self, dataset_name, nifti_type="niftis"):
        if dataset_name not in self.dataset_paths_dict:
            raise KeyError(f"Unknown dataset: {dataset_name}")
        if self._dataset_cache is None:
            self._dataset_cache = self._load_dataset()

        data = {
            "niftis": self._dataset_cache["niftis"],
            "indep_var": self._dataset_cache["indep_var"],
            "covariates": self._dataset_cache["covariates"],
        }
        if nifti_type == "niftis_ranked":
            data["niftis"] = self._rank_niftis(data["niftis"])
        elif nifti_type != "niftis":
            raise ValueError("nifti_type must be 'niftis' or 'niftis_ranked'")
        return data

    def _load_dataset(self):
        niftis = self._load_and_mask_niftis(self.data_df[self.nifti_col].tolist())
        indep_var = pd.to_numeric(self.data_df[self.outcome_col], errors="raise").to_numpy(dtype=float)[:, np.newaxis]

        if self.covariates_list:
            covariates = self.data_df[self.covariates_list].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
        else:
            covariates = np.empty((len(self.data_df), 0), dtype=float)

        niftis = self._transform_niftis(niftis)
        indep_var = self._transform_outcome(indep_var)
        niftis, indep_var, covariates = self._handle_nans(niftis, indep_var, covariates)
        return {"niftis": niftis, "indep_var": indep_var, "covariates": covariates}

    def _load_and_mask_niftis(self, nifti_paths):
        mask_indices = None
        if self.mask_path is not None:
            mask = nib.load(self.mask_path)
            mask_indices = mask.get_fdata().flatten() > 0

        arrays = []
        for path in tqdm(nifti_paths, desc=f"Loading NIFTI files for {self.outcome_col}"):
            img = nib.load(str(path))
            data = img.get_fdata().flatten()
            if mask_indices is not None:
                data = data[mask_indices]
            arrays.append(data)
        return np.asarray(arrays, dtype=float)

    def _transform_niftis(self, niftis):
        if self.data_transform_method == "standardize":
            mean = np.nanmean(niftis, axis=0, keepdims=True)
            std = np.nanstd(niftis, axis=0, keepdims=True)
            return (niftis - mean) / (std + 1e-8)
        if self.data_transform_method == "rank":
            return self._rank_niftis(niftis)
        return niftis

    def _transform_outcome(self, indep_var):
        if self.data_transform_method == "standardize":
            return (indep_var - np.nanmean(indep_var, axis=0, keepdims=True)) / (np.nanstd(indep_var, axis=0, keepdims=True) + 1e-8)
        if self.data_transform_method == "rank":
            return rankdata(indep_var.flatten())[:, np.newaxis]
        return indep_var

    @staticmethod
    def _rank_niftis(arr):
        return np.apply_along_axis(rankdata, 0, arr)

    @staticmethod
    def _handle_nans(*arrs, value=0):
        processed = []
        for arr in arrs:
            if arr.size == 0:
                processed.append(arr)
                continue
            finite = arr[np.isfinite(arr)]
            if finite.size == 0:
                processed.append(np.nan_to_num(arr, nan=value, posinf=value, neginf=value))
                continue
            processed.append(np.nan_to_num(arr, nan=value, posinf=np.nanmax(finite), neginf=np.nanmin(finite)))
        return tuple(processed)


class PredictionDataPreparer:
    """Owns spreadsheet intake, row filtering, validation, and loader creation."""

    def __init__(
        self,
        input_path,
        sheet,
        out_dir,
        outcome_col,
        nifti_col,
        mask_path,
        path_replacements=None,
        keep_rows=None,
        drop_rows=None,
        covariates_list=None,
        data_transform_method="standardize",
        invert_outcome=False,
    ):
        self.input_path = input_path
        self.sheet = sheet
        self.out_dir = out_dir
        self.outcome_col = str(outcome_col)
        self.nifti_col = nifti_col
        self.mask_path = mask_path
        self.path_replacements = list(path_replacements or [])
        self.keep_rows = list(keep_rows or [])
        self.drop_rows = list(drop_rows or [])
        self.covariates_list = list(covariates_list or [])
        self.data_transform_method = data_transform_method
        self.invert_outcome = invert_outcome
        self.skipped_rows = []

    def prepare(self):
        os.makedirs(self.out_dir, exist_ok=True)
        outcome_dir = os.path.join(self.out_dir, safe_name(self.outcome_col))
        os.makedirs(outcome_dir, exist_ok=True)

        cal_palm, raw_df = self._read_spreadsheet()
        data_df = self._prepare_spreadsheet_rows(cal_palm, raw_df)
        if not self._has_enough_data(data_df):
            self._record_skip(data_df)
            print(f"Skipping {self.outcome_col}: fewer than 3 usable rows or no outcome variation after filtering.")
            return None

        data_loader = SpreadsheetDataLoader(
            data_df=data_df,
            outcome_col=self.outcome_col,
            nifti_col=self.nifti_col,
            mask_path=self.mask_path,
            covariates_list=self.covariates_list,
            data_transform_method=self.data_transform_method,
            dataset_name=self.outcome_col,
        )
        return PreparedPredictionData(
            outcome=self.outcome_col,
            outcome_dir=outcome_dir,
            data_loader=data_loader,
        )

    def write_skips(self):
        if self.skipped_rows:
            pd.DataFrame(self.skipped_rows).to_csv(os.path.join(self.out_dir, "skipped_outcomes.csv"), index=False)

    def _read_spreadsheet(self):
        cal_palm = CalvinStatsmodelsPalm(input_csv_path=self.input_path, output_dir=self.out_dir, sheet=self.sheet)
        return cal_palm, cal_palm.read_and_display_data()

    def _prepare_spreadsheet_rows(self, cal_palm, data_df):
        data_df = data_df.copy()
        self._apply_path_replacements(data_df)
        data_df = self._apply_keep_rows(data_df)
        data_df = self._apply_outcome_inversion(data_df)

        cal_palm.df = data_df
        drop_nan_list = [self.outcome_col, self.nifti_col] + self.covariates_list
        data_df = cal_palm.drop_nans_from_columns(columns_to_drop_from=drop_nan_list)

        for column, condition, value in self.drop_rows:
            data_df, _ = cal_palm.drop_rows_based_on_value(column, condition, value)
        return data_df

    def _apply_path_replacements(self, data_df):
        for old, new in self.path_replacements:
            data_df[self.nifti_col] = data_df[self.nifti_col].astype(str).str.replace(old, new, regex=False)

    def _apply_keep_rows(self, data_df):
        for column, value in self.keep_rows:
            before = len(data_df)
            data_df = data_df[data_df[column] == value].copy()
            print(f"Keeping {column} == {value}: {len(data_df)} of {before} rows")
        return data_df

    def _apply_outcome_inversion(self, data_df):
        if self.invert_outcome:
            print(f"INVERT_OUTCOME=True, MULTIPLYING {self.outcome_col} BY -1")
            data_df[self.outcome_col] = data_df[self.outcome_col] * -1
        return data_df

    def _has_enough_data(self, data_df):
        if len(data_df) < 3:
            return False
        values = pd.to_numeric(data_df[self.outcome_col], errors="coerce")
        return values.notna().sum() >= 3 and values.nunique(dropna=True) > 1

    def _record_skip(self, data_df):
        usable_values = pd.to_numeric(data_df[self.outcome_col], errors="coerce")
        self.skipped_rows.append(
            {
                "outcome": self.outcome_col,
                "rows_after_filter": len(data_df),
                "non_null_outcome_rows": int(usable_values.notna().sum()),
                "unique_outcome_values": int(usable_values.nunique(dropna=True)),
                "reason": "fewer than 3 usable rows or no outcome variation after filtering",
            }
        )


class MapPredictionRunner:
    """Generate prediction vectors for all configured maps."""

    def __init__(self, maps_to_predict, *, mask_path, similarity):
        self.maps_to_predict = tuple(map_items(maps_to_predict))
        self.mask_path = mask_path
        self.similarity = similarity

    def run(self, prepared_data):
        if prepared_data is None:
            return []

        data = prepared_data.data_loader.load_dataset(prepared_data.data_loader.dataset_names_list[0])
        niftis = CorrelationCalculator._check_for_nans(data["niftis"], nanpolicy="remove", verbose=False)
        outcome_values = CorrelationCalculator._check_for_nans(data["indep_var"], nanpolicy="remove", verbose=False).flatten()

        results = []
        for map_name, map_path in self.maps_to_predict:
            print("Predicting outcome with map: ", map_name)
            map_out_dir = os.path.join(prepared_data.outcome_dir, safe_name(map_name))
            os.makedirs(map_out_dir, exist_ok=True)

            prediction_map = CorrelationCalculator._check_for_nans(
                load_masked_map(map_path, self.mask_path),
                nanpolicy="remove",
                verbose=False,
            )
            prediction = map_prediction_vector(niftis, prediction_map, self.similarity)
            results.append(
                MapPrediction(
                    outcome=prepared_data.outcome,
                    map_name=map_name,
                    map_out_dir=map_out_dir,
                    prediction=np.asarray(prediction, dtype=float),
                    outcome_values=np.asarray(outcome_values, dtype=float),
                )
            )
        return results


class PredictionPlotter:
    """Write prediction CSVs and scatterplots."""

    def __init__(self, *, y_label=None):
        self.y_label = y_label

    def plot(self, prediction_results):
        for result in prediction_results:
            self._plot_one(result)

    def _plot_one(self, result):
        scatter_dir = os.path.join(result.map_out_dir, "prediction_scatter")
        os.makedirs(scatter_dir, exist_ok=True)

        scatter_df = pd.DataFrame(
            {
                "map_prediction": result.prediction,
                "outcome": result.outcome_values,
            }
        )
        scatter_df.to_csv(os.path.join(scatter_dir, "map_prediction_vs_outcome.csv"), index=False)

        if not (is_variable_vector(scatter_df["map_prediction"]) and is_variable_vector(scatter_df["outcome"])):
            skip_reason = (
                "Skipped scatterplot because the map prediction or outcome vector "
                "was constant or had fewer than 3 finite values.\n"
            )
            with open(os.path.join(scatter_dir, "scatterplot_skipped.txt"), "w") as f:
                f.write(skip_reason)
            print(skip_reason.strip())
            return

        plotter = SimpleScatterPlotWrapper(scatter_df)
        plotter.plot(
            x_col="map_prediction",
            y_col="outcome",
            dataset_name=f"{safe_name(result.map_name)}_prediction_vs_{safe_name(result.outcome)}",
            out_dir=scatter_dir,
            x_label=f"{result.map_name} prediction",
            y_label=self.y_label or result.outcome,
            show=False,
        )
        plt.close("all")


def map_items(map_paths):
    """Return (name, path) tuples from map paths or explicit (name, path) pairs."""
    if not isinstance(map_paths, list):
        raise TypeError("MAPS_TO_PREDICT must be a list: ['/path/to/map.nii.gz']")
    if not map_paths:
        raise ValueError("MAPS_TO_PREDICT is empty.")

    items = []
    for item in map_paths:
        if isinstance(item, tuple) and len(item) == 2:
            name, path = item
        else:
            path = item
            name = Path(str(path)).name.replace(".nii.gz", "").replace(".nii", "")
        items.append((str(name), str(path)))
    return items


def safe_name(value):
    """Make a readable string safe for output paths."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def load_masked_map(map_path, mask_path):
    """Load a map and return the same masked vector used for subject images."""
    data = nib.load(map_path).get_fdata().flatten()
    if mask_path is None:
        return data
    mask = nib.load(mask_path).get_fdata().flatten() > 0
    return data[mask]


def cosine_similarity_matrix(niftis, prediction_map):
    """Compute cosine similarity between each subject image and one map."""
    numerator = np.dot(niftis, prediction_map)
    denominator = np.linalg.norm(niftis, axis=1) * np.linalg.norm(prediction_map)
    return numerator / (denominator + 1e-8)


def map_prediction_vector(niftis, prediction_map, similarity):
    """Convert subject maps into one prediction vector from a single map."""
    niftis = np.asarray(niftis, dtype=float)
    prediction_map = np.asarray(prediction_map, dtype=float).flatten()

    if similarity == "cosine":
        return cosine_similarity_matrix(niftis, prediction_map)
    if similarity == "spatial_correl":
        return np.asarray([pearsonr(arr.flatten(), prediction_map)[0] for arr in niftis], dtype=float)
    if similarity == "avg_in_subject":
        values = []
        for arr in niftis:
            arr = arr.flatten()
            denom = np.count_nonzero(np.isfinite(arr) & (arr != 0))
            values.append(np.dot(arr, prediction_map) / denom if denom else np.nan)
        return np.asarray(values, dtype=float)
    if similarity == "avg_in_target":
        denom = np.count_nonzero(np.isfinite(prediction_map) & (prediction_map != 0))
        if denom == 0:
            return np.full(niftis.shape[0], np.nan, dtype=float)
        return np.asarray([np.dot(arr.flatten(), prediction_map) / denom for arr in niftis], dtype=float)
    raise ValueError("SIMILARITY must be 'cosine', 'spatial_correl', 'avg_in_subject', or 'avg_in_target'.")


def is_variable_vector(arr):
    """Return True when an array has enough finite, non-constant values to fit/correlate."""
    arr = np.asarray(arr, dtype=float).flatten()
    arr = arr[np.isfinite(arr)]
    return arr.size >= 3 and np.nanstd(arr) > 0


class MapPredictionAnalysis:
    """Top-level workflow: prep data, predict outcomes with maps, plot scatterplots."""

    def __init__(self):
        self.preparer = PredictionDataPreparer(
            input_path=INPUT_PATH,
            sheet=SHEET,
            out_dir=OUT_DIR,
            outcome_col=SYMPTOM_COLUMN,
            nifti_col=NIFTI_COL,
            mask_path=MASK_PATH,
            path_replacements=None,
            keep_rows=KEEP_ROWS,
            drop_rows=DROP_ROWS,
            covariates_list=COVARIATES_LIST,
            data_transform_method=DATA_TRANSFORM_METHOD,
            invert_outcome=INVERT_OUTCOME,
        )
        self.prediction_runner = MapPredictionRunner(
            MAPS_TO_PREDICT,
            mask_path=MASK_PATH,
            similarity=SIMILARITY,
        )
        self.plotter = PredictionPlotter(y_label=Y_LABEL)

    def run(self):
        print("Running symptom: ", self.preparer.outcome_col)
        prepared_data = self.preparer.prepare()
        prediction_results = self.prediction_runner.run(prepared_data)
        self.preparer.write_skips()
        self.plotter.plot(prediction_results)


def main():
    MapPredictionAnalysis().run()


if __name__ == "__main__":
    main()
