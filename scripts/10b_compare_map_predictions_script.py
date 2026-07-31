#!/usr/bin/env python3
"""
Plain, edit-the-variables-at-the-top runner for map prediction comparisons.

This is intentionally not a CLI. Change the values in the CONFIG section, then
run this file directly.
"""

from __future__ import annotations
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from scipy.stats import pearsonr, spearmanr, ttest_ind
from tqdm import tqdm
import nibabel as nib
from calvin_utils.permutation_analysis_utils.statsmodels_palm import CalvinStatsmodelsPalm
from calvin_utils.neuroimaging_utils.ccm_utils.stat_utils import CorrelationCalculator
from calvin_utils.neuroimaging_utils.ccm_utils.convergent_loocv import LOOCVAnalyzer
from calvin_utils.neuroimaging_utils.ccm_utils.delta_scatterplot import DeltaCorrelationScatter
from calvin_utils.plotting_utils.pair_superiority_plot import PairSuperiorityPlot
from calvin_utils.plotting_utils.simple_box_plot import SimpleBoxPlotWrapper
from calvin_utils.statistical_utils.scatterplot import SimpleScatterPlotWrapper

# Redirect outputs to tmpdir
TMPDIR = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(TMPDIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMPDIR / "xdg_cache"))
matplotlib.use("Agg")

# Ensure script is seen on Python path
CIRCUIT_PYPER_DIR = Path(__file__).resolve().parents[1]
if str(CIRCUIT_PYPER_DIR) not in sys.path:
    sys.path.insert(0, str(CIRCUIT_PYPER_DIR))



# =============================================================================
# CONFIG
# =============================================================================


# Input/output paths.
INPUT_PATH = "/Volumes/HowExp/datasets/02a_Corbetta_Stroke_Lesion/Study_Metadata/3month_arm_1clean.csv" # Form: "/path/to/input.csv" or "/path/to/input.xlsx"
SHEET = None                            # Specify sheet if using excel (i.e. "Sheet1")
OUT_DIR = "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimzation/comparisons/corbetta_3mo/gdss_10-cerebellumOnly" # Form: "/path/to/output_dir"
MASK_PATH = "/Users/cu135/Software_Local/calvin_utils_project/circuit_pyper/resources/MNI152_T1_2mm_brain_mask.nii"

# Symptom setup.
SYMPTOM_COLUMN = "gdss_10"       # Spreadsheet column to analyze. Run one symptom at a time.
NIFTI_COL = "roi_path"           # Column containing patient neuroimaging files.
Y_LABEL = None                          # Defaults to SYMPTOM_COLUMN. Example: "Memory Outcome"
PATH_REPLACEMENTS = []                  # Example: [("/Volumes/HowExp/", "/Volumes/HowExp2/")]

# Maps to compare.
# Candidate maps are the maps being evaluated for SYMPTOM_COLUMN.
# Comparator maps are the reference/prior maps every candidate is tested against.
CONTROL_MAPS_DIR = "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimzation/comparisons/control_maps"
CANDIDATE_MAPS = {
    "CNRSTotScore": '/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimzation/symptom_on_lhs/proper_regressions/CNRSTotScore-on-Nifti_File_Path/regression/contrast_tval_0.nii.gz',
    "ScoreCol2B": '/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimzation/symptom_on_lhs/proper_regressions/ScoreCol2B-on-Nifti_File_Path/regression/contrast_tval_0.nii.gz',
    "CNRS_Fibers": "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimzation/symptom_on_lhs/standardized-perm1000-fibfilt/CNRSTotScore/regression_0/contrast_tval_0.nii.gz",
}
COMPARATOR_MAPS = [
    '/Users/cu135/Partners HealthCare Dropbox/Calvin Howard/studies/raynor_network_mapping/data/mikes_maps/emotion_regulation.nii',
    '/Users/cu135/Partners HealthCare Dropbox/Calvin Howard/studies/raynor_network_mapping/data/nettekoven_maps/rois/resampled_cerebellum_roi_1-4_to_17-20_bilateral.nii.gz',
    '/Users/cu135/Partners HealthCare Dropbox/Calvin Howard/studies/raynor_network_mapping/data/nettekoven_maps/rois/resampled_cerebellum_roi_5-7_to_21-23_bilateral.nii.gz',
    '/Users/cu135/Partners HealthCare Dropbox/Calvin Howard/studies/raynor_network_mapping/data/nettekoven_maps/rois/resampled_cerebellum_roi_8-11_to_24-27_bilateral.nii.gz',
    '/Users/cu135/Partners HealthCare Dropbox/Calvin Howard/studies/raynor_network_mapping/data/nettekoven_maps/rois/resampled_cerebellum_roi_12-16_to_28-32_bilateral.nii.gz'
]

# Optional preprocessing.
DROP_ROWS = [("focal_cerebellum", "equal", 0)]                          # Conditions for dropping rows. Example: [("group", "equal", "control"), ("age", "below", 18)]
KEEP_ROWS = []                          # Conditions for keeping rows. Example: [("focal_cerebellum", 1)]
COVARIATES_LIST = []                    # Loaded for compatibility with the CCM data API, but the ROI comparison does not use covariates.
DATA_TRANSFORM_METHOD = "standardize"   # Options: 'standardize' | 'rank' | None
INVERT_OUTCOME = False                  # Multiply outcome by -1. Default: False

# Comparison settings.
CORRELATION = "spearman"                # Options used by CorrelationCalculator/DeltaCorrelationScatter: 'spearman' | 'pearson'
SIMILARITY = "cosine"                   # Damage/similarity metric.
RESAMPLE_METHOD = "bootstrap"           # Options: 'bootstrap' | 'permutation'
N_ITER = 1000                           # Default 1000
SEED = 42
DELTA_R2 = True                         # Compare explained variance instead of raw r.
SKIP_EXISTING = False                   # Skip comparison folders that already exist.
DRAW_INDIVIDUAL_PLOTS = True            # Pair-superiority and scatter plots for every map-vs-map comparison.
DRAW_AGGREGATE_PLOTS = True             # Aggregate boxplots/distribution plots for each candidate map.


class SpreadsheetDataLoader:
    """
    Spreadsheet-backed replacement for ccm_utils.npy_utils.DataLoader.

    It supplies the attributes/methods used by CorrelationCalculator and
    LOOCVAnalyzer while reading subject images and outcomes from a spreadsheet.
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


@dataclass(frozen=True)
class StatsReadyMapPredictionData:
    """Minimal prepared payload consumed by the map-comparison statistics."""

    outcome: str
    outcome_dir: str
    data_loader: SpreadsheetDataLoader
    corr_map_dict: dict


@dataclass(frozen=True)
class MapComparisonStats:
    """Stats output for one candidate-vs-comparator map comparison."""

    outcome: str
    candidate_name: str
    comparator_name: str
    comparison_out_dir: str
    summary_df: pd.DataFrame
    resample_df: pd.DataFrame
    observed_x1: np.ndarray
    observed_x2: np.ndarray
    observed_y: np.ndarray
    r_values: dict
    observed_r_values: dict


@dataclass(frozen=True)
class CandidateMapStats:
    """Combined stats output for one candidate map against all comparators."""

    candidate_name: str
    candidate_out_dir: str
    comparisons: tuple[MapComparisonStats, ...]
    summary_df: pd.DataFrame
    resample_df: pd.DataFrame


@dataclass(frozen=True)
class MapPredictionStatsResults:
    """All map-comparison statistics generated from one prepared dataset."""

    outcome: str
    candidate_results: tuple[CandidateMapStats, ...]


class MapPredictionDataPreparer:
    """
    Owns spreadsheet intake, row filtering, validation, and stats-loader creation.

    Downstream comparison code should not know how rows were filtered or how the
    spreadsheet was turned into the CCM data API. It receives only the small
    StatsReadyMapPredictionData payload needed to run statistics.
    """

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
        correlation="spearman",
        similarity="cosine",
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
        self.correlation = correlation
        self.similarity = similarity
        self.skipped_rows = []

    def prepare(self):
        """Return stats-ready data, or None when the configured outcome is unusable."""
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
        corr_map_dict = self._build_correlation_maps(data_loader)
        return StatsReadyMapPredictionData(
            outcome=self.outcome_col,
            outcome_dir=outcome_dir,
            data_loader=data_loader,
            corr_map_dict=corr_map_dict,
        )

    def write_skips(self):
        """Persist skip records collected during preparation."""
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

    def _build_correlation_maps(self, data_loader):
        if self.similarity == "cosine":
            return {self.outcome_col: np.array([0.0])}
        correlation_calculator = CorrelationCalculator(method=self.correlation, verbose=False)
        return correlation_calculator.generate_correlation_maps(data_loader)


def map_items(map_dict, label):
    """Return explicit (name, path) map tuples from one config dictionary."""
    if not isinstance(map_dict, dict):
        raise TypeError(f"{label} must be a dictionary: {{'Map Name': '/path/to/map.nii.gz'}}")
    if not map_dict:
        raise ValueError(f"{label} is empty.")
    return [(str(name), str(path)) for name, path in map_dict.items()]


def comparator_items(map_paths):
    """Return (derived_name, path) tuples from a simple list of comparator paths."""
    if not isinstance(map_paths, list):
        raise TypeError("COMPARATOR_MAPS must be a list: ['/path/to/map1.nii.gz', '/path/to/map2.nii.gz']")
    if not map_paths:
        raise ValueError("COMPARATOR_MAPS is empty.")
    return [
        (Path(str(path)).name.replace(".nii.gz", "_nii_gz").replace(".nii", "_nii"), str(path))
        for path in map_paths
    ]


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


def cosine_similarity_matrix(niftis, roi_map):
    """Compute cosine similarity between each subject image and one ROI/map."""
    numerator = np.dot(niftis, roi_map)
    denominator = np.linalg.norm(niftis, axis=1) * np.linalg.norm(roi_map)
    return numerator / (denominator + 1e-8)


def correlate_vectors(x, y, correlation):
    """Correlate vectors using the configured outcome-correlation method."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 3:
        return np.nan
    x = x[mask]
    y = y[mask]
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return 0.0
    if correlation == "pearson":
        return pearsonr(x, y)[0]
    if correlation == "spearman":
        return spearmanr(x, y, nan_policy="omit")[0]
    raise ValueError("CORRELATION must be 'pearson' or 'spearman' for map comparisons.")


def shuffle_for_comparison(sim1, sim2, y, method):
    """Apply the configured resampling method to precomputed similarity vectors."""
    if method == "bootstrap":
        idx = np.random.choice(len(y), size=len(y), replace=True)
        return sim1[idx], sim2[idx], y[idx]
    if method == "permutation":
        idx = np.random.permutation(len(y))
        return sim1, sim2, y[idx]
    if method == "observed":
        return sim1, sim2, y
    raise ValueError("RESAMPLE_METHOD must be 'bootstrap' or 'permutation'.")


def compute_comparison_stats(sim1, sim2, y, method, n_iter, delta_r2, correlation):
    """Compute observed and resampled map-prediction statistics from precomputed similarities."""
    observed_stat1 = correlate_vectors(sim1, y, correlation)
    observed_stat2 = correlate_vectors(sim2, y, correlation)
    if delta_r2:
        observed_stat1 = observed_stat1 ** 2
        observed_stat2 = observed_stat2 ** 2

    stat1_values = []
    stat2_values = []
    iter_count = 0
    with tqdm(total=n_iter) as pbar:
        while iter_count < n_iter:
            sub_sim1, sub_sim2, sub_y = shuffle_for_comparison(sim1, sim2, y, method)
            stat1 = correlate_vectors(sub_sim1, sub_y, correlation)
            stat2 = correlate_vectors(sub_sim2, sub_y, correlation)
            if np.isnan(stat1) or np.isnan(stat2):
                continue
            if delta_r2:
                stat1 = stat1 ** 2
                stat2 = stat2 ** 2
            stat1_values.append(stat1)
            stat2_values.append(stat2)
            iter_count += 1
            pbar.update(1)

    return {
        "observed_stat1": observed_stat1,
        "observed_stat2": observed_stat2,
        "stat1_values": np.asarray(stat1_values, dtype=float),
        "stat2_values": np.asarray(stat2_values, dtype=float),
    }


def is_variable_vector(arr):
    """Return True when an array has enough finite, non-constant values to fit/correlate."""
    arr = np.asarray(arr, dtype=float).flatten()
    arr = arr[np.isfinite(arr)]
    return arr.size >= 3 and np.nanstd(arr) > 0


class MapPredictionStatsRunner:
    """Runs map-prediction statistics and returns only stats artifacts."""

    def __init__(
        self,
        candidate_maps,
        comparator_maps,
        *,
        mask_path,
        similarity,
        correlation,
        resample_method,
        n_iter,
        seed,
        delta_r2,
        skip_existing,
        y_label=None,
    ):
        self.candidate_maps = tuple(candidate_maps)
        self.comparator_maps = tuple(comparator_maps)
        self.mask_path = mask_path
        self.similarity = similarity
        self.correlation = correlation
        self.resample_method = resample_method
        self.n_iter = n_iter
        self.seed = seed
        self.delta_r2 = delta_r2
        self.skip_existing = skip_existing
        self.y_label = y_label

    def run_stats(self, prepared_data):
        if prepared_data is None:
            return MapPredictionStatsResults(outcome="", candidate_results=())

        candidate_results = []
        for candidate_name, candidate_path in self.candidate_maps:
            candidate_out_dir = os.path.join(prepared_data.outcome_dir, safe_name(candidate_name))
            os.makedirs(candidate_out_dir, exist_ok=True)
            comparisons = []

            for comparator_name, comparator_path in self.comparator_maps:
                print("Comparing maps: ", candidate_name, " vs ", comparator_name)
                comparison = self._run_single_comparison(
                    prepared_data,
                    candidate_name,
                    candidate_path,
                    comparator_name,
                    comparator_path,
                    candidate_out_dir,
                )
                if comparison is not None:
                    comparisons.append(comparison)

            if comparisons:
                candidate_results.append(
                    CandidateMapStats(
                        candidate_name=candidate_name,
                        candidate_out_dir=candidate_out_dir,
                        comparisons=tuple(comparisons),
                        summary_df=pd.concat([item.summary_df for item in comparisons], ignore_index=True),
                        resample_df=pd.concat([item.resample_df for item in comparisons], ignore_index=True),
                    )
                )

        return MapPredictionStatsResults(
            outcome=prepared_data.outcome,
            candidate_results=tuple(candidate_results),
        )

    def _run_single_comparison(self, prepared_data, candidate_name, candidate_path, comparator_name, comparator_path, candidate_out_dir):
        comparison_name = f"{safe_name(prepared_data.outcome)}__{safe_name(candidate_name)}-vs-{safe_name(comparator_name)}"
        comparison_out_dir = os.path.join(candidate_out_dir, comparison_name)
        if self.skip_existing and os.path.isdir(comparison_out_dir):
            print(f"Skipping existing comparison: {comparison_out_dir}")
            return None

        os.makedirs(comparison_out_dir, exist_ok=True)
        np.random.seed(self.seed)

        if self.similarity == "cosine":
            observed_x1, observed_x2, observed_y, stats = self._run_fast_cosine_comparison(
                prepared_data.data_loader,
                candidate_path,
                comparator_path,
            )
            r_values = {"roi1": stats["stat1_values"], "roi2": stats["stat2_values"]}
            observed_r_values = {"roi1": [stats["observed_stat1"]], "roi2": [stats["observed_stat2"]]}
        else:
            observed_x1, observed_x2, observed_y, r_values, observed_r_values = self._run_loocv_comparison(
                prepared_data,
                candidate_path,
                comparator_path,
                comparison_out_dir,
            )

        summary_df, resample_df = self._build_result_tables(
            prepared_data.outcome,
            prepared_data.data_loader,
            candidate_name,
            comparator_name,
            comparison_out_dir,
            r_values,
            observed_r_values,
        )
        summary_df.to_csv(os.path.join(comparison_out_dir, "observed_comparison_values.csv"), index=False)
        resample_df.to_csv(os.path.join(comparison_out_dir, "resampled_comparison_values.csv"), index=False)

        return MapComparisonStats(
            outcome=prepared_data.outcome,
            candidate_name=candidate_name,
            comparator_name=comparator_name,
            comparison_out_dir=comparison_out_dir,
            summary_df=summary_df,
            resample_df=resample_df,
            observed_x1=np.asarray(observed_x1, dtype=float),
            observed_x2=np.asarray(observed_x2, dtype=float),
            observed_y=np.asarray(observed_y, dtype=float),
            r_values=r_values,
            observed_r_values=observed_r_values,
        )

    def _run_fast_cosine_comparison(self, data_loader, candidate_path, comparator_path):
        data = data_loader.load_dataset(data_loader.dataset_names_list[0])
        niftis = CorrelationCalculator._check_for_nans(data["niftis"], nanpolicy="remove", verbose=False)
        y = CorrelationCalculator._check_for_nans(data["indep_var"], nanpolicy="remove", verbose=False).flatten()
        roi1_map = CorrelationCalculator._check_for_nans(load_masked_map(candidate_path, self.mask_path), nanpolicy="remove", verbose=False)
        roi2_map = CorrelationCalculator._check_for_nans(load_masked_map(comparator_path, self.mask_path), nanpolicy="remove", verbose=False)

        sim1 = cosine_similarity_matrix(niftis, roi1_map)
        sim2 = cosine_similarity_matrix(niftis, roi2_map)
        stats = compute_comparison_stats(
            sim1,
            sim2,
            y,
            self.resample_method,
            self.n_iter,
            self.delta_r2,
            self.correlation,
        )
        return sim1, sim2, y, stats

    def _run_loocv_comparison(self, prepared_data, candidate_path, comparator_path, comparison_out_dir):
        loocv_analyzer = LOOCVAnalyzer(
            prepared_data.corr_map_dict,
            prepared_data.data_loader,
            similarity=self.similarity,
            method=self.correlation,
            out_dir=comparison_out_dir,
            mask_path=self.mask_path,
            roi_path=None,
            ylabel=self.y_label or prepared_data.outcome,
        )
        loocv_analyzer.compare_roi_correlations(
            roi1=candidate_path,
            roi2=comparator_path,
            method=self.resample_method,
            n_iter=self.n_iter,
            seed=self.seed,
            delta_r2=self.delta_r2,
        )
        return (
            loocv_analyzer.observed_x1,
            loocv_analyzer.observed_x2,
            loocv_analyzer.observed_y,
            loocv_analyzer.r_values,
            loocv_analyzer.observed_r_values,
        )

    @staticmethod
    def _build_result_tables(outcome, data_loader, candidate_name, comparator_name, comparison_out_dir, r_values, observed_r_values):
        new_observed = np.asarray(observed_r_values["roi1"], dtype=float)
        old_observed = np.asarray(observed_r_values["roi2"], dtype=float)
        new_resampled = np.asarray(r_values["roi1"], dtype=float)
        old_resampled = np.asarray(r_values["roi2"], dtype=float)

        summary_rows = []
        for dataset_idx, dataset_name in enumerate(data_loader.dataset_names_list):
            if dataset_idx < len(new_observed) and dataset_idx < len(old_observed):
                summary_rows.append(
                    {
                        "outcome": outcome,
                        "dataset": dataset_name,
                        "new_map": candidate_name,
                        "old_map": comparator_name,
                        "new_observed_stat": new_observed[dataset_idx],
                        "old_observed_stat": old_observed[dataset_idx],
                        "delta_observed_stat": new_observed[dataset_idx] - old_observed[dataset_idx],
                        "comparison_out_dir": comparison_out_dir,
                    }
                )

        summary_df = pd.DataFrame(summary_rows)
        resample_df = pd.DataFrame(
            {
                "outcome": outcome,
                "new_map": candidate_name,
                "old_map": comparator_name,
                "new_resampled_stat": new_resampled,
                "old_resampled_stat": old_resampled,
                "delta_resampled_stat": new_resampled - old_resampled,
            }
        )
        return summary_df, resample_df


def plot_bootstrapped_performance_distribution(resample_df, aggregate_dir):
    """Plot per-map bootstrap distributions and return pooled candidate/comparator values."""
    if resample_df.empty:
        return np.array([], dtype=float), np.array([], dtype=float)

    first_old_map = resample_df["old_map"].iloc[0]
    new_map_name = resample_df["new_map"].iloc[0]
    candidate_values = (
        pd.to_numeric(
            resample_df.loc[resample_df["old_map"] == first_old_map, "new_resampled_stat"],
            errors="coerce",
        )
        .dropna()
        .to_numpy(dtype=float)
    )
    rows = [{"map": new_map_name, "type": "new", "value": value} for value in candidate_values]
    for old_map, old_df in resample_df.groupby("old_map", sort=False):
        old_values = pd.to_numeric(old_df["old_resampled_stat"], errors="coerce").dropna()
        rows += [{"map": old_map, "type": "prior", "value": value} for value in old_values]

    pooled_comparator_values = (
        pd.to_numeric(resample_df["old_resampled_stat"], errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
    )
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return candidate_values, pooled_comparator_values

    means = plot_df.groupby("map")["value"].mean()
    old_order = [map_name for map_name in means.sort_values(ascending=False).index if map_name != new_map_name]
    order = [new_map_name] + old_order
    data = [plot_df.loc[plot_df["map"] == map_name, "value"].to_numpy(dtype=float) for map_name in order]
    colors = ["#211D1E"] + ["#8E8E8E"] * len(old_order)

    fig_height = max(7, 0.42 * len(order))
    fig, ax = plt.subplots(figsize=(9, fig_height))
    box = ax.boxplot(
        data,
        vert=False,
        widths=0.62,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#FFFFFF", "linewidth": 2},
        boxprops={"linewidth": 1.6},
        whiskerprops={"linewidth": 1.4},
        capprops={"linewidth": 1.4},
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.9)

    t_stat, p_val = ttest_ind(candidate_values, pooled_comparator_values, equal_var=False)
    delta = float(np.mean(candidate_values) - np.mean(pooled_comparator_values))
    ax.text(
        0.98,
        0.04,
        f"Candidate vs pooled comparators\nMean delta = {delta:.4f}\nt = {t_stat:.2f}, p = {p_val:.2e}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=14,
    )
    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels(order, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("R2" if DELTA_R2 else "r", fontsize=18)
    ax.set_title("Bootstrapped Performance by Map", fontsize=20)
    ax.tick_params(axis="x", labelsize=14)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    fig.tight_layout()
    fig.savefig(os.path.join(aggregate_dir, "overall_resampled_performance_distribution.svg"), bbox_inches="tight")
    plt.close(fig)
    return candidate_values, pooled_comparator_values


def plot_overall_results(summary_df, resample_df, map_out_dir):
    """Plot the aggregate candidate-map vs comparator-map performance."""
    aggregate_dir = os.path.join(map_out_dir, "aggregate")
    os.makedirs(aggregate_dir, exist_ok=True)

    summary_df.to_csv(os.path.join(aggregate_dir, "observed_comparison_summary.csv"), index=False)
    resample_df.to_csv(os.path.join(aggregate_dir, "resampled_comparison_summary.csv"), index=False)
    candidate_values, pooled_comparator_values = plot_bootstrapped_performance_distribution(resample_df, aggregate_dir)

    pooled_distribution_df = pd.DataFrame(
        {
            "candidate_map": pd.Series(candidate_values, dtype=float),
            "comparator_maps": pd.Series(pooled_comparator_values, dtype=float),
        }
    )
    pooled_distribution_df.to_csv(os.path.join(aggregate_dir, "pooled_bootstrap_distributions.csv"), index=False)
    plotter = SimpleBoxPlotWrapper(pooled_distribution_df)
    plotter.plot(
        columns=[("candidate_map", "comparator_maps")],
        dataset_name="Candidate Map vs Pooled Comparator Maps",
        group_labels=["Bootstrap"],
        pair_names=["Candidate map", "Comparator maps"],
        out_dir=os.path.join(aggregate_dir, "new_vs_other_map_boxplot.svg"),
        ylabel="R2" if DELTA_R2 else "r",
    )


class MapPredictionFigurePlotter:
    """Draws figures and top-level reports from already-computed stats."""

    def __init__(
        self,
        *,
        out_dir,
        correlation,
        resample_method,
        delta_r2,
        n_iter,
        y_label=None,
        draw_individual=True,
        draw_aggregate=True,
    ):
        self.out_dir = out_dir
        self.correlation = correlation
        self.resample_method = resample_method
        self.delta_r2 = delta_r2
        self.n_iter = n_iter
        self.y_label = y_label
        self.draw_individual = draw_individual
        self.draw_aggregate = draw_aggregate

    def plot_figures(self, stats_results):
        for candidate_result in stats_results.candidate_results:
            self._plot_candidate_scatter(candidate_result)
            if self.draw_individual:
                for comparison in candidate_result.comparisons:
                    self._plot_individual_comparison(comparison)
            self._plot_candidate_aggregate(candidate_result)
        write_outperformance_reports(self.out_dir)

    def _plot_candidate_scatter(self, candidate_result):
        if not candidate_result.comparisons:
            return

        comparison = candidate_result.comparisons[0]
        scatter_dir = os.path.join(candidate_result.candidate_out_dir, "candidate_scatter")
        os.makedirs(scatter_dir, exist_ok=True)

        if not (is_variable_vector(comparison.observed_x1) and is_variable_vector(comparison.observed_y)):
            skip_reason = (
                "Skipped SimpleScatterPlot because the candidate prediction or outcome "
                "vector was constant or had fewer than 3 finite values.\n"
            )
            with open(os.path.join(scatter_dir, "simple_scatterplot_skipped.txt"), "w") as f:
                f.write(skip_reason)
            print(skip_reason.strip())
            return

        scatter_df = pd.DataFrame(
            {
                "candidate_prediction": comparison.observed_x1,
                "outcome": comparison.observed_y,
            }
        )
        scatter_df.to_csv(os.path.join(scatter_dir, "candidate_prediction_vs_outcome.csv"), index=False)

        plotter = SimpleScatterPlotWrapper(scatter_df)
        plotter.plot(
            x_col="candidate_prediction",
            y_col="outcome",
            dataset_name=f"{safe_name(candidate_result.candidate_name)}_prediction_vs_{safe_name(comparison.outcome)}",
            out_dir=scatter_dir,
            x_label=f"{candidate_result.candidate_name} prediction",
            y_label=self.y_label or comparison.outcome,
            show=False,
        )
        plt.close("all")

    def _plot_individual_comparison(self, comparison):
        visualizer = PairSuperiorityPlot(
            stat_array_1=np.array(comparison.r_values["roi1"]),
            stat_array_2=np.array(comparison.r_values["roi2"]),
            model1_name=comparison.candidate_name,
            model2_name=comparison.comparator_name,
            out_dir=comparison.comparison_out_dir,
            observed_stat_array=[
                np.array(comparison.observed_r_values["roi1"]),
                np.array(comparison.observed_r_values["roi2"]),
            ],
            method=self.resample_method,
        )
        visualizer.draw(verbose=False)
        plt.close("all")

        if (
            is_variable_vector(comparison.observed_x1)
            and is_variable_vector(comparison.observed_x2)
            and is_variable_vector(comparison.observed_y)
        ):
            vis = DeltaCorrelationScatter(
                x_array_1=comparison.observed_x1,
                x_array_2=comparison.observed_x2,
                y_array=comparison.observed_y,
                y_label=self.y_label or comparison.outcome,
                label_1=comparison.candidate_name,
                label_2=comparison.comparator_name,
                stat_label="r",
                out_dir=comparison.comparison_out_dir,
                method=self.correlation,
            )
            vis.draw(show=False)
            plt.close("all")
            return

        skip_reason = (
            "Skipped DeltaCorrelationScatter because at least one observed "
            "similarity/outcome vector was constant or had fewer than 3 finite values.\n"
        )
        with open(os.path.join(comparison.comparison_out_dir, "delta_scatterplot_skipped.txt"), "w") as f:
            f.write(skip_reason)
        print(skip_reason.strip())

    def _plot_candidate_aggregate(self, candidate_result):
        if self.draw_aggregate:
            plot_overall_results(candidate_result.summary_df, candidate_result.resample_df, candidate_result.candidate_out_dir)
            return

        aggregate_dir = os.path.join(candidate_result.candidate_out_dir, "aggregate")
        os.makedirs(aggregate_dir, exist_ok=True)
        candidate_result.summary_df.to_csv(os.path.join(aggregate_dir, "observed_comparison_summary.csv"), index=False)
        candidate_result.resample_df.to_csv(os.path.join(aggregate_dir, "resampled_comparison_summary.csv"), index=False)


def summarize_distribution(values, prefix):
    """Return compact distribution stats for report rows."""
    values = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_ci025": np.nan,
            f"{prefix}_ci975": np.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_ci025": float(np.percentile(values, 2.5)),
        f"{prefix}_ci975": float(np.percentile(values, 97.5)),
    }


def write_outperformance_reports(out_dir):
    """Write top-level CSV/TXT summaries from all aggregate comparison outputs."""
    aggregate_paths = sorted(Path(out_dir).glob("*/*/aggregate/resampled_comparison_summary.csv"))
    if not aggregate_paths:
        empty_aggregate_columns = [
            "outcome",
            "new_map",
            "n_prior_maps",
            "n_pairwise_resamples",
            "n_prior_maps_beaten_by_mean",
            "empirical_p_new_greater_pooled_priors",
            "empirical_p_prior_greater_new",
            "empirical_p_two_sided",
        ]
        empty_pair_columns = [
            "outcome",
            "new_map",
            "old_map",
            "n_resamples",
            "empirical_p_new_greater_old",
            "empirical_p_old_greater_new",
            "empirical_p_two_sided",
            "new_beats_old_mean",
        ]
        pd.DataFrame(columns=empty_aggregate_columns).to_csv(
            os.path.join(out_dir, "new_map_outperformance_with_empirical_pvalues.csv"),
            index=False,
        )
        pd.DataFrame(columns=empty_pair_columns).to_csv(
            os.path.join(out_dir, "new_map_pairwise_empirical_pvalues.csv"),
            index=False,
        )
        pd.DataFrame(columns=empty_pair_columns).to_csv(
            os.path.join(out_dir, "new_map_individual_outperforming_comparisons.csv"),
            index=False,
        )
        with open(os.path.join(out_dir, "new_map_outperformance_with_empirical_pvalues.txt"), "w") as f:
            f.write(f"Outperformance report for: {out_dir}\n")
            f.write("No aggregate resampled summaries were found. All configured outcomes were skipped or produced no analyzable comparisons.\n")
        print(f"No aggregate resampled summaries found in {out_dir}; wrote empty top-level report files.")
        return

    resampled_frames = []
    observed_frames = []
    for resampled_path in aggregate_paths:
        resampled_frames.append(pd.read_csv(resampled_path))
        observed_path = resampled_path.with_name("observed_comparison_summary.csv")
        if observed_path.exists():
            observed_frames.append(pd.read_csv(observed_path))

    all_resampled = pd.concat(resampled_frames, ignore_index=True)
    all_observed = pd.concat(observed_frames, ignore_index=True) if observed_frames else pd.DataFrame()
    all_resampled.to_csv(os.path.join(out_dir, "all_resampled_comparison_values.csv"), index=False)
    if not all_observed.empty:
        all_observed.to_csv(os.path.join(out_dir, "all_observed_comparison_values.csv"), index=False)

    pair_rows = []
    for (outcome, new_map, old_map), group in all_resampled.groupby(["outcome", "new_map", "old_map"], sort=False):
        delta = group["delta_resampled_stat"].to_numpy(dtype=float)
        row = {
            "outcome": outcome,
            "new_map": new_map,
            "old_map": old_map,
            "n_resamples": int(np.isfinite(delta).sum()),
            "empirical_p_new_greater_old": float(np.mean(delta <= 0)),
            "empirical_p_old_greater_new": float(np.mean(delta >= 0)),
            "empirical_p_two_sided": float(min(1.0, 2.0 * min(np.mean(delta <= 0), np.mean(delta >= 0)))),
            "new_beats_old_mean": bool(np.nanmean(delta) > 0),
        }
        row.update(summarize_distribution(group["new_resampled_stat"], "new_resampled"))
        row.update(summarize_distribution(group["old_resampled_stat"], "old_resampled"))
        row.update(summarize_distribution(delta, "delta_resampled"))
        if not all_observed.empty:
            observed_match = all_observed[
                (all_observed["outcome"] == outcome)
                & (all_observed["new_map"] == new_map)
                & (all_observed["old_map"] == old_map)
            ]
            if not observed_match.empty:
                observed_row = observed_match.iloc[0]
                row.update(
                    {
                        "new_observed_stat": observed_row.get("new_observed_stat", np.nan),
                        "old_observed_stat": observed_row.get("old_observed_stat", np.nan),
                        "delta_observed_stat": observed_row.get("delta_observed_stat", np.nan),
                        "comparison_out_dir": observed_row.get("comparison_out_dir", ""),
                    }
                )
        pair_rows.append(row)

    pair_df = pd.DataFrame(pair_rows)
    pair_path = os.path.join(out_dir, "new_map_pairwise_empirical_pvalues.csv")
    pair_df.to_csv(pair_path, index=False)

    aggregate_rows = []
    for (outcome, new_map), group in all_resampled.groupby(["outcome", "new_map"], sort=False):
        delta = group["delta_resampled_stat"].to_numpy(dtype=float)
        row = {
            "outcome": outcome,
            "new_map": new_map,
            "n_prior_maps": int(group["old_map"].nunique()),
            "n_pairwise_resamples": int(np.isfinite(delta).sum()),
            "n_prior_maps_beaten_by_mean": int(
                pair_df[(pair_df["outcome"] == outcome) & (pair_df["new_map"] == new_map)]["new_beats_old_mean"].sum()
            ),
            "empirical_p_new_greater_pooled_priors": float(np.mean(delta <= 0)),
            "empirical_p_prior_greater_new": float(np.mean(delta >= 0)),
            "empirical_p_two_sided": float(min(1.0, 2.0 * min(np.mean(delta <= 0), np.mean(delta >= 0)))),
        }
        row.update(summarize_distribution(group["new_resampled_stat"], "new_resampled"))
        row.update(summarize_distribution(group["old_resampled_stat"], "pooled_prior_resampled"))
        row.update(summarize_distribution(delta, "delta_resampled"))
        aggregate_rows.append(row)

    aggregate_df = pd.DataFrame(aggregate_rows).sort_values(
        ["outcome", "delta_resampled_mean"], ascending=[True, False]
    )
    aggregate_path = os.path.join(out_dir, "new_map_outperformance_with_empirical_pvalues.csv")
    aggregate_df.to_csv(aggregate_path, index=False)

    wins_df = pair_df[pair_df["new_beats_old_mean"]].sort_values(
        ["outcome", "new_map", "delta_resampled_mean"], ascending=[True, True, False]
    )
    wins_path = os.path.join(out_dir, "new_map_individual_outperforming_comparisons.csv")
    wins_df.to_csv(wins_path, index=False)

    txt_path = os.path.join(out_dir, "new_map_outperformance_with_empirical_pvalues.txt")
    with open(txt_path, "w") as f:
        f.write(f"Outperformance report for: {out_dir}\n")
        f.write(f"Metric: {'R2' if DELTA_R2 else 'r'}; resampling: {RESAMPLE_METHOD}; n_iter: {N_ITER}\n\n")
        for _, row in aggregate_df.iterrows():
            f.write(
                f"{row['outcome']} | {row['new_map']} | "
                f"delta_mean={row['delta_resampled_mean']:.6f}, "
                f"p_new>prior={row['empirical_p_new_greater_pooled_priors']:.6f}, "
                f"p_two_sided={row['empirical_p_two_sided']:.6f}, "
                f"wins={int(row['n_prior_maps_beaten_by_mean'])}/{int(row['n_prior_maps'])}\n"
            )

    print(f"Wrote top-level reports to {out_dir}")


class MapPredictionAnalysis:
    """Top-level workflow: prep data, run stats, then plot figures."""

    def __init__(self):
        self.preparer = MapPredictionDataPreparer(
            input_path=INPUT_PATH,
            sheet=SHEET,
            out_dir=OUT_DIR,
            outcome_col=SYMPTOM_COLUMN,
            nifti_col=NIFTI_COL,
            mask_path=MASK_PATH,
            path_replacements=PATH_REPLACEMENTS,
            keep_rows=KEEP_ROWS,
            drop_rows=DROP_ROWS,
            covariates_list=COVARIATES_LIST,
            data_transform_method=DATA_TRANSFORM_METHOD,
            invert_outcome=INVERT_OUTCOME,
            correlation=CORRELATION,
            similarity=SIMILARITY,
        )
        self.stats_runner = MapPredictionStatsRunner(
            map_items(CANDIDATE_MAPS, "CANDIDATE_MAPS"),
            comparator_items(COMPARATOR_MAPS),
            mask_path=MASK_PATH,
            similarity=SIMILARITY,
            correlation=CORRELATION,
            resample_method=RESAMPLE_METHOD,
            n_iter=N_ITER,
            seed=SEED,
            delta_r2=DELTA_R2,
            skip_existing=SKIP_EXISTING,
            y_label=Y_LABEL,
        )
        self.figure_plotter = MapPredictionFigurePlotter(
            out_dir=OUT_DIR,
            correlation=CORRELATION,
            resample_method=RESAMPLE_METHOD,
            delta_r2=DELTA_R2,
            n_iter=N_ITER,
            y_label=Y_LABEL,
            draw_individual=DRAW_INDIVIDUAL_PLOTS,
            draw_aggregate=DRAW_AGGREGATE_PLOTS,
        )

    def prep_data(self):
        print("Running symptom: ", self.preparer.outcome_col)
        return self.preparer.prepare()

    def run_stats(self, prepared_data):
        return self.stats_runner.run_stats(prepared_data)

    def plot_figures(self, stats_results):
        self.preparer.write_skips()
        self.figure_plotter.plot_figures(stats_results)


def main():
    analysis = MapPredictionAnalysis()
    prepared_data = analysis.prep_data()
    stats_results = analysis.run_stats(prepared_data)
    analysis.plot_figures(stats_results)


if __name__ == "__main__":
    main()
