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
from pathlib import Path
from itertools import product

TMPDIR = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(TMPDIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMPDIR / "xdg_cache"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from scipy.stats import pearsonr, spearmanr, ttest_ind
from tqdm import tqdm

if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [np.bool_, np.bytes_, np.str_, np.object_],
    }
if not hasattr(np, "maximum_sctype"):
    def _maximum_sctype(t):
        dtype = np.dtype(t)
        if np.issubdtype(dtype, np.complexfloating):
            return np.complex128
        if np.issubdtype(dtype, np.floating):
            return np.float64
        if np.issubdtype(dtype, np.unsignedinteger):
            return np.uint64
        if np.issubdtype(dtype, np.integer):
            return np.int64
        return dtype.type

    np.maximum_sctype = _maximum_sctype

import nibabel as nib

CIRCUIT_PYPER_DIR = Path(__file__).resolve().parents[1]
if str(CIRCUIT_PYPER_DIR) not in sys.path:
    sys.path.insert(0, str(CIRCUIT_PYPER_DIR))

from calvin_utils.permutation_analysis_utils.statsmodels_palm import CalvinStatsmodelsPalm
from calvin_utils.neuroimaging_utils.ccm_utils.stat_utils import CorrelationCalculator
from calvin_utils.neuroimaging_utils.ccm_utils.convergent_loocv import LOOCVAnalyzer
from calvin_utils.neuroimaging_utils.ccm_utils.delta_scatterplot import DeltaCorrelationScatter
from calvin_utils.plotting_utils.pair_superiority_plot import PairSuperiorityPlot
from calvin_utils.plotting_utils.simple_box_plot import SimpleBoxPlotWrapper


# =============================================================================
# CONFIG
# =============================================================================


# Input/output paths.
INPUT_PATH = "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimzation/optimized_master_list.csv" # Form: "/path/to/input.csv" or "/path/to/input.xlsx"
SHEET = None                            # Specify sheet if using excel (i.e. "Sheet1")
OUT_DIR = "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimzation/comparisons" # Form: "/path/to/output_dir"
WHOLE_BRAIN_MASK_PATH = "/Users/cu135/Software_Local/calvin_utils_project/circuit_pyper/resources/MNI152_T1_2mm_brain_mask.nii"
CEREBELLUM_MASK_PATH = "/Volumes/HowExp2/resources/atlases/atlases/Diedrichsen_2009/atl-Anatom_space-SUIT_dseg_coverage.nii.gz"
CEREBELLUM_ONLY = False                 # True masks patient images and compared maps to cerebellum and appends -cerebellumOnly to OUT_DIR.
MASK_PATH = CEREBELLUM_MASK_PATH if CEREBELLUM_ONLY else WHOLE_BRAIN_MASK_PATH
if CEREBELLUM_ONLY and not OUT_DIR.endswith("-cerebellumOnly"):
    OUT_DIR = f"{OUT_DIR}-cerebellumOnly"

# Model setup.
OUTCOME_LIST = []                       # Used only for flat NEW_MAPS_LIST. Nested NEW_MAPS_LIST below is preferred for batching.
NIFTI_COL = "Nifti_File_Path"           # Column containing patient neuroimaging files.
Y_LABEL = None                          # Defaults to each outcome name. Example: "Memory Outcome"
PATH_REPLACEMENTS = []                  # Example: [("/Volumes/HowExp/", "/Volumes/HowExp2/")]

# Maps to compare. Preferred form is {"DV_COLUMN": {"Map Name": "/path/to/map.nii.gz"}}.
CONTROL_MAPS_DIR = "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimzation/comparisons/control_maps"
TOTALBARS_MAPS = {
    "TotalBarsScore_New_Localization": "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimzation/symptom_on_lhs/totalbars/standardized-perm1000/regression_0/contrast_tval_0.nii.gz",
    "TotalBarsScore_New_Localization-FWE": "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimzation/symptom_on_lhs/totalbars/standardized-perm1000/regression_0/contrast_tval_FWE_0.nii.gz",
    "TotalBarsScore_Diedrichsen_Wholebrain": "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimzation/symptom_on_lhs/proper_regressions/TotalBarsScore-on-Nifti_File_Path/parcellated_files/contrast_tval_0_diedrichsen-wholebrain.nii.gz",
    "TotalBarsScore_Diedrichsen_Wholebrain-FWE": "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimzation/symptom_on_lhs/proper_regressions/TotalBarsScore-on-Nifti_File_Path/parcellated_files/contrast_tval_FWE_0_diedrichsen-wholebrain.nii.gz",
}
CNRS_MAPS = {
    "CNRS_New_Localization": "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimzation/symptom_on_lhs/cnrs/standardized-perm1000/regression_0/contrast_tval_0.nii.gz",
    "CNRS_New_Localization-FWE": "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimzation/symptom_on_lhs/cnrs/standardized-perm1000/regression_0/contrast_tval_FWE_0.nii.gz",
    "CNRS_Diedrichsen_Wholebrain": "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimzation/symptom_on_lhs/proper_regressions/CNRSTotScore-on-Nifti_File_Path/parcellated_files/contrast_tval_0_diedrichsen-wholebrain.nii.gz",
    "CNRS_Diedrichsen_Wholebrain-FWE": "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimzation/symptom_on_lhs/proper_regressions/CNRSTotScore-on-Nifti_File_Path/parcellated_files/contrast_tval_FWE_0_diedrichsen-wholebrain.nii.gz",
}
CCAS_MAPS = {
    "CCAS_New_Localization": "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimzation/symptom_on_lhs/ccas/standardized-perm1000/regression_0/contrast_tval_0.nii.gz",
    "CCAS_New_Localization-FWE": "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimzation/symptom_on_lhs/ccas/standardized-perm1000/regression_0/contrast_tval_FWE_0.nii.gz",
    "CCAS_Diedrichsen_Wholebrain": "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimzation/symptom_on_lhs/proper_regressions/TotalCCASFailScore-on-Nifti_File_Path/parcellated_files/contrast_tval_0_diedrichsen-wholebrain.nii.gz",
    "CCAS_Diedrichsen_Wholebrain-FWE": "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimzation/symptom_on_lhs/proper_regressions/TotalCCASFailScore-on-Nifti_File_Path/parcellated_files/contrast_tval_FWE_0_diedrichsen-wholebrain.nii.gz",
}
NEW_MAPS_LIST = {
    "TotalBarsScore": TOTALBARS_MAPS,
    "CNRSTotScore": CNRS_MAPS,
    "TotalCCASFailScore": CCAS_MAPS,
}
EXPAND_OUTCOME_GROUPS = False           # True for batches like Corbetta with many GDSS/NIH7 columns.
OUTCOME_GROUP_MAPS = {                  # Used only when EXPAND_OUTCOME_GROUPS=True.
    "gdss": CCAS_MAPS,                  # Any column starting with "gdss".
    "nih7": TOTALBARS_MAPS,             # Any column starting with "nih7".
    "animal_raw_acute": CNRS_MAPS,      # Exact column name.
}
OLD_MAPS_LIST = {
    path.name.replace(".nii.gz", "_nii_gz").replace(".nii", "_nii"): str(path)
    for path in sorted(Path(CONTROL_MAPS_DIR).glob("*"))
    if path.name.endswith((".nii", ".nii.gz"))
}
NEW_MAP_NAMES = None                    # Optional list matching NEW_MAPS_LIST if NEW_MAPS_LIST is not a dict.
OLD_MAP_NAMES = None                    # Optional list matching OLD_MAPS_LIST if OLD_MAPS_LIST is not a dict.

# Optional preprocessing.
DROP_ROWS = []                          # Conditions for dropping rows. Example: [("group", "equal", "control"), ("age", "below", 18)]
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
DRAW_AGGREGATE_PLOTS = True             # Aggregate boxplots/distribution plots for each new map.


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


def read_spreadsheet(out_dir):
    """Read the input spreadsheet using the same CalvinStatsmodelsPalm importer pattern as 05b."""
    os.makedirs(out_dir, exist_ok=True)
    cal_palm = CalvinStatsmodelsPalm(input_csv_path=INPUT_PATH, output_dir=out_dir, sheet=SHEET)
    return cal_palm, cal_palm.read_and_display_data()


def prepare_data(cal_palm, data_df, outcome):
    """Apply the configured spreadsheet-level preprocessing."""
    data_df = data_df.copy()
    for old, new in PATH_REPLACEMENTS:
        data_df[NIFTI_COL] = data_df[NIFTI_COL].astype(str).str.replace(old, new, regex=False)

    for column, value in KEEP_ROWS:
        before = len(data_df)
        data_df = data_df[data_df[column] == value].copy()
        print(f"Keeping {column} == {value}: {len(data_df)} of {before} rows")

    if INVERT_OUTCOME:
        print(f"INVERT_OUTCOME=True, MULTIPLYING {outcome} BY -1")
        data_df[outcome] = data_df[outcome] * -1

    cal_palm.df = data_df
    drop_nan_list = [outcome, NIFTI_COL] + list(COVARIATES_LIST)
    data_df = cal_palm.drop_nans_from_columns(columns_to_drop_from=drop_nan_list)

    if DROP_ROWS:
        for column, condition, value in DROP_ROWS:
            data_df, _ = cal_palm.drop_rows_based_on_value(column, condition, value)

    return data_df


def has_enough_data(data_df, outcome):
    """Return True when an outcome has enough rows and variation to analyze."""
    if len(data_df) < 3:
        return False
    values = pd.to_numeric(data_df[outcome], errors="coerce")
    return values.notna().sum() >= 3 and values.nunique(dropna=True) > 1


def normalize_maps(map_list, name_list=None):
    """Return a list of (name, path) tuples from dict or list config."""
    if isinstance(map_list, dict):
        return [(str(name), str(path)) for name, path in map_list.items()]

    if name_list is not None and len(name_list) != len(map_list):
        raise ValueError("Name list must match map list length.")

    normalized = []
    for i, path in enumerate(map_list):
        name = str(name_list[i]) if name_list is not None else Path(str(path)).name.split(".nii")[0]
        normalized.append((name, str(path)))
    return normalized


def normalize_new_maps_by_outcome():
    """Return [(outcome, [(map_name, map_path), ...]), ...] from flat or nested config."""
    if EXPAND_OUTCOME_GROUPS:
        return [(str(outcome), normalize_maps(maps)) for outcome, maps in build_expanded_outcome_maps().items()]

    if isinstance(NEW_MAPS_LIST, dict):
        if all(isinstance(value, dict) for value in NEW_MAPS_LIST.values()):
            return [(str(outcome), normalize_maps(maps)) for outcome, maps in NEW_MAPS_LIST.items()]
        if not OUTCOME_LIST:
            raise ValueError("Flat NEW_MAPS_LIST requires OUTCOME_LIST. Prefer {'DV': {'Map Name': '/path'}}.")
        return [(str(outcome), normalize_maps(NEW_MAPS_LIST, NEW_MAP_NAMES)) for outcome in OUTCOME_LIST]

    if not OUTCOME_LIST:
        raise ValueError("List NEW_MAPS_LIST requires OUTCOME_LIST. Prefer {'DV': {'Map Name': '/path'}}.")
    return [(str(outcome), normalize_maps(NEW_MAPS_LIST, NEW_MAP_NAMES)) for outcome in OUTCOME_LIST]


def build_expanded_outcome_maps():
    """Build outcome-to-map config from spreadsheet columns and OUTCOME_GROUP_MAPS."""
    columns = pd.read_csv(INPUT_PATH, nrows=0).columns
    expanded = {}
    for pattern, maps in OUTCOME_GROUP_MAPS.items():
        matched_columns = [
            col for col in columns
            if col == pattern or col.lower().startswith(f"{str(pattern).lower()}_")
        ]
        for col in matched_columns:
            expanded[col] = maps
    if not expanded:
        raise ValueError("EXPAND_OUTCOME_GROUPS=True, but no spreadsheet columns matched OUTCOME_GROUP_MAPS.")
    return expanded


def safe_name(value):
    """Make a readable string safe for output paths."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def build_correlation_maps(data_loader):
    """Generate the CCM correlation-map dictionary from spreadsheet-loaded data."""
    correlation_calculator = CorrelationCalculator(method=CORRELATION, verbose=False)
    return correlation_calculator.generate_correlation_maps(data_loader)


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


def correlate_vectors(x, y):
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
    if CORRELATION == "pearson":
        return pearsonr(x, y)[0]
    if CORRELATION == "spearman":
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


def compute_comparison_stats(sim1, sim2, y, method, n_iter, delta_r2):
    """Compute observed and resampled map-prediction statistics from precomputed similarities."""
    observed_stat1 = correlate_vectors(sim1, y)
    observed_stat2 = correlate_vectors(sim2, y)
    if delta_r2:
        observed_stat1 = observed_stat1 ** 2
        observed_stat2 = observed_stat2 ** 2

    stat1_values = []
    stat2_values = []
    iter_count = 0
    with tqdm(total=n_iter) as pbar:
        while iter_count < n_iter:
            sub_sim1, sub_sim2, sub_y = shuffle_for_comparison(sim1, sim2, y, method)
            stat1 = correlate_vectors(sub_sim1, sub_y)
            stat2 = correlate_vectors(sub_sim2, sub_y)
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


def run_fast_cosine_comparison(outcome, data_loader, map1_name, map_one, map2_name, map_two):
    """Run the cosine comparison without recomputing voxelwise similarities per resample."""
    data = data_loader.load_dataset(data_loader.dataset_names_list[0])
    niftis = CorrelationCalculator._check_for_nans(data["niftis"], nanpolicy="remove", verbose=False)
    y = CorrelationCalculator._check_for_nans(data["indep_var"], nanpolicy="remove", verbose=False).flatten()
    roi1_map = CorrelationCalculator._check_for_nans(load_masked_map(map_one, MASK_PATH), nanpolicy="remove", verbose=False)
    roi2_map = CorrelationCalculator._check_for_nans(load_masked_map(map_two, MASK_PATH), nanpolicy="remove", verbose=False)

    sim1 = cosine_similarity_matrix(niftis, roi1_map)
    sim2 = cosine_similarity_matrix(niftis, roi2_map)
    stats = compute_comparison_stats(sim1, sim2, y, RESAMPLE_METHOD, N_ITER, DELTA_R2)
    return sim1, sim2, y, stats


def run_single_comparison(outcome, data_loader, corr_map_dict, map1_name, map_one, map2_name, map_two, map_out_dir):
    """Run the resampling comparison and save individual visualizations."""
    comparison_name = f"{safe_name(outcome)}__{safe_name(map1_name)}-vs-{safe_name(map2_name)}"
    comparison_out_dir = os.path.join(map_out_dir, comparison_name)
    if SKIP_EXISTING and os.path.isdir(comparison_out_dir):
        print(f"Skipping existing comparison: {comparison_out_dir}")
        return None

    os.makedirs(comparison_out_dir, exist_ok=True)
    np.random.seed(SEED)

    if SIMILARITY == "cosine":
        observed_x1, observed_x2, observed_y, stats = run_fast_cosine_comparison(outcome, data_loader, map1_name, map_one, map2_name, map_two)
        r_values = {"roi1": stats["stat1_values"], "roi2": stats["stat2_values"]}
        observed_r_values = {"roi1": [stats["observed_stat1"]], "roi2": [stats["observed_stat2"]]}
    else:
        loocv_analyzer = LOOCVAnalyzer(
            corr_map_dict,
            data_loader,
            similarity=SIMILARITY,
            method=CORRELATION,
            out_dir=comparison_out_dir,
            mask_path=MASK_PATH,
            roi_path=None,
            ylabel=Y_LABEL or outcome,
        )
        loocv_analyzer.compare_roi_correlations(
            roi1=map_one,
            roi2=map_two,
            method=RESAMPLE_METHOD,
            n_iter=N_ITER,
            seed=SEED,
            delta_r2=DELTA_R2,
        )
        observed_x1 = loocv_analyzer.observed_x1
        observed_x2 = loocv_analyzer.observed_x2
        observed_y = loocv_analyzer.observed_y
        r_values = loocv_analyzer.r_values
        observed_r_values = loocv_analyzer.observed_r_values

    if DRAW_INDIVIDUAL_PLOTS:
        visualizer = PairSuperiorityPlot(
            stat_array_1=np.array(r_values["roi1"]),
            stat_array_2=np.array(r_values["roi2"]),
            model1_name=map1_name,
            model2_name=map2_name,
            out_dir=comparison_out_dir,
            observed_stat_array=[
                np.array(observed_r_values["roi1"]),
                np.array(observed_r_values["roi2"]),
            ],
            method=RESAMPLE_METHOD,
        )
        visualizer.draw(verbose=False)
        plt.close("all")

        if is_variable_vector(observed_x1) and is_variable_vector(observed_x2) and is_variable_vector(observed_y):
            vis = DeltaCorrelationScatter(
                x_array_1=observed_x1,
                x_array_2=observed_x2,
                y_array=observed_y,
                y_label=Y_LABEL or outcome,
                label_1=map1_name,
                label_2=map2_name,
                stat_label="r",
                out_dir=comparison_out_dir,
                method=CORRELATION,
            )
            vis.draw(show=False)
            plt.close("all")
        else:
            skip_reason = (
                "Skipped DeltaCorrelationScatter because at least one observed "
                "similarity/outcome vector was constant or had fewer than 3 finite values.\n"
            )
            with open(os.path.join(comparison_out_dir, "delta_scatterplot_skipped.txt"), "w") as f:
                f.write(skip_reason)
            print(skip_reason.strip())

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
                    "new_map": map1_name,
                    "old_map": map2_name,
                    "new_observed_stat": new_observed[dataset_idx],
                    "old_observed_stat": old_observed[dataset_idx],
                    "delta_observed_stat": new_observed[dataset_idx] - old_observed[dataset_idx],
                    "comparison_out_dir": comparison_out_dir,
                }
            )

    resample_df = pd.DataFrame(
        {
            "outcome": outcome,
            "new_map": map1_name,
            "old_map": map2_name,
            "new_resampled_stat": new_resampled,
            "old_resampled_stat": old_resampled,
            "delta_resampled_stat": new_resampled - old_resampled,
        }
    )
    resample_df.to_csv(os.path.join(comparison_out_dir, "resampled_comparison_values.csv"), index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(comparison_out_dir, "observed_comparison_values.csv"), index=False)
    return summary_df, resample_df


def plot_bootstrapped_performance_distribution(resample_df, aggregate_dir):
    """Plot one bootstrap R/R2 distribution for each tested map."""
    if resample_df.empty:
        return

    rows = []
    first_old_map = resample_df["old_map"].iloc[0]
    new_map_name = resample_df["new_map"].iloc[0]
    new_values = resample_df.loc[resample_df["old_map"] == first_old_map, "new_resampled_stat"].dropna()
    rows += [{"map": new_map_name, "type": "new", "value": value} for value in new_values]
    for old_map, old_df in resample_df.groupby("old_map", sort=False):
        rows += [{"map": old_map, "type": "prior", "value": value} for value in old_df["old_resampled_stat"].dropna()]

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return

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

    prior_values = plot_df.loc[plot_df["type"] == "prior", "value"].to_numpy(dtype=float)
    new_values = plot_df.loc[plot_df["type"] == "new", "value"].to_numpy(dtype=float)
    t_stat, p_val = ttest_ind(new_values, prior_values, equal_var=False)
    delta = float(np.mean(new_values) - np.mean(prior_values))
    ax.text(
        0.98,
        0.04,
        f"New vs pooled priors\nMean delta = {delta:.4f}\nt = {t_stat:.2f}, p = {p_val:.2e}",
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


def plot_overall_results(summary_df, resample_df, map_out_dir):
    """Plot the aggregate new-map vs old-map performance across all comparisons."""
    aggregate_dir = os.path.join(map_out_dir, "aggregate")
    os.makedirs(aggregate_dir, exist_ok=True)

    summary_df.to_csv(os.path.join(aggregate_dir, "observed_comparison_summary.csv"), index=False)
    resample_df.to_csv(os.path.join(aggregate_dir, "resampled_comparison_summary.csv"), index=False)

    plotter = SimpleBoxPlotWrapper(summary_df)
    plotter.plot(
        columns=["delta_observed_stat"],
        dataset_name="New Map Minus Prior Maps",
        group_labels=["Delta"],
        out_dir=os.path.join(aggregate_dir, "overall_observed_delta_boxplot.svg"),
        ylabel="Delta R2" if DELTA_R2 else "Delta r",
    )

    plotter = SimpleBoxPlotWrapper(summary_df)
    plotter.plot(
        columns=[("new_observed_stat", "old_observed_stat")],
        dataset_name="New Map vs Prior Maps",
        group_labels=["Observed"],
        pair_names=["New map", "Prior maps"],
        out_dir=os.path.join(aggregate_dir, "overall_observed_pair_boxplot.svg"),
        ylabel="R2" if DELTA_R2 else "r",
    )

    plot_bootstrapped_performance_distribution(resample_df, aggregate_dir)

    plotter = SimpleBoxPlotWrapper(resample_df)
    plotter.plot(
        columns=["delta_resampled_stat"],
        dataset_name="Resampled New Map Minus Prior Maps",
        group_labels=["Delta"],
        out_dir=os.path.join(aggregate_dir, "overall_resampled_delta_boxplot.svg"),
        ylabel="Delta R2" if DELTA_R2 else "Delta r",
    )


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


def main():
    analyses = normalize_new_maps_by_outcome()
    old_maps = normalize_maps(OLD_MAPS_LIST, OLD_MAP_NAMES)
    if not analyses:
        raise ValueError("NEW_MAPS_LIST is empty. Add at least one DV-specific new localization map.")
    if not old_maps:
        raise ValueError("OLD_MAPS_LIST is empty. Add at least one prior-work map.")

    cal_palm, raw_df = read_spreadsheet(OUT_DIR)
    skipped_rows = []

    for outcome, new_maps in analyses:
        print("Running outcome: ", outcome)
        outcome_dir = os.path.join(OUT_DIR, safe_name(outcome))
        os.makedirs(outcome_dir, exist_ok=True)
        if not new_maps:
            print(f"No new maps configured for {outcome}. Skipping.")
            continue

        data_df = prepare_data(cal_palm, raw_df, outcome)
        if not has_enough_data(data_df, outcome):
            usable_values = pd.to_numeric(data_df[outcome], errors="coerce")
            skipped_rows.append(
                {
                    "outcome": outcome,
                    "rows_after_filter": len(data_df),
                    "non_null_outcome_rows": int(usable_values.notna().sum()),
                    "unique_outcome_values": int(usable_values.nunique(dropna=True)),
                    "reason": "fewer than 3 usable rows or no outcome variation after filtering",
                }
            )
            print(f"Skipping {outcome}: fewer than 3 usable rows or no outcome variation after filtering.")
            continue

        data_loader = SpreadsheetDataLoader(
            data_df=data_df,
            outcome_col=outcome,
            nifti_col=NIFTI_COL,
            mask_path=MASK_PATH,
            covariates_list=COVARIATES_LIST,
            data_transform_method=DATA_TRANSFORM_METHOD,
            dataset_name=outcome,
        )
        if SIMILARITY == "cosine":
            corr_map_dict = {outcome: np.array([0.0])}
        else:
            corr_map_dict = build_correlation_maps(data_loader)

        for map1_name, map_one in new_maps:
            map_out_dir = os.path.join(outcome_dir, safe_name(map1_name))
            os.makedirs(map_out_dir, exist_ok=True)
            map_summary_dfs = []
            map_resample_dfs = []

            for map2_name, map_two in old_maps:
                print("Comparing maps: ", map1_name, " vs ", map2_name)
                result = run_single_comparison(outcome, data_loader, corr_map_dict, map1_name, map_one, map2_name, map_two, map_out_dir)
                if result is None:
                    continue
                summary_df, resample_df = result
                map_summary_dfs.append(summary_df)
                map_resample_dfs.append(resample_df)

            if map_summary_dfs and map_resample_dfs:
                map_summary_df = pd.concat(map_summary_dfs, ignore_index=True)
                map_resample_df = pd.concat(map_resample_dfs, ignore_index=True)
                if DRAW_AGGREGATE_PLOTS:
                    plot_overall_results(map_summary_df, map_resample_df, map_out_dir)
                else:
                    aggregate_dir = os.path.join(map_out_dir, "aggregate")
                    os.makedirs(aggregate_dir, exist_ok=True)
                    map_summary_df.to_csv(os.path.join(aggregate_dir, "observed_comparison_summary.csv"), index=False)
                    map_resample_df.to_csv(os.path.join(aggregate_dir, "resampled_comparison_summary.csv"), index=False)

    if skipped_rows:
        pd.DataFrame(skipped_rows).to_csv(os.path.join(OUT_DIR, "skipped_outcomes.csv"), index=False)
    write_outperformance_reports(OUT_DIR)


if __name__ == "__main__":
    main()
