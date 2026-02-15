import os
from typing import List

import numpy as np
import pandas as pd

from nilearn import image
from calvin_utils.file_utils.import_functions import GiiNiiFileImport
from calvin_utils.statistical_utils.scatterplot import simple_scatter
from calvin_utils.nifti_utils.generate_mask import GenerateMask


class DiceVTA:
    """
    Compute pairwise Dice overlap between two VTA columns in a CSV and
    correlate overlap with a dependent variable.

    The CSV is expected to contain:
    - subject column
    - dependent variable column
    - two VTA path columns
    """

    def __init__(
        self,
        csv_path: str,
        *,
        vta_col1: str,
        vta_col2: str,
        dep_var_col: str,
        subject_col: str = "subject",
        output_dir: str | None = None,
        mask_path: str | None = None,
        threshold: float = 0.0,
        empty_value: float = 1.0,
        scatter_title: str | None = None,
        scatter_x_label: str | None = None,
        scatter_y_label: str | None = None,
    ):
        self.csv_path = csv_path
        self.vta_col1 = vta_col1
        self.vta_col2 = vta_col2
        self.dep_var_col = dep_var_col
        self.subject_col = subject_col
        self.output_dir = output_dir
        self.mask_path = mask_path
        self.threshold = threshold
        self.empty_value = empty_value
        self.scatter_title = scatter_title
        self.scatter_x_label = scatter_x_label
        self.scatter_y_label = scatter_y_label

        self._df = None
        self._vta_df = None
        self._overlap_df = None
        self._vta_a_cols = None
        self._vta_b_cols = None
        self._mask_img = None
        self._mask_indices = None

    # ---- properties ----
    @property
    def csv_path(self):
        return self._csv_path

    @csv_path.setter
    def csv_path(self, value):
        if not isinstance(value, str) or not value:
            raise ValueError("csv_path must be a non-empty string")
        self._csv_path = value

    @property
    def subject_col(self):
        return self._subject_col

    @subject_col.setter
    def subject_col(self, value):
        if not isinstance(value, str) or not value:
            raise ValueError("subject_col must be a non-empty string")
        self._subject_col = value

    @property
    def vta_col1(self):
        return self._vta_col1

    @vta_col1.setter
    def vta_col1(self, value):
        if not isinstance(value, str) or not value:
            raise ValueError("vta_col1 must be a non-empty string")
        self._vta_col1 = value

    @property
    def vta_col2(self):
        return self._vta_col2

    @vta_col2.setter
    def vta_col2(self, value):
        if not isinstance(value, str) or not value:
            raise ValueError("vta_col2 must be a non-empty string")
        self._vta_col2 = value

    @property
    def dep_var_col(self):
        return self._dep_var_col

    @dep_var_col.setter
    def dep_var_col(self, value):
        if not isinstance(value, str) or not value:
            raise ValueError("dep_var_col must be a non-empty string")
        self._dep_var_col = value

    @property
    def output_dir(self):
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value):
        if value is None:
            self._output_dir = None
            return
        if not isinstance(value, str) or not value:
            raise ValueError("output_dir must be a non-empty string or None")
        self._output_dir = value

    @property
    def mask_path(self):
        return self._mask_path

    @mask_path.setter
    def mask_path(self, value):
        if value is None:
            self._mask_path = None
            return
        if not isinstance(value, str) or not value:
            raise ValueError("mask_path must be a non-empty string or None")
        self._mask_path = value

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        value = float(value)
        self._threshold = value

    @property
    def empty_value(self):
        return self._empty_value

    @empty_value.setter
    def empty_value(self, value):
        self._empty_value = float(value)

    @property
    def scatter_title(self):
        return self._scatter_title

    @scatter_title.setter
    def scatter_title(self, value):
        if value is None:
            self._scatter_title = None
            return
        if not isinstance(value, str) or not value:
            raise ValueError("scatter_title must be a non-empty string or None")
        self._scatter_title = value

    @property
    def scatter_x_label(self):
        return self._scatter_x_label

    @scatter_x_label.setter
    def scatter_x_label(self, value):
        if value is None:
            self._scatter_x_label = None
            return
        if not isinstance(value, str) or not value:
            raise ValueError("scatter_x_label must be a non-empty string or None")
        self._scatter_x_label = value

    @property
    def scatter_y_label(self):
        return self._scatter_y_label

    @scatter_y_label.setter
    def scatter_y_label(self, value):
        if value is None:
            self._scatter_y_label = None
            return
        if not isinstance(value, str) or not value:
            raise ValueError("scatter_y_label must be a non-empty string or None")
        self._scatter_y_label = value

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, value):
        self._df = value

    @property
    def vta_df(self):
        return self._vta_df

    @vta_df.setter
    def vta_df(self, value):
        self._vta_df = value

    @property
    def overlap_df(self):
        return self._overlap_df

    @overlap_df.setter
    def overlap_df(self, value):
        self._overlap_df = value

    # ---- main API ----
    def run(self):
        self._load_inputs()
        self._import_vtas()
        self._compute_overlap()
        self._save_and_correlate()
        return self.overlap_df

    # ---- main internal methods ----
    def _load_inputs(self):
        self.df = self._read_csv()
        self._validate_columns()
        self._drop_missing_vtas()

    def _import_vtas(self):
        vta_paths = self._stack_vta_paths()
        if self.mask_path is None:
            self._generate_mask_from_paths(vta_paths)
        self.vta_df, mapped_cols = self._import_vta_dataframe_with_mask(vta_paths)
        split_idx = len(self._col1_paths)
        self._vta_a_cols, self._vta_b_cols = mapped_cols[:split_idx], mapped_cols[split_idx:]

    def _compute_overlap(self):
        overlap = self._pairwise_dice()
        self.overlap_df = self._build_overlap_df(overlap)

    def _save_and_correlate(self):
        out_path = self._save_overlap_csv()
        self._run_scatterplot()
        return out_path

    # ---- helpers: load ----
    def _read_csv(self) -> pd.DataFrame:
        return pd.read_csv(self.csv_path)

    def _validate_columns(self):
        required = {self.subject_col, self.dep_var_col, self.vta_col1, self.vta_col2}
        missing = required - set(self._df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    def _drop_missing_vtas(self):
        self._df = self._df.dropna(subset=[self.vta_col1, self.vta_col2])

    # ---- helpers: import ----
    def _stack_vta_paths(self) -> List[str]:
        col1_paths = self._df[self.vta_col1].tolist()
        col2_paths = self._df[self.vta_col2].tolist()
        self._col1_paths = col1_paths
        self._col2_paths = col2_paths
        return col1_paths + col2_paths

    def _import_vta_dataframe(self, vta_paths: List[str]) -> pd.DataFrame:
        importer = GiiNiiFileImport(import_path=None, file_column=None, file_pattern=None)
        return importer.import_matrices(vta_paths)

    def _import_vta_dataframe_with_mask(self, vta_paths: List[str]) -> tuple[pd.DataFrame, List[str]]:
        mask_img, mask_idx = self._load_mask()
        data_dict = {}
        mapped_cols = []
        used = {}
        for path in vta_paths:
            col_name = self._unique_column_name(path, used)
            vta_img = image.load_img(path)
            resampled = image.resample_to_img(vta_img, mask_img, interpolation="nearest")
            data = resampled.get_fdata().flatten()
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            data_dict[col_name] = data[mask_idx]
            mapped_cols.append(col_name)
        return pd.DataFrame(data_dict), mapped_cols

    def _generate_mask_from_paths(self, vta_paths: List[str]):
        gen = GenerateMask(vta_paths, threshold=None, verbose=False)
        mask_img, mask_idx = gen.run()
        self._mask_img = mask_img
        self._mask_indices = mask_idx

    def _unique_column_name(self, path: str, used: dict) -> str:
        base = os.path.basename(path)
        count = used.get(base, 0)
        used[base] = count + 1
        if count == 0:
            return base
        return f"{base}__dup{count}"

    def _load_mask(self):
        if self._mask_img is None or self._mask_indices is None:
            if self.mask_path is None:
                raise ValueError("mask_path is None and no mask has been generated.")
            mask_img = image.load_img(self.mask_path)
            mask_data = mask_img.get_fdata().flatten()
            mask_idx = mask_data > 0
            self._mask_img = mask_img
            self._mask_indices = mask_idx
        return self._mask_img, self._mask_indices

    def _resolve_imported_columns(self, vta_paths: List[str]) -> tuple[list[str], list[str]]:
        columns = list(self._vta_df.columns)
        used = {}
        mapped_cols = [self._resolve_column_name(path, columns, used) for path in vta_paths]
        split_idx = len(self._col1_paths)
        return mapped_cols[:split_idx], mapped_cols[split_idx:]

    def _resolve_column_name(self, path: str, columns: List[str], used: dict) -> str:
        candidates = self._match_columns_for_path(path, columns)
        if candidates:
            idx = used.get(path, 0)
            if idx < len(candidates):
                used[path] = idx + 1
                return candidates[idx]
        raise ValueError(
            "Could not resolve imported VTA column for path: "
            f"{path}. Available columns: {columns[:5]}..."
        )

    def _match_columns_for_path(self, path: str, columns: List[str]) -> List[str]:
        base = os.path.basename(path)
        normalized = os.path.abspath(os.path.expanduser(path))
        matches = []
        for col in columns:
            if col == path or col == base:
                matches.append(col)
                continue
            if col.startswith(path) or col.startswith(base):
                matches.append(col)
                continue
            col_norm = os.path.abspath(os.path.expanduser(col))
            if col_norm == normalized:
                matches.append(col)
        return matches

    # ---- helpers: compute ----
    def _pairwise_dice(self) -> np.ndarray:
        dice_vals = []
        for idx in range(len(self._col1_paths)):
            col_a = self._vta_a_cols[idx]
            col_b = self._vta_b_cols[idx]
            arr_a = self._vta_df[col_a].to_numpy()
            arr_b = self._vta_df[col_b].to_numpy()
            dice_vals.append(self._dice_from_arrays(arr_a, arr_b))
        return np.asarray(dice_vals, dtype=float)

    def _dice_from_arrays(self, arr_a: np.ndarray, arr_b: np.ndarray) -> float:
        a = arr_a > self.threshold
        b = arr_b > self.threshold
        a_sum = int(a.sum())
        b_sum = int(b.sum())
        if a_sum == 0 and b_sum == 0:
            return self.empty_value
        inter = int(np.logical_and(a, b).sum())
        return (2.0 * inter) / (a_sum + b_sum)

    def _build_overlap_df(self, overlap: np.ndarray) -> pd.DataFrame:
        overlap_col = self._overlap_col_name()
        out_df = self._df[[self.subject_col, self.dep_var_col, self.vta_col1, self.vta_col2]].copy()
        out_df[overlap_col] = overlap
        return out_df

    # ---- helpers: output + correlation ----
    def _overlap_col_name(self) -> str:
        return f"overlap_{self.vta_col1}_vs_{self.vta_col2}"

    def _resolve_output_dir(self) -> str:
        if self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            return self.output_dir
        return os.path.dirname(os.path.abspath(self.csv_path))

    def _save_overlap_csv(self) -> str:
        filename = f"overlap-{self.vta_col1}-{self.vta_col2}.csv"
        out_path = os.path.join(self._resolve_output_dir(), filename)
        self._overlap_df.to_csv(out_path, index=False)
        return out_path

    def _run_scatterplot(self):
        overlap_col = self._overlap_col_name()
        dataset_name = self.scatter_title or f"overlap-{self.vta_col1}-{self.vta_col2}"
        x_label = self.scatter_x_label or "Dice overlap"
        y_label = self.scatter_y_label or self.dep_var_col
        simple_scatter(
            df=self._overlap_df,
            x_col=overlap_col,
            y_col=self.dep_var_col,
            dataset_name=dataset_name,
            out_dir=self._resolve_output_dir(),
            x_label=x_label,
            y_label=y_label,
        )
