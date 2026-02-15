import json
import os
from typing import Iterable, List

import numpy as np
import pandas as pd


class ModifyStimParams:
    """
    Modify stimulation parameter JSON files referenced in a CSV.

    Public methods
    --------------
    - average_pairs: average two JSON columns elementwise and write new JSONs.
    - index_series: keep only specified indices and write new JSONs.
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._df = None

    # ---- setters/getters ----
    @property
    def csv_path(self):
        return self._csv_path

    @csv_path.setter
    def csv_path(self, value):
        if not isinstance(value, str) or not value:
            raise ValueError("csv_path must be a non-empty string")
        self._csv_path = value

    @property
    def df(self):
        return self._df

    # ---- public API ----
    def average_pairs(
        self,
        left_col: str,
        right_col: str,
        output_col: str | None = None,
        suffix: str = "_average",
    ) -> pd.DataFrame:
        """
        Average two JSON columns elementwise and write out new JSONs.
        The new column defaults to f"{left_col}{suffix}".
        """
        self._load_csv()
        self._validate_columns([left_col, right_col])
        out_col = output_col or f"{self._common_base(left_col, right_col)}{suffix}"

        new_paths = []
        for _, row in self._df.iterrows():
            left_path = row[left_col]
            right_path = row[right_col]
            if not self._valid_path(left_path) or not self._valid_path(right_path):
                new_paths.append("")
                continue
            left_vals = self._read_json_list(left_path)
            right_vals = self._read_json_list(right_path)
            self._validate_same_length(left_vals, right_vals, left_path, right_path)
            avg_vals = (np.array(left_vals, dtype=float) + np.array(right_vals, dtype=float)) / 2.0
            out_path = self._derive_output_path_pair(left_path, right_path, suffix)
            self._write_json_list(out_path, avg_vals.tolist())
            new_paths.append(out_path)

        self._df[out_col] = new_paths
        self._save_csv()
        return self._df

    def index_series(
        self,
        json_col: str,
        indices: Iterable[int],
        output_col: str | None = None,
        suffix: str = "_indexed",
    ) -> pd.DataFrame:
        """
        Keep only specified indices from a JSON column and write out new JSONs.
        The new column defaults to f"{json_col}{suffix}".
        """
        self._load_csv()
        self._validate_columns([json_col])
        out_col = output_col or f"{json_col}{suffix}"
        idx_list = self._normalize_indices(indices)

        new_paths = []
        for _, row in self._df.iterrows():
            src_path = row[json_col]
            if not self._valid_path(src_path):
                new_paths.append("")
                continue
            vals = self._read_json_list(src_path)
            kept = [vals[i] for i in idx_list if i < len(vals)]
            out_path = self._derive_output_path(src_path, suffix)
            self._write_json_list(out_path, kept)
            new_paths.append(out_path)

        self._df[out_col] = new_paths
        self._save_csv()
        return self._df

    # ---- internal helpers ----
    def _load_csv(self):
        if self._df is None:
            self._df = pd.read_csv(self.csv_path)

    def _save_csv(self):
        self._df.to_csv(self.csv_path, index=False)

    def _validate_columns(self, cols: List[str]):
        missing = set(cols) - set(self._df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    @staticmethod
    def _valid_path(path) -> bool:
        return isinstance(path, str) and path.strip()

    @staticmethod
    def _read_json_list(path: str) -> List[float]:
        path = os.path.expanduser(path)
        with open(path, "r") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"JSON does not contain a list: {path}")
        return [float(x) for x in data]

    @staticmethod
    def _write_json_list(path: str, data: List[float]):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    @staticmethod
    def _derive_output_path(src_path: str, suffix: str) -> str:
        src_path = os.path.expanduser(src_path)
        base, ext = os.path.splitext(src_path)
        if ext.lower() != ".json":
            return f"{src_path}{suffix}.json"
        return f"{base}{suffix}{ext}"

    @staticmethod
    def _common_base(a: str, b: str) -> str:
        prefix = os.path.commonprefix([a, b])
        return prefix.rstrip("_-./")

    def _derive_output_path_pair(self, left_path: str, right_path: str, suffix: str) -> str:
        left_path = os.path.expanduser(left_path)
        right_path = os.path.expanduser(right_path)
        left_base, left_ext = os.path.splitext(left_path)
        right_base, right_ext = os.path.splitext(right_path)
        common = self._common_base(left_base, right_base)
        if common:
            ext = left_ext if left_ext.lower() == ".json" else ".json"
            return f"{common}{suffix}{ext}"
        return self._derive_output_path(left_path, suffix)

    @staticmethod
    def _validate_same_length(a: List[float], b: List[float], a_path: str, b_path: str):
        if len(a) != len(b):
            raise ValueError(
                f"Mismatched list lengths: {len(a)} vs {len(b)} for {a_path} and {b_path}"
            )

    @staticmethod
    def _normalize_indices(indices: Iterable[int]) -> List[int]:
        if indices is None:
            raise ValueError("indices must be provided")
        idx_list = [int(i) for i in indices]
        if not idx_list:
            raise ValueError("indices must contain at least one index")
        if any(i < 0 for i in idx_list):
            raise ValueError("indices must be non-negative")
        return idx_list
