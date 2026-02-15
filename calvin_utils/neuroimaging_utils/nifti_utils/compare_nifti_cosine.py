import os
from itertools import combinations, product
import pprint

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


class NiftiCosineStats:
    """
    Compare within-column cosine similarity to cross-column cosine similarity.

    Parameters
    ----------
    csv_path : str
        Path to CSV containing NIfTI paths.
    col_a, col_b : str
        Column names for two NIfTI groups.
    threshold : float or None, default None
        If set, binarize data via data > threshold before cosine similarity.
    drop_zeros : bool, default True
        If True, zero vectors get dropped from comparisons.
    """

    def __init__(
        self,
        csv_path: str,
        *,
        col_a: str = "nifti_a",
        col_b: str = "nifti_b",
        threshold: float | None = None,
        drop_zeros: bool = True,
    ):
        self.csv_path = csv_path
        self.col_a = col_a
        self.col_b = col_b
        self.threshold = threshold
        self.drop_zeros = drop_zeros

        self.df = pd.read_csv(csv_path)
        self._validate_columns()

        self.within_a = None
        self.cross_ab = None
        self.ttest = None

    def _validate_columns(self):
        missing = {self.col_a, self.col_b} - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    def _load_vector(self, path: str) -> np.ndarray:
        if not isinstance(path, str) or not path:
            raise ValueError(f"Invalid NIfTI path: {path}")
        path = os.path.expanduser(path)
        data = nib.load(path).get_fdata().astype(np.float64)
        if self.threshold is not None:
            data = (data > self.threshold).astype(np.float64)
        return data.ravel()

    @staticmethod
    def _cosine(u: np.ndarray, v: np.ndarray) -> float:
        denom = np.linalg.norm(u) * np.linalg.norm(v)
        if denom == 0:
            return np.nan
        return float(np.dot(u, v) / denom)

    def _get_vectors(self, col: str) -> list[np.ndarray]:
        paths = [p for p in self.df[col].dropna().tolist() if isinstance(p, str) and p]
        vecs = [self._load_vector(p) for p in paths]
        if self.drop_zeros:
            vecs = [v for v in vecs if np.linalg.norm(v) > 0]
        return vecs

    def _compute_within(self, vecs: list[np.ndarray]) -> np.ndarray:
        sims = [self._cosine(a, b) for a, b in combinations(vecs, 2)]
        return np.asarray(sims, dtype=float)

    def _compute_cross(self, vecs_a: list[np.ndarray], vecs_b: list[np.ndarray]) -> np.ndarray:
        sims = [self._cosine(a, b) for a, b in product(vecs_a, vecs_b)]
        return np.asarray(sims, dtype=float)

    def run(self, name: str = "Cosine similarity: within A vs cross A-B"):
        vecs_a = self._get_vectors(self.col_a)
        vecs_b = self._get_vectors(self.col_b)
        self.within_a = self._compute_within(vecs_a)
        self.cross_ab = self._compute_cross(vecs_a, vecs_b)

        self.ttest = ttest_ind(
            self.within_a,
            self.cross_ab,
            equal_var=False,
            nan_policy="omit",
        )

        summary = {
            "n_within_a_pairs": int(len(self.within_a)),
            "n_cross_ab_pairs": int(len(self.cross_ab)),
            "within_a_mean": float(np.nanmean(self.within_a)) if len(self.within_a) else np.nan,
            "within_a_std": float(np.nanstd(self.within_a, ddof=1)) if len(self.within_a) > 1 else np.nan,
            "cross_ab_mean": float(np.nanmean(self.cross_ab)) if len(self.cross_ab) else np.nan,
            "cross_ab_std": float(np.nanstd(self.cross_ab, ddof=1)) if len(self.cross_ab) > 1 else np.nan,
            "welch_t": float(self.ttest.statistic) if self.ttest else np.nan,
            "welch_p": float(self.ttest.pvalue) if self.ttest else np.nan,
        }

        self._show_results(summary, name)
        return summary

    @staticmethod
    def _show_results(summary, name):
        print(f"\n----- {name} -----")
        pprint.pprint(summary, compact=True)


if __name__ == "__main__":
    stats = NiftiCosineStats(
        csv_path="/path/to/paths.csv",
        col_a="nifti_a",
        col_b="nifti_b",
        threshold=None,
    )
    stats.run()
