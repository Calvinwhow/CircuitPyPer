"""Correlation bar-plot helpers."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


class CorrelationBarPlot:
    """
    Bar-plot SciPy-based Pearson correlations between one column (x_col) and
    every numeric column in a DataFrame, with an optional thresholded view.
    """

    def __init__(self, data_df: pd.DataFrame, x_col: str, method: str = "spearman",
                 figsize=(10, 10), hi_figsize=(20, 20)):
        self.method = method
        self.data_df = data_df
        self.x_col = x_col
        self.figsize = figsize
        self.hi_figsize = hi_figsize

    # ---------- internal helpers ----------
    @staticmethod
    def _clean_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mask = ~np.isnan(a) & ~np.isnan(b)
        return a[mask], b[mask]

    @classmethod
    def _safe_spearman(cls, a: np.ndarray, b: np.ndarray) -> float:
        a, b = cls._clean_pair(a, b)
        if a.size < 2 or np.all(a == a[0]) or np.all(b == b[0]):
            return np.nan
        return stats.spearmanr(a, b)[0]

    @classmethod
    def _safe_pearson(cls, a: np.ndarray, b: np.ndarray) -> float:
        a, b = cls._clean_pair(a, b)
        if a.size < 2 or np.all(a == a[0]) or np.all(b == b[0]):
            return np.nan
        return stats.pearsonr(a, b)[0]

    def _compute_correlations(self) -> pd.Series:
        numeric_df = self.data_df.select_dtypes(include="number")
        if self.method == "spearman":
            corr_func = self._safe_spearman
        elif self.method == "pearson":
            corr_func = self._safe_pearson
        return (
            numeric_df
            .corr(method=corr_func)[self.x_col]
            .sort_values(ascending=False)
        )

    # ---------- public API ----------
    def run(self, save_path: str | None = None, show: bool = True) -> pd.Series:
        """
        Full bar plot of correlations with x_col.
        """
        corr = self._compute_correlations()
        self._plot(corr, self.figsize, save_path, show,
                   title=f"Correlation of {self.x_col} with All Numeric Columns")
        return corr

    def plot_threshold(self, threshold: float = 0.4,
                       save_path: str | None = None, show: bool = True) -> pd.Series:
        """
        Bar plot of correlations whose |r| ≥ threshold (x_col itself excluded).
        """
        corr = self._compute_correlations().drop(self.x_col, errors="ignore")
        filt = corr[corr.abs() >= threshold]
        self._plot(filt, self.hi_figsize, save_path, show,
                   title=f"Correlations with {self.x_col} (|r| ≥ {threshold})")
        return filt

    # ---------- plotting ----------
    def _plot(self, series: pd.Series, figsize, save_path, show, title):
        plt.figure(figsize=figsize, dpi=120)
        series.plot(kind="bar")
        plt.title(title)
        plt.ylabel("R")
        plt.xlabel("Columns")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close()
