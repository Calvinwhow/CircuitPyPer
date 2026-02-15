import os
from typing import Iterable, List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_1samp


class SimpleHistogramPlot:
    """
    Create histograms with KDE overlays in a consistent style.

    - If one column: one-sample t-test vs ttest_ref.
    - If multiple columns: one-way ANOVA across columns.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        value_cols: Iterable[str],
        dataset_name: str,
        out_dir: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        figsize: tuple[int, int] = (6, 6),
        title_fontsize: int = 20,
        label_fontsize: int = 20,
        tick_fontsize: int = 16,
        spine_width: int = 2,
        ttest_ref: float = 0.0,
        alpha: float = 0.35,
    ):
        self.df = df
        self.value_cols = list(value_cols)
        self.dataset_name = dataset_name
        self.out_dir = out_dir
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.figsize = figsize
        self.title_fontsize = title_fontsize
        self.label_fontsize = label_fontsize
        self.tick_fontsize = tick_fontsize
        self.spine_width = spine_width
        self.ttest_ref = ttest_ref
        self.alpha = alpha

        self._ax = None

    # ---- setters/getters ----
    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, value):
        if not isinstance(value, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        self._df = value

    @property
    def value_cols(self):
        return self._value_cols

    @value_cols.setter
    def value_cols(self, value):
        cols = list(value)
        if not cols or not all(isinstance(c, str) and c for c in cols):
            raise ValueError("value_cols must contain at least one column name.")
        self._value_cols = cols

    @property
    def dataset_name(self):
        return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, value):
        if not isinstance(value, str):
            raise ValueError("dataset_name must be a string")
        self._dataset_name = value

    @property
    def out_dir(self):
        return self._out_dir

    @out_dir.setter
    def out_dir(self, value):
        if value is None:
            self._out_dir = None
            return
        if not isinstance(value, str) or not value:
            raise ValueError("out_dir must be a non-empty string or None")
        self._out_dir = value

    # ---- public API ----
    def run(self, ax=None):
        self._validate_columns()
        self._plot(ax=ax)
        return ax

    # ---- internal ----
    def _validate_columns(self):
        missing = set(self.value_cols) - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    def _plot(self, ax=None):
        sns.set_style("white")
        if ax is None:
            plt.figure(figsize=self.figsize)
            ax = plt.gca()
        self._ax = ax

        for col in self.value_cols:
            vals = self.df[col].dropna().values
            if vals.size == 0:
                continue
            sns.histplot(
                vals,
                kde=True,
                bins="auto",
                stat="density",
                element="step",
                fill=True,
                alpha=self.alpha,
                ax=ax,
                label=col,
            )

        ax.set_title(self.dataset_name, fontsize=self.title_fontsize)
        ax.set_xlabel(self.xlabel if self.xlabel is not None else "Value", fontsize=self.label_fontsize)
        ax.set_ylabel(self.ylabel if self.ylabel is not None else "Density", fontsize=self.label_fontsize)

        ax.tick_params(axis="both", labelsize=self.tick_fontsize)
        for spine in ax.spines.values():
            spine.set_linewidth(self.spine_width)

        if len(self.value_cols) > 1:
            ax.legend(fontsize=self.tick_fontsize - 2, frameon=False)

        self._annotate_stats(ax)

        if self.out_dir and ax is not None and ax.figure is not None:
            os.makedirs(os.path.join(self.out_dir, "hist_plots"), exist_ok=True)
            ax.figure.savefig(
                os.path.join(self.out_dir, f"hist_plots/{self.dataset_name}_hist.svg"),
                bbox_inches="tight",
            )
            plt.show()
        elif ax is None:
            plt.tight_layout()
            plt.show()

    def _annotate_stats(self, ax):
        groups = []
        for col in self.value_cols:
            vals = self.df[col].dropna().values
            if vals.size:
                groups.append(vals)
            else:
                groups.append(np.array([], dtype=float))

        if len(groups) <= 1:
            vals = groups[0] if groups else np.array([], dtype=float)
            if vals.size:
                t_stat, p_val = ttest_1samp(vals, popmean=self.ttest_ref, nan_policy="omit")
                text = f"t = {t_stat:.2f}, p = {p_val:.2e}"
            else:
                text = "t = NaN, p = NaN"
        else:
            try:
                f_stat, p_val = f_oneway(*groups)
                text = f"F = {f_stat:.2f}, p = {p_val:.2e}"
            except Exception:
                text = "F = NaN, p = NaN"

        ax.text(
            0.05,
            0.95,
            text,
            transform=ax.transAxes,
            fontsize=self.tick_fontsize,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0, edgecolor="none"),
        )
