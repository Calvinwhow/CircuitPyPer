import os
from typing import Iterable

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import kruskal, kstest, ttest_ind
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm


class SimpleCdfPlot:
    """
    Create cumulative distribution plots using sns.histplot(cumulative=True).

    - If one column: one-sample Kolmogorov-Smirnov test vs normal distribution.
    - If multiple columns: Kruskal-Wallis across columns.
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
        alpha: float = 0.8,
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
                kde=False,
                stat="density",
                element="step",
                fill=False,
                alpha=self.alpha,
                ax=ax,
                label=col,
                cumulative=True,
            )

        ax.set_title(self.dataset_name, fontsize=self.title_fontsize)
        ax.set_xlabel(self.xlabel if self.xlabel is not None else "Value", fontsize=self.label_fontsize)
        ax.set_ylabel(self.ylabel if self.ylabel is not None else "Cumulative Density", fontsize=self.label_fontsize)

        ax.tick_params(axis="both", labelsize=self.tick_fontsize)
        for spine in ax.spines.values():
            spine.set_linewidth(self.spine_width)

        if len(self.value_cols) > 1:
            ax.legend(fontsize=self.tick_fontsize - 2, frameon=False)

        self._annotate_stats(ax)

        if self.out_dir and ax is not None and ax.figure is not None:
            os.makedirs(os.path.join(self.out_dir, "cdf_plots"), exist_ok=True)
            ax.figure.savefig(
                os.path.join(self.out_dir, f"cdf_plots/{self.dataset_name}_cdf.svg"),
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
                v = (vals - np.mean(vals)) / (np.std(vals) + 1e-8)
                stat, p_val = kstest(v, "norm")
                text = f"KS = {stat:.2f}, p = {p_val:.2e}"
            else:
                text = "KS = NaN, p = NaN"
        else:
            try:
                h_stat, p_val = kruskal(*groups)
                text = f"H = {h_stat:.2f}, p = {p_val:.2e}"
            except Exception:
                text = "H = NaN, p = NaN"

        ax.text(
            0.05,
            0.95,
            text,
            transform=ax.transAxes,
            fontsize=self.tick_fontsize,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0, edgecolor="none"),
        )


class SimpleCdfPlotPair(SimpleCdfPlot):
    """
    Pairwise CDF plot with grouped distributions.
    Uses a two-way ANOVA for category plots and Welch t-test for overall plots.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        group_col: str,
        value_col: str,
        dataset_name: str,
        category_col: str | None = None,
        out_dir: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        figsize: tuple[int, int] = (6, 6),
        title_fontsize: int = 20,
        label_fontsize: int = 20,
        tick_fontsize: int = 16,
        spine_width: int = 2,
        alpha: float = 0.8,
    ):
        super().__init__(
            df,
            value_cols=[value_col],
            dataset_name=dataset_name,
            out_dir=out_dir,
            xlabel=xlabel,
            ylabel=ylabel,
            figsize=figsize,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            spine_width=spine_width,
            alpha=alpha,
        )
        self.group_col = group_col
        self.value_col = value_col
        self.category_col = category_col

    def run(self, ax=None):
        self._validate_columns_pair()
        self._plot_pair(ax=ax)
        return ax

    def _validate_columns_pair(self):
        required = {self.group_col, self.value_col}
        if self.category_col:
            required.add(self.category_col)
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    def _plot_pair(self, ax=None):
        sns.set_style("white")
        if ax is None:
            plt.figure(figsize=self.figsize)
            ax = plt.gca()

        if self.category_col:
            for grp in self.df[self.group_col].dropna().unique():
                for cat in self.df[self.category_col].dropna().unique():
                    vals = self.df.loc[
                        (self.df[self.group_col] == grp) & (self.df[self.category_col] == cat),
                        self.value_col,
                    ].dropna().values
                    if vals.size == 0:
                        continue
                    label = f"{grp} {cat}"
                    sns.histplot(
                        vals,
                        kde=False,
                        stat="density",
                        element="step",
                        fill=False,
                        alpha=self.alpha,
                        ax=ax,
                        label=label,
                        cumulative=True,
                    )
            self._annotate_two_way_anova(ax)
            ax.set_xlabel(self.xlabel if self.xlabel is not None else self.value_col, fontsize=self.label_fontsize)
        else:
            groups = self.df[self.group_col].dropna().unique()
            for grp in groups:
                vals = self.df.loc[self.df[self.group_col] == grp, self.value_col].dropna().values
                if vals.size == 0:
                    continue
                sns.histplot(
                    vals,
                    kde=False,
                    stat="density",
                    element="step",
                    fill=False,
                    alpha=self.alpha,
                    ax=ax,
                    label=str(grp),
                    cumulative=True,
                )
            self._annotate_welch_ttest(ax, list(groups))
            ax.set_xlabel(self.xlabel if self.xlabel is not None else self.group_col, fontsize=self.label_fontsize)

        ax.set_title(self.dataset_name, fontsize=self.title_fontsize)
        ax.set_ylabel(self.ylabel if self.ylabel is not None else "Cumulative Density", fontsize=self.label_fontsize)
        ax.tick_params(axis="both", labelsize=self.tick_fontsize)
        for spine in ax.spines.values():
            spine.set_linewidth(self.spine_width)
        if ax.get_legend() is not None:
            ax.legend(fontsize=self.tick_fontsize - 2, frameon=False)

    def _annotate_two_way_anova(self, ax):
        try:
            formula = f"{self.value_col} ~ C({self.group_col}) * C({self.category_col})"
            model = smf.ols(formula, data=self.df).fit()
            table = anova_lm(model, typ=2)
            term = f"C({self.group_col}):C({self.category_col})"
            f_stat = table.loc[term, "F"]
            p_val = table.loc[term, "PR(>F)"]
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

    def _annotate_welch_ttest(self, ax, groups):
        if len(groups) != 2:
            text = "t = NaN, p = NaN"
        else:
            g1 = self.df.loc[self.df[self.group_col] == groups[0], self.value_col].dropna()
            g2 = self.df.loc[self.df[self.group_col] == groups[1], self.value_col].dropna()
            if len(g1) and len(g2):
                t_stat, p_val = ttest_ind(g1, g2, equal_var=False)
                text = f"t = {t_stat:.2f}, p = {p_val:.2e}"
            else:
                text = "t = NaN, p = NaN"
        ax.text(
            0.05,
            0.95,
            text,
            transform=ax.transAxes,
            fontsize=self.tick_fontsize,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0, edgecolor="none"),
        )
