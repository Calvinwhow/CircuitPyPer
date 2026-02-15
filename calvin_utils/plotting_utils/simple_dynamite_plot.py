import os
from typing import Iterable, List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, ttest_1samp, ttest_ind
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm


class SimpleDynamitePlot:
    """
    Create a single-group dynamite plot (mean ± SEM) for a categorical grouping.

    This mirrors the formatting of simple_scatter for consistent styling.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        group_col: str,
        value_col: str,
        dataset_name: str,
        out_dir: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        order: Iterable[str] | None = None,
        bar_color: str = "#8E8E8E",
        figsize: tuple[int, int] = (6, 6),
        title_fontsize: int = 20,
        label_fontsize: int = 20,
        tick_fontsize: int = 16,
        spine_width: int = 2,
        ttest_ref: float = 0.0,
    ):
        self.df = df
        self.group_col = group_col
        self.value_col = value_col
        self.dataset_name = dataset_name
        self.out_dir = out_dir
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.order = list(order) if order is not None else None
        self.bar_color = bar_color
        self.figsize = figsize
        self.title_fontsize = title_fontsize
        self.label_fontsize = label_fontsize
        self.tick_fontsize = tick_fontsize
        self.spine_width = spine_width
        self.ttest_ref = ttest_ref

        self._means = None
        self._sems = None
        self._order = None
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
    def group_col(self):
        return self._group_col

    @group_col.setter
    def group_col(self, value):
        if not isinstance(value, str) or not value:
            raise ValueError("group_col must be a non-empty string")
        self._group_col = value

    @property
    def value_col(self):
        return self._value_col

    @value_col.setter
    def value_col(self, value):
        if not isinstance(value, str) or not value:
            raise ValueError("value_col must be a non-empty string")
        self._value_col = value

    @property
    def dataset_name(self):
        return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, value):
        if not isinstance(value, str) or not value:
            raise ValueError("dataset_name must be a non-empty string")
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
        self._compute_stats()
        self._plot(ax=ax)
        return self._means, self._sems

    # ---- internal ----
    def _validate_columns(self):
        missing = {self.group_col, self.value_col} - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    def _compute_stats(self):
        grouped = self.df.groupby(self.group_col)[self.value_col]
        self._means = grouped.mean()
        self._sems = grouped.sem()
        self._order = self.order if self.order is not None else list(self._means.index)

    def _plot(self, ax=None):
        sns.set_style("white")
        if ax is None:
            plt.figure(figsize=self.figsize)
            ax = plt.gca()
        self._ax = ax

        ax = sns.barplot(
            x=self._order,
            y=[self._means.loc[g] for g in self._order],
            color=self.bar_color,
            ci=None,
            ax=ax,
            edgecolor="white",
            linewidth=2,
            alpha=0.98,
        )
        ax.errorbar(
            x=np.arange(len(self._order)),
            y=[self._means.loc[g] for g in self._order],
            yerr=[self._sems.loc[g] for g in self._order],
            fmt="none",
            ecolor="black",
            capsize=4,
            lw=2,
            zorder=3,
        )

        ax.set_title(self.dataset_name, fontsize=self.title_fontsize)
        ax.set_xlabel(self.xlabel if self.xlabel is not None else self.group_col, fontsize=self.label_fontsize)
        ax.set_ylabel(self.ylabel if self.ylabel is not None else self.value_col, fontsize=self.label_fontsize)

        ax.tick_params(axis="both", labelsize=self.tick_fontsize)
        for spine in ax.spines.values():
            spine.set_linewidth(self.spine_width)

        self._annotate_stats(ax)

        if self.out_dir and ax is not None and ax.figure is not None:
            os.makedirs(os.path.join(self.out_dir, "dynamite_plots"), exist_ok=True)
            ax.figure.savefig(
                os.path.join(self.out_dir, f"dynamite_plots/{self.dataset_name}_dynamite.svg"),
                bbox_inches="tight",
            )
            plt.show()
        elif ax is None:
            plt.tight_layout()
            plt.show()

    def _annotate_stats(self, ax):
        groups = []
        for g in self._order:
            vals = self.df.loc[self.df[self.group_col] == g, self.value_col].dropna().values
            if vals.size:
                groups.append(vals)
            else:
                groups.append(np.array([], dtype=float))

        if len(groups) <= 1:
            vals = groups[0] if groups else self.df[self.value_col].dropna().values
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


class SimpleDynamitePlotPair(SimpleDynamitePlot):
    """
    Pairwise dynamite plot with grouped bars (e.g., Group A vs Group B).
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
        order: Iterable[str] | None = None,
        hue_order: Iterable[str] | None = None,
        bar_palette: Iterable[str] | None = None,
        figsize: tuple[int, int] = (6, 6),
        title_fontsize: int = 20,
        label_fontsize: int = 20,
        tick_fontsize: int = 16,
        spine_width: int = 2,
    ):
        super().__init__(
            df,
            group_col=group_col,
            value_col=value_col,
            dataset_name=dataset_name,
            out_dir=out_dir,
            xlabel=xlabel,
            ylabel=ylabel,
            order=order,
            figsize=figsize,
            title_fontsize=title_fontsize,
            label_fontsize=label_fontsize,
            tick_fontsize=tick_fontsize,
            spine_width=spine_width,
        )
        self.category_col = category_col
        self.hue_order = list(hue_order) if hue_order is not None else None
        self.bar_palette = list(bar_palette) if bar_palette is not None else ["#8E8E8E", "#B0B0B0"]

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
            order = self.order if self.order is not None else list(self.df[self.category_col].dropna().unique())
            hue_order = self.hue_order if self.hue_order is not None else list(self.df[self.group_col].dropna().unique())
            sns.barplot(
                data=self.df,
                x=self.category_col,
                y=self.value_col,
                hue=self.group_col,
                order=order,
                hue_order=hue_order,
                palette=self.bar_palette,
                ci=None,
                edgecolor="white",
                linewidth=2,
                alpha=0.98,
                ax=ax,
            )

            means = self.df.groupby([self.category_col, self.group_col])[self.value_col].mean()
            sems = self.df.groupby([self.category_col, self.group_col])[self.value_col].sem()
            self._add_grouped_errorbars(ax, order, hue_order, means, sems)
            self._annotate_two_way_anova(ax)
            ax.set_xlabel(self.xlabel if self.xlabel is not None else self.category_col, fontsize=self.label_fontsize)
        else:
            order = self.hue_order if self.hue_order is not None else list(self.df[self.group_col].dropna().unique())
            sns.barplot(
                data=self.df,
                x=self.group_col,
                y=self.value_col,
                order=order,
                palette=self.bar_palette,
                ci=None,
                edgecolor="white",
                linewidth=2,
                alpha=0.98,
                ax=ax,
            )
            means = self.df.groupby(self.group_col)[self.value_col].mean()
            sems = self.df.groupby(self.group_col)[self.value_col].sem()
            self._add_simple_errorbars(ax, order, means, sems)
            self._annotate_welch_ttest(ax, order)
            ax.set_xlabel(self.xlabel if self.xlabel is not None else self.group_col, fontsize=self.label_fontsize)

        ax.set_title(self.dataset_name, fontsize=self.title_fontsize)
        ax.set_ylabel(self.ylabel if self.ylabel is not None else self.value_col, fontsize=self.label_fontsize)
        ax.tick_params(axis="both", labelsize=self.tick_fontsize)
        for spine in ax.spines.values():
            spine.set_linewidth(self.spine_width)
        if ax.get_legend() is not None:
            ax.legend(fontsize=self.tick_fontsize - 2, frameon=False)

    def _add_grouped_errorbars(self, ax, order, hue_order, means, sems):
        containers = getattr(ax, "containers", [])
        if not containers:
            return
        for h_i, container in enumerate(containers[: len(hue_order)]):
            grp = hue_order[h_i]
            for c_i, patch in enumerate(container.patches):
                if c_i >= len(order):
                    continue
                cat = order[c_i]
                if (cat, grp) not in means.index:
                    continue
                x = patch.get_x() + patch.get_width() / 2
                y = means.loc[(cat, grp)]
                yerr = sems.loc[(cat, grp)]
                ax.errorbar(x=x, y=y, yerr=yerr, fmt="none", ecolor="black", capsize=4, lw=2, zorder=3)

    def _add_simple_errorbars(self, ax, order, means, sems):
        for idx, grp in enumerate(order):
            if grp not in means.index:
                continue
            x = ax.patches[idx].get_x() + ax.patches[idx].get_width() / 2
            y = means.loc[grp]
            yerr = sems.loc[grp]
            ax.errorbar(x=x, y=y, yerr=yerr, fmt="none", ecolor="black", capsize=4, lw=2, zorder=3)

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

    def _annotate_welch_ttest(self, ax, order):
        if len(order) != 2:
            text = "t = NaN, p = NaN"
        else:
            g1 = self.df.loc[self.df[self.group_col] == order[0], self.value_col].dropna()
            g2 = self.df.loc[self.df[self.group_col] == order[1], self.value_col].dropna()
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


class SimpleDynamitePlotWrapper:
    """
    Wide-format wrapper for SimpleDynamitePlot and SimpleDynamitePlotPair.

    Usage
    -----
    Non-pairwise (list of columns):
        plotter.plot(columns=["group_a", "group_b"], group_labels=["A", "B"])

    Pairwise (list of tuples):
        plotter.plot(columns=[("x_a","x_b"), ("y_a","y_b")], group_labels=["X","Y"], pair_names=["A","B"])
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def plot(self, columns, dataset_name="Dynamite Plot", group_labels=None, pair_names=None, **kwargs):
        # Pairwise: list of tuples [(col_a, col_b), ...]
        if any(isinstance(c, (list, tuple)) for c in columns):
            if pair_names is None:
                pair_names = ["A", "B"]
            if group_labels is None:
                group_labels = [f"metric_{i}" for i in range(len(columns))]
            if len(group_labels) != len(columns):
                raise ValueError("group_labels must match the length of columns.")
            rows = []
            for i, pair in enumerate(columns):
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ValueError("Each paired entry must be a 2-item list/tuple of column names.")
                left_col, right_col = pair
                if left_col not in self.df.columns or right_col not in self.df.columns:
                    raise ValueError(f"Missing columns for metric '{group_labels[i]}': {left_col}, {right_col}")
                rows += [{"metric": group_labels[i], "group": pair_names[0], "value": v} for v in self.df[left_col].dropna()]
                rows += [{"metric": group_labels[i], "group": pair_names[1], "value": v} for v in self.df[right_col].dropna()]
            df_long = pd.DataFrame(rows)
            category_col = "metric" if len(columns) > 1 else None
            plotter = SimpleDynamitePlotPair(
                df_long,
                group_col="group",
                category_col=category_col,
                value_col="value",
                dataset_name=dataset_name,
                hue_order=pair_names,
                order=group_labels if category_col else None,
                **kwargs,
            )
            return plotter.run()

        # Non-pairwise: list of columns
        df_long = self.df.melt(value_vars=columns, var_name="group", value_name="value")
        if group_labels is not None:
            if len(group_labels) != len(columns):
                raise ValueError("group_labels must match the length of columns.")
            label_map = dict(zip(columns, group_labels))
            df_long["group"] = df_long["group"].map(label_map)
        plotter = SimpleDynamitePlot(
            df_long,
            group_col="group",
            value_col="value",
            dataset_name=dataset_name,
            **kwargs,
        )
        return plotter.run()
