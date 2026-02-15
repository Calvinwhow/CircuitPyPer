import os
from typing import Iterable

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class SimpleLinePlot:
    """
    Create line plots with optional fill (mean ± SEM) and consistent styling.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        x_col: str,
        y_col: str,
        hue_col: str | None = None,
        dataset_name: str = "",
        out_dir: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        figsize: tuple[int, int] = (6, 6),
        title_fontsize: int = 20,
        label_fontsize: int = 20,
        tick_fontsize: int = 16,
        spine_width: int = 2,
        fill: bool = True,
        fill_alpha: float = 0.2,
        palette: Iterable[str] | str = "tab10",
        sort_x: bool = True,
    ):
        self.df = df
        self.x_col = x_col
        self.y_col = y_col
        self.hue_col = hue_col
        self.dataset_name = dataset_name
        self.out_dir = out_dir
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.figsize = figsize
        self.title_fontsize = title_fontsize
        self.label_fontsize = label_fontsize
        self.tick_fontsize = tick_fontsize
        self.spine_width = spine_width
        self.fill = fill
        self.fill_alpha = fill_alpha
        self.palette = palette
        self.sort_x = sort_x

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
    def x_col(self):
        return self._x_col

    @x_col.setter
    def x_col(self, value):
        if not isinstance(value, str) or not value:
            raise ValueError("x_col must be a non-empty string")
        self._x_col = value

    @property
    def y_col(self):
        return self._y_col

    @y_col.setter
    def y_col(self, value):
        if not isinstance(value, str) or not value:
            raise ValueError("y_col must be a non-empty string")
        self._y_col = value

    # ---- public API ----
    def run(self, ax=None):
        self._validate_columns()
        self._plot(ax=ax)
        return ax

    # ---- internal ----
    def _validate_columns(self):
        required = {self.x_col, self.y_col}
        if self.hue_col:
            required.add(self.hue_col)
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    def _plot(self, ax=None):
        sns.set_style("white")
        if ax is None:
            plt.figure(figsize=self.figsize)
            ax = plt.gca()
        self._ax = ax

        data = self.df.copy()
        if self.sort_x:
            data = data.sort_values(by=self.x_col)

        sns.lineplot(
            data=data,
            x=self.x_col,
            y=self.y_col,
            hue=self.hue_col,
            palette=self.palette,
            ax=ax,
            errorbar=None,
        )

        if self.fill:
            self._fill_sem(ax, data)

        ax.set_title(self.dataset_name, fontsize=self.title_fontsize)
        ax.set_xlabel(self.xlabel if self.xlabel is not None else self.x_col, fontsize=self.label_fontsize)
        ax.set_ylabel(self.ylabel if self.ylabel is not None else self.y_col, fontsize=self.label_fontsize)

        ax.tick_params(axis="both", labelsize=self.tick_fontsize)
        for spine in ax.spines.values():
            spine.set_linewidth(self.spine_width)

        if ax.get_legend() is not None:
            ax.legend(fontsize=self.tick_fontsize - 2, frameon=False)

        if self.out_dir and ax is not None and ax.figure is not None:
            os.makedirs(os.path.join(self.out_dir, "line_plots"), exist_ok=True)
            ax.figure.savefig(
                os.path.join(self.out_dir, f"line_plots/{self.dataset_name}_lineplot.svg"),
                bbox_inches="tight",
            )
            plt.show()
        elif ax is None:
            plt.tight_layout()
            plt.show()

    def _fill_sem(self, ax, data: pd.DataFrame):
        if self.hue_col:
            groups = data[self.hue_col].dropna().unique()
        else:
            groups = [None]

        palette = sns.color_palette(self.palette, n_colors=len(groups)) if self.hue_col else [sns.color_palette(self.palette)[0]]

        for idx, grp in enumerate(groups):
            if grp is None:
                sub = data
            else:
                sub = data.loc[data[self.hue_col] == grp]
            grouped = sub.groupby(self.x_col)[self.y_col]
            mean = grouped.mean()
            sem = grouped.sem()
            x = mean.index.to_numpy()
            y = mean.to_numpy()
            yerr = sem.to_numpy()
            ax.fill_between(x, y - yerr, y + yerr, color=palette[idx], alpha=self.fill_alpha, linewidth=0)


class SimpleLinePlotWrapper:
    """
    Wide-format wrapper for SimpleLinePlot.

    Usage
    -----
    plotter.plot(x_col="x", series_cols=None)
    - If x_col exists, all other columns are treated as series.
    - If x_col is None, the index is used as x.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def plot(self, x_col=None, series_cols=None, dataset_name="Line Plot", **kwargs):
        data = self.df.copy()
        if x_col is not None and x_col in data.columns:
            id_vars = [x_col]
            value_cols = series_cols or [c for c in data.columns if c != x_col]
            df_long = data.melt(id_vars=id_vars, value_vars=value_cols, var_name="series", value_name="value")
            plotter = SimpleLinePlot(
                df_long,
                x_col=x_col,
                y_col="value",
                hue_col="series",
                dataset_name=dataset_name,
                **kwargs,
            )
            return plotter.run()
        df_long = data.reset_index().melt(id_vars=["index"], var_name="series", value_name="value")
        plotter = SimpleLinePlot(
            df_long,
            x_col="index",
            y_col="value",
            hue_col="series",
            dataset_name=dataset_name,
            **kwargs,
        )
        return plotter.run()
