import os
import re
from collections.abc import Callable
from typing import Iterable, Literal

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

Orientation = Literal["vertical", "horizontal", "v", "h", "x", "y"]


class SimpleBarPlot:
    """
    Plot every value in one dataframe column as its own bar.

    The bar labels come from a second dataframe column. Unlike seaborn's default
    categorical barplot behavior, this does not aggregate repeated labels.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        value_col: str,
        label_col: str,
        dataset_name: str = "Bar Plot",
        out_dir: str | None = None,
        output_path: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        order: Iterable[str] | None = None,
        orientation: Orientation = "vertical",
        bar_color: str = "#4C78A8",
        palette: str | Iterable[str] | None = None,
        figsize: tuple[int, int] = (10, 6),
        title_fontsize: int = 20,
        label_fontsize: int = 18,
        tick_fontsize: int = 12,
        spine_width: int = 2,
        label_rotation: int | None = None,
        value_format: str | Callable[[float], str] | None = "{:.2f}",
        annotate_values: bool = False,
        value_label_padding: float = 0.01,
        bar_width: float = 0.85,
    ):
        self.df = df
        self.value_col = value_col
        self.label_col = label_col
        self.dataset_name = dataset_name
        self.out_dir = out_dir
        self.output_path = output_path
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.order = list(order) if order is not None else None
        self.orientation = orientation
        self.bar_color = bar_color
        self.palette = palette
        self.figsize = figsize
        self.title_fontsize = title_fontsize
        self.label_fontsize = label_fontsize
        self.tick_fontsize = tick_fontsize
        self.spine_width = spine_width
        self.label_rotation = label_rotation
        self.value_format = value_format
        self.annotate_values = annotate_values
        self.value_label_padding = value_label_padding
        self.bar_width = bar_width

        self._plot_df = None
        self._ax = None

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, value):
        if not isinstance(value, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        self._df = value

    @property
    def value_col(self):
        return self._value_col

    @value_col.setter
    def value_col(self, value):
        if not isinstance(value, str) or not value:
            raise ValueError("value_col must be a non-empty string")
        self._value_col = value

    @property
    def label_col(self):
        return self._label_col

    @label_col.setter
    def label_col(self, value):
        if not isinstance(value, str) or not value:
            raise ValueError("label_col must be a non-empty string")
        self._label_col = value

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

    @property
    def output_path(self):
        return self._output_path

    @output_path.setter
    def output_path(self, value):
        if value is None:
            self._output_path = None
            return
        if not isinstance(value, str) or not value:
            raise ValueError("output_path must be a non-empty string or None")
        self._output_path = value

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, value):
        valid = {"vertical", "horizontal", "v", "h", "x", "y"}
        if value not in valid:
            raise ValueError(f"orientation must be one of {sorted(valid)}")
        if value in {"horizontal", "h", "y"}:
            self._orientation = "horizontal"
        else:
            self._orientation = "vertical"

    def run(self, ax=None):
        self._validate_columns()
        self._prepare_plot_df()
        self._plot(ax=ax)
        return ax if ax is not None else self._ax

    def _validate_columns(self):
        missing = {self.value_col, self.label_col} - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        if not pd.api.types.is_numeric_dtype(self.df[self.value_col]):
            raise ValueError(f"value_col must be numeric: {self.value_col}")

    def _prepare_plot_df(self):
        plot_df = self.df[[self.label_col, self.value_col]].dropna().copy()
        if plot_df.empty:
            raise ValueError("No rows remain after dropping missing labels/values.")
        if self.order is not None:
            order_lookup = {label: idx for idx, label in enumerate(self.order)}
            plot_df["_sort_key"] = plot_df[self.label_col].map(order_lookup)
            plot_df = plot_df.loc[plot_df["_sort_key"].notna()].sort_values("_sort_key")
            if plot_df.empty:
                raise ValueError("No rows match the requested order.")
        plot_df["_bar_position"] = range(len(plot_df))
        plot_df["_bar_label"] = plot_df[self.label_col].astype(str)
        self._plot_df = plot_df

    def _plot(self, ax=None):
        sns.set_theme(style="white", context="notebook")
        created_fig = ax is None
        if ax is None:
            plt.figure(figsize=self.figsize)
            ax = plt.gca()
        self._ax = ax

        colors = self._resolve_bar_colors()
        if self.orientation == "horizontal":
            ax.barh(
                self._plot_df["_bar_position"],
                self._plot_df[self.value_col],
                height=self.bar_width,
                color=colors,
                edgecolor="white",
                linewidth=1.2,
                alpha=0.95,
            )
            ax.set_yticks(self._plot_df["_bar_position"])
            ax.set_yticklabels(self._plot_df["_bar_label"])
            ax.set_xlabel(self.xlabel if self.xlabel is not None else self.value_col, fontsize=self.label_fontsize)
            ax.set_ylabel(self.ylabel if self.ylabel is not None else self.label_col, fontsize=self.label_fontsize)
            if self.annotate_values:
                self._annotate_horizontal(ax)
        else:
            ax.bar(
                self._plot_df["_bar_position"],
                self._plot_df[self.value_col],
                width=self.bar_width,
                color=colors,
                edgecolor="white",
                linewidth=1.2,
                alpha=0.95,
            )
            ax.set_xticks(self._plot_df["_bar_position"])
            ax.set_xticklabels(self._plot_df["_bar_label"])
            rotation = 45 if self.label_rotation is None else self.label_rotation
            ax.tick_params(axis="x", rotation=rotation)
            ax.set_xlabel(self.xlabel if self.xlabel is not None else self.label_col, fontsize=self.label_fontsize)
            ax.set_ylabel(self.ylabel if self.ylabel is not None else self.value_col, fontsize=self.label_fontsize)
            if self.annotate_values:
                self._annotate_vertical(ax)

        ax.set_title(self.dataset_name, fontsize=self.title_fontsize)
        ax.tick_params(axis="both", labelsize=self.tick_fontsize)
        for spine in ax.spines.values():
            spine.set_linewidth(self.spine_width)
        sns.despine(ax=ax)

        if self.orientation == "horizontal":
            ax.grid(axis="x", color="#D9D9D9", linewidth=0.8, alpha=0.8)
        else:
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)

        if ax is not None and ax.figure is not None:
            if self.output_path:
                self._save_figure(ax, self.output_path)
            elif self.out_dir:
                os.makedirs(os.path.join(self.out_dir, "bar_plots"), exist_ok=True)
                output_path = os.path.join(self.out_dir, "bar_plots", f"{self._safe_dataset_name()}_barplot.svg")
                self._save_figure(ax, output_path)
        if created_fig:
            plt.tight_layout()
            if plt.get_backend().lower() != "agg":
                plt.show()

    def _annotate_vertical(self, ax):
        for patch, value in zip(ax.patches, self._plot_df[self.value_col]):
            label = self._format_value(value)
            y_min, y_max = ax.get_ylim()
            offset = (y_max - y_min) * self.value_label_padding
            y = patch.get_height()
            va = "bottom" if y >= 0 else "top"
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                y + offset if y >= 0 else y - offset,
                label,
                ha="center",
                va=va,
                fontsize=self.tick_fontsize,
            )

    def _annotate_horizontal(self, ax):
        for patch, value in zip(ax.patches, self._plot_df[self.value_col]):
            label = self._format_value(value)
            x_min, x_max = ax.get_xlim()
            offset = (x_max - x_min) * self.value_label_padding
            x = patch.get_width()
            ha = "left" if x >= 0 else "right"
            ax.text(
                x + offset if x >= 0 else x - offset,
                patch.get_y() + patch.get_height() / 2,
                label,
                ha=ha,
                va="center",
                fontsize=self.tick_fontsize,
            )

    def _format_value(self, value):
        if self.value_format is None:
            return str(value)
        if callable(self.value_format):
            return self.value_format(value)
        return self.value_format.format(value)

    def _resolve_bar_colors(self):
        n_bars = len(self._plot_df)
        if self.palette is None:
            return self.bar_color
        if isinstance(self.palette, str):
            return sns.color_palette(self.palette, n_colors=n_bars)
        colors = list(self.palette)
        if not colors:
            raise ValueError("palette must not be empty.")
        repeats = (n_bars // len(colors)) + 1
        return (colors * repeats)[:n_bars]

    def _safe_dataset_name(self):
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.dataset_name).strip("_")
        return safe_name or "bar_plot"

    @staticmethod
    def _save_figure(ax, output_path):
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        ax.figure.savefig(output_path, bbox_inches="tight")


class SimpleBarPlotWrapper:
    """
    Wrapper for plotting one value column against one label column from a wide dataframe.

    Usage
    -----
        plotter = SimpleBarPlotWrapper(df)
        plotter.plot(bars="r_squared", labels="predictor", orientation="horizontal")
    """

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")
        self.df = df

    def plot(
        self,
        bars: str,
        labels: str,
        dataset_name: str = "Bar Plot",
        flip: bool | None = None,
        sort_by_value: bool = False,
        ascending: bool = False,
        top_n: int | None = None,
        **kwargs,
    ):
        output_path = None
        if "out_dir" in kwargs and isinstance(kwargs["out_dir"], str):
            _, ext = os.path.splitext(kwargs["out_dir"])
            if ext:
                output_path = kwargs["out_dir"]
                kwargs["out_dir"] = None

        missing = {bars, labels} - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        if flip is not None:
            kwargs["orientation"] = "horizontal" if flip else "vertical"

        plot_df = self.df[[labels, bars]].dropna().copy()
        if plot_df.empty:
            raise ValueError("No rows remain after dropping missing labels/values.")
        if sort_by_value:
            plot_df = plot_df.sort_values(bars, ascending=ascending, kind="mergesort")
        if top_n is not None:
            if not isinstance(top_n, int) or top_n <= 0:
                raise ValueError("top_n must be a positive integer or None")
            plot_df = plot_df.head(top_n)

        plotter = SimpleBarPlot(
            plot_df,
            value_col=bars,
            label_col=labels,
            dataset_name=dataset_name,
            output_path=output_path,
            **kwargs,
        )
        return plotter.run()
