import os
from typing import Iterable

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class SimplePiePlot:
    """
    Create a pretty pie chart with SimpleXX-style formatting.

    Expects a dataframe with:
    - label_col: category labels
    - value_col: numeric values (summed by label)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        label_col: str,
        value_col: str,
        dataset_name: str = "",
        out_dir: str | None = None,
        figsize: tuple[int, int] = (6, 6),
        title_fontsize: int = 20,
        label_fontsize: int = 16,
        tick_fontsize: int = 16,
        spine_width: int = 2,
        palette: Iterable[str] | str = "tab10",
        autopct: str | None = "%1.1f%%",
        startangle: float = 90,
        sort: bool = True,
        show_legend: bool = False,
        legend_loc: str = "center left",
        legend_bbox: tuple[float, float] = (1.0, 0.5),
    ):
        self.df = df
        self.label_col = label_col
        self.value_col = value_col
        self.dataset_name = dataset_name
        self.out_dir = out_dir
        self.figsize = figsize
        self.title_fontsize = title_fontsize
        self.label_fontsize = label_fontsize
        self.tick_fontsize = tick_fontsize
        self.spine_width = spine_width
        self.palette = palette
        self.autopct = autopct
        self.startangle = startangle
        self.sort = sort
        self.show_legend = show_legend
        self.legend_loc = legend_loc
        self.legend_bbox = legend_bbox

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
    def label_col(self):
        return self._label_col

    @label_col.setter
    def label_col(self, value):
        if not isinstance(value, str) or not value:
            raise ValueError("label_col must be a non-empty string")
        self._label_col = value

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
        required = {self.label_col, self.value_col}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    def _plot(self, ax=None):
        sns.set_style("white")
        plt.rcParams["font.family"] = "Helvetica"
        plt.rcParams["font.size"] = self.label_fontsize

        data = self.df[[self.label_col, self.value_col]].dropna()
        if data.empty:
            raise ValueError("No valid rows after dropping NaNs.")

        grouped = data.groupby(self.label_col, dropna=False)[self.value_col].sum()
        if self.sort:
            grouped = grouped.sort_values(ascending=False)

        labels = grouped.index.astype(str).tolist()
        values = grouped.values
        colors = sns.color_palette(self.palette, n_colors=len(values))

        if ax is None:
            plt.figure(figsize=self.figsize)
            ax = plt.gca()
        self._ax = ax

        label_texts = labels if not self.show_legend else None
        wedges, texts, autotexts = ax.pie(
            values,
            labels=label_texts,
            colors=colors,
            startangle=self.startangle,
            autopct=self.autopct,
            pctdistance=0.75,
            labeldistance=1.08,
            textprops={"fontsize": self.label_fontsize},
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )

        for text in autotexts:
            text.set_fontsize(self.label_fontsize)

        ax.set_aspect("equal")
        if self.dataset_name:
            ax.set_title(self.dataset_name, fontsize=self.title_fontsize)

        if self.show_legend:
            ax.legend(
                wedges,
                labels,
                loc=self.legend_loc,
                bbox_to_anchor=self.legend_bbox,
                fontsize=self.tick_fontsize,
                frameon=False,
            )

        for spine in ax.spines.values():
            spine.set_linewidth(self.spine_width)

        if self.out_dir and ax is not None and ax.figure is not None:
            os.makedirs(os.path.join(self.out_dir, "pie_charts"), exist_ok=True)
            ax.figure.savefig(
                os.path.join(self.out_dir, f"pie_charts/{self.dataset_name}_piechart.svg"),
                bbox_inches="tight",
            )
            plt.show()
        elif ax is None:
            plt.tight_layout()
            plt.show()
