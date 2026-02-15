import os
from math import pi
from typing import Iterable

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class SimpleRadarPlot:
    """
    Simple radar plot with SimpleScatter-style formatting.

    Expects a dataframe with:
    - a column for variable names (variable_col)
    - metric columns (metric_cols)
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        *,
        variable_col: str,
        metric_cols: Iterable[str],
        dataset_name: str = "Comparison of Metrics across Variables",
        out_dir: str | None = None,
        figsize: tuple[int, int] = (10, 8),
        title_fontsize: int = 20,
        label_fontsize: int = 16,
        tick_fontsize: int = 16,
        spine_width: int = 2,
        palette: str = "tab10",
        y_min: float = 0.0,
        y_max: float = 1.0,
    ):
        self.dataframe = dataframe
        self.variable_col = variable_col
        self.metric_cols = list(metric_cols)
        self.dataset_name = dataset_name
        self.out_dir = out_dir
        self.figsize = figsize
        self.title_fontsize = title_fontsize
        self.label_fontsize = label_fontsize
        self.tick_fontsize = tick_fontsize
        self.spine_width = spine_width
        self.palette = palette
        self.y_min = y_min
        self.y_max = y_max

    # ---- public API ----
    def run(self):
        self._validate_inputs()
        return self._plot_radar()

    # ---- internal ----
    def _validate_inputs(self):
        missing = {self.variable_col, *self.metric_cols} - set(self.dataframe.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    def _plot_radar(self):
        sns.set_style("white")
        fig = plt.figure(figsize=self.figsize)
        ax = plt.subplot(111, polar=True)

        metrics = self.metric_cols
        num_vars = len(metrics)
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]

        palette = sns.color_palette(self.palette, self.dataframe[self.variable_col].nunique())

        for idx, variable in enumerate(self.dataframe[self.variable_col].unique()):
            row = self.dataframe[self.dataframe[self.variable_col] == variable].iloc[0]
            values = [row[m] for m in metrics]
            values += values[:1]
            ax.plot(angles, values, "o-", linewidth=2, label=str(variable), color=palette[idx])

        ax.set_thetagrids([a * 180 / pi for a in angles[:-1]], labels=metrics, fontsize=self.label_fontsize)
        ax.set_ylim(self.y_min, self.y_max)
        ax.tick_params(labelsize=self.tick_fontsize)
        ax.set_title(self.dataset_name, fontsize=self.title_fontsize)
        for spine in ax.spines.values():
            spine.set_linewidth(self.spine_width)

        ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1), fontsize=self.tick_fontsize - 2, frameon=False)

        if self.out_dir:
            radar_plots_subdir = "radar_plots"
            os.makedirs(os.path.join(self.out_dir, radar_plots_subdir), exist_ok=True)
            out_svg = os.path.join(self.out_dir, radar_plots_subdir, f"{self.variable_col}_radar.svg")
            ax.figure.savefig(out_svg, bbox_inches="tight")

        plt.show()
        return fig, ax
