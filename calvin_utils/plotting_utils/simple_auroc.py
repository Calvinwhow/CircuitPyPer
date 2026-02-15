import os
from typing import Iterable, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score


class SimpleAUROC:
    """
    Plot AUROC curves for one outcome column and multiple probability columns.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        outcome_col: str,
        prob_cols: Iterable[str],
        dataset_name: str = "AUROC",
        out_dir: str | None = None,
        figsize: tuple[int, int] = (6, 6),
        title_fontsize: int = 20,
        label_fontsize: int = 20,
        tick_fontsize: int = 16,
        spine_width: int = 2,
        palette: str = "tab10",
    ):
        self.df = df
        self.outcome_col = outcome_col
        self.prob_cols = list(prob_cols)
        self.dataset_name = dataset_name
        self.out_dir = out_dir
        self.figsize = figsize
        self.title_fontsize = title_fontsize
        self.label_fontsize = label_fontsize
        self.tick_fontsize = tick_fontsize
        self.spine_width = spine_width
        self.palette = palette

    # ---- public API ----
    def run(self):
        self._validate_inputs()
        return self._plot()

    # ---- internal ----
    def _validate_inputs(self):
        missing = {self.outcome_col, *self.prob_cols} - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    def _plot(self):
        sns.set_style("white")
        fig, ax = plt.subplots(figsize=self.figsize)

        y_true = self.df[self.outcome_col].astype(float).values
        palette = sns.color_palette(self.palette, len(self.prob_cols))

        for idx, col in enumerate(self.prob_cols):
            scores = self.df[col].astype(float).values
            fpr, tpr, _ = roc_curve(y_true, scores)
            auc = roc_auc_score(y_true, scores)
            ax.plot(fpr, tpr, linewidth=2, color=palette[idx], label=f"{col} (AUC={auc:.2f})")

        ax.plot([0, 1], [0, 1], linestyle="--", color="#8E8E8E", linewidth=1)

        ax.set_title(self.dataset_name, fontsize=self.title_fontsize)
        ax.set_xlabel("False Positive Rate", fontsize=self.label_fontsize)
        ax.set_ylabel("True Positive Rate", fontsize=self.label_fontsize)
        ax.tick_params(axis="both", labelsize=self.tick_fontsize)
        for spine in ax.spines.values():
            spine.set_linewidth(self.spine_width)
        ax.legend(fontsize=self.tick_fontsize - 2, frameon=False)

        if self.out_dir:
            os.makedirs(os.path.join(self.out_dir, "roc_plots"), exist_ok=True)
            fig.savefig(os.path.join(self.out_dir, "roc_plots/auroc.svg"), bbox_inches="tight")

        plt.show()
        return fig, ax
