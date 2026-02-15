import os
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from calvin_utils.plotting_utils.simple_heatmap import simple_heatmap


class SimpleConfusionMatrix:
    """
    Simple confusion matrix plotter supporting multinomial classification.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        true_col: str,
        pred_col: str,
        labels: Iterable[str] | None = None,
        normalize: str | None = None,
        dataset_name: str = "Confusion Matrix",
        out_dir: str | None = None,
        cmap: str = "Blues",
        fmt: str = ".2f",
    ):
        self.df = df
        self.true_col = true_col
        self.pred_col = pred_col
        self.labels = list(labels) if labels is not None else None
        self.normalize = normalize
        self.dataset_name = dataset_name
        self.out_dir = out_dir
        self.cmap = cmap
        self.fmt = fmt

    def run(self):
        self._validate()
        return self._plot()

    def _validate(self):
        missing = {self.true_col, self.pred_col} - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    def _plot(self):
        sns.set_style("white")
        y_true = self.df[self.true_col].values
        y_pred = self.df[self.pred_col].values

        if self.labels is None:
            labels = sorted(set(y_true) | set(y_pred))
        else:
            labels = self.labels

        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=self.normalize)
        fig, ax = plt.subplots(figsize=(6, 6))
        simple_heatmap(
            cm,
            dataset_name=self.dataset_name,
            ax=ax,
            palette=self.cmap,
            annot=True,
            fmt=self.fmt,
            labels=labels,
        )
        ax.set_xlabel("Predicted", fontsize=20)
        ax.set_ylabel("Actual", fontsize=20)

        if self.out_dir:
            os.makedirs(os.path.join(self.out_dir, "confusion_matrix"), exist_ok=True)
            fig.savefig(os.path.join(self.out_dir, "confusion_matrix/confusion_matrix.svg"), bbox_inches="tight")

        plt.show()
        return fig, ax
