import os
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def simple_heatmap(
    data,
    *,
    dataset_name: str = "",
    out_dir: str | None = None,
    output_name: str = "heatmap.svg",
    ax=None,
    palette: Literal["similarity", "pvals", "redblue", "viridis"] | str = "similarity",
    mask_half: bool = False,
    limit: float | None = None,
    labels: Iterable[str] | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    annot: bool = False,
    fmt: str = ".2f",
    cbar: bool = True,
    square: bool = True,
    linewidths: float = 1.0,
    linecolor: str | None = None,
    cbar_kws: dict | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    title_fontsize: int = 20,
    label_fontsize: int = 20,
    tick_fontsize: int = 16,
    spine_width: int = 2,
):
    """
    Simple heatmap with consistent styling and flexible colormap logic.
    """
    sns.set_style("white")

    if isinstance(data, pd.DataFrame):
        matrix = data.copy()
    else:
        matrix = pd.DataFrame(np.asarray(data))

    if mask_half:
        matrix = pd.DataFrame(np.tril(matrix.to_numpy()), index=matrix.index, columns=matrix.columns)

    np.fill_diagonal(matrix.values, np.nan)

    cmap = None
    norm = None
    if palette == "similarity":
        cmap = LinearSegmentedColormap.from_list(
            "RedBlackGreen",
            [(0, "red"), (0.5, "black"), (0.5, "black"), (1.0, "green")],
        )
        if limit is None:
            minimum = np.nanmin(np.abs(matrix.values))
            maximum = np.nanmax(np.abs(matrix.values))
            limit = np.max(np.array([minimum, maximum]))
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    elif palette == "pvals":
        bounds = [0, 0.0001, 0.001, 0.01, 0.05, 1]
        cmap = cm.get_cmap("viridis", len(bounds) - 1)
        norm = cm.colors.BoundaryNorm(bounds, cmap.N)
    elif palette == "redblue":
        cmap = "coolwarm"
    elif palette == "viridis":
        cmap = "viridis"
    else:
        cmap = palette

    if vmin is None and vmax is None and norm is None and palette not in {"pvals", "similarity"}:
        vmin, vmax = np.nanmin(matrix.values), np.nanmax(matrix.values)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        matrix,
        square=square,
        linewidths=linewidths,
        linecolor=linecolor,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        cbar=cbar,
        annot=annot,
        fmt=fmt,
        cbar_kws=cbar_kws,
    )

    if labels is None:
        x_labels = list(matrix.columns)
        y_labels = list(matrix.index)
    else:
        x_labels = list(labels)
        y_labels = list(labels)

    ax.set_xticklabels(x_labels, rotation=90)
    ax.set_yticklabels(y_labels, rotation=0)

    ax.set_title(dataset_name, fontsize=title_fontsize)
    ax.set_xlabel(xlabel or "", fontsize=label_fontsize)
    ax.set_ylabel(ylabel or "", fontsize=label_fontsize)

    ax.tick_params(axis="both", labelsize=tick_fontsize)
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        ax.figure.savefig(os.path.join(out_dir, output_name), bbox_inches="tight")

    return limit
