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
    figsize: tuple[float, float] | None = None,
    cell_width: float | None = None,
    cell_height: float | None = None,
    linewidths: float = 1.0,
    linecolor: str | None = None,
    cbar_kws: dict | None = None,
    cbar_range: tuple[float, float] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    title_fontsize: int = 20,
    label_fontsize: int = 20,
    tick_fontsize: int = 16,
    annot_fontsize: int | None = None,
    x_tick_rotation: int = 90,
    y_tick_rotation: int = 0,
    x_tick_ha: str = "center",
    y_tick_va: str = "center",
    spine_width: int = 2,
    remove_diag: bool = False,
    tight_layout: bool = True,
    dpi: int = 300,
):
    """
    Simple heatmap with consistent styling and flexible colormap logic.

    Use cbar_range=(vmin, vmax) to set the colorbar range in one argument.
    For palette="similarity", cbar_range creates a zero-centered TwoSlopeNorm.
    For other palettes, cbar_range is passed as seaborn/matplotlib vmin/vmax.
    """
    sns.set_style("white")

    if cbar_range is not None:
        if len(cbar_range) != 2:
            raise ValueError("cbar_range must be a 2-value tuple: (vmin, vmax).")
        if vmin is not None or vmax is not None:
            raise ValueError("Provide either cbar_range or vmin/vmax, not both.")
        vmin, vmax = cbar_range
        if vmin >= vmax:
            raise ValueError(f"cbar_range must be increasing; got {cbar_range}.")

    if isinstance(data, pd.DataFrame):
        matrix = data.copy()
    else:
        matrix = pd.DataFrame(np.asarray(data))

    if mask_half:
        matrix = pd.DataFrame(np.tril(matrix.to_numpy()), index=matrix.index, columns=matrix.columns)

    if remove_diag:
        np.fill_diagonal(matrix.values, np.nan)

    cmap = None
    norm = None
    if palette == "similarity":
        cmap = LinearSegmentedColormap.from_list(
            "RedBlackGreen",
            [(0, "red"), (0.5, "black"), (0.5, "black"), (1.0, "green")],
        )
        if cbar_range is not None:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        elif limit is None:
            minimum = np.nanmin(np.abs(matrix.values))
            maximum = np.nanmax(np.abs(matrix.values))
            limit = np.max(np.array([minimum, maximum]))
            norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
        else:
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

    if cbar_range is not None and norm is None:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    if vmin is None and vmax is None and norm is None and palette not in {"pvals", "similarity"}:
        vmin, vmax = np.nanmin(matrix.values), np.nanmax(matrix.values)

    if figsize is None and (cell_width is not None or cell_height is not None):
        width = matrix.shape[1] * (cell_width if cell_width is not None else 0.35)
        height = matrix.shape[0] * (cell_height if cell_height is not None else 0.5)
        figsize = (max(width, 6), max(height, 3))

    if ax is None:
        _, ax = plt.subplots(figsize=figsize or (14, 6))

    annot_kws = {"fontsize": annot_fontsize} if annot_fontsize is not None else None

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
        annot_kws=annot_kws,
        cbar_kws=cbar_kws,
    )

    if labels is None:
        x_labels = list(matrix.columns)
        y_labels = list(matrix.index)
    else:
        x_labels = list(labels)
        y_labels = list(labels)

    ax.set_xticks(np.arange(matrix.shape[1]) + 0.5)
    ax.set_yticks(np.arange(matrix.shape[0]) + 0.5)
    ax.set_xticklabels(x_labels, rotation=x_tick_rotation, ha=x_tick_ha)
    ax.set_yticklabels(y_labels, rotation=y_tick_rotation, va=y_tick_va)

    ax.set_title(dataset_name, fontsize=title_fontsize)
    ax.set_xlabel(xlabel or "", fontsize=label_fontsize)
    ax.set_ylabel(ylabel or "", fontsize=label_fontsize)

    ax.tick_params(axis="both", labelsize=tick_fontsize)
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)

    if tight_layout:
        ax.figure.tight_layout()

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        ax.figure.savefig(os.path.join(out_dir, output_name), bbox_inches="tight", dpi=dpi)

    return limit
