"""Reusable publication-style three-dimensional scatter plots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Hashable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, MaxNLocator

from calvin_utils.plotting_utils.scatter_ellipsoids import (
    add_cluster_ellipsoids_matplotlib,
    calculate_cluster_ellipsoids,
)
from calvin_utils.plotting_utils.hotelling_cluster_separation import (
    pairwise_hotelling_tests,
)


class SimpleScatterplot3D:
    """Plot any three-column point cloud with optional clustered ellipsoids.

    Observations are rows and spatial dimensions are columns. The class has no
    dependency on UMAP: callers may provide any three-dimensional coordinates,
    arbitrary hashable group labels, optional per-point opacity values, color
    and display-name mappings, axis labels, and camera settings.
    """

    def __init__(self, observations, labels=None, probabilities=None):
        points = np.asarray(observations, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(
                "observations must have shape (n_observations, 3), with "
                "dimensions in columns."
            )
        if not np.isfinite(points).all():
            raise ValueError("observations must contain only finite values.")

        if labels is None:
            labels = np.zeros(points.shape[0], dtype=int)
        labels = np.asarray(labels)
        if labels.ndim != 1 or labels.shape[0] != points.shape[0]:
            raise ValueError("labels must contain one value per observation.")

        if probabilities is None:
            probabilities = np.ones(points.shape[0], dtype=float)
        probabilities = np.asarray(probabilities, dtype=float)
        if probabilities.ndim != 1 or probabilities.shape[0] != points.shape[0]:
            raise ValueError(
                "probabilities must contain one value per observation."
            )
        if not np.isfinite(probabilities).all():
            raise ValueError("probabilities must contain only finite values.")

        self.observations = points
        self.labels = labels
        self.probabilities = probabilities
        self.figure = None
        self.axis = None
        self.cluster_separation_results = None
        self.cluster_separation_path = None

    @staticmethod
    def _ordered_labels(labels):
        return list(dict.fromkeys(labels.tolist()))

    @staticmethod
    def _separation_path_for_figure(out_path):
        path = Path(out_path)
        return path.with_name(
            f"{path.stem}_cluster_separation_hotelling.csv"
        )

    def analyze_cluster_separation(
        self,
        *,
        names: Mapping[Hashable, str] | None = None,
        ellipsoid_n_std: float = 3.0,
        ellipsoid_ddof: int = 1,
        p_adjust: str | None = "holm",
        alpha: float = 0.05,
        print_results: bool = True,
        out_path=None,
    ):
        """Calculate, print, and optionally save pairwise Hotelling tests.

        One multivariate test is produced for every pair of non-noise
        clusters. It compares their complete centroid vectors using the
        pooled within-cluster spatial covariance.
        """
        results = pairwise_hotelling_tests(
            self.observations,
            self.labels,
            p_adjust=p_adjust,
            alpha=alpha,
            ellipsoid_n_std=ellipsoid_n_std,
            ddof=ellipsoid_ddof,
        )
        if names is not None and not results.empty:
            results.insert(
                1,
                "cluster_a_name",
                results["cluster_a"].map(
                    lambda label: names.get(label, f"Cluster {label}")
                ),
            )
            results.insert(
                3,
                "cluster_b_name",
                results["cluster_b"].map(
                    lambda label: names.get(label, f"Cluster {label}")
                ),
            )

        self.cluster_separation_results = results
        if print_results and not results.empty:
            columns = [
                column
                for column in (
                    "cluster_a_name",
                    "cluster_a",
                    "cluster_b_name",
                    "cluster_b",
                    "n_a",
                    "n_b",
                    "centroid_distance_l2",
                    "hotelling_t_squared",
                    "f_statistic",
                    "df1",
                    "df2",
                    "p_value",
                    "p_value_adjusted",
                    "reject_null",
                    "ellipsoid_axis_gap",
                    "ellipsoid_axis_projection_overlap",
                )
                if column in results
            ]
            print(
                "\n[SimpleScatterplot3D] Pairwise Hotelling tests "
                f"({ellipsoid_n_std:g}-SD ellipsoids):"
            )
            print(results.loc[:, columns].to_string(index=False))

        self.cluster_separation_path = None
        if out_path is not None:
            path = Path(out_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            results.to_csv(path, index=False)
            self.cluster_separation_path = str(path)
            if print_results:
                print(
                    "[SimpleScatterplot3D] Cluster-separation results saved "
                    f"to: {path}"
                )
        return results

    def plot(
        self,
        *,
        colors: Mapping[Hashable, object] | None = None,
        names: Mapping[Hashable, str] | None = None,
        legend_order: Sequence[Hashable] | None = None,
        point_size: float = 50,
        outlines: bool = False,
        point_edge_color: object = "black",
        minimum_point_opacity: float = 0.15,
        ellipsoids: bool = True,
        ellipsoid_n_std: float = 3.0,
        ellipsoid_opacity: float = 0.18,
        ellipsoid_resolution: int = 36,
        ellipsoid_ddof: int = 1,
        lighting: bool = True,
        light_azimuth: float = 315.0,
        light_altitude: float = 45.0,
        light_strength: float = 0.22,
        ground_shadows: bool = False,
        ground_level: float | None = None,
        shadow_opacity: float = 0.10,
        shadow_stems: bool = False,
        shadow_stem_opacity: float = 0.24,
        show_ground_plane: bool = False,
        ground_plane_color: object = "#f6f7fa",
        elevation: float = 24,
        azimuth: float = -55,
        roll: float = 0,
        projection: str = "ortho",
        figsize: tuple[float, float] = (12, 6),
        dpi: int = 300,
        box_aspect: Sequence[float] | None = None,
        zoom: float = 1.0,
        axis_labels: Sequence[str] = ("Dimension 1", "Dimension 2", "Dimension 3"),
        axes: bool = True,
        grid: bool = False,
        axis_color: object = "black",
        axis_label_fontsize: float = 20,
        tick_label_fontsize: float = 16,
        axis_linewidth: float = 2.0,
        max_ticks: Sequence[int] = (6, 4, 6),
        tick_values: Sequence[Sequence[float] | None] | None = None,
        legend: bool = True,
        legend_fontsize: float = 16,
        legend_title: str | None = None,
        figure_legend: bool = False,
        title: str | None = None,
        subtitle: str | None = None,
        title_fontsize: float = 20,
        subtitle_fontsize: float = 16,
        background_color: object = "white",
        cluster_separation: bool = True,
        cluster_separation_print: bool = True,
        cluster_separation_out_path=None,
        cluster_separation_p_adjust: str | None = "holm",
        cluster_separation_alpha: float = 0.05,
    ):
        """Build and return a Matplotlib figure."""
        if len(axis_labels) != 3:
            raise ValueError("axis_labels must contain exactly three strings.")
        if len(max_ticks) != 3:
            raise ValueError("max_ticks must contain exactly three integers.")
        if tick_values is not None and len(tick_values) != 3:
            raise ValueError("tick_values must contain exactly three entries.")
        if not 0.0 <= minimum_point_opacity <= 1.0:
            raise ValueError("minimum_point_opacity must be between 0 and 1.")

        labels = self._ordered_labels(self.labels)
        base = list(plt.cm.tab10.colors)
        palette = {
            label: to_rgb(colors[label]) if colors and label in colors else base[index % len(base)]
            for index, label in enumerate(labels)
        }
        display_names = {
            label: names[label] if names and label in names else f"Cluster {label}"
            for label in labels
        }

        if cluster_separation:
            self.analyze_cluster_separation(
                names=display_names,
                ellipsoid_n_std=ellipsoid_n_std,
                ellipsoid_ddof=ellipsoid_ddof,
                p_adjust=cluster_separation_p_adjust,
                alpha=cluster_separation_alpha,
                print_results=cluster_separation_print,
                out_path=cluster_separation_out_path,
            )
        else:
            self.cluster_separation_results = None
            self.cluster_separation_path = None

        geometry = {}
        if ellipsoids:
            geometry = calculate_cluster_ellipsoids(
                self.observations,
                self.labels,
                n_std=ellipsoid_n_std,
                ddof=ellipsoid_ddof,
                resolution=ellipsoid_resolution,
            )
        shadow_floor = None
        if ground_shadows and geometry:
            shadow_floor = (
                float(ground_level)
                if ground_level is not None
                else min(item.center[2] - item.radii[2] for item in geometry.values())
            )

        figure = plt.figure(figsize=figsize, dpi=dpi, facecolor=background_color)
        bounds = [0.015, 0.025, 0.97, 0.835] if title else [0.04, 0.04, 0.92, 0.92]
        axis = figure.add_axes(bounds, projection="3d", facecolor=background_color)
        axis.set_proj_type(projection)

        for label in labels:
            keep = self.labels == label
            rgba = np.tile(np.r_[palette[label], 1.0], (int(keep.sum()), 1))
            rgba[:, 3] = np.clip(
                self.probabilities[keep], minimum_point_opacity, 1.0
            )
            axis.scatter(
                *self.observations[keep].T,
                s=point_size,
                c=rgba,
                depthshade=False,
                edgecolors=point_edge_color if outlines else "none",
                linewidths=0.65 if outlines else 0,
                zorder=10,
            )

        if ellipsoids:
            geometry = add_cluster_ellipsoids_matplotlib(
                axis,
                self.observations,
                self.labels,
                n_std=ellipsoid_n_std,
                ddof=ellipsoid_ddof,
                resolution=ellipsoid_resolution,
                opacity=ellipsoid_opacity,
                colors=palette,
                lighting=lighting,
                light_azimuth=light_azimuth,
                light_altitude=light_altitude,
                light_strength=light_strength,
                ground_shadows=ground_shadows,
                ground_level=shadow_floor,
                shadow_opacity=shadow_opacity,
                shadow_stems=shadow_stems,
                shadow_stem_opacity=shadow_stem_opacity,
            )

        data_min = self.observations.min(axis=0)
        data_max = self.observations.max(axis=0)
        for item in geometry.values():
            data_min = np.minimum(data_min, item.center - item.radii)
            data_max = np.maximum(data_max, item.center + item.radii)
        if shadow_floor is not None:
            data_min[2] = min(data_min[2], shadow_floor)
        ranges = data_max - data_min
        ranges[ranges == 0] = 1.0
        padding = ranges * 0.08
        axis.set_xlim(data_min[0] - padding[0], data_max[0] + padding[0])
        axis.set_ylim(data_min[1] - padding[1], data_max[1] + padding[1])
        axis.set_zlim(
            shadow_floor if shadow_floor is not None else data_min[2] - padding[2],
            data_max[2] + padding[2],
        )
        aspect = np.asarray(box_aspect if box_aspect is not None else ranges / ranges.max())
        if aspect.shape != (3,) or not np.isfinite(aspect).all() or np.any(aspect <= 0):
            raise ValueError("box_aspect must contain three positive finite values.")
        axis.set_box_aspect(aspect, zoom=zoom)
        axis.view_init(elev=elevation, azim=azimuth, roll=roll)

        if axes:
            axis.set_xlabel(axis_labels[0], labelpad=14, color=axis_color, fontsize=axis_label_fontsize)
            axis.set_ylabel(axis_labels[1], labelpad=17, color=axis_color, fontsize=axis_label_fontsize)
            axis.set_zlabel(axis_labels[2], labelpad=14, color=axis_color, fontsize=axis_label_fontsize)
            for index, current_axis in enumerate((axis.xaxis, axis.yaxis, axis.zaxis)):
                if tick_values is not None and tick_values[index] is not None:
                    current_axis.set_major_locator(FixedLocator(tick_values[index]))
                else:
                    current_axis.set_major_locator(MaxNLocator(nbins=max_ticks[index]))
                current_axis.pane.fill = False
                current_axis.pane.set_edgecolor((1, 1, 1, 0))
                current_axis.line.set_color(axis_color)
                current_axis.line.set_linewidth(axis_linewidth)
                current_axis._axinfo["grid"].update({"linewidth": 0 if not grid else 0.35})
                current_axis._axinfo["tick"]["linewidth"] = {
                    True: axis_linewidth,
                    False: axis_linewidth,
                }
            if show_ground_plane and shadow_floor is not None:
                axis.zaxis.pane.fill = True
                axis.zaxis.pane.set_facecolor(to_rgb(ground_plane_color) + (1.0,))
            axis.tick_params(
                axis="both",
                which="major",
                labelsize=tick_label_fontsize,
                pad=3,
                colors=axis_color,
                width=axis_linewidth,
            )
            axis.grid(grid)
        else:
            axis.set_axis_off()

        order = list(legend_order) if legend_order is not None else labels
        unknown = [label for label in order if label not in palette]
        if unknown:
            raise ValueError(f"legend_order contains unknown labels: {unknown}")
        if axes and legend:
            handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markerfacecolor=palette[label],
                    markeredgecolor=point_edge_color if outlines else "none",
                    markeredgewidth=0.65 if outlines else 0,
                    markersize=9,
                )
                for label in order
            ]
            legend_labels = [display_names[label] for label in order]
            if figure_legend:
                figure.legend(
                    handles,
                    legend_labels,
                    loc="upper right",
                    bbox_to_anchor=(0.95, 0.935),
                    ncol=len(order),
                    frameon=False,
                    columnspacing=1.5,
                    handletextpad=0.45,
                    fontsize=legend_fontsize,
                    title=legend_title,
                )
            else:
                axis.legend(
                    handles,
                    legend_labels,
                    frameon=False,
                    title=legend_title,
                    loc="upper left",
                    fontsize=legend_fontsize,
                )

        if title:
            figure.text(
                0.05,
                0.925,
                title,
                ha="left",
                va="center",
                fontsize=title_fontsize,
                color=axis_color,
            )
        if subtitle:
            figure.text(
                0.05,
                0.875,
                subtitle,
                ha="left",
                va="center",
                fontsize=subtitle_fontsize,
                color=axis_color,
            )

        self.figure = figure
        self.axis = axis
        return figure

    def export(
        self,
        out_path,
        *,
        transparent: bool = False,
        bbox_inches=None,
        close: bool = True,
        **plot_kwargs,
    ):
        """Build and save a figure, preserving the requested canvas shape."""
        path = Path(out_path)
        if (
            plot_kwargs.get("cluster_separation", True)
            and "cluster_separation_out_path" not in plot_kwargs
        ):
            plot_kwargs["cluster_separation_out_path"] = (
                self._separation_path_for_figure(path)
            )
        figure = self.plot(**plot_kwargs)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            path,
            dpi=plot_kwargs.get("dpi", 300),
            bbox_inches=bbox_inches,
            transparent=transparent,
            facecolor="none" if transparent else plot_kwargs.get("background_color", "white"),
        )
        if close:
            plt.close(figure)
        return str(path)

