"""Standard-deviation ellipses and ellipsoids for clustered point clouds.

Input arrays use the conventional tabular layout: observations are rows and
dimensions are columns. Two-dimensional inputs produce ellipses and
three-dimensional inputs produce ellipsoids. Each axis-aligned boundary is
defined by

    sum(((coordinate - cluster_mean) / (n_std * cluster_std)) ** 2) = 1

across all input dimensions. The module also provides pairwise Welch tests for
cluster separation. The preferred test projects both clusters onto the line
joining their centroids: the difference between projected means
is exactly the L2 distance between the centroids, while the denominator uses
the observed spatial standard deviations along that line.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Hashable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind


@dataclass(frozen=True)
class ClusterEllipsoid:
    """Geometry of one axis-aligned 2D ellipse or 3D ellipsoid."""

    label: Hashable
    center: np.ndarray
    standard_deviation: np.ndarray
    radii: np.ndarray
    n_points: int
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray | None = None

    @property
    def n_dimensions(self):
        """Number of coordinate dimensions represented by this geometry."""
        return int(self.center.size)


def _validate_cluster_data(observations, labels):
    points = np.asarray(observations, dtype=float)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError(
            "observations must have shape (n_observations, 2) or "
            "(n_observations, 3), with dimensions in columns."
        )
    if labels is None:
        labels = np.zeros(points.shape[0], dtype=int)
    else:
        labels = np.asarray(labels)
    if labels.ndim != 1 or labels.shape[0] != points.shape[0]:
        raise ValueError("labels must be one-dimensional with one label per point.")
    if not np.isfinite(points).all():
        raise ValueError("points must contain only finite values.")
    return points, labels


def _cluster_labels(labels, exclude_noise):
    unique = list(pd.unique(labels))
    if exclude_noise:
        unique = [label for label in unique if label != -1]
    return unique


def calculate_cluster_ellipsoids(
    observations,
    labels=None,
    *,
    n_std: float = 3.0,
    ddof: int = 1,
    resolution: int = 36,
    exclude_noise: bool = True,
):
    """Calculate one axis-aligned SD ellipse or ellipsoid per cluster.

    Parameters
    ----------
    observations : array-like, shape (n_observations, 2 or 3)
        Point coordinates, with observations in rows and dimensions in
        columns.
    labels : array-like, shape (n_observations,), optional
        Group membership. If omitted, all observations form one group labeled
        ``0``. A ``-1`` noise label is excluded by default.
    n_std : float, default 3
        Number of standard deviations represented by each semi-axis.
    ddof : int, default 1
        Delta degrees of freedom used to calculate each axis standard
        deviation.  The default gives sample standard deviations.
    resolution : int, default 36
        Number of samples in each angular direction of the surface mesh.

    Returns
    -------
    dict
        Labels mapped to :class:`ClusterEllipsoid` instances. For 2D input,
        each instance's ``z`` attribute is ``None``.

    Notes
    -----
    A zero-variance dimension has a zero radius, producing a valid degenerate
    (flat) surface.  Clusters too small for the requested ``ddof`` are omitted.
    """
    points, labels = _validate_cluster_data(observations, labels)
    if not np.isfinite(n_std) or n_std <= 0:
        raise ValueError("n_std must be a positive finite number.")
    if not isinstance(ddof, (int, np.integer)) or ddof < 0:
        raise ValueError("ddof must be a non-negative integer.")
    if not isinstance(resolution, (int, np.integer)) or resolution < 4:
        raise ValueError("resolution must be an integer of at least 4.")

    if points.shape[1] == 2:
        angle = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=True)
        unit_x = np.cos(angle)
        unit_y = np.sin(angle)
        unit_z = None
    else:
        azimuth = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=True)
        polar = np.linspace(0.0, np.pi, resolution, endpoint=True)
        azimuth, polar = np.meshgrid(azimuth, polar)
        unit_x = np.cos(azimuth) * np.sin(polar)
        unit_y = np.sin(azimuth) * np.sin(polar)
        unit_z = np.cos(polar)

    ellipsoids = {}
    for label in _cluster_labels(labels, exclude_noise):
        cluster = points[labels == label]
        if len(cluster) <= ddof:
            continue
        center = cluster.mean(axis=0)
        standard_deviation = cluster.std(axis=0, ddof=ddof)
        radii = n_std * standard_deviation
        ellipsoids[label] = ClusterEllipsoid(
            label=label,
            center=center,
            standard_deviation=standard_deviation,
            radii=radii,
            n_points=len(cluster),
            x=center[0] + radii[0] * unit_x,
            y=center[1] + radii[1] * unit_y,
            z=(center[2] + radii[2] * unit_z) if unit_z is not None else None,
        )
    return ellipsoids


def _rgba(color, opacity):
    """Return a Plotly-compatible color with the requested opacity."""
    if isinstance(color, str) and color.startswith("rgba("):
        values = color[5:-1].split(",")[:3]
        return f"rgba({','.join(value.strip() for value in values)},{opacity})"
    if isinstance(color, str) and color.startswith("rgb("):
        return f"rgba({color[4:-1]},{opacity})"
    if isinstance(color, str) and color.startswith("#") and len(color) == 7:
        red, green, blue = (
            int(color[index:index + 2], 16) for index in (1, 3, 5)
        )
        return f"rgba({red},{green},{blue},{opacity})"
    return color


def add_cluster_ellipsoids_plotly(
    figure,
    observations,
    labels=None,
    *,
    n_std: float = 3.0,
    ddof: int = 1,
    resolution: int = 36,
    opacity: float = 0.18,
    colors: Mapping[Hashable, str] | None = None,
    exclude_noise: bool = True,
    show_legend: bool = True,
):
    """Add SD ellipses or ellipsoids to a compatible Plotly figure."""
    import plotly.graph_objects as go
    from plotly.colors import qualitative

    if not 0.0 <= opacity <= 1.0:
        raise ValueError("opacity must be between 0 and 1.")
    ellipsoids = calculate_cluster_ellipsoids(
        observations,
        labels,
        n_std=n_std,
        ddof=ddof,
        resolution=resolution,
        exclude_noise=exclude_noise,
    )
    palette = qualitative.Plotly
    for index, (label, ellipsoid) in enumerate(ellipsoids.items()):
        color = colors[label] if colors and label in colors else palette[index % len(palette)]
        # Keep the colorscale opaque and let the trace opacity control the
        # transparency; otherwise RGBA alpha and trace alpha are multiplied.
        surface_color = _rgba(color, 1.0)
        common = dict(
            name=f"Cluster {label}: {n_std:g} SD",
            legendgroup=f"cluster-{label}",
            showlegend=show_legend,
            hovertemplate=(
                f"Cluster {label}<br>{n_std:g}-SD boundary<extra></extra>"
            ),
        )
        if ellipsoid.n_dimensions == 2:
            figure.add_trace(
                go.Scatter(
                    x=ellipsoid.x,
                    y=ellipsoid.y,
                    mode="lines",
                    line=dict(color=surface_color),
                    fill="toself",
                    fillcolor=_rgba(color, opacity),
                    **common,
                )
            )
        else:
            figure.add_trace(
                go.Surface(
                    x=ellipsoid.x,
                    y=ellipsoid.y,
                    z=ellipsoid.z,
                    surfacecolor=np.zeros_like(ellipsoid.x),
                    colorscale=[[0.0, surface_color], [1.0, surface_color]],
                    cmin=0,
                    cmax=1,
                    opacity=opacity,
                    showscale=False,
                    **common,
                )
            )
    return ellipsoids


def add_cluster_ellipsoids_matplotlib(
    axis,
    observations,
    labels=None,
    *,
    n_std: float = 3.0,
    ddof: int = 1,
    resolution: int = 36,
    opacity: float = 0.18,
    colors: Mapping[Hashable, object] | None = None,
    exclude_noise: bool = True,
    lighting: bool = True,
    light_azimuth: float = 315.0,
    light_altitude: float = 45.0,
    light_strength: float = 0.22,
    ground_shadows: bool = False,
    ground_level: float | None = None,
    shadow_opacity: float = 0.10,
    shadow_stems: bool = False,
    shadow_stem_opacity: float = 0.24,
):
    """Add SD ellipses or ellipsoids to a compatible Matplotlib axis.

    Three-dimensional surfaces use a soft, hue-preserving upper-left light by
    default. Unlike Matplotlib's built-in surface shading, this renderer keeps
    shadows within the cluster color instead of driving them toward gray.
    ``light_azimuth`` and ``light_altitude`` use Matplotlib's
    :class:`~matplotlib.colors.LightSource` convention, in degrees. Set
    ``lighting=False`` to restore a flat, unshaded surface.

    For three-dimensional data, ``ground_shadows=True`` projects each
    ellipsoid's x/y footprint onto a shared z plane. If ``ground_level`` is
    omitted, the plane is placed at the lowest ellipsoid boundary. Optional
    centroid stems connect each cluster center to its projected footprint.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource

    if not 0.0 <= opacity <= 1.0:
        raise ValueError("opacity must be between 0 and 1.")
    if not np.isfinite(light_azimuth):
        raise ValueError("light_azimuth must be finite.")
    if not np.isfinite(light_altitude) or not -90.0 <= light_altitude <= 90.0:
        raise ValueError("light_altitude must be finite and between -90 and 90 degrees.")
    if not np.isfinite(light_strength) or not 0.0 <= light_strength <= 1.0:
        raise ValueError("light_strength must be between 0 and 1.")
    if ground_level is not None and not np.isfinite(ground_level):
        raise ValueError("ground_level must be finite when provided.")
    if not np.isfinite(shadow_opacity) or not 0.0 <= shadow_opacity <= 1.0:
        raise ValueError("shadow_opacity must be between 0 and 1.")
    if not np.isfinite(shadow_stem_opacity) or not 0.0 <= shadow_stem_opacity <= 1.0:
        raise ValueError("shadow_stem_opacity must be between 0 and 1.")
    light_source = (
        LightSource(azdeg=light_azimuth, altdeg=light_altitude)
        if lighting
        else None
    )
    ellipsoids = calculate_cluster_ellipsoids(
        observations,
        labels,
        n_std=n_std,
        ddof=ddof,
        resolution=resolution,
        exclude_noise=exclude_noise,
    )
    palette = list(plt.cm.tab10.colors)
    if (
        ground_shadows
        and ellipsoids
        and next(iter(ellipsoids.values())).n_dimensions == 3
    ):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        if ground_level is None:
            ground_level = min(
                ellipsoid.center[2] - ellipsoid.radii[2]
                for ellipsoid in ellipsoids.values()
            )
        angle = np.linspace(0.0, 2.0 * np.pi, max(resolution, 24), endpoint=True)
        for index, (label, ellipsoid) in enumerate(ellipsoids.items()):
            color = colors[label] if colors and label in colors else palette[index % len(palette)]
            vertices = np.column_stack(
                [
                    ellipsoid.center[0] + ellipsoid.radii[0] * np.cos(angle),
                    ellipsoid.center[1] + ellipsoid.radii[1] * np.sin(angle),
                    np.full(angle.shape, ground_level),
                ]
            )
            shadow = Poly3DCollection(
                [vertices],
                facecolors=[color],
                edgecolors="none",
                linewidths=0,
                alpha=shadow_opacity,
                zorder=0,
            )
            axis.add_collection3d(shadow)
            if shadow_stems:
                axis.plot(
                    [ellipsoid.center[0], ellipsoid.center[0]],
                    [ellipsoid.center[1], ellipsoid.center[1]],
                    [ground_level, ellipsoid.center[2]],
                    color=color,
                    alpha=shadow_stem_opacity,
                    linewidth=0.9,
                    linestyle=(0, (2, 3)),
                    zorder=1,
                )
    for index, (label, ellipsoid) in enumerate(ellipsoids.items()):
        color = colors[label] if colors and label in colors else palette[index % len(palette)]
        if ellipsoid.n_dimensions == 2:
            axis.fill(ellipsoid.x, ellipsoid.y, color=color, alpha=opacity)
            axis.plot(ellipsoid.x, ellipsoid.y, color=color, linewidth=1.0)
        else:
            surface_options = {
                "linewidth": 0,
                "antialiased": True,
                "shade": False,
            }
            if lighting:
                from matplotlib.colors import to_rgb

                radii_squared = np.square(ellipsoid.radii)
                safe_radii_squared = np.where(radii_squared == 0.0, 1.0, radii_squared)
                normals = np.stack(
                    [
                        (ellipsoid.x - ellipsoid.center[0]) / safe_radii_squared[0],
                        (ellipsoid.y - ellipsoid.center[1]) / safe_radii_squared[1],
                        (ellipsoid.z - ellipsoid.center[2]) / safe_radii_squared[2],
                    ],
                    axis=-1,
                )
                normal_lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
                normals = np.divide(
                    normals,
                    normal_lengths,
                    out=np.zeros_like(normals),
                    where=normal_lengths > 0,
                )
                diffuse = np.clip(normals @ light_source.direction, 0.0, 1.0)
                base_color = np.asarray(to_rgb(color), dtype=float)
                # Ambient light keeps the original hue; a restrained white
                # highlight gives curvature without the muddy dark facets of
                # Matplotlib's default shader.
                ambient = 1.0 - light_strength * (1.0 - diffuse)
                highlight = 0.30 * light_strength * diffuse
                shaded_rgb = (
                    base_color[None, None, :] * ambient[..., None]
                    + (1.0 - base_color[None, None, :]) * highlight[..., None]
                )
                facecolors = np.concatenate(
                    [
                        np.clip(shaded_rgb, 0.0, 1.0),
                        np.full((*diffuse.shape, 1), opacity),
                    ],
                    axis=-1,
                )
                surface_options["facecolors"] = facecolors
            else:
                surface_options.update(color=color, alpha=opacity)
            axis.plot_surface(
                ellipsoid.x,
                ellipsoid.y,
                ellipsoid.z,
                **surface_options,
            )
    return ellipsoids


def _adjust_pvalues(p_values, method):
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_values, np.nan)
    finite = np.isfinite(p_values)
    values = p_values[finite]
    if method in (None, "none"):
        adjusted[finite] = values
    elif method == "bonferroni":
        adjusted[finite] = np.minimum(values * len(values), 1.0)
    elif method == "holm":
        order = np.argsort(values)
        ordered = values[order]
        corrected = np.maximum.accumulate((len(values) - np.arange(len(values))) * ordered)
        corrected = np.minimum(corrected, 1.0)
        restored = np.empty_like(corrected)
        restored[order] = corrected
        adjusted[finite] = restored
    else:
        raise ValueError("p_adjust must be one of: 'holm', 'bonferroni', or None.")
    return adjusted


def pairwise_cluster_separation_tests(
    observations,
    labels=None,
    *,
    score: str = "centroid_axis",
    alternative: str = "two-sided",
    p_adjust: str | None = "holm",
    alpha: float = 0.05,
    ellipsoid_n_std: float = 3.0,
    ddof: int = 1,
    exclude_noise: bool = True,
):
    """Run pairwise Welch tests between clusters in a 2D or 3D point cloud.

    ``score="centroid_axis"`` (recommended) projects each pair onto the line
    joining its centroids.  The projected mean difference equals the L2
    centroid distance, and Welch's denominator uses each cluster's spatial
    standard deviation along that line.  This is invariant to translating and
    rotating the point cloud.

    ``score="l2_origin"`` performs the literal requested comparison of each
    point's L2 norm from the coordinate origin. It is provided for completeness
    but is origin-dependent and should not be interpreted as a test of overlap.

    A non-significant p-value is not evidence that clusters overlap.  The
    returned ``ellipsoid_axis_gap`` is positive only when the selected SD
    ellipsoids are separated along their centroid axis; a non-positive value
    means their projections overlap, not necessarily the full 2D/3D regions.

    Because ``centroid_axis`` estimates the projection direction from the same
    observations it tests, its p-value is exploratory rather than a calibrated
    multivariate confirmatory test.  Use held-out data, a permutation test, or
    a multivariate two-sample procedure for confirmatory inference.
    """
    points, labels = _validate_cluster_data(observations, labels)
    if score not in {"centroid_axis", "l2_origin"}:
        raise ValueError("score must be 'centroid_axis' or 'l2_origin'.")
    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError("alternative must be 'two-sided', 'less', or 'greater'.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    if p_adjust not in {None, "none", "holm", "bonferroni"}:
        raise ValueError("p_adjust must be one of: 'holm', 'bonferroni', or None.")

    cluster_labels = _cluster_labels(labels, exclude_noise)
    ellipsoids = calculate_cluster_ellipsoids(
        points,
        labels,
        n_std=ellipsoid_n_std,
        ddof=ddof,
        exclude_noise=exclude_noise,
    )
    rows = []
    for label_a, label_b in combinations(cluster_labels, 2):
        cluster_a = points[labels == label_a]
        cluster_b = points[labels == label_b]
        if len(cluster_a) <= ddof or len(cluster_b) <= ddof:
            continue
        center_a = cluster_a.mean(axis=0)
        center_b = cluster_b.mean(axis=0)
        delta = center_b - center_a
        centroid_distance = float(np.linalg.norm(delta))
        if centroid_distance > 0:
            direction = delta / centroid_distance
        else:
            direction = np.zeros(points.shape[1], dtype=float)
            direction[0] = 1.0

        if score == "centroid_axis":
            values_a = cluster_a @ direction
            values_b = cluster_b @ direction
        else:
            values_a = np.linalg.norm(cluster_a, axis=1)
            values_b = np.linalg.norm(cluster_b, axis=1)

        result = ttest_ind(
            values_a,
            values_b,
            equal_var=False,
            nan_policy="omit",
            alternative=alternative,
        )
        ellipsoid_a = ellipsoids[label_a]
        ellipsoid_b = ellipsoids[label_b]
        support_a = float(np.sqrt(np.sum((ellipsoid_a.radii * direction) ** 2)))
        support_b = float(np.sqrt(np.sum((ellipsoid_b.radii * direction) ** 2)))
        axis_gap = centroid_distance - support_a - support_b
        row = {
            "cluster_a": label_a,
            "cluster_b": label_b,
            "n_a": len(cluster_a),
            "n_b": len(cluster_b),
            "score": score,
            "mean_a": float(np.mean(values_a)),
            "mean_b": float(np.mean(values_b)),
            "standard_deviation_a": float(np.std(values_a, ddof=ddof)),
            "standard_deviation_b": float(np.std(values_b, ddof=ddof)),
            "centroid_distance_l2": centroid_distance,
            "t_statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "ellipsoid_axis_gap": axis_gap,
            "ellipsoid_axis_projection_overlap": bool(axis_gap <= 0),
        }
        row.update(
            {
                f"axis_dimension_{index + 1}": float(value)
                for index, value in enumerate(direction)
            }
        )
        rows.append(row)

    axis_columns = [
        f"axis_dimension_{index + 1}" for index in range(points.shape[1])
    ]
    columns = [
        "cluster_a", "cluster_b", "n_a", "n_b", "score", "mean_a", "mean_b",
        "standard_deviation_a", "standard_deviation_b", "centroid_distance_l2",
        *axis_columns, "t_statistic", "p_value",
        "p_value_adjusted", "reject_null", "ellipsoid_axis_gap",
        "ellipsoid_axis_projection_overlap",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    frame["p_value_adjusted"] = _adjust_pvalues(frame["p_value"], p_adjust)
    frame["reject_null"] = frame["p_value_adjusted"] < alpha
    return frame.loc[:, columns]


__all__ = [
    "ClusterEllipsoid",
    "add_cluster_ellipsoids_matplotlib",
    "add_cluster_ellipsoids_plotly",
    "calculate_cluster_ellipsoids",
    "pairwise_cluster_separation_tests",
]
