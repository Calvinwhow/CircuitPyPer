"""Pairwise Hotelling tests for labeled multidimensional point clouds.

Observations are rows and dimensions are columns. The two-sample Hotelling
test is the multivariate analogue of the pooled two-sample t-test: it tests
whether two centroid vectors are equal while accounting for the pooled spatial
covariance of the point clouds.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_distribution


def _validate_inputs(observations, labels):
    points = np.asarray(observations, dtype=float)
    if points.ndim != 2 or points.shape[1] < 1:
        raise ValueError(
            "observations must have shape (n_observations, n_dimensions)."
        )
    if not np.isfinite(points).all():
        raise ValueError("observations must contain only finite values.")
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != len(points):
        raise ValueError("labels must contain one value per observation.")
    return points, labels


def _adjust_pvalues(p_values, method):
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(values, np.nan)
    finite = np.isfinite(values)
    current = values[finite]
    if method in (None, "none"):
        adjusted[finite] = current
    elif method == "bonferroni":
        adjusted[finite] = np.minimum(current * len(current), 1.0)
    elif method == "holm":
        order = np.argsort(current)
        ordered = current[order]
        corrected = np.maximum.accumulate(
            (len(ordered) - np.arange(len(ordered))) * ordered
        )
        restored = np.empty_like(corrected)
        restored[order] = np.minimum(corrected, 1.0)
        adjusted[finite] = restored
    else:
        raise ValueError(
            "p_adjust must be one of: 'holm', 'bonferroni', or None."
        )
    return adjusted


def pairwise_hotelling_tests(
    observations,
    labels,
    *,
    p_adjust: str | None = "holm",
    alpha: float = 0.05,
    ellipsoid_n_std: float = 3.0,
    ddof: int = 1,
    exclude_noise: bool = True,
):
    """Run a two-sample Hotelling test for every pair of labeled groups.

    The returned ``hotelling_t_squared`` is Hotelling's :math:`T^2`, not a
    signed univariate t statistic. ``f_statistic`` is its exact pooled-
    covariance F transformation with ``df1=n_dimensions`` and
    ``df2=n_a+n_b-n_dimensions-1``.

    ``ellipsoid_axis_gap`` compares the displayed SD ellipsoids along the line
    joining the centroids. A positive value proves that their projections on
    that line do not overlap.
    """
    points, labels = _validate_inputs(observations, labels)
    if p_adjust not in {None, "none", "holm", "bonferroni"}:
        raise ValueError(
            "p_adjust must be one of: 'holm', 'bonferroni', or None."
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    if not np.isfinite(ellipsoid_n_std) or ellipsoid_n_std <= 0:
        raise ValueError("ellipsoid_n_std must be a positive finite number.")
    if not isinstance(ddof, (int, np.integer)) or ddof < 0:
        raise ValueError("ddof must be a non-negative integer.")

    group_labels = list(pd.unique(labels))
    if exclude_noise:
        group_labels = [label for label in group_labels if label != -1]

    n_dimensions = points.shape[1]
    rows = []
    for label_a, label_b in combinations(group_labels, 2):
        group_a = points[labels == label_a]
        group_b = points[labels == label_b]
        n_a, n_b = len(group_a), len(group_b)
        if n_a < 2 or n_b < 2:
            continue

        centroid_a = group_a.mean(axis=0)
        centroid_b = group_b.mean(axis=0)
        delta = centroid_a - centroid_b
        centroid_distance = float(np.linalg.norm(delta))

        # The classical two-sample Hotelling statistic is defined using
        # unbiased within-group sample covariance matrices (ddof=1). ``ddof``
        # remains configurable for the plotted ellipsoid projection only.
        covariance_a = np.atleast_2d(np.cov(group_a, rowvar=False, ddof=1))
        covariance_b = np.atleast_2d(np.cov(group_b, rowvar=False, ddof=1))
        pooled_denominator = n_a + n_b - 2
        pooled_covariance = (
            (n_a - 1) * covariance_a + (n_b - 1) * covariance_b
        ) / pooled_denominator
        covariance_rank = int(np.linalg.matrix_rank(pooled_covariance))
        inverse_covariance = np.linalg.pinv(pooled_covariance)
        hotelling_t_squared = float(
            (n_a * n_b / (n_a + n_b))
            * delta.T
            @ inverse_covariance
            @ delta
        )

        df1 = n_dimensions
        df2 = n_a + n_b - n_dimensions - 1
        if df2 > 0 and covariance_rank == n_dimensions:
            f_statistic = float(
                (df2 / (n_dimensions * pooled_denominator))
                * hotelling_t_squared
            )
            p_value = float(f_distribution.sf(f_statistic, df1, df2))
        else:
            f_statistic = np.nan
            p_value = np.nan

        if centroid_distance > 0:
            direction = (centroid_b - centroid_a) / centroid_distance
        else:
            direction = np.zeros(n_dimensions, dtype=float)
            direction[0] = 1.0
        projected_sd_a = float(np.std(group_a @ direction, ddof=ddof))
        projected_sd_b = float(np.std(group_b @ direction, ddof=ddof))
        ellipsoid_axis_gap = float(
            centroid_distance
            - ellipsoid_n_std * projected_sd_a
            - ellipsoid_n_std * projected_sd_b
        )

        row = {
            "cluster_a": label_a,
            "cluster_b": label_b,
            "n_a": n_a,
            "n_b": n_b,
            "n_dimensions": n_dimensions,
            "centroid_distance_l2": centroid_distance,
            "hotelling_t_squared": hotelling_t_squared,
            "f_statistic": f_statistic,
            "df1": df1,
            "df2": df2,
            "p_value": p_value,
            "covariance_rank": covariance_rank,
            "pooled_covariance_singular": covariance_rank < n_dimensions,
            "projected_standard_deviation_a": projected_sd_a,
            "projected_standard_deviation_b": projected_sd_b,
            "ellipsoid_axis_gap": ellipsoid_axis_gap,
            "ellipsoid_axis_projection_overlap": ellipsoid_axis_gap <= 0,
        }
        row.update(
            {
                f"centroid_a_dimension_{index + 1}": float(value)
                for index, value in enumerate(centroid_a)
            }
        )
        row.update(
            {
                f"centroid_b_dimension_{index + 1}": float(value)
                for index, value in enumerate(centroid_b)
            }
        )
        rows.append(row)

    centroid_columns = [
        *(f"centroid_a_dimension_{index + 1}" for index in range(n_dimensions)),
        *(f"centroid_b_dimension_{index + 1}" for index in range(n_dimensions)),
    ]
    columns = [
        "cluster_a",
        "cluster_b",
        "n_a",
        "n_b",
        "n_dimensions",
        *centroid_columns,
        "centroid_distance_l2",
        "hotelling_t_squared",
        "f_statistic",
        "df1",
        "df2",
        "p_value",
        "p_value_adjusted",
        "reject_null",
        "covariance_rank",
        "pooled_covariance_singular",
        "projected_standard_deviation_a",
        "projected_standard_deviation_b",
        "ellipsoid_axis_gap",
        "ellipsoid_axis_projection_overlap",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    results = pd.DataFrame(rows)
    results["p_value_adjusted"] = _adjust_pvalues(
        results["p_value"], p_adjust
    )
    results["reject_null"] = results["p_value_adjusted"] < alpha
    return results.loc[:, columns]


def main(argv=None):
    """Run the pairwise test on a CSV containing coordinates and labels."""
    parser = argparse.ArgumentParser(
        description="Pairwise Hotelling tests for a labeled point cloud."
    )
    parser.add_argument("input_csv", help="CSV containing point coordinates.")
    parser.add_argument(
        "--coordinate-columns",
        nargs="+",
        default=["umap_x", "umap_y", "umap_z"],
        help="Coordinate columns (default: umap_x umap_y umap_z).",
    )
    parser.add_argument(
        "--label-column",
        default="cluster_label",
        help="Cluster-label column (default: cluster_label).",
    )
    parser.add_argument(
        "--output-csv",
        help=(
            "Destination CSV. Defaults to "
            "<input_stem>_cluster_separation_hotelling.csv."
        ),
    )
    parser.add_argument("--ellipsoid-n-std", type=float, default=3.0)
    parser.add_argument(
        "--p-adjust", choices=["holm", "bonferroni", "none"], default="holm"
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input_csv)
    frame = pd.read_csv(input_path)
    required = [*args.coordinate_columns, args.label_column]
    missing = [column for column in required if column not in frame]
    if missing:
        parser.error(f"missing CSV columns: {missing}")
    results = pairwise_hotelling_tests(
        frame.loc[:, args.coordinate_columns].to_numpy(),
        frame[args.label_column].to_numpy(),
        p_adjust=args.p_adjust,
        ellipsoid_n_std=args.ellipsoid_n_std,
    )
    output_path = (
        Path(args.output_csv)
        if args.output_csv
        else input_path.with_name(
            f"{input_path.stem}_cluster_separation_hotelling.csv"
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(results.to_string(index=False))
    print(f"Saved: {output_path}")
    return results


if __name__ == "__main__":
    main()


__all__ = ["pairwise_hotelling_tests", "main"]
