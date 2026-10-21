"""Label-invariant cluster recovery scored as N-class classification.

The primary statistic is Hungarian-aligned macro-F1. True classes and
discovered clusters are matched one-to-one by maximizing their pairwise F1
scores. This makes the score invariant to arbitrary cluster labels while
penalizing merged classes, fragmented classes, extra clusters, and noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class HungarianMacroF1Result:
    """Complete output of a Hungarian-aligned macro-F1 calculation."""

    macro_f1: float
    per_class: pd.DataFrame
    per_cluster: pd.DataFrame
    contingency: pd.DataFrame
    n_observations: int
    n_classes: int
    n_clusters: int
    n_noise: int

    @property
    def perfect_parcellation(self) -> bool:
        """Whether the discovered partition exactly recovers every class."""
        return bool(np.isclose(self.macro_f1, 1.0, rtol=0.0, atol=1e-12))

    def summary(self) -> dict:
        """Return scalar results suitable for a CSV row."""
        return {
            "hungarian_macro_f1": self.macro_f1,
            "perfect_parcellation": self.perfect_parcellation,
            "n_observations": self.n_observations,
            "n_classes": self.n_classes,
            "n_clusters": self.n_clusters,
            "n_noise": self.n_noise,
        }


def _ordered_unique(values):
    return list(pd.unique(np.asarray(values, dtype=object)))


def hungarian_macro_f1(
    true_classes,
    predicted_clusters,
    *,
    noise_label: Hashable = -1,
) -> HungarianMacroF1Result:
    """Score a clustering using one-to-one, label-invariant macro-F1.

    Parameters
    ----------
    true_classes : array-like, shape (n_observations,)
        Prespecified class/domain for every observation.
    predicted_clusters : array-like, shape (n_observations,)
        Discovered cluster label for every observation. Labels need not match
        the true class names and may change across clustering runs.
    noise_label : hashable, default -1
        Predicted label treated as unassigned noise. Noise observations remain
        in every true-class denominator and therefore reduce recall.

    Returns
    -------
    HungarianMacroF1Result
        Overall macro-F1, per-class and per-cluster F1 tables, and the full
        true-class-by-predicted-cluster contingency table.

    Notes
    -----
    For true class ``c`` and predicted cluster ``k``, pairwise F1 is

        ``2 * n_ck / (n_c + n_k)``.

    The Hungarian algorithm selects a one-to-one matching that maximizes the
    sum of these scores. Unmatched true classes, unmatched predicted clusters,
    and noise receive F1 zero. Consequently, macro-F1 equals one only for an
    exact one-cluster-per-class parcellation with no noise.
    """
    true = np.asarray(true_classes, dtype=object)
    predicted = np.asarray(predicted_clusters, dtype=object)
    if true.ndim != 1 or predicted.ndim != 1:
        raise ValueError("true_classes and predicted_clusters must be one-dimensional.")
    if len(true) != len(predicted):
        raise ValueError("true_classes and predicted_clusters must have equal length.")
    if len(true) == 0:
        raise ValueError("at least one observation is required.")
    if pd.isna(true).any():
        raise ValueError("true_classes cannot contain missing values.")
    if pd.isna(predicted).any():
        raise ValueError(
            "predicted_clusters cannot contain missing values; use noise_label instead."
        )

    class_labels = _ordered_unique(true)
    cluster_labels = [
        label for label in _ordered_unique(predicted) if label != noise_label
    ]
    n_classes = len(class_labels)
    n_clusters = len(cluster_labels)
    class_counts = np.array([(true == label).sum() for label in class_labels], int)
    cluster_counts = np.array(
        [(predicted == label).sum() for label in cluster_labels], int
    )
    counts = np.zeros((n_classes, n_clusters), dtype=int)
    for class_index, class_label in enumerate(class_labels):
        for cluster_index, cluster_label in enumerate(cluster_labels):
            counts[class_index, cluster_index] = int(
                np.sum((true == class_label) & (predicted == cluster_label))
            )

    if n_clusters:
        denominators = class_counts[:, None] + cluster_counts[None, :]
        pairwise_f1 = np.divide(
            2.0 * counts,
            denominators,
            out=np.zeros_like(counts, dtype=float),
            where=denominators > 0,
        )
        matched_class_indices, matched_cluster_indices = linear_sum_assignment(
            -pairwise_f1
        )
        class_to_cluster = dict(
            zip(matched_class_indices.tolist(), matched_cluster_indices.tolist())
        )
        cluster_to_class = {
            cluster_index: class_index
            for class_index, cluster_index in class_to_cluster.items()
        }
    else:
        pairwise_f1 = np.zeros((n_classes, 0), dtype=float)
        class_to_cluster = {}
        cluster_to_class = {}

    per_class_columns = [
        "true_class",
        "matched_cluster",
        "n_true_class",
        "n_matched_cluster",
        "true_positives",
        "precision",
        "recall",
        "f1",
    ]
    per_cluster_columns = [
        "predicted_cluster",
        "matched_true_class",
        "n_cluster",
        "n_true_class",
        "true_positives",
        "precision",
        "recall",
        "f1",
    ]
    per_class_rows = []
    class_f1 = np.zeros(n_classes, dtype=float)
    for class_index, class_label in enumerate(class_labels):
        cluster_index = class_to_cluster.get(class_index)
        if cluster_index is None:
            matched_cluster = None
            n_cluster = 0
            true_positives = 0
        else:
            matched_cluster = cluster_labels[cluster_index]
            n_cluster = int(cluster_counts[cluster_index])
            true_positives = int(counts[class_index, cluster_index])
        precision = true_positives / n_cluster if n_cluster else 0.0
        recall = true_positives / class_counts[class_index]
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        class_f1[class_index] = f1
        per_class_rows.append(
            {
                "true_class": class_label,
                "matched_cluster": matched_cluster,
                "n_true_class": int(class_counts[class_index]),
                "n_matched_cluster": n_cluster,
                "true_positives": true_positives,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    per_cluster_rows = []
    for cluster_index, cluster_label in enumerate(cluster_labels):
        class_index = cluster_to_class.get(cluster_index)
        if class_index is None:
            matched_class = None
            n_true_class = 0
            true_positives = 0
        else:
            matched_class = class_labels[class_index]
            n_true_class = int(class_counts[class_index])
            true_positives = int(counts[class_index, cluster_index])
        n_cluster = int(cluster_counts[cluster_index])
        precision = true_positives / n_cluster if n_cluster else 0.0
        recall = true_positives / n_true_class if n_true_class else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_cluster_rows.append(
            {
                "predicted_cluster": cluster_label,
                "matched_true_class": matched_class,
                "n_cluster": n_cluster,
                "n_true_class": n_true_class,
                "true_positives": true_positives,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    contingency = pd.DataFrame(
        counts,
        index=pd.Index(class_labels, name="true_class"),
        columns=pd.Index(cluster_labels, name="predicted_cluster"),
    )
    return HungarianMacroF1Result(
        macro_f1=float(class_f1.mean()),
        per_class=pd.DataFrame(per_class_rows, columns=per_class_columns),
        per_cluster=pd.DataFrame(per_cluster_rows, columns=per_cluster_columns),
        contingency=contingency,
        n_observations=len(true),
        n_classes=n_classes,
        n_clusters=n_clusters,
        n_noise=int(np.sum(predicted == noise_label)),
    )


__all__ = ["HungarianMacroF1Result", "hungarian_macro_f1"]
