"""Identifiability projection onto unit pooled prediction variance."""

import numpy as np
from functools import partial


def gram(X, dataset_weights):
    """Pool patient-score covariance across datasets.

    X: dict[str, np.ndarray] with D dataset-name keys; each value has shape (N, B), where N is patients and B is basis maps.
    dataset_weights: dict[str, float] of length D; keys match X and scalar values weight each dataset's covariance.
    """
    n_components = next(iter(X.values())).shape[1]
    covariance = np.zeros((n_components, n_components))
    for name, values in X.items():
        centered = values - values.mean(axis=0)
        covariance += dataset_weights[name] * centered.T @ centered / len(centered)
    return covariance


def project(weights, covariance):
    variance = float(weights @ covariance @ weights)
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("Weights have zero or invalid prediction variance.")
    return np.asarray(weights) / np.sqrt(variance)


def normalize(weights, X, dataset_weights):
    return project(weights, gram(X, dataset_weights))


def main(X, dataset_weights):
    """Return pooled prediction covariance and its unit-variance projection.
    X: dict[str, np.ndarray] of length D; each value is patient-by-basis with shape (N, B).
    dataset_weights: dict[str, float] of length D; values weight dataset covariances.
    """
    covariance = gram(X, dataset_weights)
    return covariance, partial(project, covariance=covariance)
