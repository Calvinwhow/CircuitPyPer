"""Generalized-Rayleigh initialization and normalized sparse objective."""

import numpy as np
from functools import partial
from scipy.stats import rankdata


def prepare(X, y, dataset_weights):
    B_by_dataset = {}
    Q_by_dataset = {}
    n_components = next(iter(X.values())).shape[1]
    B = np.zeros((n_components, n_components))
    G = np.zeros_like(B)
    for name in X:
        ranks = rankdata(y[name], method="average").astype(float)
        ranks -= ranks.mean()
        association = X[name].T @ ranks
        centered = X[name] - X[name].mean(axis=0)
        B_by_dataset[name] = np.outer(association, association) / float(ranks @ ranks)
        Q_by_dataset[name] = centered.T @ centered
        B += dataset_weights[name] * B_by_dataset[name]
        G += dataset_weights[name] * Q_by_dataset[name]
    return B_by_dataset, Q_by_dataset, B, G, dataset_weights


def initialize(B, G, singular_tolerance):
    eigenvalues, eigenvectors = np.linalg.eigh(G)
    supported = eigenvalues > singular_tolerance * eigenvalues[-1]
    whitening = eigenvectors[:, supported] / np.sqrt(eigenvalues[supported])
    values, vectors = np.linalg.eigh(whitening.T @ B @ whitening)
    weights = whitening @ vectors[:, -1]
    weights /= np.sqrt(weights @ G @ weights)
    if weights[np.argmax(np.abs(weights))] < 0:
        weights *= -1
    return weights, float(values[-1]), int(supported.sum())


def merit(weights, prepared, denominator_tolerance):
    B_by_dataset, Q_by_dataset, _, _, dataset_weights = prepared
    value = 0.0
    for name in B_by_dataset:
        denominator = float(weights @ Q_by_dataset[name] @ weights)
        if denominator > denominator_tolerance:
            value += dataset_weights[name] * float(weights @ B_by_dataset[name] @ weights) / denominator
    return value


def evaluate(weights, X, y, dataset_weights, denominator_tolerance=1e-14):
    return merit(weights, prepare(X, y, dataset_weights), denominator_tolerance)


def l1(weights, strength):
    return strength * np.abs(weights).sum()


def l2(weights, strength):
    return 0.5 * strength * float(weights @ weights)


def target(weights, prepared, l1_strength, l2_strength, denominator_tolerance):
    return merit(weights, prepared, denominator_tolerance) - l1(weights, l1_strength) - l2(weights, l2_strength)


def gradient(weights, prepared, l1_strength, l2_strength, denominator_tolerance):
    B_by_dataset, Q_by_dataset, _, _, dataset_weights = prepared
    derivative = -l1_strength * np.sign(weights) - l2_strength * weights
    for name in B_by_dataset:
        B_w = B_by_dataset[name] @ weights
        Q_w = Q_by_dataset[name] @ weights
        numerator = float(weights @ B_w)
        denominator = float(weights @ Q_w)
        if denominator > denominator_tolerance:
            derivative += dataset_weights[name] * 2 * (B_w * denominator - Q_w * numerator) / denominator**2
    return derivative


def main(X, y, *, solver, dataset_weights, project=None, l1_strength=0.0, l2_strength=0.0, learning_rate=0.01, max_iter=5000, tol=1e-9, singular_tolerance=1e-10, denominator_tolerance=1e-14, zero_tolerance=1e-10):
    """Initialize analytically and delegate refinement to the chosen solver.
    X: dict[str, np.ndarray] of length D; each value is patient-by-basis with shape (N, B).
    y: dict[str, np.ndarray] of length D; each value is an outcome vector with shape (N,).
    solver: callable; gradient-ascent solver receiving the prepared target and gradient.
    dataset_weights: dict[str, float] of length D; values weight each dataset's Rayleigh merit.
    project: callable | None; optional coefficient projection applied after each update.
    l1_strength: float; coefficient of the L1 penalty.
    l2_strength: float; coefficient of the half-squared-L2 penalty.
    learning_rate: float; solver step size.
    max_iter: int; maximum solver iterations.
    tol: float; solver convergence tolerance.
    singular_tolerance: float; relative eigenvalue cutoff when whitening the pooled covariance.
    denominator_tolerance: float; minimum valid dataset-specific Rayleigh denominator.
    zero_tolerance: float; absolute cutoff used to set final coefficients to zero.
    """
    prepared = prepare(X, y, dataset_weights)
    _, _, B, G, _ = prepared
    initial, eigenvalue, supported_rank = initialize(B, G, singular_tolerance)
    initial = initial if project is None else project(initial)
    objective = partial(target, prepared=prepared, l1_strength=l1_strength, l2_strength=l2_strength, denominator_tolerance=denominator_tolerance)
    derivative = partial(gradient, prepared=prepared, l1_strength=l1_strength, l2_strength=l2_strength, denominator_tolerance=denominator_tolerance)
    result = solver(objective, initial, gradient=derivative, project=project, learning_rate=learning_rate, max_iter=max_iter, tol=tol)
    weights = result["weights"]
    weights[np.abs(weights) <= zero_tolerance] = 0
    weights = weights if project is None else project(weights)
    if weights[np.argmax(np.abs(weights))] < 0:
        weights *= -1
    result.update({"weights": weights, "objective": float(objective(weights)), "merit": merit(weights, prepared, denominator_tolerance), "null_merit": 0.0, "initial_weights": initial, "leading_generalized_eigenvalue": eigenvalue, "pooled_g_rank": supported_rank, "B": B, "G": G})
    return result
