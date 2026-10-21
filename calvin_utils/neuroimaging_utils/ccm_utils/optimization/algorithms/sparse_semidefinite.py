"""Sparse semidefinite objective for sign-invariant map weights."""

import numpy as np
from scipy.stats import rankdata
from ..projections import prediction_variance


def clinical_rank(outcomes):
    ranks = rankdata(outcomes, method="average").astype(float)
    ranks -= ranks.mean()
    return ranks / np.linalg.norm(ranks)


def prepare(X, y, dataset_weights, ridge):
    n_components = next(iter(X.values())).shape[1]
    B = np.zeros((n_components, n_components))
    for name in X:
        ranks = clinical_rank(y[name])
        association = X[name].T @ ranks
        B += dataset_weights[name] * np.outer(association, association)
    G = prediction_variance.gram(X, dataset_weights)
    return B, G + ridge * np.eye(n_components)


def merit(theta, B, trace):
    return trace(B @ theta)


def evaluate(weights, X, y, dataset_weights, ridge=1e-6):
    B, _ = prepare(X, y, dataset_weights, ridge)
    return float(merit(np.outer(weights, weights), B, np.trace))


def l1(theta, strength, norm):
    return strength * norm(theta)


def l2(theta, strength, norm):
    return 0.5 * strength * norm(theta)


def target(theta, B, l1_strength, l2_strength, trace, l1_norm, l2_norm):
    return merit(theta, B, trace) - l1(theta, l1_strength, l1_norm) - l2(theta, l2_strength, l2_norm)


def project_constraint(theta, G, trace):
    return [trace(G @ theta) == 1, theta >> 0]


def recover(theta, G, rank_tolerance, spectral_mass_threshold):
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (theta + theta.T))
    positive = np.clip(eigenvalues, 0, None)
    leading = float(positive[-1])
    weights = np.sqrt(leading) * eigenvectors[:, -1]
    weights /= np.sqrt(weights @ G @ weights)
    if weights[np.argmax(np.abs(weights))] < 0:
        weights *= -1
    spectral_mass = leading / positive.sum()
    return {"weights": weights, "eigenvalues": eigenvalues, "leading_eigenvalue": leading, "positive_eigenvalue_sum": float(positive.sum()), "leading_eigenvalue_fraction": float(spectral_mass), "numerical_rank": int(np.sum(positive > rank_tolerance * leading)), "leading_vector_recovery_acceptable": bool(spectral_mass >= spectral_mass_threshold)}


def main(X, y, *, solver, dataset_weights, l1_strength=0.0, l2_strength=0.0, ridge=1e-6, conic_solver="SCS", max_iter=20_000, tolerance=1e-5, rank_tolerance=1e-5, spectral_mass_threshold=0.95):
    """Prepare the lifted objective and delegate its numerical solution.
    X: dict[str, np.ndarray] of length D; each value is patient-by-basis with shape (N, B).
    y: dict[str, np.ndarray] of length D; each value is an outcome vector with shape (N,).
    solver: callable; semidefinite solver receiving the lifted target and constraints.
    dataset_weights: dict[str, float] of length D; values weight each dataset's lifted association.
    l1_strength: float; coefficient of the matrix L1 penalty.
    l2_strength: float; coefficient of the half-squared-Frobenius penalty.
    ridge: float; diagonal stabilization added to pooled prediction covariance.
    conic_solver: str; CVXPY solver name.
    max_iter: int; maximum conic-solver iterations.
    tolerance: float; conic-solver numerical tolerance.
    rank_tolerance: float; relative eigenvalue cutoff for numerical rank.
    spectral_mass_threshold: float; minimum leading-eigenvalue fraction accepted for vector recovery.
    """
    B, G = prepare(X, y, dataset_weights, ridge)
    objective = lambda theta, trace, l1_norm, l2_norm: target(theta, B, l1_strength, l2_strength, trace, l1_norm, l2_norm)
    constraint = lambda theta, trace: project_constraint(theta, G, trace)
    solved = solver(objective, constraint, B.shape, solver_name=conic_solver, max_iter=max_iter, tolerance=tolerance)
    theta = solved.pop("parameters")
    recovered = recover(theta, G, rank_tolerance, spectral_mass_threshold)
    recovered.update(solved)
    recovered.update({"theta": theta, "B": B, "G": G, "merit": float(np.trace(B @ theta)), "null_merit": 0.0, "spectral_mass_threshold": spectral_mass_threshold, "recovery_recommendation": "Leading-vector recovery passed the spectral-mass threshold." if recovered["leading_vector_recovery_acceptable"] else f"Spectral mass is below {100 * spectral_mass_threshold:g}%; use a different optimization algorithm."})
    return recovered
