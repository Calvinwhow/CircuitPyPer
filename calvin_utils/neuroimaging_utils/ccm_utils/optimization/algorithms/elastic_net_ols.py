"""Centered ordinary least squares with calibrated elastic-net penalties."""

import numpy as np
from functools import partial
from sklearn.metrics import r2_score


def center(values, axis=0):
    values = np.asarray(values, dtype=float)
    return values - values.mean(axis=axis, keepdims=True)


def prepare(X, y):
    names = tuple(X)
    centered_X = [center(X[name]) for name in names]
    centered_y = [center(y[name]).reshape(-1) for name in names]
    return names, centered_X, centered_y, np.vstack(centered_X), np.concatenate(centered_y)


def merit(beta, prepared):
    residual = prepared[3] @ beta - prepared[4]
    return -float(residual @ residual)


def evaluate(beta, X, y, dataset_weights):
    return merit(beta, prepare(X, y))


def l1(beta, strength):
    return strength * np.abs(beta).sum()


def l2(beta, strength):
    return 0.5 * strength * float(beta @ beta)


def target(beta, prepared, l1_strength, l2_strength):
    return merit(beta, prepared) - l1(beta, l1_strength) - l2(beta, l2_strength)


def gradient(beta, prepared, l2_strength):
    return -2 * prepared[3].T @ (prepared[3] @ beta - prepared[4]) - l2_strength * beta


def proximal(beta, step, l1_strength):
    return np.sign(beta) * np.maximum(np.abs(beta) - step * l1_strength, 0.0)


def main(X, y, *, solver, project=None, l1_strength=0.0, l2_strength=0.0, learning_rate=1.0, max_iter=5000, tol=1e-8):
    """Fit elastic-net OLS to the vertically stacked datasets.
    X: dict[str, np.ndarray] of length D; each value is patient-by-basis with shape (N, B).
    y: dict[str, np.ndarray] of length D; each value is an outcome vector with shape (N,).
    solver: callable; proximal-gradient solver receiving the prepared objective functions.
    project: callable | None; optional coefficient projection applied after each update.
    l1_strength: float; coefficient of the L1 penalty.
    l2_strength: float; coefficient of the half-squared-L2 penalty.
    learning_rate: float; initial proximal-gradient step size.
    max_iter: int; maximum solver iterations.
    tol: float; coefficient-change convergence tolerance.
    """
    prepared = prepare(X, y)
    names, centered_X, centered_y, stacked_X, stacked_y = prepared
    initial = np.linalg.lstsq(stacked_X, stacked_y, rcond=None)[0]
    objective = partial(target, prepared=prepared, l1_strength=l1_strength, l2_strength=l2_strength)
    smooth = lambda beta: merit(beta, prepared) - l2(beta, l2_strength)
    derivative = partial(gradient, prepared=prepared, l2_strength=l2_strength)
    threshold = partial(proximal, l1_strength=l1_strength)
    result = solver(objective, smooth, initial, gradient=derivative, proximal=threshold, project=project, learning_rate=learning_rate, max_iter=max_iter, tol=tol)
    beta = result["weights"]
    prediction = stacked_X @ beta
    result.update({"merit": merit(beta, prepared), "null_merit": merit(np.zeros_like(beta), prepared), "r2": float(r2_score(stacked_y, prediction)), "dataset_r2": {name: float(r2_score(target_y, design @ beta)) for name, design, target_y in zip(names, centered_X, centered_y)}})
    return result
