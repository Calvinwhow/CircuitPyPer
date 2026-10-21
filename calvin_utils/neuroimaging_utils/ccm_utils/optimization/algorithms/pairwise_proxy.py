"""Sign-invariant pairwise softplus objective."""

import numpy as np
from functools import partial
from scipy.special import expit
from scipy.stats import rankdata


def pairwise_differences(X, y):
    ranks = rankdata(y, method="average")
    left, right = np.triu_indices(len(ranks), k=1)
    delta_ranks = ranks[left] - ranks[right]
    keep = delta_ranks != 0
    return np.sign(delta_ranks[keep])[:, None] * (X[left[keep]] - X[right[keep]])


def prepare(X, y):
    return {name: pairwise_differences(X[name], y[name]) for name in X}


def merit(beta, differences, dataset_weights, signs):
    loss = 0.0
    for name, delta_X in differences.items():
        margin = signs[name] * delta_X @ beta
        loss += dataset_weights[name] * np.logaddexp(0.0, -margin).mean()
    return -float(loss)


def l1(beta, strength):
    return strength * np.abs(beta).sum()


def l2(beta, strength):
    return 0.5 * strength * float(beta @ beta)


def target(beta, differences, dataset_weights, signs, l1_strength, l2_strength):
    return merit(beta, differences, dataset_weights, signs) - l1(beta, l1_strength) - l2(beta, l2_strength)


def smooth(beta, differences, dataset_weights, signs, l2_strength):
    return merit(beta, differences, dataset_weights, signs) - l2(beta, l2_strength)


def gradient(beta, differences, dataset_weights, signs, l2_strength):
    derivative = -l2_strength * beta
    for name, delta_X in differences.items():
        margin = signs[name] * delta_X @ beta
        derivative += dataset_weights[name] * signs[name] * delta_X.T @ expit(-margin) / len(delta_X)
    return derivative


def proximal(beta, step, l1_strength):
    return np.sign(beta) * np.maximum(np.abs(beta) - step * l1_strength, 0.0)


def orient(beta, differences):
    return {name: 1.0 if np.sum(delta_X @ beta) >= 0 else -1.0 for name, delta_X in differences.items()}


def evaluate(beta, X, y, dataset_weights):
    differences = prepare(X, y)
    return merit(beta, differences, dataset_weights, orient(beta, differences))


def fit(beta, differences, dataset_weights, signs, solver, project, l1_strength, l2_strength, learning_rate, max_iter, tol):
    objective = partial(target, differences=differences, dataset_weights=dataset_weights, signs=signs, l1_strength=l1_strength, l2_strength=l2_strength)
    smooth_objective = partial(smooth, differences=differences, dataset_weights=dataset_weights, signs=signs, l2_strength=l2_strength)
    derivative = partial(gradient, differences=differences, dataset_weights=dataset_weights, signs=signs, l2_strength=l2_strength)
    threshold = partial(proximal, l1_strength=l1_strength)
    return solver(objective, smooth_objective, beta, gradient=derivative, proximal=threshold, project=project, learning_rate=learning_rate, max_iter=max_iter, tol=tol)


def refine(beta, differences, dataset_weights, solver, project, l1_strength, l2_strength, learning_rate, max_iter, tol, max_orientations):
    signs = orient(beta, differences)
    for iteration in range(1, max_orientations + 1):
        result = fit(beta, differences, dataset_weights, signs, solver, project, l1_strength, l2_strength, learning_rate, max_iter, tol)
        updated = orient(result["weights"], differences)
        if updated == signs:
            return result, signs, iteration
        beta, signs = result["weights"], updated
    result = fit(beta, differences, dataset_weights, signs, solver, project, l1_strength, l2_strength, learning_rate, max_iter, tol)
    return result, signs, max_orientations + 1


def main(X, y, *, solver, dataset_weights, project=None, l1_strength=0.0, l2_strength=0.0, learning_rate=1.0, max_iter=5000, tol=1e-8, max_orientations=20):
    """Alternate one joint coefficient fit with analytical dataset orientations.
    X: dict[str, np.ndarray] of length D; each value is patient-by-basis with shape (N, B).
    y: dict[str, np.ndarray] of length D; each value is an outcome vector with shape (N,).
    solver: callable; proximal-gradient solver receiving the prepared objective functions.
    dataset_weights: dict[str, float] of length D; values weight each dataset's mean pairwise loss.
    project: callable | None; optional coefficient projection applied after each update.
    l1_strength: float; coefficient of the L1 penalty.
    l2_strength: float; coefficient of the half-squared-L2 penalty.
    learning_rate: float; initial proximal-gradient step size.
    max_iter: int; maximum solver iterations per orientation.
    tol: float; coefficient-change convergence tolerance.
    max_orientations: int; maximum analytical sign updates.
    """
    differences = prepare(X, y)
    n_components = next(iter(X.values())).shape[1]
    result, signs, orientations = refine(np.ones(n_components), differences, dataset_weights, solver, project, l1_strength, l2_strength, learning_rate, max_iter, tol, max_orientations)
    weights = result["weights"]
    if weights[np.argmax(np.abs(weights))] < 0:
        weights, signs = -weights, {name: -sign for name, sign in signs.items()}
    result.update({"weights": weights, "objective": target(weights, differences, dataset_weights, signs, l1_strength, l2_strength), "merit": merit(weights, differences, dataset_weights, signs), "null_merit": -float(np.log(2)), "dataset_signs": signs, "orientation_iterations": orientations})
    return result
