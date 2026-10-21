"""Proximal-gradient maximization with scalar backtracking steps."""

import numpy as np


def main(target, smooth, initial, *, gradient, proximal, project=None, learning_rate=1.0, max_iter=5000, tol=1e-8):
    """Maximize a composite target with proximal gradient ascent.
    target: callable; complete scalar objective accepting coefficients with shape (B,).
    smooth: callable; differentiable scalar portion of target.
    initial: np.ndarray with shape (B,); starting coefficients.
    gradient: callable; gradient of smooth with shape (B,).
    proximal: callable; nonsmooth update receiving coefficients and step size.
    project: callable | None; optional constraint projection applied after each update.
    learning_rate: float; initial backtracking step size.
    max_iter: int; maximum optimization iterations.
    tol: float; relative coefficient-change convergence tolerance.
    """
    weights = np.asarray(initial, dtype=float).copy()
    weights = weights if project is None else project(weights)
    best_weights, best_value = weights.copy(), float(target(weights))
    history = []
    for iteration in range(1, max_iter + 1):
        derivative = np.asarray(gradient(weights), dtype=float)
        step = learning_rate
        while True:
            candidate = proximal(weights + step * derivative, step)
            candidate = candidate if project is None else project(candidate)
            change = candidate - weights
            if smooth(candidate) >= smooth(weights) + derivative @ change - change @ change / (2 * step) or step <= 1e-14:
                break
            step *= 0.5
        value = float(target(candidate))
        history.append((weights.copy(), float(target(weights))))
        if value > best_value:
            best_weights, best_value = candidate.copy(), value
        if np.linalg.norm(change) <= tol * max(1.0, np.linalg.norm(weights)):
            return {"weights": best_weights, "objective": best_value, "converged": True, "n_iterations": iteration, "history": history}
        weights = candidate
    return {"weights": best_weights, "objective": best_value, "converged": False, "n_iterations": max_iter, "history": history}
