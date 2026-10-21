"""ADAM maximization of an arbitrary scalar target."""

import numpy as np


class AdamOptimizer:
    """Stateful ADAM update written for gradient ascent."""

    def __init__(self, weights, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.weights = np.asarray(weights, dtype=float).copy()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = np.zeros(self.weights.shape)
        self.v = np.zeros(self.weights.shape)
        self.t = 0

    def step(self, gradient):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * gradient**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        self.weights += self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return self.weights


def finite_difference(target, weights, step):
    base = target(weights)
    candidates = np.repeat(weights[None, :], len(weights), axis=0)
    candidates[np.arange(len(weights)), np.arange(len(weights))] += step
    return np.asarray([(target(candidate) - base) / step for candidate in candidates])


def main(target, initial, *, gradient=None, proximal=None, project=None, learning_rate=0.01, max_iter=500, tol=1e-6, finite_step=1e-5, gradient_clip=None):
    """Maximize a scalar target with ADAM.
    target: callable; scalar objective accepting one coefficient vector with shape (B,).
    initial: np.ndarray with shape (B,); starting coefficients.
    gradient: callable | None; analytical gradient or finite differences when omitted.
    proximal: callable | None; optional proximal update receiving coefficients and step size.
    project: callable | None; optional constraint projection applied after each update.
    learning_rate: float; ADAM step size.
    max_iter: int; maximum optimization iterations.
    tol: float; gradient, coefficient, or objective-change convergence tolerance.
    finite_step: float; forward-difference step when gradient is omitted.
    gradient_clip: float | None; optional elementwise gradient magnitude limit.
    """
    optimizer = AdamOptimizer(initial, lr=learning_rate)
    if project is not None:
        optimizer.weights = project(optimizer.weights)
    best_weights = optimizer.weights.copy()
    best_value = float(target(best_weights))
    history = []
    previous = None
    for iteration in range(1, max_iter + 1):
        weights = optimizer.weights.copy()
        value = float(target(weights))
        derivative = finite_difference(target, weights, finite_step) if gradient is None else np.asarray(gradient(weights), dtype=float)
        if gradient_clip is not None:
            derivative = np.clip(derivative, -gradient_clip, gradient_clip)
        updated = optimizer.step(derivative)
        if proximal is not None:
            optimizer.weights = proximal(updated, learning_rate)
        if project is not None:
            optimizer.weights = project(optimizer.weights)
        history.append((weights, value))
        if value > best_value:
            best_weights, best_value = weights.copy(), value
        converged = iteration > 25 and (np.linalg.norm(optimizer.weights - weights) < tol or np.linalg.norm(derivative) < tol or (previous is not None and abs(value - previous) < tol))
        if converged:
            return {"weights": best_weights, "objective": best_value, "converged": True, "n_iterations": iteration, "history": history}
        previous = value
    value = float(target(optimizer.weights))
    if value > best_value:
        best_weights, best_value = optimizer.weights.copy(), value
    return {"weights": best_weights, "objective": best_value, "converged": False, "n_iterations": max_iter, "history": history}
