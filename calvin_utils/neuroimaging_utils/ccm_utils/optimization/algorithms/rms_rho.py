"""Sign-invariant RMS Spearman objective."""

import numpy as np
from functools import partial
from scipy.stats import rankdata
from ..projections import prediction_variance


def spearman(prediction, outcome):
    left = rankdata(prediction, method="average")
    right = rankdata(outcome, method="average")
    left -= left.mean()
    right -= right.mean()
    return float(left @ right / np.sqrt((left @ left) * (right @ right)))


def prepare(X, y):
    return X, y


def merit(weights, prepared):
    X, y = prepared
    correlations = [spearman(X[name] @ weights, y[name]) for name in X]
    return float(np.sqrt(np.mean(np.square(correlations))))


def evaluate(weights, X, y, dataset_weights):
    return merit(weights, prepare(X, y))


def l1(weights, strength):
    return strength * np.abs(weights).sum()


def l2(weights, strength):
    return 0.5 * strength * float(weights @ weights)


def target(weights, prepared, l1_strength, l2_strength):
    return merit(weights, prepared) - l1(weights, l1_strength) - l2(weights, l2_strength)


def project_constraint(weights, covariance):
    return prediction_variance.project(weights, covariance)


def starts(n_components, initial, n_starts, random_state, mean, std):
    sequence = np.random.SeedSequence(random_state)
    children = sequence.spawn(n_starts - 1)
    values = [("configured", random_state, np.asarray(initial, dtype=float))]
    for child in children:
        seed = int(child.generate_state(1, dtype=np.uint32)[0])
        values.append(("wide_gaussian", seed, np.random.default_rng(child).normal(mean, std, n_components)))
    return values


def main(X, y, *, solver, dataset_weights, initial_weights=None, n_starts=12, gaussian_mean=0.1, gaussian_std=5.0, random_state=None, learning_rate=0.1, max_iter=500, tol=1e-6, l1_strength=0.0, l2_strength=0.0):
    """Prepare RMS rho and delegate each broad start to one solver.
    X: dict[str, np.ndarray] of length D; each value is patient-by-basis with shape (N, B).
    y: dict[str, np.ndarray] of length D; each value is an outcome vector with shape (N,).
    solver: callable; gradient-ascent solver receiving the RMS-rho target.
    dataset_weights: dict[str, float] of length D; values weight the pooled prediction variance.
    initial_weights: np.ndarray | None with shape (B,); configured first starting coefficients.
    n_starts: int; number of broad optimization starts, at least ten.
    gaussian_mean: float; mean of randomized starting coefficients.
    gaussian_std: float; standard deviation of randomized starting coefficients.
    random_state: int | None; seed for randomized starts.
    learning_rate: float; solver step size.
    max_iter: int; maximum iterations per start.
    tol: float; solver convergence tolerance.
    l1_strength: float; coefficient of the L1 penalty.
    l2_strength: float; coefficient of the half-squared-L2 penalty.
    """
    n_components = next(iter(X.values())).shape[1]
    if n_starts < 10:
        raise ValueError("RMS-rho multistart requires n_starts >= 10.")
    initial_weights = np.ones(n_components) if initial_weights is None else initial_weights
    prepared = prepare(X, y)
    covariance = prediction_variance.gram(X, dataset_weights)
    project = partial(project_constraint, covariance=covariance)
    objective = partial(target, prepared=prepared, l1_strength=l1_strength, l2_strength=l2_strength)
    fitted = []
    records = []
    for index, (kind, seed, initial) in enumerate(starts(n_components, initial_weights, n_starts, random_state, gaussian_mean, gaussian_std), start=1):
        result = solver(objective, initial, project=project, learning_rate=learning_rate, max_iter=max_iter, tol=tol, finite_step=max(1e-5, 0.5 / n_components), gradient_clip=0.5)
        fitted.append(result)
        records.append({"start": index, "initialization": kind, "seed": seed, "training_rms_rho": merit(result["weights"], prepared), "iterations": result["n_iterations"]})
    best = int(np.argmax([record["training_rms_rho"] for record in records]))
    for index, record in enumerate(records):
        record["selected"] = index == best
    return {"weights": fitted[best]["weights"], "objective": records[best]["training_rms_rho"], "merit": records[best]["training_rms_rho"], "null_merit": 0.0, "best_start": best + 1, "starts": records, "start_weights": np.stack([result["weights"] for result in fitted]), "history": fitted[best]["history"], "converged": fitted[best]["converged"], "n_iterations": fitted[best]["n_iterations"], "G": covariance}
