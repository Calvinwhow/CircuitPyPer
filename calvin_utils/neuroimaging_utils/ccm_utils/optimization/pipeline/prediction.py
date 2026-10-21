"""Patient-level prediction from patient-by-basis damage."""

import numpy as np
from scipy.stats import spearmanr


def score(damage, weights):
    """Return patient scores Xw without changing the fitted coefficient scale."""
    return np.asarray(damage) @ np.asarray(weights, dtype=float).reshape(-1)


def correlation(predicted, observed):
    result = spearmanr(predicted, observed)
    return {"spearman_rho": float(getattr(result, "statistic", result.correlation)), "p_value": float(result.pvalue)}


def main(datasets, weights):
    """Predict and correlate every fitting dataset.
    datasets: dict[str, dict] of length D; each entry contains X with shape (N, B) and y with shape (N,).
    weights: np.ndarray with shape (B,); fitted basis coefficients.
    """
    predictions = {name: score(data["X"], weights) for name, data in datasets.items()}
    performance = {name: correlation(predictions[name], data["y"]) for name, data in datasets.items()}
    return predictions, performance


def fitting_ids(datasets):
    return set(np.concatenate([data["ids"] for data in datasets.values()]).astype(str))


def test(datasets, weights, excluded_ids):
    """Evaluate frozen coefficients on independent datasets.
    datasets: dict[str, dict] of length D; each entry contains X with shape (N, B), y with shape (N,), and ids with shape (N,).
    weights: np.ndarray with shape (B,); fitted prediction-variance-normalized basis coefficients.
    excluded_ids: set[str]; patient identifiers used to fit the coefficients.
    """
    records = []
    summary = {}
    for name, data in datasets.items():
        outcomes = np.asarray(data["y"], dtype=float).reshape(-1)
        ids = np.asarray(data["ids"]).astype(str).reshape(-1)
        if set(ids) & excluded_ids:
            raise ValueError(f"Test dataset {name!r} shares patients with fitting data.")
        valid = np.isfinite(outcomes)
        predicted = score(data["X"][valid], weights)
        observed = outcomes[valid]
        summary[name] = {"n_patients": int(valid.sum()), "n_missing_outcomes": int((~valid).sum()), **correlation(predicted, observed)}
        records.extend({"dataset": name, "patient_id": patient_id, "observed": float(outcome), "prediction": float(value)} for patient_id, outcome, value in zip(ids[valid], observed, predicted))
    return records, summary
