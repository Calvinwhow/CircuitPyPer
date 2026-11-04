"""Score patients against a fitted map, correlate the scores with outcomes, and rank models."""

import numpy as np
from scipy.stats import spearmanr


def normalize(patients):
    """Every patient map divided by its L2 norm."""
    patients = np.asarray(patients, dtype=np.float32)
    norms = np.linalg.norm(patients, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("Patient maps have zero or invalid L2 norms.")
    return patients / norms[:, None]


def score(patients, symptom_map):
    """patient . map / ||patient||: for an algorithm, exactly damage @ weights."""
    return normalize(patients) @ np.asarray(symptom_map, dtype=np.float32)


def correlation(predicted, observed):
    result = spearmanr(predicted, observed)
    return {"spearman_rho": float(getattr(result, "statistic", result.correlation)), "p_value": float(result.pvalue)}


def fitting_ids(datasets):
    return set(np.concatenate([data["ids"] for data in datasets.values()]).astype(str))


def evaluate(datasets, symptom_map, excluded_ids=frozenset()):
    """Score every dataset's patients with a frozen map; patients without an outcome are skipped and patients in excluded_ids are refused."""
    records, summary = [], {}
    for name, data in datasets.items():
        ids = np.asarray(data["ids"]).astype(str)
        if set(ids) & excluded_ids:
            raise ValueError(f"Test dataset {name!r} shares patients with fitting data.")
        observed = np.asarray(data["y"], dtype=float)
        valid = np.isfinite(observed)
        predicted = score(np.asarray(data["patients"])[valid], symptom_map)
        summary[name] = {"n_patients": int(valid.sum()), "n_missing_outcomes": int((~valid).sum()), **correlation(predicted, observed[valid])}
        records.extend({"dataset": name, "patient_id": patient_id, "observed": float(value), "prediction": float(prediction)} for patient_id, value, prediction in zip(ids[valid], observed[valid], predicted))
    return {"predictions": records, "summary": summary}


def main(results, damage):
    """Add each fitted map's apparent performance on the fitting patients and, when test datasets exist, its test performance; cross-validated models have no map."""
    excluded = fitting_ids(damage["datasets"])
    return {model: {**result, "apparent": None if result["symptom_map"] is None else evaluate(damage["datasets"], result["symptom_map"]), "test": None if result["symptom_map"] is None or damage["test_datasets"] is None else evaluate(damage["test_datasets"], result["symptom_map"], excluded)} for model, result in results.items()}


def rank(results):
    """RMS Spearman rho across datasets for every model, on the test datasets when given, otherwise cross-validation, otherwise the fitting patients; best first."""
    summaries = {model: (result["test"] or result["cross_validation"] or result["apparent"])["summary"] for model, result in results.items()}
    performance = {model: float(np.sqrt(np.mean([values["spearman_rho"] ** 2 for values in summary.values()]))) for model, summary in summaries.items()}
    ranking = sorted(performance, key=performance.get, reverse=True)
    for model in ranking:
        print(f"{model}: {performance[model]:.6f}")
    print(f"Winner: {ranking[0]}")
    return {"winner": ranking[0], "ranking": ranking, "performance": performance, "summaries": summaries}
