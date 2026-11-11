"""Typed patient prediction, evaluation, and model ranking."""

import numpy as np
from scipy.stats import spearmanr

from .model_fitter_utils.model_fitter_schema import DatasetPerformance, ModelComparison, ModelEvaluation, PredictionRecord


def normalize(patients):
    """Return patient maps divided by their L2 norms."""
    patients = np.asarray(patients, dtype=np.float32)
    norms = np.linalg.norm(patients, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("Patient maps have zero or invalid L2 norms.")
    return patients / norms[:, None]


def score(patients, symptom_map):
    """Return normalized patient-map projections onto one symptom map."""
    return normalize(patients) @ np.asarray(symptom_map, dtype=np.float32)


def correlation(predicted, observed):
    """Return typed Spearman performance for paired predictions and outcomes."""
    result = spearmanr(predicted, observed)
    rho = float(getattr(result, "statistic", result.correlation))
    return DatasetPerformance(len(predicted), rho, float(result.pvalue))


def fitting_ids(datasets):
    """Return every fitting patient identifier."""
    return set(np.concatenate([data.patient_ids for data in datasets]).astype(str))


def evaluate(datasets, symptom_map, excluded_ids=frozenset()):
    """Return typed predictions and performance for every supplied dataset."""
    records, summaries = [], {}
    for data in datasets:
        ids = data.patient_ids
        if set(ids) & excluded_ids:
            raise ValueError(f"Test dataset {data.dataset_name!r} shares patients with fitting data.")
        observed = data.outcome_values
        valid = np.isfinite(observed)
        predicted = score(data.patient_maps[valid], symptom_map)
        performance = correlation(predicted, observed[valid])
        summaries[data.dataset_name] = DatasetPerformance(performance.n_patients, performance.spearman_rho, performance.p_value, int((~valid).sum()))
        records.extend(PredictionRecord(data.dataset_name, str(patient_id), float(value), float(prediction)) for patient_id, value, prediction in zip(ids[valid], observed[valid], predicted))
    return ModelEvaluation(tuple(records), summaries)


def main(results, damage):
    """Attach apparent and optional external evaluation to each final typed result."""
    excluded = fitting_ids(damage.fitting_datasets)
    for result in results.models:
        result.apparent = _apparent_evaluation(result, damage)
        result.test = _test_evaluation(result, damage, excluded)
    return results


def _apparent_evaluation(result, damage):
    """Resolve apparent evaluation only for a final fitted map."""
    if result.symptom_map is None:
        return None
    return evaluate(damage.fitting_datasets, result.symptom_map)


def _test_evaluation(result, damage, excluded_patient_ids):
    """Resolve external evaluation only when a final map and test datasets exist."""
    if result.symptom_map is None or damage.testing_datasets is None:
        return None
    return evaluate(damage.testing_datasets, result.symptom_map, excluded_patient_ids)


def rank(results):
    """Rank typed model results by RMS Spearman correlation across datasets."""
    summaries = {result.method: _selected_evaluation(result).summary for result in results.models}
    performance = {method: float(np.sqrt(np.mean([values.spearman_rho ** 2 for values in summary.values()]))) for method, summary in summaries.items()}
    ranking = tuple(sorted(performance, key=performance.get, reverse=True))
    for method in ranking:
        print(f"{method}: {performance[method]:.6f}")
    print(f"Winner: {ranking[0]}")
    return ModelComparison(ranking[0], ranking, performance, summaries)


def _selected_evaluation(result):
    """Resolve test, cross-validation, or apparent performance in priority order."""
    if result.test is not None:
        return result.test
    if result.cross_validation is not None:
        return result.cross_validation
    if result.apparent is not None:
        return result.apparent
    raise ValueError(f"Model {result.method!r} has no evaluation to rank.")
