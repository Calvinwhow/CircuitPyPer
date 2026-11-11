"""Cross-validation of typed model-fitting results."""

from dataclasses import replace

import numpy as np

from .model_fitter_utils.model_fitter_schema import AlgorithmResult, ModelEvaluation, ModelResult, ModelResults, PredictionRecord, WeightRecord
from .prediction import correlation, score


def make_folds(ids, folds, seed):
    shuffled = np.random.default_rng(seed).permutation(np.asarray(ids).astype(str))
    if str(folds).lower() == "loocv":
        return [np.asarray([patient_id]) for patient_id in shuffled]
    if not isinstance(folds, (int, np.integer)) or not 2 <= folds <= len(shuffled):
        raise ValueError("Cross-validation must be 'lodo', 'loocv', or an integer from 2 to the smallest dataset size.")
    return [part for part in np.array_split(shuffled, folds) if len(part)]


def patient_folds(datasets, folds, seed):
    """Return one held-out patient-ID mapping per fold."""
    parts = {data.dataset_name: make_folds(data.patient_ids, folds, seed + index) for index, data in enumerate(datasets)}
    return [{name: values[fold] for name, values in parts.items() if fold < len(values)} for fold in range(max(len(values) for values in parts.values()))]


def dataset_folds(datasets):
    """Return one whole-dataset holdout per fitting dataset."""
    if len(datasets) < 2:
        raise ValueError("Leave-one-dataset-out validation requires at least two fitting datasets.")
    return [{data.dataset_name: data.patient_ids} for data in datasets]


def split(datasets, held):
    """Return valid training and held-out typed dataset views."""
    training, evaluated = [], []
    for data in datasets:
        rows = np.isin(data.patient_ids, np.asarray(held.get(data.dataset_name, ()), dtype=str))
        if (~rows).sum() >= 3 and np.unique(data.outcome_values[~rows]).size > 1:
            training.append(data.subset(~rows))
        if rows.any():
            evaluated.append(data.subset(rows))
    return tuple(training), tuple(evaluated)


def held_out_predictions(fitted, data):
    """Return predictions using coefficients for algorithms and voxel maps for neural results."""
    if isinstance(fitted, AlgorithmResult):
        return data.basis_damage_scores @ fitted.weights
    return score(data.patient_maps, fitted.symptom_map)


def held_out_records(fitted, datasets, fold):
    """Return typed held-out prediction records for one fold."""
    return tuple(PredictionRecord(data.dataset_name, str(patient_id), float(observed), float(value), fold) for data in datasets for patient_id, observed, value in zip(data.patient_ids, data.outcome_values, held_out_predictions(fitted, data)))


def weight_records(fitted, fold):
    """Return algorithm coefficients for one fold; neural integration has no single coefficient vector."""
    if not isinstance(fitted, AlgorithmResult):
        return ()
    return tuple(WeightRecord(fold, name, float(weight)) for name, weight in zip(fitted.damage.map_names, fitted.weights))


def summary(records, datasets, by_fold=False):
    """Return typed per-dataset held-out performance."""
    result = {}
    for name in (data.dataset_name for data in datasets):
        selected = tuple(record for record in records if record.dataset == name)
        pooled = correlation([record.prediction for record in selected], [record.observed for record in selected])
        if by_fold:
            folds = sorted({record.fold for record in selected})
            fold_performance = [correlation([record.prediction for record in selected if record.fold == fold], [record.observed for record in selected if record.fold == fold]) for fold in folds if sum(record.fold == fold for record in selected) >= 3]
            fold_rhos = tuple(value.spearman_rho for value in fold_performance if np.isfinite(value.spearman_rho))
            rho_value, rho_method = _fold_correlation(fold_rhos, pooled.spearman_rho)
            pooled = replace(pooled, spearman_rho=rho_value, rho_method=rho_method, n_folds=len(folds), fold_rhos=fold_rhos, pooled_spearman_rho=pooled.spearman_rho)
        result[name] = pooled
    return result


def _fold_correlation(fold_values, pooled_rho):
    """Resolve mean fold correlation, falling back to pooled correlation for tiny folds."""
    if fold_values:
        return float(np.mean(fold_values)), "mean_of_folds"
    return pooled_rho, "pooled"


def validate(damage, scheme, model_fitter, seed):
    """Fit every model on each fold and return one typed validation result per method."""
    datasets = damage.fitting_datasets
    leave_one_dataset_out = str(scheme).lower() == "lodo"
    folds = _validation_folds(datasets, scheme, seed, leave_one_dataset_out)
    records = {}
    for fold, held in enumerate(folds, start=1):
        print(f"Fold {fold}/{len(folds)}", flush=True)
        training, evaluated = split(datasets, held)
        for fitted in model_fitter.fit(seed + fold, False, training).models:
            method = fitted.method
            predictions, weights = records.setdefault(method, ([], []))
            predictions.extend(held_out_records(fitted, evaluated, fold))
            weights.extend(weight_records(fitted, fold))
    results = []
    for method, (predictions, weights) in records.items():
        evaluation = ModelEvaluation(tuple(predictions), summary(predictions, datasets, by_fold=not leave_one_dataset_out), tuple(weights), str(scheme))
        results.append(ModelResult(method=method, damage=damage, symptom_map=None, cross_validation=evaluation))
    return ModelResults(tuple(results))


def _validation_folds(datasets, scheme, seed, leave_one_dataset_out):
    """Resolve whole-dataset or within-dataset validation folds."""
    if leave_one_dataset_out:
        return dataset_folds(datasets)
    return patient_folds(datasets, scheme, seed)


def main(damage, scheme, model_fitter, seed):
    """Cross-validate all models or fit each model once on every fitting patient."""
    if scheme is not None:
        return validate(damage, scheme, model_fitter, seed)
    print("Fitting every patient", flush=True)
    return model_fitter.fit(seed, True)
