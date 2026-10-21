"""Within-dataset and cross-dataset validation of symptom-basis weights."""

import numpy as np
from .prediction import correlation, fitting_ids, score, test


def make_folds(ids, folds, seed):
    shuffled = np.random.default_rng(seed).permutation(np.asarray(ids).astype(str))
    if str(folds).lower() == "loocv":
        return [np.asarray([patient_id]) for patient_id in shuffled]
    if not isinstance(folds, (int, np.integer)) or not 2 <= folds <= len(shuffled):
        raise ValueError("Within-dataset folds must be 'loocv' or an integer from 2 to N.")
    return [part for part in np.array_split(shuffled, folds) if len(part)]


def summary(records, datasets):
    result = {}
    for name in datasets:
        selected = [record for record in records if record["dataset"] == name]
        result[name] = {"n_patients": len(selected), **correlation([record["prediction"] for record in selected], [record["observed"] for record in selected])}
    return result


def record_predictions(records, data, rows, weights, name, fold):
    predicted = score(data["X"][rows], weights)
    records.extend({"dataset": name, "patient_id": data["ids"][row], "fold": fold, "observed": float(data["y"][row]), "prediction": float(value)} for row, value in zip(rows, predicted))


def record_weights(records, weights, map_names, dataset, fold):
    records.extend({"dataset": dataset, "fold": fold, "map": name, "weight": float(weight)} for name, weight in zip(map_names, weights))


def within_datasets(damage, fit, folds, seed):
    records = []
    weights = []
    for dataset_index, (name, data) in enumerate(damage["datasets"].items()):
        for fold, held_out_ids in enumerate(make_folds(data["ids"], folds, seed + dataset_index), start=1):
            held = np.isin(data["ids"], held_out_ids)
            train = ~held
            fitted = fit({name: data["X"][train]}, {name: data["y"][train]}, damage["map_names"], damage["sample_sizes"], seed + dataset_index + fold)
            record_predictions(records, data, np.flatnonzero(held), fitted["weights"], name, fold)
            record_weights(weights, fitted["weights"], damage["map_names"], name, fold)
    return {"predictions": records, "summary": summary(records, damage["datasets"]), "weights": weights}


def across_datasets(damage, fit, seed):
    datasets = damage["datasets"]
    if len(datasets) < 2:
        raise ValueError("Cross-dataset validation requires at least two fitting datasets.")
    records = []
    summaries = {}
    weights = []
    for fold, (held_out, data) in enumerate(datasets.items(), start=1):
        training = {name: values for name, values in datasets.items() if name != held_out}
        fitted = fit({name: values["X"] for name, values in training.items()}, {name: values["y"] for name, values in training.items()}, damage["map_names"], damage["sample_sizes"], seed + fold)
        predicted, evaluated = test({held_out: data}, fitted["weights"], fitting_ids(training))
        records.extend({**record, "fold": fold} for record in predicted)
        summaries.update(evaluated)
        record_weights(weights, fitted["weights"], damage["map_names"], held_out, fold)
    return {"predictions": records, "summary": summaries, "weights": weights}


def main(damage, fit, *, within_dataset_folds=None, cross_dataset=False, seed=2026):
    """Validate weights against fixed bases, then fit all datasets jointly.
    damage: dict; scored library containing D datasets with X shape (N, B), y shape (N,), and IDs shape (N,).
    fit: callable; coefficient fitter accepting X, y, B map names, sample sizes, and seed.
    within_dataset_folds: int | "loocv" | None; isolated patient folds per dataset or no patient CV.
    cross_dataset: bool; whether to fit D-1 datasets and test the omitted dataset.
    seed: int; random seed for patient folds and fitted models.
    """
    within = None if within_dataset_folds is None else within_datasets(damage, fit, within_dataset_folds, seed)
    across = across_datasets(damage, fit, seed) if cross_dataset else None
    datasets = damage["datasets"]
    fitted = fit({name: data["X"] for name, data in datasets.items()}, {name: data["y"] for name, data in datasets.items()}, damage["map_names"], damage["sample_sizes"], seed)
    return {"fit": fitted, "within_dataset": within, "cross_dataset": across}
