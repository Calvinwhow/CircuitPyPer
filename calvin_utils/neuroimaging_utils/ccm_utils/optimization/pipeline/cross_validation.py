"""Cross-validate every model on the same splits, or fit every model once on every fitting patient.

"lodo" fits D-1 datasets and predicts the omitted one, once per dataset. "loocv" or an integer k holds the same share of
every dataset out per fold, fits all remaining patients jointly, and predicts each dataset's held-out patients; each
dataset's rho is then the mean of its fold rhos. Every fold scores its patients against the one basis library built before
validation and refits only the weights. Without a scheme, every model is fitted once on every patient; only that fit has a
voxel map, so only it is tested on external datasets and passed to basis-map regression.
"""

import numpy as np
from .prediction import correlation, score


def make_folds(ids, folds, seed):
    shuffled = np.random.default_rng(seed).permutation(np.asarray(ids).astype(str))
    if str(folds).lower() == "loocv":
        return [np.asarray([patient_id]) for patient_id in shuffled]
    if not isinstance(folds, (int, np.integer)) or not 2 <= folds <= len(shuffled):
        raise ValueError("Cross-validation must be 'lodo', 'loocv', or an integer from 2 to the smallest dataset size.")
    return [part for part in np.array_split(shuffled, folds) if len(part)]


def patient_folds(datasets, folds, seed):
    """One {dataset: held-out IDs} per fold; fold j holds out part j of every dataset."""
    parts = {name: make_folds(data["ids"], folds, seed + index) for index, (name, data) in enumerate(datasets.items())}
    return [{name: values[fold] for name, values in parts.items() if fold < len(values)} for fold in range(max(len(values) for values in parts.values()))]


def dataset_folds(datasets):
    """One {dataset: every ID} per dataset, for leave-one-dataset-out."""
    if len(datasets) < 2:
        raise ValueError("Leave-one-dataset-out validation requires at least two fitting datasets.")
    return [{name: np.asarray(data["ids"]).astype(str)} for name, data in datasets.items()]


def split(datasets, held):
    """Training and held-out rows of every dataset; a dataset trains only if at least three patients with varying outcomes remain."""
    training, evaluated = {}, {}
    for name, data in datasets.items():
        rows = np.isin(np.asarray(data["ids"]).astype(str), np.asarray(held.get(name, ()), dtype=str))
        if (~rows).sum() >= 3 and np.unique(np.asarray(data["y"])[~rows]).size > 1:
            training[name] = {key: np.asarray(value)[~rows] for key, value in data.items()}
        if rows.any():
            evaluated[name] = {key: np.asarray(value)[rows] for key, value in data.items()}
    return training, evaluated


def held_out_predictions(fitted, data):
    """An algorithm predicts damage @ weights; the network scores the patients' voxel maps against its map."""
    return data["X"] @ fitted["weights"] if np.ndim(fitted["weights"]) == 1 else score(data["patients"], fitted["symptom_map"])


def held_out_records(fitted, datasets, fold):
    return [{"dataset": name, "patient_id": patient_id, "fold": fold, "observed": float(observed), "prediction": float(value)} for name, data in datasets.items() for patient_id, observed, value in zip(data["ids"], data["y"], held_out_predictions(fitted, data))]

def weight_records(fitted, fold):
    return [{"fold": fold, "map": name, "weight": float(weight)} for name, weight in zip(fitted["map_names"], fitted["weights"])] if np.ndim(fitted["weights"]) == 1 else []


def rho(records):
    return correlation([record["prediction"] for record in records], [record["observed"] for record in records])


def summary(records, datasets, by_fold=False):
    """Per-dataset held-out performance.
    by_fold: patient folds report spearman_rho as the mean of each fold's rho (folds with at least three held-out patients), alongside the rho of the pooled out-of-fold predictions; LOOCV folds hold one patient, so their spearman_rho is the pooled rho.
    """
    result = {}
    for name in datasets:
        selected = [record for record in records if record["dataset"] == name]
        pooled = rho(selected)
        row = {"n_patients": len(selected), **pooled}
        if by_fold:
            folds = sorted({record["fold"] for record in selected})
            values = [rho([record for record in selected if record["fold"] == fold])["spearman_rho"] for fold in folds if sum(record["fold"] == fold for record in selected) >= 3]
            values = [value for value in values if np.isfinite(value)]
            row = {**row, "spearman_rho": float(np.mean(values)) if values else pooled["spearman_rho"], "rho_method": "mean_of_folds" if values else "pooled", "n_folds": len(folds), "fold_rhos": values, "pooled_spearman_rho": pooled["spearman_rho"]}
        result[name] = row
    return result


def validate(damage, scheme, fit_models, seed):
    """Fit every model on each fold's training rows and predict its held-out rows."""
    whole = str(scheme).lower() == "lodo"
    folds = dataset_folds(damage["datasets"]) if whole else patient_folds(damage["datasets"], scheme, seed)
    records = {}
    for fold, held in enumerate(folds, start=1):
        print(f"Fold {fold}/{len(folds)}", flush=True)
        training, evaluated = split(damage["datasets"], held)
        for model, fitted in fit_models({**damage, "datasets": training, "test_datasets": evaluated}, seed + fold, False).items():
            values = records.setdefault(model, {"predictions": [], "weights": []})
            values["predictions"] += held_out_records(fitted, evaluated, fold)
            values["weights"] += weight_records(fitted, fold)
    return {model: {"method": model, "symptom_map": None, "cross_validation": {"scheme": str(scheme), **values, "summary": summary(values["predictions"], damage["datasets"], by_fold=not whole)}} for model, values in records.items()}


def main(damage, scheme, fit_models, seed):
    """Cross-validate every model with the scheme and nothing else, or, with no scheme, fit every model once on every fitting patient.
    fit_models(damage, seed, final) -> {model: dict with weights, and symptom_map when final}.
    """
    if scheme is not None:
        return validate(damage, scheme, fit_models, seed)
    print("Fitting every patient", flush=True)
    return {model: {**result, "cross_validation": None} for model, result in fit_models(damage, seed, True).items()}
