"""Inner weight selection, outer validation, and final regression-map fitting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from calvin_utils.file_utils.import_functions import GiiNiiFileImport
from calvin_utils.permutation_analysis_utils.map_damage_cv import load_native_patient_vectors
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.convergent_map_optimizer import LocalizationOptimizer
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.history import OptimizationHistory
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.inner_cv import prepare_inner_folds
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.fold_assignment import assign_id_folds
from calvin_utils.neuroimaging_utils.output_functions import NeuroimageFileOutporter


class ArrayLoader:
    """Give the optimizer scoring arrays without writing temporary manifests."""

    def __init__(self, datasets):
        self.datasets = datasets
        self.dataset_names_list = tuple(datasets)

    def load_dataset(self, name, mmap_mode=None):
        return self.datasets[name]


FoldArrayLoader = ArrayLoader  # Existing notebook and test imports.


def datasets_from_loader(loader, *, mmap_mode="r"):
    """Load optimizer datasets and their patient IDs from a DataLoader."""
    datasets = {}
    for name in loader.dataset_names_list:
        paths = loader.dataset_paths_dict[name]
        if "ids" not in paths:
            raise ValueError(f"Dataset {name!r} needs an ids array in its JSON.")
        data = loader.load_dataset(name, mmap_mode=mmap_mode)
        X = data["niftis"]
        y = np.asarray(data["indep_var"], dtype=float).reshape(-1)
        ids = np.load(paths["ids"], allow_pickle=False).astype(str).reshape(-1)
        if X.ndim != 2 or y.shape != (len(X),) or ids.shape != (len(X),):
            raise ValueError(f"Dataset {name!r} has misaligned images, outcomes, and IDs.")
        datasets[name] = {"X": X, "y": y, "ids": ids}
    return datasets


def map_inputs_from_loader(loader, *, mmap_mode="r"):
    """Merge map-generating symptom datasets into one patient image matrix."""
    datasets = datasets_from_loader(loader, mmap_mode=mmap_mode)
    if len(datasets) < 2:
        raise ValueError("Map generation needs at least two symptom datasets.")
    ids, vectors, outcomes = [], {}, {}
    n_features = None
    for name, data in datasets.items():
        X = data["X"]
        n_features = X.shape[1] if n_features is None else n_features
        if X.shape[1] != n_features:
            raise ValueError(f"Map dataset {name!r} uses a different feature space.")
        outcomes[name] = dict(zip(data["ids"], data["y"]))
        for patient_id, vector in zip(data["ids"], X):
            if patient_id in vectors:
                if not np.allclose(vectors[patient_id], vector, rtol=0, atol=1e-6):
                    raise ValueError(
                        f"Patient {patient_id!r} has different images across map datasets."
                    )
            else:
                ids.append(patient_id)
                vectors[patient_id] = np.asarray(vector, dtype=np.float32).copy()
    ids = np.asarray(ids, dtype=str)
    X = np.stack([vectors[patient_id] for patient_id in ids])
    y = {
        name: np.asarray([values.get(patient_id, np.nan) for patient_id in ids])
        for name, values in outcomes.items()
    }
    return X, ids, y


def load_patient_table(path, *, subject_col, image_col, map_outcome_cols,
                       scoring_datasets, sheet=None):
    """Read the configured patient table and resolve its native image paths."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Patient table does not exist: {path}")
    df = (pd.read_excel(path, sheet_name=sheet or 0)
          if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path))
    outcome_cols = set(map_outcome_cols)
    for name, spec in scoring_datasets.items():
        if not {"image_col", "outcome_col"} <= spec.keys():
            raise ValueError(f"Scoring dataset {name!r} needs image_col and outcome_col.")
        if spec["image_col"] != image_col:
            raise ValueError(
                f"Scoring dataset {name!r} must use {image_col!r}, the image "
                "column used to build its component maps."
            )
        outcome_cols.add(spec["outcome_col"])
    missing = sorted(({subject_col, image_col} | outcome_cols) - set(df.columns))
    if missing:
        raise ValueError(f"Patient table is missing columns: {missing}")
    if df[subject_col].isna().any() or df[subject_col].duplicated().any():
        raise ValueError(f"{subject_col!r} must contain one nonmissing ID per patient.")
    if df[image_col].isna().any():
        raise ValueError(f"{image_col!r} contains missing image paths.")
    df = df.copy()
    paths = []
    for value in df[image_col]:
        image_path = Path(str(value).strip()).expanduser()
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"Patient image does not exist: {image_path}")
        paths.append(str(image_path.resolve()))
    df[image_col] = paths
    for name in outcome_cols:
        values = pd.to_numeric(df[name], errors="coerce")
        if (df[name].notna() & values.isna()).any():
            raise ValueError(f"Outcome {name!r} contains nonnumeric values.")
        df[name] = values
    return df


def load_patient_images(df, image_col, mask_path, *, output_type=None):
    """Import one patient-by-feature matrix in the map's native space."""
    paths = df[image_col].tolist()
    detected = detect_image_type(paths, mask_path)
    X = load_native_patient_vectors(paths, mask_path, output_type or detected)
    if X.ndim != 2 or X.shape[0] != len(df):
        raise ValueError("Imported patient images must be patient-by-feature.")
    return X


def detect_image_type(paths, mask_path):
    """Return the generic output type for a uniform collection of images."""
    importer = GiiNiiFileImport(import_path=None, mask_path=mask_path)
    detected = importer._import_type_switch(list(paths))
    if detected == "npy":
        raise ValueError("Use individual patient images, not a cohort .npy matrix.")
    return {"gii": "surface", "freesurfer": "surface"}.get(detected, detected)


def fit_weights_with_inner_cv(
        map_X, map_ids, map_outcomes, scoring_datasets, *, inner_folds=5,
        seed=2026, data_transform_method="rank", weight_mode="unweighted",
        max_iters=500, store_iters=False, return_optimizer=False,
        score_mode="reference_z"):
    """Select one weight vector from inner out-of-fold regression maps.

    ``reference_z`` calibrates each inner-validation cosine against scores from
    that inner fold's training patients. It removes fold-specific score offsets
    and works for inner leave-one-out validation.
    """
    if max_iters < 1:
        raise ValueError("max_iters must be positive.")
    full_maps, fold_specs, cohorts, map_sizes = prepare_inner_folds(
        map_X, map_ids, map_outcomes, scoring_datasets,
        inner_folds=inner_folds, seed=seed,
        data_transform_method=data_transform_method,
    )
    loader = ArrayLoader({
        name: {"niftis": data["X"], "indep_var": data["y"]}
        for name, data in cohorts.items()
    })
    optimizer = LocalizationOptimizer(
        full_maps, loader, data_mode="ram", inner_folds=fold_specs,
        mode=weight_mode, map_sample_sizes=map_sizes,
        random_state=seed,
        inner_score_mode=score_mode,
    )
    optimizer.engine.convergence_monitor.max_iterations = max_iters
    optimized_map, _ = optimizer.optimize(
        second_stage=False, store_iters=store_iters
    )
    weights = optimizer.engine.best_W
    scores = optimizer.engine.predict_inner(weights)
    adjusted_scores = optimizer.engine.predict_inner(weights, adjusted=True)
    final_map = np.asarray(optimized_map, dtype=float).reshape(-1)
    final_norm = np.linalg.norm(final_map)
    if not np.isfinite(final_norm) or final_norm == 0:
        raise ValueError("The final full-cohort map has zero or nonfinite norm.")
    records, summaries = [], []
    for name, data in cohorts.items():
        y = data["y"]
        rho_result = spearmanr(scores[name], y)
        adjusted_result = spearmanr(adjusted_scores[name], y)
        rho = getattr(rho_result, "statistic", rho_result.correlation)
        adjusted_rho = getattr(adjusted_result, "statistic", adjusted_result.correlation)
        patient_norms = np.linalg.norm(data["X"], axis=1)
        if np.any(patient_norms == 0):
            raise ValueError(f"Scoring cohort {name!r} contains a zero-norm patient.")
        final_scores = (data["X"] @ final_map) / (patient_norms * final_norm)
        final_result = spearmanr(final_scores, y)
        final_rho = getattr(final_result, "statistic", final_result.correlation)
        fold_numbers = np.zeros(len(y), dtype=int)
        for fold_number, (rows, _, _) in enumerate(
            optimizer.engine._inner_projections[name], start=1
        ):
            fold_numbers[rows] = fold_number
        summaries.append({
            "dataset": name, "n_patients": len(y),
            "n_maps": len(full_maps), "score_mode": score_mode,
            "inner_cv_weight_fit_spearman_rho": float(adjusted_rho),
            "final_map_apparent_spearman_rho": float(final_rho),
            "raw_pooled_inner_fold_spearman_rho": float(rho),
        })
        for patient_id, fold, observed, score, adjusted_score, final_score in zip(
            data["ids"], fold_numbers, y, scores[name], adjusted_scores[name], final_scores
        ):
            records.append({
                "dataset": name, "patient_id": patient_id,
                "inner_fold": int(fold),
                "observed": float(observed), "damage": float(score),
                "inner_calibrated_damage": float(adjusted_score),
                "final_map_damage": float(final_score),
            })
    predictions = pd.DataFrame.from_records(records)
    summary = pd.DataFrame.from_records(summaries)
    weight_table = pd.DataFrame({
        "map": optimizer.corr_map_names,
        "weight": weights.reshape(-1),
        "map_sample_size": [map_sizes[name] for name in optimizer.corr_map_names],
    })
    result = optimized_map, predictions, summary, weight_table
    return (*result, optimizer) if return_optimizer else result


# Compatibility for notebooks that imported the old ambiguous name.
def fit_oof(map_X, map_ids, map_outcomes, scoring_datasets, *, folds=5, **kwargs):
    return fit_weights_with_inner_cv(
        map_X, map_ids, map_outcomes, scoring_datasets,
        inner_folds=folds, **kwargs,
    )


def evaluate_regression_pipeline_outer_cv(
        map_X, map_ids, map_outcomes, scoring_datasets, *, inner_folds,
        outer_folds, seed=2026, data_transform_method="rank",
        weight_mode="unweighted", max_iters=500,
        score_mode="reference_z"):
    """Test the full regression-plus-weight-selection process in outer folds.

    For every outer split, all inner folds run inside the outer-training set to
    select one weight vector. Component maps are then rebuilt on the complete
    outer-training set, combined with those weights, and applied once to the
    untouched outer-validation patients.
    """
    map_X = np.asarray(map_X)
    map_ids = np.asarray(map_ids).astype(str)
    all_ids = np.unique(np.concatenate([
        np.asarray(data["ids"]).astype(str) for data in scoring_datasets.values()
    ]))
    held_out_sets = assign_id_folds(all_ids, outer_folds, seed=seed, level="outer")
    if held_out_sets is None:
        print(f"OUTER_FOLDS = {held_out_sets}. Skipping in-data cross-validation.")
        return None

    outer_records = []
    outer_weight_tables = []
    inner_prediction_tables = []
    inner_summary_tables = []
    scored_rows = {
        name: np.zeros(len(data["y"]), dtype=int)
        for name, data in scoring_datasets.items()
    }
    excluded_folds = {name: 0 for name in scoring_datasets}

    for outer_fold, held_out_ids in enumerate(held_out_sets, start=1):
        map_train = ~np.isin(map_ids, held_out_ids)
        outer_map_X = map_X[map_train]
        outer_map_ids = map_ids[map_train]
        outer_map_outcomes = {
            name: np.asarray(values)[map_train]
            for name, values in map_outcomes.items()
        }
        outer_training = {}
        held_rows = {}
        for name, data in scoring_datasets.items():
            ids = np.asarray(data["ids"]).astype(str)
            held = np.isin(ids, held_out_ids)
            train = ~held
            held_rows[name] = np.flatnonzero(held)
            if train.sum() < 5 or np.unique(np.asarray(data["y"])[train]).size < 2:
                # This outcome is constant once this fold is withheld, so it
                # cannot contribute to weight selection here. Exclude it from
                # THIS fold's inner CV only; its held-out patients are still
                # scored below with the map the other outcomes fitted, so
                # every row is scored exactly once.
                excluded_folds[name] += 1
                continue
            outer_training[name] = {
                "X": np.asarray(data["X"])[train],
                "y": np.asarray(data["y"])[train],
                "ids": ids[train],
            }
        if len(outer_training) < 1:
            raise ValueError(
                f"Outer fold {outer_fold} leaves no scoring dataset with varying "
                "training patients, so no weights can be selected."
            )

        fitted_map, inner_predictions, inner_summary, weights = (
            fit_weights_with_inner_cv(
                outer_map_X, outer_map_ids, outer_map_outcomes, outer_training,
                inner_folds=inner_folds, seed=seed + outer_fold,
                data_transform_method=data_transform_method,
                weight_mode=weight_mode, max_iters=max_iters,
                score_mode=score_mode,
            )
        )
        inner_predictions.insert(0, "outer_fold", outer_fold)
        inner_summary.insert(0, "outer_fold", outer_fold)
        inner_prediction_tables.append(inner_predictions)
        inner_summary_tables.append(inner_summary)
        weights.insert(0, "outer_fold", outer_fold)
        outer_weight_tables.append(weights)

        fitted_map = np.asarray(fitted_map, dtype=float).reshape(-1)
        fitted_norm = np.linalg.norm(fitted_map)
        if not np.isfinite(fitted_norm) or fitted_norm == 0:
            raise ValueError(f"Outer fold {outer_fold} produced an invalid map.")
        for name, data in scoring_datasets.items():
            rows = held_rows[name]
            if not len(rows):
                continue
            X = np.asarray(data["X"])
            y = np.asarray(data["y"], dtype=float).reshape(-1)
            ids = np.asarray(data["ids"]).astype(str)
            training_rows = ~np.isin(ids, held_out_ids)
            held_norms = np.linalg.norm(X[rows], axis=1)
            reference_norms = np.linalg.norm(X[training_rows], axis=1)
            if np.any(held_norms == 0) or np.any(reference_norms == 0):
                raise ValueError(
                    f"Outer fold {outer_fold} contains a zero-norm patient in {name!r}."
                )
            raw = (X[rows] @ fitted_map) / (held_norms * fitted_norm)
            reference = ((X[training_rows] @ fitted_map)
                         / (reference_norms * fitted_norm))
            spread = reference.std()
            if not np.isfinite(spread) or spread <= 0:
                raise ValueError(
                    f"Outer fold {outer_fold} has no reference-score variation "
                    f"in {name!r}."
                )
            adjusted = (raw - reference.mean()) / spread
            scored_rows[name][rows] += 1
            outer_records.extend(
                {
                    "dataset": name,
                    "patient_id": patient_id,
                    "outer_fold": outer_fold,
                    "observed": float(observed),
                    "damage": float(damage),
                    "outer_calibrated_damage": float(calibrated),
                }
                for patient_id, observed, damage, calibrated in zip(
                    ids[rows], y[rows], raw, adjusted
                )
            )

    outer_predictions = pd.DataFrame.from_records(outer_records)
    summaries = []
    for name, data in scoring_datasets.items():
        if not np.all(scored_rows[name] == 1):
            raise ValueError(
                f"Outer CV did not score every patient in {name!r} exactly once."
            )
        rows = outer_predictions[outer_predictions["dataset"] == name]
        raw_result = spearmanr(rows["damage"], rows["observed"])
        adjusted_result = spearmanr(
            rows["outer_calibrated_damage"], rows["observed"]
        )
        raw_rho = getattr(raw_result, "statistic", raw_result.correlation)
        adjusted_rho = getattr(
            adjusted_result, "statistic", adjusted_result.correlation
        )
        summaries.append({
            "dataset": name,
            "n_patients": len(rows),
            "n_outer_folds": len(held_out_sets),
            "n_folds_excluded_from_weight_fitting": excluded_folds[name],
            "outer_cv_spearman_rho": float(adjusted_rho),
            "raw_outer_cv_spearman_rho": float(raw_rho),
        })
    return (
        outer_predictions,
        pd.DataFrame.from_records(summaries),
        pd.concat(outer_weight_tables, ignore_index=True),
        pd.concat(inner_prediction_tables, ignore_index=True),
        pd.concat(inner_summary_tables, ignore_index=True),
    )


def save_nested_cv(out_dir, outer_predictions, outer_summary, outer_weights,
                   inner_predictions, inner_summary):
    """Save outer generalization results and their inner-fit audit trail."""
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    outer_predictions.to_csv(out_dir / "outer_predictions.csv", index=False)
    outer_summary.to_csv(out_dir / "outer_summary.csv", index=False)
    outer_weights.to_csv(out_dir / "outer_fold_weights.csv", index=False)
    inner_predictions.to_csv(
        out_dir / "outer_fold_inner_predictions.csv", index=False
    )
    inner_summary.to_csv(out_dir / "outer_fold_inner_summary.csv", index=False)
    print(outer_summary.to_string(index=False))
    print(f"Saved nested outer validation to: {out_dir}")
    return out_dir


def save_fit(out_dir, optimized_map, predictions, summary, weights, output_type,
             *, mask_path, prefix="inner_cv"):
    """Save inner-fit diagnostics, fitted weights, and the final native map."""
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_dir / f"{prefix}_predictions.csv", index=False)
    summary.to_csv(out_dir / f"{prefix}_summary.csv", index=False)
    weights.to_csv(out_dir / "optimized_weights.csv", index=False)
    exporter = NeuroimageFileOutporter(output_ftype=output_type, mask_path=mask_path)
    exporter.validate_for_output()
    exporter.save_map(np.asarray(optimized_map).reshape(-1), "optimized_map", str(out_dir))
    print(summary.to_string(index=False))
    print(f"Saved {prefix} fit and {output_type} map to: {out_dir}")


def save_history(optimizer, path, *, output_type, mask_path):
    """Save compact weight history with the full-cohort component maps."""
    history = OptimizationHistory(
        weights=np.asarray(optimizer.engine.iter_weights).reshape(-1, optimizer.MAPS.shape[0]),
        losses=np.asarray(optimizer.engine.iter_losses),
        maps=optimizer.MAPS,
        map_names=tuple(optimizer.corr_map_names),
        output_type=output_type,
        mask_path=str(Path(mask_path).expanduser().resolve()) if mask_path else "",
        objective_label="Inner-CV RMS Spearman rho (weight selection)",
    )
    return history.save(path)
