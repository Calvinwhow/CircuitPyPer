"""Held-out weight fitting for component maps fixed before scoring patients."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from calvin_utils.neuroimaging_utils.ccm_utils.stat_utils import CorrelationCalculator
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.convergent_map_optimizer import LocalizationOptimizer
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.nested_cv import ArrayLoader
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.fold_assignment import assign_id_folds


def _load_cohorts(data_loader, *, data_mode, patient_ids=None):
    """Open each cohort once and keep its X, y, and patient IDs aligned."""
    if data_mode not in {"ram", "memmap"}:
        raise ValueError("data_mode must be 'ram' or 'memmap'.")
    names = tuple(data_loader.dataset_names_list)
    if not names:
        raise ValueError("At least one scoring dataset is required.")
    paths_by_name = getattr(data_loader, "dataset_paths_dict", {})
    image_paths = [paths_by_name.get(name, {}).get("niftis") for name in names]
    shared_image_rows = (
        len(names) > 1 and all(image_paths)
        and len(set(image_paths)) == 1
    )
    cohorts = {}
    for name in names:
        data = data_loader.load_dataset(
            name, mmap_mode="r" if data_mode == "memmap" else None
        )
        X = data["niftis"]
        y = np.asarray(data["indep_var"], dtype=float).reshape(-1)
        if X.ndim != 2 or y.shape != (len(X),) or len(X) < 5:
            raise ValueError(
                f"Scoring dataset {name!r} needs at least five aligned patient rows."
            )
        LocalizationOptimizer._require_finite(X, name, "niftis")
        if not np.isfinite(y).all() or np.unique(y).size < 2:
            raise ValueError(f"Scoring dataset {name!r} needs finite, varying outcomes.")
        configured_ids = None if patient_ids is None else patient_ids.get(name)
        ids_path = paths_by_name.get(name, {}).get("ids")
        if configured_ids is not None:
            ids = np.asarray(configured_ids).astype(str).reshape(-1)
        elif ids_path:
            ids = np.load(ids_path, allow_pickle=False).astype(str).reshape(-1)
        elif len(names) == 1 or shared_image_rows:
            ids = np.arange(len(X)).astype(str)
        else:
            raise ValueError(
                "Multiple scoring datasets need patient IDs in every dataset "
                "manifest when their X files differ, so the same patient is "
                "held out everywhere."
            )
        if ids.shape != (len(X),) or len(np.unique(ids)) != len(ids):
            raise ValueError(f"Scoring dataset {name!r} needs one unique ID per row.")
        cohorts[name] = {"X": X, "y": y, "ids": ids}
    return cohorts


def _project_cohorts(corr_map_dict, cohorts, data_loader):
    """Project each large patient matrix onto fixed maps only once for all folds."""
    names = tuple(corr_map_dict)
    maps = []
    for name in names:
        values = CorrelationCalculator._check_for_nans(
            np.asarray(corr_map_dict[name], dtype=float).reshape(-1),
            nanpolicy="remove", verbose=False,
        )
        norm = np.linalg.norm(values)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError(f"Component map {name!r} has zero or nonfinite norm.")
        maps.append(values / norm)
    matrix = np.stack(maps)
    projected_by_path = {}
    paths_by_name = getattr(data_loader, "dataset_paths_dict", {})
    for name, data in cohorts.items():
        X = data["X"]
        if X.shape[1] != matrix.shape[1]:
            raise ValueError(
                f"Scoring dataset {name!r} has {X.shape[1]} features; "
                f"component maps have {matrix.shape[1]}."
            )
        path = paths_by_name.get(name, {}).get("niftis")
        cache_key = path if path is not None else id(X)
        if cache_key in projected_by_path:
            data["projected"] = projected_by_path[cache_key]
            continue
        projected = np.empty((len(X), len(names)), dtype=float)
        rows_per_chunk = max(1, 10_000_000 // matrix.shape[1])
        for start in range(0, len(X), rows_per_chunk):
            stop = min(start + rows_per_chunk, len(X))
            chunk = X[start:stop]
            norms = np.linalg.norm(chunk, axis=1)
            if np.any(norms == 0):
                raise ValueError(f"Scoring dataset {name!r} contains a zero-norm patient.")
            projected[start:stop] = (chunk @ matrix.T) / norms[:, None]
        projected_by_path[cache_key] = projected
        data["projected"] = projected


def evaluate_fixed_maps_outer_cv(
        corr_map_dict, data_loader, *, outer_folds=5, seed=2026,
        data_mode="memmap", weight_mode="unweighted", map_sample_sizes=None,
        max_iters=500, second_stage=False, patient_ids=None):
    """Test fixed-map weight fitting on untouched outer-validation patients.

    If a patient occurs in several scoring datasets, their ID selects the same
    outer-validation fold in all of them. Component maps are never rebuilt.
    """
    if max_iters < 1:
        raise ValueError("max_iters must be positive.")
    cohorts = _load_cohorts(
        data_loader, data_mode=data_mode, patient_ids=patient_ids
    )
    _project_cohorts(corr_map_dict, cohorts, data_loader)
    all_ids = np.unique(np.concatenate([data["ids"] for data in cohorts.values()]))
    held_out_sets = assign_id_folds( all_ids, outer_folds, seed=seed, level="outer")
    if held_out_sets is None:
        print(f"OUTER_FOLDS = {held_out_sets}. Skipping in-data cross-validation.")
        return None

    scores = {name: np.full(len(data["y"]), np.nan) for name, data in cohorts.items()}
    calibrated_scores = {
        name: np.full(len(data["y"]), np.nan) for name, data in cohorts.items()
    }
    outer_fold_numbers = {
        name: np.zeros(len(data["y"]), dtype=int)
        for name, data in cohorts.items()
    }
    weight_records = []
    excluded_folds = {name: 0 for name in cohorts}
    for outer_fold, held_out in enumerate(held_out_sets, start=1):
        train_data = {}
        train_projections = {}
        held_rows = {}
        for name, data in cohorts.items():
            held = np.isin(data["ids"], held_out)
            train = ~held
            held_rows[name] = np.flatnonzero(held)
            if train.sum() < 3 or np.unique(data["y"][train]).size < 2:
                # This outcome is constant once this fold is withheld, so it
                # cannot contribute to the objective here. Exclude it from
                # THIS fold's weight fitting only; its held-out patients are
                # still scored below with the map the other outcomes fitted,
                # so every row is scored exactly once.
                excluded_folds[name] += 1
                continue
            train_data[name] = {
                "niftis": data["X"][train], "indep_var": data["y"][train]
            }
            train_projections[name] = (
                data["projected"][train], data["y"][train]
            )
        if not train_data:
            raise ValueError(
                f"Outer fold {outer_fold} leaves no scoring dataset with varying "
                "training outcomes, so no weights can be fitted."
            )
        optimizer = LocalizationOptimizer(
            corr_map_dict, ArrayLoader(train_data), data_mode="ram",
            mode=weight_mode, map_sample_sizes=map_sample_sizes,
            random_state=seed + outer_fold,
            precomputed_projections=train_projections,
        )
        optimizer.engine.convergence_monitor.max_iterations = max_iters
        optimized_map, blended_map = optimizer.optimize(second_stage=second_stage)
        scored_map = blended_map if second_stage else optimized_map
        scored_weights = (optimizer.W_final if second_stage
                          else optimizer.engine.best_W).reshape(-1)
        if not np.isfinite(scored_map).all() or np.linalg.norm(scored_map) == 0:
            raise ValueError(
                f"Outer fold {outer_fold} produced a zero or nonfinite map."
            )
        for name, data in cohorts.items():
            rows = held_rows[name]
            if len(rows):
                raw = np.asarray(optimizer._calculate_similarity(
                    data["X"][rows], scored_map
                )).reshape(-1)
                training = ~np.isin(data["ids"], held_out)
                reference = np.asarray(optimizer._calculate_similarity(
                    data["X"][training], scored_map
                )).reshape(-1)
                spread = reference.std()
                if not np.isfinite(spread) or spread <= 0:
                    raise ValueError(
                        f"Outer fold {outer_fold} has no reference-score "
                        f"variation in {name!r}."
                    )
                scores[name][rows] = raw
                calibrated_scores[name][rows] = (
                    raw - reference.mean()
                ) / spread
                outer_fold_numbers[name][rows] = outer_fold
        weight_records.extend(
            {"outer_fold": outer_fold, "map": name, "weight": float(weight)}
            for name, weight in zip(optimizer.corr_map_names, scored_weights)
        )

    prediction_records = []
    summary_records = []
    for name, data in cohorts.items():
        if (not np.isfinite(scores[name]).all()
                or not np.isfinite(calibrated_scores[name]).all()
                or np.any(outer_fold_numbers[name] == 0)):
            raise ValueError(f"Scoring dataset {name!r} was not scored in every row.")
        raw_result = spearmanr(scores[name], data["y"])
        calibrated_result = spearmanr(calibrated_scores[name], data["y"])
        raw_rho = float(getattr(raw_result, "statistic", raw_result.correlation))
        rho = float(getattr(
            calibrated_result, "statistic", calibrated_result.correlation
        ))
        if not np.isfinite(rho) or not np.isfinite(raw_rho):
            raise ValueError(f"Held-out Spearman rho is undefined for {name!r}.")
        summary_records.append({
            "dataset": name, "n_patients": len(data["y"]),
            "n_maps": len(corr_map_dict),
            "n_outer_folds": len(held_out_sets),
            "n_folds_excluded_from_weight_fitting": excluded_folds[name],
            "outer_cv_spearman_rho": rho,
            "raw_outer_cv_spearman_rho": raw_rho,
        })
        prediction_records.extend(
            {"dataset": name, "patient_id": patient_id,
             "outer_fold": int(outer_fold),
             "observed": float(observed), "damage": float(damage),
             "outer_calibrated_damage": float(calibrated)}
            for patient_id, outer_fold, observed, damage, calibrated in zip(
                data["ids"], outer_fold_numbers[name], data["y"], scores[name],
                calibrated_scores[name],
            )
        )
    return (
        pd.DataFrame.from_records(prediction_records),
        pd.DataFrame.from_records(summary_records),
        pd.DataFrame.from_records(weight_records),
    )


def cross_validate_fixed_maps(corr_map_dict, data_loader, *, folds=5, **kwargs):
    """Compatibility wrapper for the former ambiguous fold parameter."""
    return evaluate_fixed_maps_outer_cv(
        corr_map_dict, data_loader, outer_folds=folds, **kwargs
    )


def save_fixed_cv(out_dir, predictions, summary, outer_fold_weights):
    """Save outer-validation scores and weights fitted in each outer fold."""
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_dir / "outer_predictions.csv", index=False)
    summary.to_csv(out_dir / "outer_summary.csv", index=False)
    outer_fold_weights.to_csv(out_dir / "outer_fold_weights.csv", index=False)
    return out_dir
