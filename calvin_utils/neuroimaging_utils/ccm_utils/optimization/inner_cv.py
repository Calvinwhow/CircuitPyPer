"""Build maps that exclude every patient used to score each inner fold."""

from __future__ import annotations

import numpy as np

from calvin_utils.neuroimaging_utils.ccm_utils.optimization.fold_maps import build_fold_maps
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.fold_assignment import assign_id_folds


def prepare_inner_folds(map_X, map_ids, map_outcomes, scoring_datasets, *,
                        inner_folds=5, seed=2026,
                        data_transform_method="rank"):
    """Return full-training maps and inner-fold maps used to select weights.

    Map-building patients and scoring patients are matched by ID. Every scored
    patient's ID is omitted from the maps used for that patient's score. The
    caller supplies native patient vectors aligned with the map's feature space.
    """
    map_X = np.asarray(map_X)
    map_ids = np.asarray(map_ids).astype(str)
    if map_X.ndim != 2 or map_ids.shape != (len(map_X),):
        raise ValueError("Map X must be patient-by-feature with one ID per row.")
    if not np.isfinite(map_X).all() or len(np.unique(map_ids)) != len(map_ids):
        raise ValueError("Map-building patients need finite images and unique IDs.")
    if len(map_outcomes) < 2 or not scoring_datasets:
        raise ValueError("Out-of-fold fitting needs two component outcomes and a scoring cohort.")

    cohorts = {}
    shared_X = {}
    validated_X = set()
    for name, data in scoring_datasets.items():
        X = np.asarray(data["X"])
        ids = np.asarray(data["ids"]).astype(str)
        y = np.asarray(data["y"], dtype=float).reshape(-1)
        if X.ndim != 2 or X.shape[1] != map_X.shape[1] or ids.shape != (len(X),) or y.shape != (len(X),):
            raise ValueError(f"Scoring cohort {name!r} needs aligned X, IDs, and y in the map feature space.")
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"Scoring cohort {name!r} has duplicate IDs.")
        if id(X) not in validated_X:
            if not np.isfinite(X).all():
                raise ValueError(f"Scoring cohort {name!r} has nonfinite X.")
            validated_X.add(id(X))
        valid = np.isfinite(y)
        key = (id(X), valid.tobytes())
        if key not in shared_X:
            shared_X[key] = X if valid.all() else X[valid]
        X, ids, y = shared_X[key], ids[valid], y[valid]
        if len(y) < 5 or np.unique(y).size < 2:
            raise ValueError(f"Scoring cohort {name!r} needs five observed patients and varying outcomes.")
        cohorts[name] = {"X": X, "ids": ids, "y": y}

    scored_ids = np.unique(np.concatenate([data["ids"] for data in cohorts.values()]))
    held_out_sets = assign_id_folds(
        scored_ids, inner_folds, seed=seed, level="inner"
    )

    shared_names = set(map_outcomes)
    for held_out_ids in [np.asarray([], dtype=str), *held_out_sets]:
        train = ~np.isin(map_ids, held_out_ids)
        for name, outcome in map_outcomes.items():
            y = np.asarray(outcome, dtype=float).reshape(-1)
            if y.shape != (len(map_ids),):
                raise ValueError(f"Map outcome {name!r} needs one value per map-building patient.")
            valid = train & np.isfinite(y)
            if valid.sum() < 4 or np.unique(y[valid]).size < 2:
                shared_names.discard(name)
    if len(shared_names) < 2:
        raise ValueError("Fewer than two component maps can be built in every inner fold.")
    names = [name for name in map_outcomes if name in shared_names]
    selected_outcomes = {name: map_outcomes[name] for name in names}
    full_maps, full_sizes = build_fold_maps(
        map_X, selected_outcomes, np.ones(len(map_ids), dtype=bool),
        data_transform_method=data_transform_method, validate_X=False,
    )

    def fold_specs():
        # Yield one large map matrix at a time; WeightOptimizer immediately
        # reduces it to small patient-by-map products and a map Gram matrix.
        for held_out_ids in held_out_sets:
            train = ~np.isin(map_ids, held_out_ids)
            maps, _ = build_fold_maps(
                map_X, selected_outcomes, train,
                data_transform_method=data_transform_method, validate_X=False,
            )
            yield {
                "maps": maps,
                "rows": {
                    name: np.flatnonzero(np.isin(data["ids"], held_out_ids))
                    for name, data in cohorts.items()
                },
            }

    return full_maps, fold_specs(), cohorts, full_sizes


# Compatibility for notebooks that imported the old ambiguous name.
def prepare_oof_folds(map_X, map_ids, map_outcomes, scoring_datasets, *,
                      folds=5, **kwargs):
    return prepare_inner_folds(
        map_X, map_ids, map_outcomes, scoring_datasets,
        inner_folds=folds, **kwargs,
    )
