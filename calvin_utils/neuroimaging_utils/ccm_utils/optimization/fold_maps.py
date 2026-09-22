"""Build component maps from training patients only for cross-validation."""

from __future__ import annotations

import numpy as np

from calvin_utils.permutation_analysis_utils.voxelwise_regression import VoxelwiseRegression
from calvin_utils.permutation_analysis_utils.voxelwise_regression_prep import RegressionPrep


def build_fold_maps(X, outcomes, train_rows, data_transform_method="rank",
                    validate_X=True,
                    skip_invalid=False):
    """Return maps and sample sizes using only selected training rows.

    Transformations are delegated to :class:`RegressionPrep`, and t maps are
    fitted by :class:`VoxelwiseRegression`. Each outcome may have a different
    set of observed patients. Transformation is redone within each training
    subset, so held-out patients cannot influence the fitted maps.

    ``skip_invalid`` leaves out an outcome that cannot be fitted on these
    training rows (fewer than four patients, or no variation) instead of
    raising. Sparse items -- one or two non-zero patients -- lose all their
    variation whenever those patients are held out, and one such fold should
    cost that map, not the whole run.
    """
    if data_transform_method not in {"standardize", "rank", None}:
        raise ValueError(
            "data_transform_method must be 'standardize', 'rank', or None."
        )
    X = np.asarray(X)
    train_rows = np.asarray(train_rows, dtype=bool)
    if X.ndim != 2 or train_rows.shape != (X.shape[0],):
        raise ValueError("X must be patient-by-feature and train_rows one per patient.")
    if validate_X and not np.isfinite(X).all():
        raise ValueError("Map-building X must contain only finite values.")
    if not outcomes:
        raise ValueError("At least one map outcome is required.")

    groups = {}
    sample_sizes = {}
    for name, values in outcomes.items():
        y = np.asarray(values, dtype=float).reshape(-1)
        if y.shape != (X.shape[0],):
            raise ValueError(f"Map outcome {name!r} needs one value per patient.")
        valid = train_rows & np.isfinite(y)
        n = int(valid.sum())
        if n < 4 or np.unique(y[valid]).size < 2:
            if skip_invalid:
                continue
            raise ValueError(
                f"Map outcome {name!r} needs four training patients and varying values."
            )
        groups.setdefault(valid.tobytes(), (valid, []))[1].append((name, y))
        sample_sizes[name] = n

    maps = {}
    for valid, targets in groups.values():
        values = X[valid]
        transformed_images = RegressionPrep.transform_array(
            values[:, None, :], data_transform_method
        )[:, 0, :]
        for name, y in targets:
            transformed_symptom = RegressionPrep.transform_array(
                y[valid, None, None], data_transform_method,
                keep_categorical=True,
            )[:, 0, 0]
            maps[name] = VoxelwiseRegression.fit_linear_tmap(
                transformed_symptom, transformed_images, contrast=[[1.0]]
            )[0]
    return {name: maps[name] for name in outcomes if name in maps}, sample_sizes
