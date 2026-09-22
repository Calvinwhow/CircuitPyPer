"""Shared patient-ID fold assignment for inner and outer cross-validation."""

import numpy as np


def assign_id_folds(patient_ids, folds, *, seed, level):
    """Return held-out ID arrays, preserving one assignment across outcomes."""
    ids = np.unique(np.asarray(patient_ids).astype(str))
    if not len(ids):
        raise ValueError(f"{level} cross-validation needs at least one patient ID.")
    if folds == "loocv":
        return [np.asarray([patient_id]) for patient_id in ids]
    if isinstance(folds, int) and 2 <= folds <= len(ids):
        return list(np.array_split(np.random.default_rng(seed).permutation(ids), folds))
    raise ValueError(
        f"{level}_folds must be 'loocv' or an integer from 2 to {len(ids)}."
    )
