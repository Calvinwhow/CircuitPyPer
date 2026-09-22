"""Patient-held-out damage scores from maps refitted inside each fold."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from calvin_utils.file_utils.import_functions import GiiNiiFileImport
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.fold_maps import build_fold_maps


def load_native_patient_vectors(subject_files, mask_path, output_ftype, *, cache=None):
    """Load patients in the same ordered feature space as their fitted map.

    Fibers remain fiber values; NIfTI images use the same mask as the map.
    ``cache`` may be shared across symptom models to avoid repeated file reads.
    """
    paths = [str(Path(path).expanduser().resolve()) for path in subject_files]
    if not paths:
        raise ValueError("At least one patient image is required.")
    importer = GiiNiiFileImport(import_path=None, mask_path=mask_path)
    detected = importer._import_type_switch(paths)
    detected = {"gii": "surface", "freesurfer": "surface"}.get(detected, detected)
    expected = {"nifti": "nii", "gii": "surface", "freesurfer": "surface"}.get(
        output_ftype, output_ftype
    )
    if detected != expected:
        raise ValueError(
            f"CV comparator is {detected}, but the fitted map is {expected}. "
            "Use patient images in the map's native feature space."
        )
    vectors = {} if cache is None else cache
    keys = [(str(mask_path), path) for path in paths]
    missing = list(dict.fromkeys(path for path, key in zip(paths, keys) if key not in vectors))
    if missing:
        loaded = np.asarray(importer._import_matrices(missing).T, dtype=np.float32)
        if loaded.ndim != 2 or loaded.shape[0] != len(missing):
            raise ValueError("Native importer returned a malformed patient matrix.")
        for path, row in zip(missing, loaded):
            vectors[(str(mask_path), path)] = row.copy()
    result = np.vstack([vectors[key] for key in keys])
    if not np.isfinite(result).all():
        raise ValueError("Native patient vectors contain NaN or infinity.")
    return result


def cross_validated_map_damage(patient_values, outcomes, *, cv="loocv"):
    """Fit a ranked inverse-regression t map and score held-out patient vectors.

    The regression is image values ~ scalar outcome, with one predictor, no
    intercept, no covariates, and within-fold ranking. The score is cosine
    similarity in the same feature space as the fitted map. No patient outcome
    is used while building the map that scores that patient.

    Returns ``(damage_scores, fold_numbers)`` in input patient order.
    """
    X = np.asarray(patient_values)
    y = np.asarray(outcomes, dtype=float).reshape(-1)
    if X.ndim != 2 or y.shape != (len(X),):
        raise ValueError("Patient values must be a patient-by-feature matrix with one numeric outcome per row.")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Patient values and outcomes must be finite before cross-validation.")
    n = len(y)
    if n < 5:
        raise ValueError("At least five patients are needed to fit held-out t maps.")
    if cv in {"loocv", "loo"}:
        held_out_sets = [np.asarray([i]) for i in range(n)]
    elif cv == "leave_all_in":
        held_out_sets = [np.arange(n)]
    elif isinstance(cv, int) and 2 <= cv <= n:
        held_out_sets = np.array_split(np.arange(n), cv)
    else:
        raise ValueError("cv must be 'loocv', 'leave_all_in', or an integer fold count.")

    scores = np.full(n, np.nan, dtype=float)
    fold_numbers = np.full(n, -1, dtype=int)
    patient_norms = np.linalg.norm(X, axis=1)
    if np.any(patient_norms == 0):
        raise ValueError("A patient image has zero magnitude, so cosine damage is undefined.")
    for fold, held_out in enumerate(held_out_sets, start=1):
        train_rows = np.ones(n, dtype=bool)
        if cv != "leave_all_in":
            train_rows[held_out] = False
        maps, _ = build_fold_maps(
            X, {"outcome": y}, train_rows,
            data_transform_method="rank", validate_X=False,
        )
        fitted_map = maps["outcome"]
        map_norm = np.linalg.norm(fitted_map)
        if not np.isfinite(map_norm) or map_norm == 0:
            raise ValueError(f"Fold {fold} produced a zero or nonfinite t map.")
        scores[held_out] = (X[held_out] @ fitted_map) / (
            patient_norms[held_out] * map_norm
        )
        fold_numbers[held_out] = fold
    return scores, fold_numbers
