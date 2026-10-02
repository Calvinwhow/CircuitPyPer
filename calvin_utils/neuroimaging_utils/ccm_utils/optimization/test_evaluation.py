"""Score one frozen optimized map in datasets excluded from every fit.

Fitting datasets choose the weights. Test datasets are opened only after the
final map exists, are never used to build maps, select weights, or calibrate
scores, and are scored exactly once. Spearman rho is invariant to any monotone
rescaling of a single cohort's scores, so no reference calibration is applied
here; the raw cosine damage is reported directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def evaluate_test_datasets(optimized_map, test_datasets, *, fitting_ids=None):
    """Apply a frozen map to untouched cohorts and correlate with outcomes.

    ``test_datasets`` uses the ``{name: {'X': ..., 'y': ..., 'ids': ...}}``
    structure produced by :func:`datasets_from_loader`. ``fitting_ids`` is the
    set of patient IDs used anywhere in fitting; an overlap is refused because
    a shared patient makes the reported rho an in-sample number.
    """
    final_map = np.asarray(optimized_map, dtype=float).reshape(-1)
    final_norm = np.linalg.norm(final_map)
    if not np.isfinite(final_map).all():
        raise ValueError("The optimized map contains nonfinite values.")
    if not np.isfinite(final_norm) or final_norm == 0:
        raise ValueError("The optimized map has zero or nonfinite norm.")
    if not test_datasets:
        raise ValueError("No test datasets were supplied.")
    known_ids = (
        None if fitting_ids is None
        else set(np.asarray(list(fitting_ids)).astype(str).reshape(-1).tolist())
    )

    records, summaries = [], []
    for name, data in test_datasets.items():
        X = np.asarray(data["X"])
        y = np.asarray(data["y"], dtype=float).reshape(-1)
        ids = np.asarray(data["ids"]).astype(str).reshape(-1)
        if X.ndim != 2:
            raise ValueError(f"Test dataset {name!r} must be patient-by-feature.")
        if X.shape[1] != final_map.size:
            raise ValueError(
                f"Test dataset {name!r} has {X.shape[1]} features; the "
                f"optimized map has {final_map.size}."
            )
        if y.shape != (len(X),) or ids.shape != (len(X),):
            raise ValueError(
                f"Test dataset {name!r} has misaligned images, outcomes, and IDs."
            )
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"Test dataset {name!r} has duplicate IDs.")
        if known_ids is not None:
            shared = sorted(known_ids.intersection(ids.tolist()))
            if shared:
                raise ValueError(
                    f"Test dataset {name!r} shares {len(shared)} patient IDs "
                    f"with the fitting data, starting with {shared[:5]}. Test "
                    "patients must be untouched by map building and weight "
                    "selection."
                )
        valid = np.isfinite(y)
        if valid.sum() < 3:
            raise ValueError(
                f"Test dataset {name!r} needs three patients with observed outcomes."
            )
        norms = np.linalg.norm(X[valid], axis=1)
        if np.any(norms == 0):
            raise ValueError(f"Test dataset {name!r} contains a zero-norm patient.")
        damage = (np.asarray(X[valid], dtype=float) @ final_map) / (norms * final_norm)
        observed = y[valid]
        rho = pvalue = np.nan
        if np.unique(damage).size > 1 and np.unique(observed).size > 1:
            result = spearmanr(damage, observed)
            rho = float(getattr(result, "statistic", result.correlation))
            pvalue = float(result.pvalue)
        summaries.append({
            "dataset": name,
            "n_patients": int(valid.sum()),
            "n_missing_outcomes": int((~valid).sum()),
            "test_spearman_rho": rho,
            "test_spearman_p": pvalue,
        })
        records.extend(
            {"dataset": name, "patient_id": patient_id,
             "observed": float(value), "damage": float(score)}
            for patient_id, value, score in zip(ids[valid], observed, damage)
        )
    return (
        pd.DataFrame.from_records(records),
        pd.DataFrame.from_records(summaries),
    )


def fitting_ids_from_datasets(*sources):
    """Collect every patient ID that fitting touched.

    Each source is either a ``{name: {'ids': ...}}`` dataset mapping or a plain
    array of patient IDs, so map-building and weight-fitting cohorts can both
    be passed in one call.
    """
    ids = set()
    for source in sources:
        if source is None or not len(source):
            continue
        groups = (source.values() if isinstance(source, dict) else [{"ids": source}])
        for data in groups:
            ids.update(np.asarray(data["ids"]).astype(str).reshape(-1).tolist())
    return ids


def save_test_evaluation(out_dir, predictions, summary):
    """Save the one-shot test scores beside the outer-validation results."""
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out_dir / "test_predictions.csv", index=False)
    summary.to_csv(out_dir / "test_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"Saved frozen-map test evaluation to: {out_dir}")
    return out_dir
