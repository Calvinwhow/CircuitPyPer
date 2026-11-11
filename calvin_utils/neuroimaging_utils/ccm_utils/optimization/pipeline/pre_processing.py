"""All data preparation used by the optimization dispatcher."""

import numpy as np
import pandas as pd
from pathlib import Path
from .damage_matrix_utils.basis_library import FEATURE_DTYPE
from .damage_matrix_utils.damage_matrix_schema import CohortData

try:
    from calvin_utils.neuroimaging_utils.io.importers import GiiNiiFileImport
    from calvin_utils.permutation_analysis_utils.voxelwise_regression import VoxelwiseRegression
    from calvin_utils.permutation_analysis_utils.voxelwise_regression_prep import RegressionPrep
except ModuleNotFoundError:
    RegressionPrep = None
    GiiNiiFileImport = None
    VoxelwiseRegression = None

def dataset_weights(names, supplied=None):
    """Return dict[str, float] of length D with dataset-name keys and scalar contributions normalized to sum to one."""
    names = tuple(names)
    if supplied is None:
        values = np.ones(len(names), dtype=float)
    else:
        values = np.asarray([supplied[name] for name in names], dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0) or values.sum() == 0:
        raise ValueError("Dataset weights must be finite, nonnegative, and nonzero.")
    values /= values.sum()
    return dict(zip(names, values))


def initial_weights(n_maps, mode="unweighted", *, sample_sizes=None, names=None, seed=None):
    """Create one unnormalized starting vector."""
    if mode == "unweighted":
        weights = np.ones(n_maps)
    elif mode == "gaussian":
        weights = np.random.default_rng(seed).normal(0.1, 1.0, n_maps)
    elif mode == "weighted":
        weights = np.asarray([sample_sizes[name] for name in names], dtype=float)
    else:
        raise ValueError("Weight mode must be 'unweighted', 'gaussian', or 'weighted'.")
    return weights


def load_patient_images(table, image_column, mask_path, output_type=None):
    """Load one patient-by-feature matrix in the maps' native space."""
    if GiiNiiFileImport is None:
        raise ModuleNotFoundError("Patient-image loading requires the neuroimaging I/O dependencies.")
    from calvin_utils.permutation_analysis_utils.map_damage_cv import load_native_patient_vectors
    paths = table[image_column].tolist()
    detected = GiiNiiFileImport(import_path=None, mask_path=mask_path)._import_type_switch(paths)
    if detected == "npy":
        raise ValueError("Use individual patient images, not a cohort .npy matrix.")
    native_type = output_type or {"gii": "surface", "freesurfer": "surface"}.get(detected, detected)
    patients = load_native_patient_vectors(paths, mask_path, native_type)
    if patients.ndim != 2 or patients.shape[0] != len(table):
        raise ValueError("Imported patient images must be patient-by-feature.")
    return patients


def build_basis_maps(patient_maps, outcomes, train_rows=None, *, transform="rank", skip_invalid=False):
    """Fit one symptom-localization map per outcome using selected patients."""
    if RegressionPrep is None or VoxelwiseRegression is None:
        raise ModuleNotFoundError("Basis-map construction requires the regression dependencies.")
    patient_maps = np.asarray(patient_maps, dtype=FEATURE_DTYPE)
    if train_rows is None:
        train_rows = np.ones(len(patient_maps), dtype=bool)
    else:
        train_rows = np.asarray(train_rows, dtype=bool)
    groups, sample_sizes = {}, {}
    for name, values in outcomes.items():
        values = np.asarray(values, dtype=float).reshape(-1)
        valid = train_rows & np.isfinite(values)
        if valid.sum() < 4 or np.unique(values[valid]).size < 2:
            if skip_invalid:
                continue
            raise ValueError(f"Map outcome {name!r} cannot be fitted in this split.")
        groups.setdefault(valid.tobytes(), (valid, []))[1].append((name, values))
        sample_sizes[name] = int(valid.sum())
    maps = {}
    for valid, targets in groups.values():
        images = RegressionPrep.transform_array(patient_maps[valid, None, :], transform)[:, 0, :]
        for name, values in targets:
            symptom = RegressionPrep.transform_array(values[valid, None, None], transform, keep_categorical=True)[:, 0, 0]
            maps[name] = VoxelwiseRegression.fit_linear_tmap(symptom, images, contrast=[[1.0]])[0]
    return maps, sample_sizes


def load_target_cohorts(dataset_specs, targets, mask_path, subject_col=None):
    """Return typed cohorts with (N,V) maps and (N,T) outcomes."""
    cohorts = []
    for name, spec in dataset_specs.items():
        table_path = Path(spec["csv_path"]).expanduser().resolve()
        table = _read_clinical_table(table_path, spec.get("sheet"))
        image_column = spec["nifti_column"]
        id_column = spec.get("subject_column", subject_col)
        required_columns = {image_column, *targets}
        if id_column:
            required_columns.add(id_column)
        missing = required_columns - set(table.columns)
        if missing:
            raise ValueError(f"{table_path} is missing columns: {sorted(missing)}")
        outcomes = {target: pd.to_numeric(table[target], errors="coerce") for target in targets}
        invalid = [target for target in targets if (table[target].notna() & outcomes[target].isna()).any()]
        if invalid:
            raise ValueError(f"{table_path} contains nonnumeric values in {invalid}.")
        valid = _valid_patient_rows(table, image_column, id_column)
        rows = table.loc[valid].copy()
        paths = [Path(str(value)).expanduser() for value in rows[image_column]]
        rows[image_column] = [str(_resolve_image_path(path, table_path.parent)) for path in paths]
        ids = _patient_ids(rows, id_column)
        if ids.duplicated().any() or (ids == "").any():
            raise ValueError(f"Dataset {name!r} needs unique nonempty patient IDs.")
        cohorts.append(CohortData(name, load_patient_images(rows, image_column, mask_path), np.asarray([f"{name}:{value}" for value in ids]), tuple(targets), np.column_stack([outcomes[target].loc[valid] for target in targets]).astype(float)))
    return tuple(cohorts)


def _read_clinical_table(table_path, sheet):
    """Resolve spreadsheet and CSV loading in one visible switch."""
    if table_path.suffix.lower() not in {".xlsx", ".xls"}:
        return pd.read_csv(table_path)
    if sheet is None:
        sheet = 0
    return pd.read_excel(table_path, sheet_name=sheet)


def _valid_patient_rows(table, image_column, id_column):
    """Require an image path and, when configured, a patient identifier."""
    valid = table[image_column].notna()
    if id_column is not None:
        valid &= table[id_column].notna()
    return valid


def _resolve_image_path(path, table_directory):
    """Resolve relative image paths against the clinical table's directory."""
    if not path.is_absolute():
        path = table_directory / path
    return path.resolve()


def _patient_ids(rows, id_column):
    """Resolve configured IDs or stable table-row IDs."""
    if id_column is None:
        return rows.index.astype(str)
    return rows[id_column].astype(str).str.strip()
