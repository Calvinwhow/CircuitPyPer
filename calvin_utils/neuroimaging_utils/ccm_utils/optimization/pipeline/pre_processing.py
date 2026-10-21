"""All data preparation used by the optimization dispatcher."""

import numpy as np
import pandas as pd
from pathlib import Path

try:
    from calvin_utils.neuroimaging_utils.io.importers import GiiNiiFileImport
    from calvin_utils.neuroimaging_utils.ccm_utils.npy_utils import DataLoader
    from calvin_utils.permutation_analysis_utils.voxelwise_regression import VoxelwiseRegression
    from calvin_utils.permutation_analysis_utils.voxelwise_regression_prep import RegressionPrep
    from calvin_utils.permutation_analysis_utils.map_damage_cv import load_native_patient_vectors
except ModuleNotFoundError:
    DataLoader = None
    RegressionPrep = None
    GiiNiiFileImport = None
    VoxelwiseRegression = None
    load_native_patient_vectors = None

FEATURE_DTYPE = np.float32  # every array with a voxel or fiber axis; patient-by-basis arrays stay float64


def validate_datasets(datasets):
    """Return aligned finite patient maps, outcomes, and optional IDs; the voxel count is checked against the shards in basis_damage."""
    if not datasets:
        raise ValueError("At least one fitting dataset is required.")
    validated = {}
    for name, data in datasets.items():
        patients = np.asarray(data["X"], dtype=FEATURE_DTYPE)
        outcomes = np.asarray(data["y"], dtype=float).reshape(-1)
        ids = np.asarray(data.get("ids", np.arange(len(patients)))).astype(str)
        valid = np.isfinite(outcomes)
        patients, outcomes, ids = patients[valid], outcomes[valid], ids[valid]
        if patients.ndim != 2:
            raise ValueError(f"Dataset {name!r} must be a patient-by-voxel matrix.")
        if len(patients) < 3 or len(outcomes) != len(patients):
            raise ValueError(f"Dataset {name!r} needs three aligned observations.")
        if not np.isfinite(patients).all() or np.any(np.linalg.norm(patients, axis=1) == 0):
            raise ValueError(f"Dataset {name!r} contains an invalid patient map.")
        if np.unique(outcomes).size < 2:
            raise ValueError(f"Dataset {name!r} has a constant outcome.")
        validated[name] = {"X": patients, "y": outcomes, "ids": ids}
    return validated


def dataset_weights(names, supplied=None):
    """Return dict[str, float] of length D with dataset-name keys and scalar contributions normalized to sum to one."""
    names = tuple(names)
    values = np.ones(len(names), dtype=float) if supplied is None else np.asarray([supplied[name] for name in names], dtype=float)
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
    if GiiNiiFileImport is None or load_native_patient_vectors is None:
        raise ModuleNotFoundError("Patient-image loading requires the neuroimaging I/O dependencies.")
    paths = table[image_column].tolist()
    detected = GiiNiiFileImport(import_path=None, mask_path=mask_path)._import_type_switch(paths)
    if detected == "npy":
        raise ValueError("Use individual patient images, not a cohort .npy matrix.")
    native_type = output_type or {"gii": "surface", "freesurfer": "surface"}.get(detected, detected)
    patients = load_native_patient_vectors(paths, mask_path, native_type)
    if patients.ndim != 2 or patients.shape[0] != len(table):
        raise ValueError("Imported patient images must be patient-by-feature.")
    return patients


def datasets_from_loader(loader, *, data_mode="memmap", patient_ids=None):
    """Load fitting arrays and attach stable patient IDs once."""
    names = tuple(loader.dataset_names_list)
    paths = getattr(loader, "dataset_paths_dict", {})
    image_paths = [paths.get(name, {}).get("niftis") for name in names]
    shared_rows = len(names) == 1 or (all(image_paths) and len(set(image_paths)) == 1)
    datasets = {}
    for name in names:
        data = loader.load_dataset(name, mmap_mode="r" if data_mode == "memmap" else None)
        patients = data["niftis"]
        outcomes = np.asarray(data["indep_var"], dtype=float).reshape(-1)
        configured = None if patient_ids is None else patient_ids.get(name)
        ids_path = paths.get(name, {}).get("ids")
        if configured is not None:
            ids = np.asarray(configured).astype(str)
        elif ids_path:
            ids = np.load(ids_path, allow_pickle=False).astype(str)
        elif shared_rows:
            ids = np.arange(len(patients)).astype(str)
        else:
            raise ValueError("Multiple datasets with different rows require patient IDs.")
        datasets[name] = {"X": patients, "y": outcomes, "ids": ids}
    return datasets


def build_basis_maps(patient_maps, outcomes, train_rows=None, *, transform="rank", skip_invalid=False):
    """Fit one symptom-localization map per outcome using selected patients."""
    if RegressionPrep is None or VoxelwiseRegression is None:
        raise ModuleNotFoundError("Basis-map construction requires the regression dependencies.")
    patient_maps = np.asarray(patient_maps, dtype=FEATURE_DTYPE)
    train_rows = np.ones(len(patient_maps), dtype=bool) if train_rows is None else np.asarray(train_rows, dtype=bool)
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


def make_dataset_loader(out_dir, dataset_specs=None, manifest_path=None, mask_path=None, subject_col=None, data_transform_method=None):
    if DataLoader is None:
        raise ModuleNotFoundError("Dataset loading requires the neuroimaging I/O dependencies.")
    if bool(dataset_specs) == bool(manifest_path):
        raise ValueError("Set a CSV dataset dictionary or a manifest path, not both.")
    return DataLoader.from_csv_dict(dataset_specs, out_dir, mask_path=mask_path, subject_col=subject_col, data_transform_method=data_transform_method) if dataset_specs else DataLoader(Path(manifest_path).expanduser())


def optional_datasets(out_dir, dataset_specs=None, manifest_path=None, mask_path=None, subject_col=None, data_mode="memmap"):
    if not dataset_specs and not manifest_path:
        return None
    loader = make_dataset_loader(out_dir, dataset_specs=dataset_specs, manifest_path=manifest_path, mask_path=mask_path, subject_col=subject_col)
    return datasets_from_loader(loader, data_mode=data_mode)


def load_target_cohorts(dataset_specs, targets, mask_path, subject_col=None):
    """Load each CSV cohort once with exact target columns."""
    cohorts = {}
    for name, spec in dataset_specs.items():
        table_path = Path(spec["csv_path"]).expanduser().resolve()
        sheet = spec.get("sheet")
        table = pd.read_excel(table_path, sheet_name=0 if sheet is None else sheet) if table_path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(table_path)
        image_column = spec["nifti_column"]
        id_column = spec.get("subject_column", subject_col)
        missing = ({image_column, *targets} | ({id_column} if id_column else set())) - set(table.columns)
        if missing:
            raise ValueError(f"{table_path} is missing columns: {sorted(missing)}")
        outcomes = {target: pd.to_numeric(table[target], errors="coerce") for target in targets}
        invalid = [target for target in targets if (table[target].notna() & outcomes[target].isna()).any()]
        if invalid:
            raise ValueError(f"{table_path} contains nonnumeric values in {invalid}.")
        valid = table[image_column].notna() if id_column is None else table[image_column].notna() & table[id_column].notna()
        rows = table.loc[valid].copy()
        paths = [Path(str(value)).expanduser() for value in rows[image_column]]
        rows[image_column] = [str((path if path.is_absolute() else table_path.parent / path).resolve()) for path in paths]
        ids = rows.index.astype(str) if id_column is None else rows[id_column].astype(str).str.strip()
        if ids.duplicated().any() or (ids == "").any():
            raise ValueError(f"Dataset {name!r} needs unique nonempty patient IDs.")
        cohorts[name] = {"X": load_patient_images(rows, image_column, mask_path), "ids": np.asarray([f"{name}:{value}" for value in ids]), "outcomes": {target: values.loc[valid].to_numpy(float) for target, values in outcomes.items()}}
    return cohorts


def target_datasets(cohorts, target):
    return {name: {"X": cohort["X"], "y": cohort["outcomes"][target], "ids": cohort["ids"]} for name, cohort in cohorts.items()}


def main(datasets):
    """Validate fitting datasets before basis scoring.
    datasets: dict[str, dict] of length D; each entry contains patient maps X with shape (N, V), outcomes y with shape (N,), and optional IDs with shape (N,).
    """
    return {"datasets": validate_datasets(datasets)}
