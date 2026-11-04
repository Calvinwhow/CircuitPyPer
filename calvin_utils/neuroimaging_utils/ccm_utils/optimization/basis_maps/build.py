"""Write one basis shard per registry cohort with regression_pipeline.py's table preparation and regression engine.

The registry holds one recipe for the whole library (``mask_path`` and ``transform``) and one entry per cohort. An entry
names its table (``input_path``), ``pathology`` and ``subject_column``, the symptoms (``regressand_list``) and the image
columns (``voxelwise_vars``, a tuple). For each image column the builder runs one multi-output regression,
``regressand_1 + regressand_2 + ... ~ image + covariates``, and every symptom becomes one basis: the t map of the image's
coefficient, named ``{regressand}-on-{image}``. Preparation and covariate settings left out of an entry take the defaults
of ``load_table`` and ``fit_bases``. Column names are the names after regression_pipeline.py's column cleaning.
A cohort is written once, as ``shard_{cohort}.npy`` plus a sidecar holding the fully resolved entry and what the build
measured; an existing shard whose sidecar no longer matches its entry stops the build.
"""

import os
import json
import inspect
import tempfile
import functools
import numpy as np
import nibabel as nib
from scipy import sparse
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from ..pipeline import basis_damage
from ..neural_network.anatomy import canonical_representation
from calvin_utils.permutation_analysis_utils.statsmodels_palm import CalvinStatsmodelsPalm
from calvin_utils.permutation_analysis_utils.voxelwise_regression import VoxelwiseRegression
from calvin_utils.permutation_analysis_utils.voxelwise_regression_prep import RegressionPrep
from calvin_utils.neuroimaging_utils.io.importers import GiiNiiFileImport
from calvin_utils.neuroimaging_utils.tract_utils.fiber_io import FiberIO
from calvin_utils.neuroimaging_utils.tract_utils.tract_density import TractDensity
from calvin_utils.neuroimaging_utils.tract_utils.fiber_converter import FiberFormatConverter

REGISTRY_PATH = Path(__file__).with_name("registry.json")
REQUIRED_KEYS = ("pathology", "input_path", "subject_column", "regressand_list", "voxelwise_vars")
METADATA_KEYS = ("representation",)


def load_table(input_path, regressand_list, sheet=None, drop_rows=(), one_hot=(), invert_regressand=False):
    """The cohort table after regression_pipeline.py's row preparation: column cleaning, inversion, row drops, one-hot. Rows missing any symptom, the image or a covariate are dropped in fit_bases."""
    palm = CalvinStatsmodelsPalm(input_csv_path=input_path, output_dir=None, sheet=sheet)
    table = palm.read_and_display_data()
    if invert_regressand:
        table[list(regressand_list)] = table[list(regressand_list)] * -1
    for column, condition, value in drop_rows:
        table, _ = palm.drop_rows_based_on_value(column, condition, value)
    for column in one_hot:
        table = table.join(pd.get_dummies(table[column], prefix=column, dtype=int))
    return palm, table.reset_index(drop=True)


def fit_bases(palm, table, voxelwise_var, regressand_list, mask_path, transform, covariates_list=(), add_intercept=False, weights_col=None, drop_nans=True):
    """Every symptom's t map for one image column, from one multi-output regression regressand_1 + ... ~ voxelwise_var + covariates.
    With drop_nans, a patient missing any symptom, the image or a covariate is dropped, so every symptom is fitted on the same patients.
    Returns {symptom: (t map, table row positions used)}.
    """
    required = [voxelwise_var, *regressand_list, *covariates_list, *([weights_col] if weights_col else [])]
    usable = table[required].notna().all(axis=1).to_numpy() if drop_nans else np.ones(len(table), dtype=bool)
    constant = [symptom for symptom in regressand_list if table.loc[usable, symptom].nunique() < 2]
    if usable.sum() < 4 or constant:
        raise ValueError(f"{int(usable.sum())} patients have {voxelwise_var} and every symptom; constant symptoms {constant}. Fill or remove the symptoms that leave too few complete rows.")
    data = table.loc[usable]
    formula = " + ".join(regressand_list) + f" ~ {voxelwise_var}" + "".join(f" + {covariate}" for covariate in covariates_list)
    outcome, design = palm.define_design_matrix(formula, data, add_intercept=add_intercept, voxelwise_variable_list=[voxelwise_var], voxelwise_interaction_terms=None)
    contrast = palm.finalize_contrast_matrix(design_matrix=design, contrast_matrix=np.eye(len(design.columns)))
    weights = None if weights_col is None else pd.to_numeric(data[weights_col], errors="raise").astype(float).to_numpy()
    with tempfile.TemporaryDirectory() as scratch:
        prepared = RegressionPrep(design_matrix=design, contrast_matrix=contrast, outcome_df=outcome, out_dir=scratch, voxelwise_variables=[voxelwise_var], mask_path=mask_path, data_transform_method=transform, weights=weights, formula=formula)
    _, T, _ = VoxelwiseRegression.from_arrays(prepared.design_tensor, prepared.outcome_data.astype(prepared.design_tensor.dtype), contrast.to_numpy(), prepared.weights_vector).run_single_multiout_regression()
    image_row = list(design.columns).index(voxelwise_var)
    return {symptom: (np.asarray(T[image_row, :, index], dtype=basis_damage.FEATURE_DTYPE), np.flatnonzero(usable)) for index, symptom in enumerate(outcome.columns)}


def is_fiber_column(table, voxelwise_var):
    """Whether an image column holds per-fiber values (*.fib.npy / *.fib.json) rather than volumes."""
    paths = table[voxelwise_var].dropna().astype(str)
    return GiiNiiFileImport(import_path=None, mask_path=None)._import_type_switch(paths[paths.str.lower() != "nan"].tolist()) == "fiber"


@functools.lru_cache(maxsize=4)
def fiber_raster(fiber_atlas, mask_path):
    """Sparse (library mask voxels, atlas fibers) matrix that rasterizes a per-fiber map exactly as regression_pipeline.py writes its fiber NIfTIs.
    That output is TractDensity over every fiber, with no threshold and no mirroring: each fiber adds its value once to every voxel holding one of its rounded vertices.
    """
    reference = nib.load(str(mask_path))
    shape = reference.shape[:3]
    mask = np.asarray(reference.dataobj).reshape(-1) > 0
    position = np.full(mask.size, -1, dtype=np.int64)
    position[mask] = np.arange(int(mask.sum()))
    inverse = np.linalg.inv(reference.affine)
    fibers = [FiberFormatConverter._as_streamline(fiber) for fiber in FiberFormatConverter._load_fibers(fiber_atlas)]
    rows, columns = [], []
    for index, fiber in enumerate(fibers):
        ijk = TractDensity._in_bounds_ijk(TractDensity._world_to_ijk(fiber, inverse), shape)
        voxels = np.unique(position[np.ravel_multi_index(ijk.T, shape)]) if len(ijk) else np.empty(0, dtype=np.int64)
        voxels = voxels[voxels >= 0]
        rows.append(voxels)
        columns.append(np.full(len(voxels), index, dtype=np.int64))
    rows, columns = np.concatenate(rows), np.concatenate(columns)
    return sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, columns)), shape=(int(mask.sum()), len(fibers)))


def voxelize_fibers(t_maps, fiber_atlas, mask_path):
    """Per-fiber t maps over the atlas's kept fibers, shape (kept fibers, maps), rasterized to the library mask, shape (voxels, maps)."""
    fiber_io = FiberIO(mask_path=fiber_atlas)
    kept = fiber_io.fiber_mask
    full = np.zeros((len(fiber_io.reference_fibers), t_maps.shape[1]), dtype=np.float32)
    full[kept if kept is not None else slice(None)] = t_maps
    return np.asarray(fiber_raster(str(fiber_atlas), str(mask_path)) @ full, dtype=basis_damage.FEATURE_DTYPE)


def optional_parameters(function):
    return {name: parameter.default for name, parameter in inspect.signature(function).parameters.items() if parameter.default is not inspect.Parameter.empty}


def settings(function, entry):
    return {name: entry[name] for name in optional_parameters(function) if name in entry}


OPTIONAL_KEYS = {**optional_parameters(load_table), **optional_parameters(fit_bases)}


def resolved(entry):
    """The entry with every unset setting filled from the defaults of load_table and fit_bases, in JSON form, so a sidecar records exactly what was run."""
    filled = {**{name: list(default) if isinstance(default, tuple) else default for name, default in OPTIONAL_KEYS.items()}, **entry}
    return {**filled, **{key: list(filled[key]) for key in ("regressand_list", "voxelwise_vars")}}


def read_registry(registry=REGISTRY_PATH):
    """Return a validated registry from a path or an already-parsed dict."""
    registry = registry if isinstance(registry, dict) else json.loads(Path(registry).read_text())
    missing = [key for key in ("mask_path", "transform", "cohorts") if key not in registry]
    unknown = sorted(set(registry) - {"mask_path", "fiber_atlas", "transform", "cohorts"})
    if unknown:
        raise ValueError(f"The registry has unknown keys {unknown}.")
    if missing:
        raise ValueError(f"The registry is missing {missing}.")
    for cohort, entry in registry["cohorts"].items():
        if not cohort or any(character in cohort for character in "/\\:"):
            raise ValueError(f"Cohort name {cohort!r} cannot be empty or contain '/', '\\' or ':'.")
        missing = [key for key in REQUIRED_KEYS if key not in entry]
        unknown = sorted(set(entry) - set(REQUIRED_KEYS) - set(OPTIONAL_KEYS) - set(METADATA_KEYS))
        if missing or unknown:
            raise ValueError(f"Cohort {cohort!r}: missing {missing}, unknown {unknown}.")
        for key in ("regressand_list", "voxelwise_vars"):
            if not entry[key] or len(set(entry[key])) != len(entry[key]):
                raise ValueError(f"Cohort {cohort!r}: {key} must be nonempty without repeats.")
        if set(entry["regressand_list"]) & set(entry["voxelwise_vars"]):
            raise ValueError(f"Cohort {cohort!r}: a column cannot be both a regressand and a voxelwise variable.")
        representation = entry.get("representation")
        if isinstance(representation, dict):
            if set(representation) != set(entry["voxelwise_vars"]):
                raise ValueError(f"Cohort {cohort!r}: representation mapping must define every voxelwise variable exactly once.")
            for value in representation.values():
                canonical_representation(value)
        elif representation is not None:
            canonical_representation(representation)
    return registry


def recipe(registry, cohort):
    """Everything the sidecar records from the registry for this cohort."""
    return {"cohort": cohort, **resolved(registry["cohorts"][cohort]), "mask_path": registry["mask_path"], "fiber_atlas": registry.get("fiber_atlas"), "transform": registry["transform"]}


def representation_for(entry, voxelwise_var):
    value = entry.get("representation")
    if isinstance(value, dict):
        value = value[voxelwise_var]
    return None if value is None else canonical_representation(value)


def save_atomically(path, write):
    """Write next to the destination, then rename, so an interrupted build never leaves a half-written shard."""
    partial = path.with_name(f".{path.name}.partial")
    with partial.open("wb") as stream:
        write(stream)
    os.replace(partial, path)


def build_cohort(registry, cohort, directory):
    """Fit every symptom on every image column of one cohort and write its shard and sidecar."""
    entry = recipe(registry, cohort)
    palm, table = load_table(entry["input_path"], entry["regressand_list"], **settings(load_table, entry))
    needed = [entry["subject_column"], *entry["regressand_list"], *entry["voxelwise_vars"], *entry["covariates_list"], *([entry["weights_col"]] if entry["weights_col"] else [])]
    absent = sorted(set(needed) - set(table.columns))
    if absent:
        raise ValueError(f"{cohort}: {entry['input_path']} has no columns {absent} after column cleaning.")
    subjects = table[entry["subject_column"]]
    columns, bases, patients = [], [], set()
    for voxelwise_var in entry["voxelwise_vars"]:
        fibers = is_fiber_column(table, voxelwise_var)
        if fibers and not entry["fiber_atlas"]:
            raise ValueError(f"{cohort}: {voxelwise_var!r} holds fiber files; set the registry's fiber_atlas to the atlas their values index.")
        representation = representation_for(entry, voxelwise_var)
        print(f"shard_{cohort}: {len(entry['regressand_list'])} symptoms ~ {voxelwise_var} ({'fiber' if fibers else 'volume'} files, representation {representation})", flush=True)
        maps = fit_bases(palm, table, voxelwise_var, entry["regressand_list"], entry["fiber_atlas"] if fibers else entry["mask_path"], entry["transform"], **settings(fit_bases, entry))
        print(f"shard_{cohort}: {len(maps)} {voxelwise_var} bases fitted on {len(next(iter(maps.values()))[1])} patients", flush=True)
        if fibers:
            voxelized = voxelize_fibers(np.column_stack([t_map for t_map, _ in maps.values()]), entry["fiber_atlas"], entry["mask_path"])
            maps = {symptom: (voxelized[:, index], rows) for index, (symptom, (_, rows)) in enumerate(maps.items())}
        for symptom in maps:
            t_map, rows = maps[symptom]
            ids = subjects.iloc[rows]
            if ids.isna().any() or ids.astype(str).duplicated().any():
                raise ValueError(f"{cohort}: {symptom}-on-{voxelwise_var} uses rows with a missing or repeated {entry['subject_column']!r}.")
            columns.append(t_map)
            basis = {"name": f"{symptom}-on-{voxelwise_var}", "regressand": symptom, "voxelwise_var": voxelwise_var, "n": len(rows)}
            if representation is not None:
                basis["representation"] = representation
            bases.append(basis)
            patients.update(ids.astype(str))
    lengths = sorted({len(column) for column in columns})
    if len(lengths) != 1:
        raise ValueError(f"{cohort}: the fitted maps have {lengths} voxels; every voxelwise variable must share the mask.")
    values = np.stack(columns, axis=1)
    sidecar = {**entry, "patient_ids": sorted(patients), "bases": bases, "n_voxels": int(values.shape[0]), "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    path = Path(directory) / f"shard_{cohort}.npy"
    save_atomically(path, lambda stream: np.save(stream, values))
    save_atomically(path.with_suffix(".json"), lambda stream: stream.write((json.dumps(sidecar, indent=2) + "\n").encode()))
    print(f"shard_{cohort}: wrote {values.shape[1]} bases ({', '.join(entry['voxelwise_vars'])}) x {values.shape[0]} voxels to {path}", flush=True)
    return sidecar


def check_built(registry, cohort, directory):
    """Refuse an existing shard whose sidecar was built from a different registry entry."""
    sidecar = json.loads((Path(directory) / f"shard_{cohort}.json").read_text())
    expected = recipe(registry, cohort)
    changed = sorted(key for key in expected if sidecar.get(key) != expected[key])
    if changed:
        raise ValueError(f"shard_{cohort} was built from a different registry entry ({changed} differ). Delete shard_{cohort}.npy and shard_{cohort}.json to rebuild it.")


def override(directory):
    """Remove every basis shard from one basis directory before rebuilding it."""
    directory = Path(directory)
    for path in (*directory.glob("shard_*.npy"), *directory.glob("shard_*.json")):
        path.unlink()


def main(registry=REGISTRY_PATH, directory=basis_damage.BASIS_DIR, cohorts=None, overwrite=False):
    """Build selected registry cohorts, optionally replacing the entire existing basis library."""
    registry = read_registry(registry)
    selected = tuple(registry["cohorts"]) if cohorts is None else tuple(cohorts)
    unknown = sorted(set(selected) - set(registry["cohorts"]))
    if unknown:
        raise ValueError(f"Cohorts {unknown} are not in the registry.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if overwrite:
        override(directory)
    built = []
    for cohort in selected:
        if (directory / f"shard_{cohort}.npy").is_file() and (directory / f"shard_{cohort}.json").is_file():
            check_built(registry, cohort, directory)
            continue
        print(f"Building shard_{cohort}")
        build_cohort(registry, cohort, directory)
        built.append(cohort)
    if selected:
        basis_damage.load_library(selected, directory)
    return built


if __name__ == "__main__":
    print(f"Built {main()}")
