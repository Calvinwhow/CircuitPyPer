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
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from ..pipeline import basis_damage
from calvin_utils.permutation_analysis_utils.statsmodels_palm import CalvinStatsmodelsPalm
from calvin_utils.permutation_analysis_utils.voxelwise_regression import VoxelwiseRegression
from calvin_utils.permutation_analysis_utils.voxelwise_regression_prep import RegressionPrep

REGISTRY_PATH = Path(__file__).with_name("registry.json")
REQUIRED_KEYS = ("pathology", "input_path", "subject_column", "regressand_list", "voxelwise_vars")


def load_table(input_path, regressand_list, sheet=None, drop_rows=(), one_hot=(), invert_regressand=False):
    """The cohort table after regression_pipeline.py's row preparation: column cleaning, inversion, row drops, one-hot. NaNs are dropped per regression in fit_bases."""
    palm = CalvinStatsmodelsPalm(input_csv_path=input_path, output_dir=None, sheet=sheet)
    table = palm.read_and_display_data()
    if invert_regressand:
        table[list(regressand_list)] = table[list(regressand_list)] * -1
    for column, condition, value in drop_rows:
        table, _ = palm.drop_rows_based_on_value(column, condition, value)
    for column in one_hot:
        table = table.join(pd.get_dummies(table[column], prefix=column, dtype=int))
    return palm, table.reset_index(drop=True)


def outcome_groups(table, regressand_list, required, drop_nans):
    """Symptoms that use the same rows; each group is one multi-output regression, so no symptom loses rows to another's NaNs."""
    if not drop_nans:
        return [(np.ones(len(table), dtype=bool), list(regressand_list))]
    shared = table[required].notna().all(axis=1).to_numpy()
    groups = {}
    for symptom in regressand_list:
        rows = shared & table[symptom].notna().to_numpy()
        if rows.sum() < 4 or table.loc[rows, symptom].nunique() < 2:
            raise ValueError(f"{symptom!r} has {int(rows.sum())} usable rows or a constant value; it cannot be fitted.")
        groups.setdefault(rows.tobytes(), (rows, []))[1].append(symptom)
    return list(groups.values())


def fit_bases(palm, table, voxelwise_var, regressand_list, mask_path, transform, covariates_list=(), add_intercept=False, weights_col=None, drop_nans=True):
    """Every symptom's t map for one image column, from multi-output regressions regressand_1 + ... ~ voxelwise_var + covariates.

    The images are read once, untransformed; each group of symptoms sharing rows is then transformed on its own rows and
    fitted as one multi-output regression, which reuses the per-voxel design inverse across that group's symptoms.
    Returns {symptom: (t map, row positions used)}.
    """
    groups = outcome_groups(table, regressand_list, [voxelwise_var, *covariates_list, *([weights_col] if weights_col else [])], drop_nans)
    usable = np.logical_or.reduce([rows for rows, _ in groups])
    data = table.loc[usable]
    formula = " + ".join(regressand_list) + f" ~ {voxelwise_var}" + "".join(f" + {covariate}" for covariate in covariates_list)
    outcome, design = palm.define_design_matrix(formula, data, add_intercept=add_intercept, voxelwise_variable_list=[voxelwise_var], voxelwise_interaction_terms=None)
    contrast = palm.finalize_contrast_matrix(design_matrix=design, contrast_matrix=np.eye(len(design.columns)))
    weights = None if weights_col is None else pd.to_numeric(data[weights_col], errors="raise").astype(float).to_numpy()
    with tempfile.TemporaryDirectory() as scratch:
        raw = RegressionPrep(design_matrix=design, contrast_matrix=contrast, outcome_df=outcome, out_dir=scratch, voxelwise_variables=[voxelwise_var], mask_path=mask_path, data_transform_method=None, weights=weights, formula=formula)
    image_row = list(design.columns).index(voxelwise_var)
    outcome_column = {symptom: index for index, symptom in enumerate(outcome.columns)}
    maps = {}
    for rows, symptoms in groups:
        keep = rows[usable]
        X = RegressionPrep.transform_array(raw.design_tensor[keep], transform, keep_categorical=True)
        Y = RegressionPrep.transform_array(raw.outcome_data[keep][:, [outcome_column[symptom] for symptom in symptoms], :], transform).astype(X.dtype)
        W = raw.weights_vector[keep] / raw.weights_vector[keep].sum()
        _, T, _ = VoxelwiseRegression.from_arrays(X, Y, contrast.to_numpy(), W).run_single_multiout_regression()
        maps.update({symptom: (np.asarray(T[image_row, :, index], dtype=basis_damage.FEATURE_DTYPE), np.flatnonzero(rows)) for index, symptom in enumerate(symptoms)})
    return maps


def optional_parameters(function):
    return {name: parameter.default for name, parameter in inspect.signature(function).parameters.items() if parameter.default is not inspect.Parameter.empty}


def settings(function, entry):
    return {name: entry[name] for name in optional_parameters(function) if name in entry}


OPTIONAL_KEYS = {**optional_parameters(load_table), **optional_parameters(fit_bases)}


def resolved(entry):
    """The entry with every unset setting filled from the defaults of load_table and fit_bases, in JSON form, so a sidecar records exactly what was run."""
    filled = {**{name: list(default) if isinstance(default, tuple) else default for name, default in OPTIONAL_KEYS.items()}, **entry}
    return {**filled, "voxelwise_vars": list(filled["voxelwise_vars"])}


def read_registry(registry=REGISTRY_PATH):
    """Return a validated registry from a path or an already-parsed dict."""
    registry = registry if isinstance(registry, dict) else json.loads(Path(registry).read_text())
    missing = [key for key in ("mask_path", "transform", "cohorts") if key not in registry]
    if missing:
        raise ValueError(f"The registry is missing {missing}.")
    for cohort, entry in registry["cohorts"].items():
        if not cohort or any(character in cohort for character in "/\\:"):
            raise ValueError(f"Cohort name {cohort!r} cannot be empty or contain '/', '\\' or ':'.")
        missing = [key for key in REQUIRED_KEYS if key not in entry]
        unknown = sorted(set(entry) - set(REQUIRED_KEYS) - set(OPTIONAL_KEYS))
        if missing or unknown:
            raise ValueError(f"Cohort {cohort!r}: missing {missing}, unknown {unknown}.")
        for key in ("regressand_list", "voxelwise_vars"):
            if not entry[key] or len(set(entry[key])) != len(entry[key]):
                raise ValueError(f"Cohort {cohort!r}: {key} must be nonempty without repeats.")
        if set(entry["regressand_list"]) & set(entry["voxelwise_vars"]):
            raise ValueError(f"Cohort {cohort!r}: a column cannot be both a regressand and a voxelwise variable.")
    return registry


def recipe(registry, cohort):
    """Everything the sidecar records from the registry for this cohort."""
    return {"cohort": cohort, **resolved(registry["cohorts"][cohort]), "mask_path": registry["mask_path"], "transform": registry["transform"]}


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
        maps = fit_bases(palm, table, voxelwise_var, entry["regressand_list"], entry["mask_path"], entry["transform"], **settings(fit_bases, entry))
        for symptom in entry["regressand_list"]:
            t_map, rows = maps[symptom]
            ids = subjects.iloc[rows]
            if ids.isna().any() or ids.astype(str).duplicated().any():
                raise ValueError(f"{cohort}: {symptom}-on-{voxelwise_var} uses rows with a missing or repeated {entry['subject_column']!r}.")
            columns.append(t_map)
            bases.append({"name": f"{symptom}-on-{voxelwise_var}", "regressand": symptom, "voxelwise_var": voxelwise_var, "n": len(rows)})
            patients.update(ids.astype(str))
    lengths = sorted({len(column) for column in columns})
    if len(lengths) != 1:
        raise ValueError(f"{cohort}: the fitted maps have {lengths} voxels; every voxelwise variable must share the mask.")
    values = np.stack(columns, axis=1)
    sidecar = {**entry, "patient_ids": sorted(patients), "bases": bases, "n_voxels": int(values.shape[0]), "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    path = Path(directory) / f"shard_{cohort}.npy"
    save_atomically(path, lambda stream: np.save(stream, values))
    save_atomically(path.with_suffix(".json"), lambda stream: stream.write((json.dumps(sidecar, indent=2) + "\n").encode()))
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
