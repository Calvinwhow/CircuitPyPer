"""The registry-driven basis builder fits regressand_1 + ... ~ image + covariates once per image and writes loadable shards."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from calvin_utils.neuroimaging_utils.ccm_utils.optimization.basis_maps import build
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.pipeline import basis_damage, pre_processing
from calvin_utils.permutation_analysis_utils.voxelwise_regression import VoxelwiseRegression
from calvin_utils.permutation_analysis_utils.voxelwise_regression_prep import RegressionPrep


def toy_registry(tmp_path, n_patients=16):
    rng = np.random.default_rng(2)
    mask = np.zeros((3, 3, 3), dtype=np.float32)
    mask[:, :, 1:] = 1
    nib.save(nib.Nifti1Image(mask, np.eye(4)), tmp_path / "mask.nii.gz")
    rows = []
    for index in range(n_patients):
        lesion = rng.normal(size=(3, 3, 3)).astype(np.float32)
        network = rng.normal(size=(3, 3, 3)).astype(np.float32)
        nib.save(nib.Nifti1Image(lesion, np.eye(4)), tmp_path / f"lesion-{index:02d}.nii.gz")
        nib.save(nib.Nifti1Image(network, np.eye(4)), tmp_path / f"network-{index:02d}.nii.gz")
        rows.append({"subject": f"sub-{index:02d}", "lesion": str(tmp_path / f"lesion-{index:02d}.nii.gz"), "network": str(tmp_path / f"network-{index:02d}.nii.gz"), "gait": float(lesion[0, 0, 1] + rng.normal()), "speech": np.nan if index == 3 else float(network[1, 1, 2] + rng.normal()), "ataxia": float(rng.normal()), "age": 40.0 + index, "site": "AB"[index % 2], "selected": 0 if index == 5 else 1})
    pd.DataFrame(rows).to_csv(tmp_path / "table.csv", index=False)
    entry = {"pathology": "Stroke", "input_path": str(tmp_path / "table.csv"), "subject_column": "subject", "regressand_list": ["gait", "speech", "ataxia"], "voxelwise_vars": ("lesion", "network"), "drop_rows": [["selected", "equal", 0]]}
    return {"mask_path": str(tmp_path / "mask.nii.gz"), "transform": "rank", "cohorts": {"toy": entry}}


def single_output_t_map(tmp_path, registry, table, symptom, image, rows):
    data = table.iloc[rows]
    palm = build.CalvinStatsmodelsPalm(input_csv_path=None, output_dir=None)
    outcome, design = palm.define_design_matrix(f"{symptom} ~ {image}", data, add_intercept=False, voxelwise_variable_list=[image], voxelwise_interaction_terms=None)
    prep = RegressionPrep(design_matrix=design, contrast_matrix=np.eye(1), outcome_df=outcome, out_dir=str(tmp_path / "single"), voxelwise_variables=[image], mask_path=registry["mask_path"], data_transform_method="rank")
    _, t_values, _ = VoxelwiseRegression.from_arrays(prep.design_tensor, prep.outcome_data, np.eye(1), prep.weights_vector).voxelwise_regression()
    return t_values[0]


def test_builder_writes_every_symptom_on_every_image_as_one_loadable_shard(tmp_path):
    registry = toy_registry(tmp_path)
    directory = tmp_path / "basis_maps"
    assert build.main(registry, directory) == ["toy"]

    library = basis_damage.load_library(directory=directory)
    shard = library["shards"][0]
    assert shard["values"].dtype == np.float32 and shard["values"].shape == (18, 6)
    assert library["names"] == tuple(f"toy:{symptom}-on-{image}" for image in ("lesion", "network") for symptom in ("gait", "speech", "ataxia"))
    assert set(library["sample_sizes"].values()) == {14}
    sidecar = shard["sidecar"]
    assert sidecar["voxelwise_vars"] == ["lesion", "network"] and sidecar["drop_nans"] is True and sidecar["covariates_list"] == []
    assert sidecar["pathology"] == "Stroke" and "sub-05" not in sidecar["patient_ids"] and "sub-03" not in sidecar["patient_ids"] and len(sidecar["patient_ids"]) == 14


def test_each_symptom_map_equals_its_single_output_regression_on_the_complete_rows(tmp_path):
    registry = toy_registry(tmp_path)
    entry = build.recipe(registry, "toy")
    palm, table = build.load_table(entry["input_path"], entry["regressand_list"], drop_rows=entry["drop_rows"])
    for image in entry["voxelwise_vars"]:
        maps = build.fit_bases(palm, table, image, entry["regressand_list"], registry["mask_path"], registry["transform"])
        assert all(len(rows) == 14 for _, rows in maps.values())
        for symptom, (t_map, rows) in maps.items():
            reference = single_output_t_map(tmp_path, registry, table, symptom, image, rows)
            np.testing.assert_allclose(t_map, reference, rtol=2e-4, atol=1e-5)


def test_covariates_intercept_and_one_hot_covariates_enter_the_right_hand_side(tmp_path):
    registry = toy_registry(tmp_path)
    plain = tmp_path / "plain"
    build.main(registry, plain)
    registry["cohorts"]["toy"].update({"covariates_list": ["age", "site_B"], "one_hot": ["site"], "add_intercept": True})
    adjusted = tmp_path / "adjusted"
    build.main(registry, adjusted)

    first = basis_damage.load_library(directory=plain)["shards"][0]
    second = basis_damage.load_library(directory=adjusted)["shards"][0]
    assert second["sidecar"]["covariates_list"] == ["age", "site_B"] and second["sidecar"]["add_intercept"] is True
    assert not np.allclose(first["values"], second["values"])


def test_builder_skips_built_cohorts_and_refuses_a_changed_entry(tmp_path):
    registry = toy_registry(tmp_path)
    directory = tmp_path / "basis_maps"
    build.main(registry, directory)
    assert build.main(registry, directory) == []

    registry["cohorts"]["toy"]["drop_rows"] = []
    with pytest.raises(ValueError, match="drop_rows"):
        build.main(registry, directory)


def test_override_replaces_the_basis_library(tmp_path):
    registry = toy_registry(tmp_path)
    directory = tmp_path / "basis_maps"
    build.main(registry, directory)
    np.save(directory / "shard_extra.npy", np.ones((2, 2), dtype=np.float32))
    (directory / "shard_extra.json").write_text("{}")
    registry["cohorts"]["toy"]["drop_rows"] = []

    assert build.main(registry, directory, overwrite=True) == ["toy"]
    assert not (directory / "shard_extra.npy").exists()
    assert not (directory / "shard_extra.json").exists()
    assert basis_damage.load_library(directory=directory)["shards"][0]["sidecar"]["drop_rows"] == []


def test_registry_entries_are_checked_for_missing_unknown_and_conflated_keys(tmp_path):
    registry = toy_registry(tmp_path)
    del registry["cohorts"]["toy"]["pathology"]
    with pytest.raises(ValueError, match="pathology"):
        build.read_registry(registry)
    registry = toy_registry(tmp_path)
    registry["cohorts"]["toy"]["regressor_list"] = ["gait"]
    with pytest.raises(ValueError, match="regressor_list"):
        build.read_registry(registry)
    registry = toy_registry(tmp_path)
    registry["cohorts"]["toy"]["regressand_list"] = ["gait", "lesion"]
    with pytest.raises(ValueError, match="both a regressand and a voxelwise"):
        build.read_registry(registry)


def test_repository_registry_is_valid():
    registry = build.read_registry()
    assert json.loads(build.REGISTRY_PATH.read_text()) == registry


def toy_fitting_dataset(tmp_path, registry):
    table = pd.read_csv(tmp_path / "table.csv")
    table = table[table["selected"] != 0].reset_index(drop=True)
    patients = pre_processing.load_patient_images(table, "lesion", registry["mask_path"])
    return {"X": patients, "y": table["gait"].to_numpy(), "ids": table["subject"].to_numpy()}


def test_a_cohort_refuses_a_symptom_its_complete_rows_leave_constant(tmp_path):
    registry = toy_registry(tmp_path)
    table = pd.read_csv(tmp_path / "table.csv")
    table["rare"] = 0.0
    table.to_csv(tmp_path / "table.csv", index=False)
    entry = registry["cohorts"]["toy"]
    registry = {**registry, "cohorts": {"toy": {**entry, "regressand_list": [*entry["regressand_list"], "rare"]}}}
    with pytest.raises(ValueError, match="constant symptoms \\['rare'\\]"):
        build.build_cohort(registry, "toy", tmp_path)


def test_test_cohorts_cannot_have_built_the_bases(tmp_path):
    from calvin_utils.neuroimaging_utils.ccm_utils.optimization.run_optimization import RunOptimization

    registry = toy_registry(tmp_path)
    directory = tmp_path / "basis_maps"
    build.main(registry, directory)
    fitting = toy_fitting_dataset(tmp_path, registry)
    with pytest.raises(ValueError, match="built selected basis cohorts"):
        RunOptimization({"elsewhere": {**fitting, "ids": np.asarray([f"elsewhere-{index}" for index in range(len(fitting["y"]))])}}, basis_dir=directory, methods=("elastic_net_ols",)).run(test_datasets={"toy": fitting})


def test_fiber_columns_are_regressed_on_the_atlas_and_rasterized_like_tract_density(tmp_path):
    from calvin_utils.neuroimaging_utils.tract_utils.tract_density import TractDensity
    from calvin_utils.neuroimaging_utils.tract_utils.fiber_io import FiberIO

    rng = np.random.default_rng(6)
    mask = np.ones((6, 6, 6), dtype=np.float32)
    mask[0] = 0
    nib.save(nib.Nifti1Image(mask, np.eye(4)), tmp_path / "mask.nii.gz")
    fibers = np.empty(25, dtype=object)
    for index in range(25):
        start = rng.uniform(0, 5, size=3)
        fibers[index] = np.clip(start + np.outer(np.linspace(0, 4, 9), rng.normal(size=3)), -1, 6).astype(np.float32)
    np.savez(tmp_path / "atlas.npz", fibers=fibers)
    rows = []
    for index in range(16):
        values = rng.normal(size=25).astype(np.float32)
        np.save(tmp_path / f"sub-{index:02d}.fib.npy", values)
        FiberIO.write_values_description(tmp_path / f"sub-{index:02d}.fib.npy", tmp_path / "atlas.npz")
        rows.append({"subject": f"sub-{index:02d}", "fibers": str(tmp_path / f"sub-{index:02d}.fib.json"), "gait": float(values[:5].mean() + rng.normal())})
    pd.DataFrame(rows).to_csv(tmp_path / "table.csv", index=False)
    entry = {"pathology": "Stroke", "input_path": str(tmp_path / "table.csv"), "subject_column": "subject", "regressand_list": ["gait"], "voxelwise_vars": ["fibers"]}
    registry = {"mask_path": str(tmp_path / "mask.nii.gz"), "transform": "rank", "cohorts": {"toy": entry}}
    build.main(registry, tmp_path / "basis_maps")
    shard = basis_damage.load_library(directory=tmp_path / "basis_maps")["shards"][0]
    column = shard["values"][:, 0]
    assert Path(shard["sidecar"]["bases"][0]["fiber_path"]).is_file()

    palm, table = build.load_table(entry["input_path"], entry["regressand_list"])
    t_map = build.fit_bases(palm, table, "fibers", ["gait"], str(tmp_path / "atlas.npz"), "rank")["gait"][0]
    density = TractDensity(fiber_path=tmp_path / "atlas.npz", reference_nifti_path=tmp_path / "mask.nii.gz", out_path=tmp_path / "density.nii.gz", values=t_map, fiberset="both", threshold=None, symmetric=False, show_progress=False).run()
    expected = np.asarray(nib.load(tmp_path / "density.nii.gz").dataobj).reshape(-1)[mask.reshape(-1) > 0]
    assert column.shape == (int(mask.sum()),) and np.any(column)
    np.testing.assert_allclose(column, expected, rtol=1e-5, atol=1e-6)
