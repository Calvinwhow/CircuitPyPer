"""Test-cohort prediction uses only the frozen weights and the patient-by-basis damage."""

import numpy as np
import pytest
from scipy.stats import spearmanr

from calvin_utils.neuroimaging_utils.ccm_utils.optimization.output import saving
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.pipeline import basis_damage, prediction
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.run_optimization import RunOptimization


def cohort(rng, n, n_features, prefix):
    return {"X": rng.normal(size=(n, n_features)), "y": rng.normal(size=n), "ids": np.asarray([f"{prefix}_{index}" for index in range(n)])}


def scored_test_cohort(basis_shards, seed, n, n_features):
    rng = np.random.default_rng(seed)
    maps = {f"basis_{index}": rng.normal(size=n_features) for index in range(3)}
    data = cohort(rng, n, n_features, "test")
    damage = basis_damage.main({"fit": cohort(rng, n, n_features, "fit")}, {"held_out": data}, directory=basis_shards(maps))
    return maps, data, damage


def weighted_map(maps, weights):
    return weights @ np.stack([values / np.linalg.norm(values) for values in maps.values()])


def test_test_prediction_is_weighted_projection_with_frozen_map(basis_shards):
    maps, data, damage = scored_test_cohort(basis_shards, 3, 9, 11)
    weights = np.array([0.6, -0.3, 0.9])
    symptom_map = weighted_map(maps, weights)
    evaluated = prediction.evaluate(damage["test_datasets"], symptom_map)
    records, summary = evaluated["predictions"], evaluated["summary"]
    np.testing.assert_allclose([record["prediction"] for record in records], damage["test_datasets"]["held_out"]["X"] @ weights, rtol=1e-5)
    expected = (data["X"] @ symptom_map) / np.linalg.norm(data["X"], axis=1)
    np.testing.assert_allclose([record["prediction"] for record in records], expected, rtol=1e-5)
    expected_correlation = spearmanr(expected, data["y"])
    expected_rho = getattr(expected_correlation, "statistic", expected_correlation.correlation)
    assert summary["held_out"]["spearman_rho"] == pytest.approx(expected_rho)


def test_rescaling_weights_does_not_change_test_rho(basis_shards):
    maps, _, damage = scored_test_cohort(basis_shards, 5, 12, 8)
    symptom_map = weighted_map(maps, np.array([0.6, -0.3, 0.9]))
    first = prediction.evaluate(damage["test_datasets"], symptom_map)
    second = prediction.evaluate(damage["test_datasets"], symptom_map * 7.5)
    np.testing.assert_allclose([record["prediction"] for record in second["predictions"]], 7.5 * np.asarray([record["prediction"] for record in first["predictions"]]), rtol=1e-5)
    assert first["summary"]["held_out"]["spearman_rho"] == pytest.approx(second["summary"]["held_out"]["spearman_rho"])


def test_fitting_patients_are_refused():
    rng = np.random.default_rng(7)
    fitting = {"fit": cohort(rng, 10, 6, "shared")}
    with pytest.raises(ValueError, match="shares"):
        prediction.evaluate({"held_out": {**cohort(rng, 10, 6, "shared"), "patients": rng.normal(size=(10, 6))}}, rng.normal(size=6), prediction.fitting_ids(fitting))


def test_folds_save_only_cross_validation_and_a_fit_on_every_patient_saves_its_model_and_test(tmp_path, basis_shards):
    rng = np.random.default_rng(19)
    maps = {"one": rng.normal(size=7), "two": rng.normal(size=7)}
    fitting = {"first": cohort(rng, 20, 7, "first"), "second": cohort(rng, 20, 7, "second")}
    test = {"held_out": cohort(rng, 8, 7, "test")}
    optimizer = RunOptimization(fitting, basis_dir=basis_shards(maps), methods=("elastic_net_ols",))
    optimizer.run(cross_validation=4, out_dir=tmp_path / "folds")
    assert (tmp_path / "folds" / "cross_validation" / "predictions.csv").is_file() and (tmp_path / "folds" / "cross_validation" / "fold_weights.csv").is_file()
    assert not (tmp_path / "folds" / "final_model").exists() and not (tmp_path / "folds" / "apparent_evaluation").exists()
    optimizer.run(cross_validation=4, test_datasets=test, n_permutations=10)
    assert optimizer.damage["test_datasets"] is None and optimizer.result["test"] is None and optimizer.result["basis_map_regression"] is None
    optimizer.run(test_datasets=test, out_dir=tmp_path / "all", store_history=True)
    assert (tmp_path / "all" / "final_model" / "optimized_weights.csv").is_file() and (tmp_path / "all" / "final_model" / "optimization_history.npz").is_file()
    assert (tmp_path / "all" / "test_evaluation" / "predictions.csv").is_file() and not (tmp_path / "all" / "cross_validation").exists()


def test_store_history_writes_the_4d_stack_and_render_gif_writes_only_the_gif(tmp_path, basis_shards):
    import nibabel as nib
    rng = np.random.default_rng(23)
    maps = {"one": rng.normal(size=8), "two": rng.normal(size=8)}
    nib.save(nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4)), tmp_path / "mask.nii.gz")
    fitting = {"fit": cohort(rng, 20, 8, "fit")}
    optimizer = RunOptimization(fitting, basis_dir=basis_shards(maps), methods=("elastic_net_ols",))
    optimizer.run(out_dir=tmp_path / "stored", output_type="nii", mask_path=tmp_path / "mask.nii.gz", store_history=True, store_map_stack=True)
    stored = tmp_path / "stored" / "final_model"
    assert (stored / "optimization_history.npz").is_file() and nib.load(stored / "optimization.nii.gz").ndim == 4 and not (stored / "optimization.gif").exists()
    saving.main(optimizer.results, optimizer.damage, tmp_path / "rendered", optimizer.comparison["winner"], False, "nii", tmp_path / "mask.nii.gz", False, False, True, 5, 10, "metrics")
    rendered = tmp_path / "rendered" / "final_model"
    assert (rendered / "optimization.gif").is_file() and not (rendered / "optimization.nii.gz").exists() and not (rendered / "optimization_history.npz").exists()
