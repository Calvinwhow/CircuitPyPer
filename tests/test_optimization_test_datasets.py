"""Test-cohort prediction uses only the frozen weights and the patient-by-basis damage."""

import numpy as np
import pytest
from scipy.stats import spearmanr

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


def test_test_prediction_is_weighted_projection_with_frozen_map(basis_shards):
    maps, data, damage = scored_test_cohort(basis_shards, 3, 9, 11)
    weights = np.array([0.6, -0.3, 0.9])
    symptom_map = weights @ np.stack([values / np.max(np.abs(values)) for values in maps.values()])
    records, summary = prediction.test(damage["test_datasets"], weights, set())
    expected = (data["X"] @ symptom_map) / np.linalg.norm(data["X"], axis=1)
    np.testing.assert_allclose([record["prediction"] for record in records], expected, rtol=1e-5)
    expected_correlation = spearmanr(expected, data["y"])
    expected_rho = getattr(expected_correlation, "statistic", expected_correlation.correlation)
    assert summary["held_out"]["spearman_rho"] == pytest.approx(expected_rho)


def test_rescaling_weights_does_not_change_test_rho(basis_shards):
    _, _, damage = scored_test_cohort(basis_shards, 5, 12, 8)
    weights = np.array([0.6, -0.3, 0.9])
    first_records, first = prediction.test(damage["test_datasets"], weights, set())
    second_records, second = prediction.test(damage["test_datasets"], weights * 7.5, set())
    np.testing.assert_allclose([record["prediction"] for record in second_records], 7.5 * np.asarray([record["prediction"] for record in first_records]))
    assert first["held_out"]["spearman_rho"] == pytest.approx(second["held_out"]["spearman_rho"])


def test_fitting_patients_are_refused():
    rng = np.random.default_rng(7)
    fitting = {"fit": cohort(rng, 10, 6, "shared")}
    with pytest.raises(ValueError, match="shares"):
        prediction.test({"held_out": cohort(rng, 10, 6, "shared")}, rng.normal(size=6), prediction.fitting_ids(fitting))


def test_run_saves_every_stage_in_one_call(tmp_path, basis_shards):
    rng = np.random.default_rng(19)
    maps = {"one": rng.normal(size=7), "two": rng.normal(size=7)}
    fitting = {"first": cohort(rng, 20, 7, "first"), "second": cohort(rng, 20, 7, "second")}
    test = {"held_out": cohort(rng, 8, 7, "test")}
    out_dir = tmp_path / "run"
    RunOptimization(fitting, basis_dir=basis_shards(maps), method="elastic_net_ols").run(within_dataset_folds=4, cross_dataset_validation=True, test_datasets=test, out_dir=out_dir, store_history=True)
    assert (out_dir / "final_model" / "optimized_weights.csv").is_file()
    assert (out_dir / "final_model" / "optimization_history.npz").is_file()
    assert (out_dir / "within_dataset_validation" / "predictions.csv").is_file()
    assert (out_dir / "cross_dataset_validation" / "predictions.csv").is_file()
    assert (out_dir / "test_evaluation" / "test_predictions.csv").is_file()
