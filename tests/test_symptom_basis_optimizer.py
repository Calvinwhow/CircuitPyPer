"""Contracts for the linear symptom-basis optimization pipeline."""

import ast
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

from calvin_utils.neuroimaging_utils.ccm_utils.optimization.algorithms import elastic_net_ols, pairwise_proxy, rayleigh_refinement, rms_rho
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.run_optimization import RunOptimization
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.output import saving
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.pipeline import basis_damage, cross_validation, pre_processing, prediction
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.projections import prediction_variance
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.selection import prepare_regularization
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.solvers import adam, proximal_gradient


def synthetic_data(seed=14):
    rng = np.random.default_rng(seed)
    maps = {f"basis_{index}": rng.normal(size=24) for index in range(4)}
    patients = rng.normal(size=(40, 24))
    outcome = patients @ maps["basis_0"] - 0.6 * patients @ maps["basis_2"]
    datasets = {
        "score": {
            "X": patients,
            "y": outcome,
            "ids": np.arange(len(patients)).astype(str),
        }
    }
    return maps, datasets


def test_algorithm_modules_define_no_classes():
    directory = Path(elastic_net_ols.__file__).parent
    for path in directory.glob("*.py"):
        tree = ast.parse(path.read_text())
        assert not any(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
        if path.name != "__init__.py":
            functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
            assert {"prepare", "merit", "l1", "l2", "target", "main"} <= functions
            assert "solve" not in functions


def test_damage_reproduces_weighted_map_projection(basis_shards):
    maps, datasets = synthetic_data()
    damage = basis_damage.main(datasets, directory=basis_shards(maps))
    beta = np.array([0.8, -0.4, 0.2, 0.1])
    normalized = np.stack([values / np.linalg.norm(values) for values in maps.values()])
    symptom_map = beta @ normalized
    patients = datasets["score"]["X"]
    projection = (patients @ symptom_map) / np.linalg.norm(patients, axis=1)

    assert damage["datasets"]["score"]["X"].dtype == np.float64
    np.testing.assert_allclose(damage["datasets"]["score"]["X"] @ beta, projection, rtol=1e-5)
    cosines = (patients / np.linalg.norm(patients, axis=1)[:, None]) @ normalized.T
    np.testing.assert_allclose(damage["datasets"]["score"]["X"], cosines, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(prediction.score(patients, symptom_map), projection, rtol=1e-5)
    np.testing.assert_allclose(basis_damage.combine(beta, damage["library"], damage["scales"]), symptom_map, rtol=1e-5, atol=1e-6)
    assert damage["map_names"] == tuple(f"synthetic:{name}" for name in maps)


def test_prediction_variance_projection_is_independent_of_penalties():
    rng = np.random.default_rng(12)
    X = {"a": rng.normal(size=(30, 4)), "b": rng.normal(size=(20, 4))}
    covariance, project = prediction_variance.main(X, {"a": 0.5, "b": 0.5})
    weights = project(np.array([1.0, -0.4, 0.2, 0.7]))
    assert weights @ covariance @ weights == pytest.approx(1.0)


def test_regularization_calibration_splits_the_target_budget_equally():
    result = prepare_regularization.main(2.0, 1.0, 4.0, 2.0, strength=0.1)
    assert result["l1_strength"] * result["l1_reference"] == pytest.approx(0.05)
    assert result["l2_strength"] * result["l2_reference"] == pytest.approx(0.05)
    assert result["elastic_net_alpha"] == pytest.approx(1 / 3)


def test_final_weights_are_normalized_after_unconstrained_optimization(basis_shards):
    maps, datasets = synthetic_data()
    optimizer = RunOptimization(datasets, basis_dir=basis_shards(maps), methods=("elastic_net_ols",))
    optimizer.run()
    result = optimizer.result
    X = {"score": optimizer.damage["datasets"]["score"]["X"]}
    covariance = prediction_variance.gram(X, {"score": 1.0})

    assert result["weights"] @ covariance @ result["weights"] == pytest.approx(1.0)
    assert np.var([record["prediction"] for record in result["apparent"]["predictions"]]) == pytest.approx(1.0, rel=1e-5)
    assert result["map_names"] == tuple(f"synthetic:{name}" for name in maps)


def test_gradient_step_updates_state_and_returns_native_merit(basis_shards):
    maps, datasets = synthetic_data()
    optimizer = RunOptimization(datasets, basis_dir=basis_shards(maps), methods=("elastic_net_ols",))
    optimizer.run()
    result = optimizer.result
    previous = result["weights"].copy()
    gradient = np.asarray([0.2, -0.1, 0.3, -0.4])
    merit = optimizer.gradient_step(gradient, learning_rate=0.5)
    basis_maps = basis_damage.stack(optimizer.damage["library"], optimizer.damage["scales"])
    X = {"score": optimizer.damage["datasets"]["score"]["X"]}
    expected = prediction_variance.normalize(previous + 0.5 * gradient, X, {"score": 1.0})

    np.testing.assert_allclose(optimizer.result["weights"], expected)
    assert optimizer.result["symptom_map"].dtype == np.float32
    np.testing.assert_allclose(optimizer.result["symptom_map"], expected @ basis_maps, rtol=1e-5, atol=1e-6)
    assert merit == pytest.approx(elastic_net_ols.evaluate(expected, X, {"score": datasets["score"]["y"]}, {"score": 1.0}))


def test_elastic_net_ols_matches_unpenalized_ols_when_penalties_are_zero():
    rng = np.random.default_rng(3)
    X = {"a": rng.normal(size=(30, 5)), "b": rng.normal(size=(22, 5))}
    beta = np.array([1.0, -0.7, 0.2, 0.0, 0.4])
    y = {name: values @ beta for name, values in X.items()}
    result = elastic_net_ols.main(X, y, solver=proximal_gradient.main)
    assert np.corrcoef(result["weights"], beta)[0, 1] > 0.999
    assert result["r2"] > 0.999


def test_rayleigh_is_invariant_to_outcome_reversal():
    rng = np.random.default_rng(7)
    X = {"a": rng.normal(size=(35, 5)), "b": rng.normal(size=(31, 5))}
    beta = np.array([1.0, 0.0, -0.7, 0.0, 0.2])
    y = {"a": X["a"] @ beta, "b": -(X["b"] @ beta)}
    weights = {"a": 0.5, "b": 0.5}
    forward = rayleigh_refinement.main(X, y, solver=adam.main, dataset_weights=weights, l1_strength=0.005, l2_strength=0.001, max_iter=1000)
    reverse = rayleigh_refinement.main(X, {name: -values for name, values in y.items()}, solver=adam.main, dataset_weights=weights, l1_strength=0.005, l2_strength=0.001, max_iter=1000)
    np.testing.assert_allclose(forward["weights"], reverse["weights"], atol=1e-6)


def test_pairwise_proxy_recovers_positive_direction():
    rng = np.random.default_rng(4)
    beta = np.array([1.2, 0.0, -0.8, 0.0])
    X = {"a": rng.normal(size=(60, 4)), "b": rng.normal(size=(55, 4))}
    y = {name: values @ beta for name, values in X.items()}
    result = pairwise_proxy.main(X, y, solver=proximal_gradient.main, dataset_weights={"a": 0.5, "b": 0.5}, l1_strength=0.05, l2_strength=0.01, max_iter=2000)
    assert np.corrcoef(result["weights"], beta)[0, 1] > 0.99


def test_pairwise_merit_is_the_stated_softplus_equation():
    X = {"a": np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0]])}
    y = {"a": np.array([3.0, 1.0, 2.0])}
    beta = np.array([0.7, -0.2])
    differences = pairwise_proxy.prepare(X, y)
    margin = -1.0 * differences["a"] @ beta
    expected = -0.4 * np.log(1 + np.exp(-margin)).mean()
    assert pairwise_proxy.merit(beta, differences, {"a": 0.4}, {"a": -1.0}) == pytest.approx(expected)


def test_pairwise_proxy_orients_reversed_datasets():
    rng = np.random.default_rng(41)
    beta = np.array([1.1, -0.8, 0.3])
    X = {"forward": rng.normal(size=(45, 3)), "reverse": rng.normal(size=(42, 3))}
    y = {"forward": X["forward"] @ beta, "reverse": -(X["reverse"] @ beta)}
    result = pairwise_proxy.main(X, y, solver=proximal_gradient.main, dataset_weights={"forward": 0.5, "reverse": 0.5}, l1_strength=0.0, l2_strength=0.001, max_iter=1500)
    assert result["dataset_signs"] == {"forward": 1.0, "reverse": -1.0}
    assert np.corrcoef(result["weights"], beta)[0, 1] > 0.99


def test_initial_weights_are_not_l1_normalized():
    weights = pre_processing.initial_weights(3, "weighted", sample_sizes={"a": 20, "b": 10, "c": 5}, names=("a", "b", "c"))
    np.testing.assert_array_equal(weights, [20.0, 10.0, 5.0])


def test_cross_validation_fits_only_its_folds_and_no_scheme_fits_every_patient(basis_shards):
    maps, datasets = synthetic_data()
    optimizer = RunOptimization(datasets, basis_dir=basis_shards(maps), methods=("elastic_net_ols",), seed=11)
    comparison = optimizer.run(cross_validation=4)
    result = optimizer.result

    assert comparison["winner"] == "elastic_net_ols"
    assert result["symptom_map"] is None and result["apparent"] is None and "weights" not in result
    assert result["cross_validation"]["summary"]["score"]["spearman_rho"] > 0.9
    assert result["cross_validation"]["summary"]["score"]["rho_method"] == "mean_of_folds"
    assert len(result["cross_validation"]["predictions"]) == len(datasets["score"]["y"])

    optimizer.run()
    result = optimizer.result
    assert result["weights"].shape == (len(maps),) and result["symptom_map"].shape == (24,) and result["cross_validation"] is None
    assert result["apparent"]["summary"]["score"]["spearman_rho"] > 0.95


def test_sparse_semidefinite_is_off_unless_requested():
    assert "sparse_semidefinite" not in RunOptimization({}).fit_models.keywords["methods"]
    assert "sparse_semidefinite" in RunOptimization({}, sparse_semidefinite=True).fit_models.keywords["methods"]


def test_held_out_algorithm_predictions_are_damage_at_weights(basis_shards):
    maps, datasets = synthetic_data()
    damage = basis_damage.main(datasets, directory=basis_shards(maps))
    weights = np.array([0.8, -0.4, 0.2, 0.1])
    voxel = prediction.score(damage["datasets"]["score"]["patients"], basis_damage.combine(weights, damage["library"], damage["scales"]))
    np.testing.assert_allclose(cross_validation.held_out_predictions({"weights": weights}, damage["datasets"]["score"]), voxel, rtol=1e-4, atol=1e-5)

def two_cohorts():
    datasets = {}
    for index, name in enumerate(("first", "second")):
        X = np.arange(30, dtype=np.float32).reshape(10, 3) + index
        datasets[name] = {"X": X, "y": np.arange(10, dtype=float), "ids": np.asarray([f"{name}_{row}" for row in range(10)])}
    return datasets


def test_patient_folds_hold_out_every_dataset_at_once_and_lodo_holds_out_whole_datasets():
    datasets = two_cohorts()
    folds = cross_validation.patient_folds(datasets, 5, seed=3)
    assert len(folds) == 5 and all(set(fold) == {"first", "second"} and all(len(ids) == 2 for ids in fold.values()) for fold in folds)
    for name in datasets:
        held = np.concatenate([fold[name] for fold in folds])
        assert sorted(held) == sorted(datasets[name]["ids"])
    training, evaluated = cross_validation.split(datasets, folds[0])
    assert {name: len(data["y"]) for name, data in training.items()} == {"first": 8, "second": 8}
    assert {name: len(data["y"]) for name, data in evaluated.items()} == {"first": 2, "second": 2}
    lodo = cross_validation.dataset_folds(datasets)
    training, evaluated = cross_validation.split(datasets, lodo[0])
    assert set(training) == {"second"} and set(evaluated) == {"first"} and len(evaluated["first"]["y"]) == 10


def test_fold_summary_is_the_mean_of_fold_rhos_and_loocv_falls_back_to_pooled():
    records = [{"dataset": "first", "fold": fold, "patient_id": f"{fold}_{row}", "observed": float(row), "prediction": float(row if fold == 1 else -row)} for fold in (1, 2) for row in range(4)]
    summary = cross_validation.summary(records, {"first": None}, by_fold=True)["first"]
    assert summary["fold_rhos"] == [1.0, -1.0] and summary["spearman_rho"] == 0.0 and summary["rho_method"] == "mean_of_folds"
    loocv = [{**record, "fold": index} for index, record in enumerate(records[:4])]
    summary = cross_validation.summary(loocv, {"first": None}, by_fold=True)["first"]
    assert summary["rho_method"] == "pooled" and summary["spearman_rho"] == summary["pooled_spearman_rho"] == 1.0


def test_patient_folds_fit_all_datasets_jointly(basis_shards):
    rng = np.random.default_rng(5)
    maps = {f"basis_{index}": rng.normal(size=3) for index in range(2)}
    damage = basis_damage.main(two_cohorts(), directory=basis_shards(maps))
    seen = []

    def fit_models(damage, seed, final):
        seen.append({name: len(data["y"]) for name, data in damage["datasets"].items()})
        return {"elastic_net_ols": {"weights": np.ones(2), "map_names": damage["map_names"]}}

    validation = cross_validation.validate(damage, 5, fit_models, 0)["elastic_net_ols"]["cross_validation"]
    assert seen == [{"first": 8, "second": 8}] * 5
    assert len(validation["predictions"]) == 20 and set(validation["summary"]) == {"first", "second"}
    seen.clear()
    validation = cross_validation.validate(damage, "lodo", fit_models, 0)["elastic_net_ols"]["cross_validation"]
    assert seen == [{"second": 10}, {"first": 10}] and "rho_method" not in validation["summary"]["first"]


def summarized(rho, dataset="score"):
    return {"summary": {dataset: {"spearman_rho": rho}}}


def test_ranking_uses_cross_validation_rms_rho(capsys):
    results = {method: {"test": None, "cross_validation": summarized(rho), "apparent": summarized(1.0)} for method, rho in (("elastic_net_ols", 0.8), ("pairwise_proxy", -0.9))}
    comparison = prediction.rank(results)

    assert comparison["winner"] == "pairwise_proxy" and comparison["ranking"] == ["pairwise_proxy", "elastic_net_ols"]
    assert comparison["performance"]["pairwise_proxy"] == pytest.approx(0.9)
    assert "pairwise_proxy: 0.900000" in capsys.readouterr().out


def test_ranking_prefers_test_performance():
    results = {method: {"test": summarized(test, "held_out"), "cross_validation": summarized(fold), "apparent": summarized(1.0)} for method, test, fold in (("elastic_net_ols", 0.95, 0.1), ("pairwise_proxy", 0.2, 0.9))}
    assert prediction.rank(results)["winner"] == "elastic_net_ols"


def test_every_model_is_saved_in_its_own_directory_unless_winner_only(monkeypatch, tmp_path):
    monkeypatch.setattr(saving, "save_model", lambda result, damage, root, *options: root)
    results = {"elastic_net_ols": {}, "pairwise_proxy": {}}
    options = (None, None, False, False, False, 5, 60, "auto")
    assert saving.main(results, None, tmp_path, "pairwise_proxy", False, *options) == {"elastic_net_ols": tmp_path / "elastic_net_ols", "pairwise_proxy": tmp_path / "pairwise_proxy"}
    assert saving.main(results, None, tmp_path, "pairwise_proxy", True, *options) == {"pairwise_proxy": tmp_path}
    assert saving.main(results, None, None, "pairwise_proxy", False, *options) == {}

def test_damage_is_independent_of_voxel_chunk_size(basis_shards):
    maps, datasets = synthetic_data()
    directory = basis_shards({name: maps[name] for name in ("basis_0", "basis_1")}, cohort="first")
    basis_shards({name: maps[name] for name in ("basis_2", "basis_3")}, cohort="second")
    small = basis_damage.main(datasets, directory=directory, chunk_voxels=5)
    whole = basis_damage.main(datasets, directory=directory, chunk_voxels=10_000)

    np.testing.assert_allclose(small["datasets"]["score"]["X"], whole["datasets"]["score"]["X"], rtol=1e-5, atol=1e-6)
    assert small["map_names"] == ("first:basis_0", "first:basis_1", "second:basis_2", "second:basis_3")


def test_library_symptoms_are_every_regressand_once_in_shard_order(basis_shards):
    maps, _ = synthetic_data()
    directory = basis_shards({name: maps[name] for name in ("basis_0", "basis_1")}, cohort="first")
    basis_shards({name: maps[name] for name in ("basis_1", "basis_2")}, cohort="second")
    assert basis_damage.symptoms(basis_damage.load_library(directory=directory)) == ("basis_0", "basis_1", "basis_2")


def test_cohort_selection_loads_only_the_named_shards(basis_shards):
    maps, datasets = synthetic_data()
    directory = basis_shards({name: maps[name] for name in ("basis_0", "basis_1")}, cohort="first")
    basis_shards({name: maps[name] for name in ("basis_2", "basis_3")}, cohort="second")
    library = basis_damage.load_library(("second",), directory)

    assert library["names"] == ("second:basis_2", "second:basis_3")
    assert library["sample_sizes"] == {"second:basis_2": 40, "second:basis_3": 40}
    with pytest.raises(FileNotFoundError):
        basis_damage.load_library(("missing",), directory)


def test_shards_are_validated_against_their_sidecars(basis_shards):
    import json
    maps, _ = synthetic_data()
    directory = basis_shards(maps)
    sidecar_path = directory / "shard_synthetic.json"
    sidecar = json.loads(sidecar_path.read_text())

    sidecar_path.write_text(json.dumps({key: value for key, value in sidecar.items() if key != "pathology"}))
    with pytest.raises(ValueError, match="pathology"):
        basis_damage.load_library(directory=directory)

    sidecar_path.write_text(json.dumps({**sidecar, "bases": sidecar["bases"][:3]}))
    with pytest.raises(ValueError, match="voxels, 3"):
        basis_damage.load_library(directory=directory)

    sidecar_path.write_text(json.dumps(sidecar))
    np.save(directory / "shard_synthetic.npy", np.stack(list(maps.values()), axis=1).astype(np.float64))
    with pytest.raises(ValueError, match="float32"):
        basis_damage.load_library(directory=directory)


def test_rms_requires_multistart_and_is_reproducible():
    rng = np.random.default_rng(8)
    X = {"score": rng.normal(size=(20, 4))}
    y = {"score": rng.normal(size=20)}
    first = rms_rho.main(X, y, solver=adam.main, dataset_weights={"score": 1.0}, n_starts=10, max_iter=2, random_state=8)
    second = rms_rho.main(X, y, solver=adam.main, dataset_weights={"score": 1.0}, n_starts=10, max_iter=2, random_state=8)
    np.testing.assert_allclose(first["weights"], second["weights"])
    assert len(first["starts"]) == 10

    X = {"score": np.ones((4, 2))}
    y = {"score": np.arange(4)}
    try:
        rms_rho.main(X, y, solver=adam.main, dataset_weights={"score": 1.0}, n_starts=9)
    except ValueError as error:
        assert "n_starts >= 10" in str(error)
    else:
        raise AssertionError("RMS rho accepted fewer than ten starts.")


def test_rms_gradient_is_the_forward_difference_of_its_target():
    rng = np.random.default_rng(9)
    prepared = ({"a": rng.normal(size=(20, 6)), "b": rng.normal(size=(15, 6))}, {"a": rng.normal(size=20), "b": rng.normal(size=15)})
    weights = rng.normal(size=6)
    expected = adam.finite_difference(lambda values: rms_rho.target(values, prepared, 0.01, 0.02), weights, 0.05)
    np.testing.assert_allclose(rms_rho.gradient(weights, prepared, 0.01, 0.02, 0.05), expected, rtol=1e-9, atol=1e-12)


def test_rms_enforces_unit_prediction_variance():
    rng = np.random.default_rng(18)
    X = {"score": rng.normal(size=(24, 4))}
    y = {"score": rng.normal(size=24)}
    result = rms_rho.main(X, y, solver=adam.main, dataset_weights={"score": 1.0}, n_starts=10, max_iter=2, random_state=18)

    assert result["weights"] @ result["G"] @ result["weights"] == pytest.approx(1.0)
