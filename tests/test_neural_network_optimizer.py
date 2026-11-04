"""Contracts for hierarchical neural integration of basis optimizers."""

import json
import numpy as np
import pytest

from calvin_utils.neuroimaging_utils.ccm_utils.optimization.neural_network import anatomy


def synthetic(seed=19):
    rng = np.random.default_rng(seed)
    maps = {f"basis_{index}": rng.normal(size=30) for index in range(6)}
    patients = rng.normal(size=(24, 30)).astype(np.float32)
    outcome = patients @ (maps["basis_0"] - 0.5 * maps["basis_4"])
    datasets = {"clinical": {"X": patients, "y": outcome, "ids": np.arange(len(patients)).astype(str)}}
    representations = {f"synthetic:basis_{index}": ("fibers", "networks", "local")[index // 2] for index in range(6)}
    return maps, datasets, representations


def test_representation_masks_are_explicit_and_include_composite(basis_shards):
    from calvin_utils.neuroimaging_utils.ccm_utils.optimization.pipeline import basis_damage

    maps, _, supplied = synthetic()
    library = basis_damage.load_library(directory=basis_shards(maps))
    labels = anatomy.basis_representations(library, supplied)
    masks = anatomy.representation_masks(labels)

    assert labels == ("fibers", "fibers", "networks", "networks", "local", "local")
    np.testing.assert_array_equal(masks[:3].sum(axis=0), 1)
    np.testing.assert_array_equal(masks[3], 1)


def test_masked_algorithm_maps_match_direct_basis_products(basis_shards):
    from calvin_utils.neuroimaging_utils.ccm_utils.optimization.pipeline import basis_damage

    maps, _, supplied = synthetic()
    damage = basis_damage.main({"dummy": {"X": np.ones((3, 30)), "y": np.arange(3), "ids": np.arange(3)}}, directory=basis_shards(maps))
    labels = anatomy.basis_representations(damage["library"], supplied)
    masks = anatomy.representation_masks(labels)
    weights = np.arange(30, dtype=np.float32).reshape(6, 5) / 10
    reconstructed = anatomy.reconstruct_algorithm_maps(weights, damage["library"], damage["scales"], masks)
    basis = anatomy.normalized_basis(damage["library"], damage["scales"])

    for group in range(4):
        np.testing.assert_allclose(reconstructed[group], basis @ (weights * masks[group, :, None]), rtol=1e-5, atol=1e-6)


def network_optimizer(basis_shards, datasets=None, **options):
    from calvin_utils.neuroimaging_utils.ccm_utils.optimization.run_optimization import RunOptimization

    maps, fitting, representations = synthetic()
    labels = {name.split(":", 1)[1]: label for name, label in representations.items()}
    return RunOptimization(datasets or fitting, basis_dir=basis_shards(maps, representations=labels), neural_network=True, learning_rate=0.02, device="cpu", seed=4, **options)


def test_fully_trainable_network_trains_its_coefficients_and_exports_parameters(basis_shards, tmp_path):
    pytest.importorskip("torch")
    optimizer = network_optimizer(basis_shards, fully_trainable=True, max_epochs=30, patience=10)
    optimizer.run(out_dir=tmp_path / "output")
    result = optimizer.results["anatomical_neural_network"]

    assert result["method"] == "anatomical_neural_network" and result["mode"] == "fine_tune"
    assert result["symptom_map"].shape == (30,) and result["symptom_map"].dtype == np.float32
    assert result["representation_maps"].shape == (4, 30)
    assert result["network_parameters"]["algorithm_weights"].shape == (6, len(result["methods"]))
    assert not np.allclose(result["network_parameters"]["algorithm_weights"], result["weights"])
    assert np.var([record["prediction"] for record in result["apparent"]["predictions"]]) == pytest.approx(1.0, rel=1e-4)
    metadata = json.loads((tmp_path / "output" / "anatomical_neural_network" / "final_model" / "model.json").read_text())
    assert metadata["mode"] == "fine_tune" and metadata["optimizer"] == "adam" and metadata["batch_mode"] == "full"


def test_frozen_network_keeps_coefficients_and_every_model_is_tested(basis_shards):
    pytest.importorskip("torch")
    _, datasets, _ = synthetic()
    test = {"held_out": {"X": datasets["clinical"]["X"][:5], "y": datasets["clinical"]["y"][:5], "ids": np.asarray([f"test:{index}" for index in range(5)])}}
    optimizer = network_optimizer(basis_shards, max_epochs=5, patience=2)
    optimizer.run(test_datasets=test)
    result = optimizer.results["anatomical_neural_network"]

    np.testing.assert_array_equal(result["network_parameters"]["algorithm_weights"], result["weights"])
    assert set(optimizer.results) == {*result["methods"], "anatomical_neural_network"}
    assert all(model["test"]["summary"]["held_out"]["n_patients"] == 5 for model in optimizer.results.values())


def test_the_network_is_compared_with_every_algorithm_on_held_out_datasets(basis_shards, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from calvin_utils.neuroimaging_utils.ccm_utils.optimization.pipeline import fitting

    fitted = []
    original = fitting.fit_algorithm
    monkeypatch.setattr(fitting, "fit_algorithm", lambda method, *arguments: fitted.append(method) or original(method, *arguments))
    _, datasets, _ = synthetic()
    patients, outcome = datasets["clinical"]["X"], datasets["clinical"]["y"]
    cohorts = {name: {"X": patients[rows], "y": outcome[rows], "ids": np.asarray([f"{name}:{index}" for index in rows])} for name, rows in (("first", np.arange(0, 12)), ("second", np.arange(12, 24)))}
    optimizer = network_optimizer(basis_shards, cohorts, methods=("elastic_net_ols", "pairwise_proxy"), max_epochs=5, patience=2)
    comparison = optimizer.run(cross_validation="lodo", out_dir=tmp_path / "output")

    assert set(comparison["performance"]) == {"elastic_net_ols", "pairwise_proxy", "anatomical_neural_network"}
    assert fitted == ["elastic_net_ols", "pairwise_proxy"] * 2
    assert set(optimizer.results["anatomical_neural_network"]["cross_validation"]["summary"]) == {"first", "second"}
    assert (tmp_path / "output" / "anatomical_neural_network" / "cross_validation" / "summary.csv").is_file()
    assert (tmp_path / "output" / "elastic_net_ols" / "cross_validation" / "summary.csv").is_file()

def test_frozen_model_can_promote_shared_filters_to_one_parameter_tensor():
    torch = pytest.importorskip("torch")
    from calvin_utils.neuroimaging_utils.ccm_utils.optimization.neural_network.model import AnatomicalNetwork

    weights = np.ones((6, 5), dtype=np.float32)
    masks = anatomy.representation_masks(("fibers", "fibers", "networks", "networks", "local", "local"))
    fixed = np.ones((4, 10, 5), dtype=np.float32)
    basis = np.ones((10, 6), dtype=np.float32)
    model = AnatomicalNetwork(weights, masks, fixed_algorithm_maps=fixed, mode="frozen")
    model.unfreeze_filters(basis)

    assert isinstance(model.algorithm_weights, torch.nn.Parameter)
    assert model.algorithm_weights.requires_grad
    assert model.masked_algorithm_maps().shape == (4, 10, 5)


def test_network_layers_follow_explicit_spatial_equations_in_float32():
    torch = pytest.importorskip("torch")
    from calvin_utils.neuroimaging_utils.ccm_utils.optimization.neural_network.layers import AlgorithmMapLayer, AnatomicalRepresentationLayer, FinalLocalizationLayer
    from calvin_utils.neuroimaging_utils.ccm_utils.optimization.neural_network.model import AnatomicalNetwork

    weights = np.ones((3, 5), dtype=np.float32)
    masks = anatomy.representation_masks(("fibers", "networks", "local"))
    algorithm_maps = np.arange(60, dtype=np.float32).reshape(4, 3, 5) - 20
    patients = torch.tensor([[1.0, 2.0, -1.0]], dtype=torch.float32)
    model = AnatomicalNetwork(weights, masks, fixed_algorithm_maps=algorithm_maps, mode="frozen")

    assert isinstance(model.algorithm_layer, AlgorithmMapLayer)
    assert isinstance(model.anatomy_layer, AnatomicalRepresentationLayer)
    assert isinstance(model.localization_layer, FinalLocalizationLayer)
    assert sum(parameter.numel() for parameter in model.parameters()) == 52
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())

    output = model({"clinical": patients})
    maps = torch.tensor(algorithm_maps)
    # Both ReLUs start inactive (b = -min_v Z, c = 0), so H is the algorithm mean plus its offset and M is the plain mean.
    expected_representations = maps.mean(dim=2) - torch.amin(maps, dim=1).mean(dim=1)[:, None]
    expected_map = maps.mean(dim=(0, 2))

    torch.testing.assert_close(output["representations"], expected_representations)
    torch.testing.assert_close(output["map"], expected_map)
    torch.testing.assert_close(output["predictions"]["clinical"], patients @ expected_map)
    assert output["map"].dtype == torch.float32


def test_network_training_is_written_as_a_4d_map_stack(basis_shards, tmp_path):
    pytest.importorskip("torch")
    import nibabel as nib
    from calvin_utils.neuroimaging_utils.ccm_utils.optimization.output import saving

    mask = np.zeros((5, 6, 2), dtype=np.float32)
    mask.reshape(-1)[:30] = 1
    nib.save(nib.Nifti1Image(mask, np.eye(4)), tmp_path / "mask.nii.gz")
    optimizer = network_optimizer(basis_shards, methods=("elastic_net_ols",), fully_trainable=True, max_epochs=40, patience=40)
    optimizer.run(out_dir=tmp_path / "output", output_type="nii", mask_path=tmp_path / "mask.nii.gz", store_history=True, gif_max_frames=10)

    final = tmp_path / "output" / "anatomical_neural_network" / "final_model"
    stack = np.asarray(nib.load(final / "optimization.nii.gz").dataobj)
    saved = np.asarray(nib.load(final / "optimized_map.nii.gz").dataobj)
    assert stack.shape == (5, 6, 2, 10)
    np.testing.assert_allclose(stack[..., -1], saved, rtol=1e-6, atol=1e-7)
    assert not np.allclose(stack[..., 0], stack[..., -1])
    assert not (final / "optimization.gif").exists() and not (final / "optimization_history.npz").exists()
    saving.main(optimizer.results, optimizer.damage, tmp_path / "rendered", "anatomical_neural_network", True, "nii", tmp_path / "mask.nii.gz", False, True, 5, 10, "metrics")
    assert not (tmp_path / "rendered" / "final_model" / "optimization.gif").exists() and not (tmp_path / "rendered" / "final_model" / "optimization.nii.gz").exists()

def test_network_basis_regression_uses_least_squares_basis_weights(basis_shards, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from calvin_utils.neuroimaging_utils.ccm_utils.optimization.inference import basis_map
    from calvin_utils.neuroimaging_utils.ccm_utils.optimization.pipeline import basis_damage

    optimizer = network_optimizer(basis_shards, methods=("elastic_net_ols",), max_epochs=10, patience=5)
    optimizer.run()
    result = optimizer.results["anatomical_neural_network"]
    captured = {}
    monkeypatch.setattr(basis_map, "main", lambda basis_maps, weights, out_dir, **options: captured.update(basis_maps=basis_maps, weights=weights) or {"regression_dir": str(out_dir)})
    basis_map.infer(result, tmp_path, "nii", None, 10)

    stacked = basis_damage.stack(result["library"], result["scales"])
    design = np.vstack([stacked, np.ones((1, stacked.shape[1]), dtype=stacked.dtype)]).T.astype(float)
    expected = np.linalg.lstsq(design, result["symptom_map"].astype(float), rcond=None)[0][:-1]
    np.testing.assert_allclose(captured["weights"], expected, rtol=1e-3, atol=1e-5)
    table = (tmp_path / "final_model" / "basis_map_regression" / "network_basis_weights.csv").read_text().splitlines()
    assert len(table) == 7 and table[0] == "map,weight,constant,projection_r2"
    assert 0 <= float(table[1].split(",")[3]) <= 1
