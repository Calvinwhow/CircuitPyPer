"""Write every fitted model's map, weights, fit details, iterations, validation, and test results."""

import csv
import copy
import json
import numpy as np
from dataclasses import asdict, is_dataclass
from pathlib import Path
from . import visualization
from .anatomy_maps import AnatomyMaps
from .history import OptimizationHistory
from ..pipeline import prediction
from ..pipeline.model_fitter_utils.model_fitter_schema import AlgorithmResult, NetworkResult
from ..neural_network.anatomy import REPRESENTATIONS
from calvin_utils.neuroimaging_utils.io.exporters import NeuroimageFileOutporter


def write_csv(path, records):
    records = list(records)
    if not records:
        return
    records = [asdict(record) if is_dataclass(record) else record for record in records]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)


def export_map(values, output_type, out_dir, mask_path):
    exporter = NeuroimageFileOutporter(output_ftype=output_type, mask_path=mask_path)
    exporter.validate_for_output()
    return exporter.save_map(map_data=np.asarray(values).reshape(-1), file_name="optimized_map", out_dir=str(out_dir))


def weight_records(result):
    """One row per basis weight; the network has one weight per basis and algorithm."""
    if isinstance(result, AlgorithmResult):
        return [{"map": name, "weight": float(weight), "selected": bool(weight != 0), "optimizer_method": result.method} for name, weight in zip(result.damage.map_names, result.weights)]
    return [{"map": name, "algorithm": method, "weight": float(weight)} for name, row in zip(result.damage.map_names, result.weights) for method, weight in zip(result.methods, row)]


def evaluation(directory, values):
    if values is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    write_csv(directory / "predictions.csv", values.predictions)
    write_csv(directory / "summary.csv", ({"dataset": name, **asdict(summary)} for name, summary in values.summary.items()))
    write_csv(directory / "fold_weights.csv", ({"fold": record.fold, "map": record.map_name, "weight": record.weight} for record in values.weights))


def algorithm_details(result, final):
    """Penalty calibration, plus the SDP's recovery diagnostics and RMS rho's multistart record."""
    write_csv(final / "elastic_net_calibration.csv", (asdict(result.regularization),))
    write_csv(final / "dataset_signs.csv", ({"dataset": name, "sign": sign} for name, sign in result.details.dataset_signs.items()))
    sparse = result.details.sparse_semidefinite
    if sparse is not None:
        diagnostic = asdict(sparse)
        diagnostic.update(objective=result.details.objective, n_iterations=result.details.n_iterations)
        theta = diagnostic.pop("theta")
        eigenvalues = diagnostic.pop("eigenvalues")
        (final / "sdp_recovery_diagnostic.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
        np.savez_compressed(final / "sdp_lifted_solution.npz", theta=theta, eigenvalues=eigenvalues, recovered_weights=result.weights)
    if result.details.starts:
        write_csv(final / "rms_multistart_summary.csv", result.details.starts)
        write_csv(final / "rms_multistart_weights.csv", ({name: float(weight) for name, weight in zip(result.damage.map_names, values)} for values in result.details.start_weights))


def network_details(result, final):
    """Representation maps, every network parameter, the integration weights, the training loss, and the model description."""
    parameters = result.parameters
    np.save(final / "anatomical_representation_maps.npy", np.asarray(result.representation_maps, dtype=np.float32))
    np.savez_compressed(final / "neural_network_parameters.npz", **asdict(parameters))
    write_csv(final / "integration_weights.csv", ({"representation": representation, "algorithm": method, "algorithm_bias": float(parameters.algorithm_bias[group, column]), "algorithm_mix": float(parameters.algorithm_mix[group, column]), "representation_bias": float(parameters.representation_bias[group]), "representation_mix": float(parameters.representation_mix[group]), "representation_baseline": float(parameters.representation_baseline[group])} for group, representation in enumerate(REPRESENTATIONS) for column, method in enumerate(result.methods)))
    write_csv(final / "training_history.csv", result.training.history)
    (final / "model.json").write_text(json.dumps({"mode": result.mode, "methods": result.methods, "representations": REPRESENTATIONS, "basis_representations": result.basis_representations, "dataset_signs": result.dataset_signs, "algorithm_orientations": result.algorithm_orientations.tolist(), "prediction_scale": result.training.prediction_scale, "optimizer": result.training.optimizer, "batch_mode": result.training.batch_mode, "device": result.training.device, "epochs": result.training.epochs, "loss": result.training.loss}, indent=2) + "\n")


def algorithm_history(result, output_type, mask_path):
    """The solver's weights and objective at every iteration, replayed through the bases."""
    trajectory = result.details.history or ((result.weights, result.details.objective),)
    resolved_mask_path = _history_mask_path(mask_path)
    normalized_maps = result.damage.normalized_basis_maps()
    return OptimizationHistory(weights=np.stack([item[0] for item in trajectory]), losses=np.asarray([item[1] for item in trajectory]), maps=normalized_maps, map_names=result.damage.map_names, output_type=output_type or "", mask_path=resolved_mask_path, objective_label=result.method)


def network_history(result, damage, max_frames, output_type, mask_path):
    """The network's map at up to max_frames - 1 evenly spaced training epochs, then the saved map."""
    from ..neural_network import training
    trajectory = result.training.parameter_trajectory
    indices = visualization._frame_indices(len(trajectory), max(int(max_frames) - 1, 1))
    frames, mixes = training.replay(copy.deepcopy(result.model), trajectory, {data.dataset_name: prediction.normalize(data.patient_maps) for data in damage.fitting_datasets}, result.dataset_weights, indices)
    losses = [result.training.history[index]["loss"] for index in indices]
    resolved_mask_path = _history_mask_path(mask_path)
    return OptimizationHistory(weights=np.stack([*mixes, result.parameters.representation_mix]), losses=np.asarray([*losses, result.training.loss]), maps=None, frames=np.stack([*frames, result.symptom_map]).astype(np.float32), steps=np.asarray([index + 1 for index in indices] + [len(trajectory)]), map_names=REPRESENTATIONS, output_type=output_type or "", mask_path=resolved_mask_path, objective_label="pairwise softplus loss")


def _history_mask_path(mask_path):
    """Resolve an optional mask path to the string stored in optimization history."""
    if not mask_path:
        return ""
    return str(Path(mask_path).expanduser().resolve())


def iterations(result, damage, final, output_type, mask_path, store_history, store_map_stack, render_gif, gif_fps, gif_max_frames, gif_view):
    """store_history: the algorithm's iteration record (optimization_history.npz). store_map_stack: the 4D map stack, for algorithms and the network. render_gif: the algorithm's weight dashboard GIF."""
    network = isinstance(result, NetworkResult)
    render_gif = render_gif and not network
    if not (store_history or store_map_stack or render_gif):
        return
    record = _iteration_history(result, damage, gif_max_frames, output_type, mask_path, network)
    stack_path = visualization.optimization_stack_path(final / "optimization.gif", output_type or "")
    if store_history and not network:
        record.save(final / "optimization_history.npz")
    if store_map_stack and stack_path is not None:
        max_frames = _map_stack_frame_limit(network, gif_max_frames)
        visualization.export_optimization_map_stack(record, stack_path, output_type=output_type, mask_path=mask_path, max_frames=max_frames)
    if render_gif:
        visualization.render_optimization_history(record, final / "optimization.gif", fps=gif_fps, max_frames=gif_max_frames, view=gif_view, write_map_stack=False)


def _iteration_history(result, damage, max_frames, output_type, mask_path, network):
    """Resolve the distinct algorithm and neural-network history formats."""
    if network:
        return network_history(result, damage, max_frames, output_type, mask_path)
    return algorithm_history(result, output_type, mask_path)


def _map_stack_frame_limit(network, max_frames):
    """Resolve frame limiting because network history is already sampled."""
    if network:
        return None
    return max_frames


def final_model(result, damage, final, output_type, mask_path, store_history, store_map_stack, render_gif, gif_fps, gif_max_frames, gif_view, save_anatomy_maps):
    """The map, weights, fit details and iterations of a fit on every patient."""
    final.mkdir(parents=True, exist_ok=True)
    if output_type is not None:
        export_map(result.symptom_map, output_type, final, mask_path)
    write_csv(final / "optimized_weights.csv", weight_records(result))
    if isinstance(result, NetworkResult):
        network_details(result, final)
    else:
        algorithm_details(result, final)
    if save_anatomy_maps:
        AnatomyMaps(result, final).run()
    iterations(result, damage, final, output_type, mask_path, store_history, store_map_stack, render_gif, gif_fps, gif_max_frames, gif_view)


def save_model(result, damage, root, output_type, mask_path, store_history, store_map_stack, render_gif, gif_fps, gif_max_frames, gif_view, save_anatomy_maps):
    """Everything one model produced, under root: final_model/ for a fit on every patient, then its evaluations."""
    if result.symptom_map is not None:
        final_model(result, damage, root / "final_model", output_type, mask_path, store_history, store_map_stack, render_gif, gif_fps, gif_max_frames, gif_view, save_anatomy_maps)
    evaluation(root / "apparent_evaluation", result.apparent)
    evaluation(root / "cross_validation", result.cross_validation)
    evaluation(root / "test_evaluation", result.test)
    return root


def main(results, damage, out_dir, winner, winner_only, output_type, mask_path, store_history, store_map_stack, render_gif, gif_fps, gif_max_frames, gif_view, save_anatomy_maps=False):
    """Save every model in its own directory, or only the winner directly in out_dir; return each saved model's directory."""
    if out_dir is None:
        return {}
    selected = _selected_results(results, winner, winner_only)
    saved = {}
    for result in selected:
        model_directory = _model_directory(out_dir, result.method, len(selected))
        saved[result.method] = save_model(result, damage, model_directory, output_type, mask_path, store_history, store_map_stack, render_gif, gif_fps, gif_max_frames, gif_view, save_anatomy_maps)
    return saved


def _selected_results(results, winner, winner_only):
    """Resolve whether saving receives only the winner or every model."""
    if winner_only:
        return (results.get(winner),)
    return results.models


def _model_directory(out_dir, model, selected_model_count):
    """Resolve whether one model writes at the root or each model gets a subdirectory."""
    root = Path(out_dir).expanduser()
    if selected_model_count == 1:
        return root
    return root / model


def target_validation(comparisons, out_dir):
    """Save held-out correlations and RMS-rho summaries across optimized targets."""
    if out_dir is None:
        return None
    records = [{"method": method, "target": target, "dataset": dataset, **asdict(performance)} for target, comparison in comparisons.items() for method, summary in comparison.summaries.items() for dataset, performance in summary.items()]
    datasets = sorted({record["dataset"] for record in records})
    summary = []
    for method in sorted({record["method"] for record in records}):
        row = {"method": method}
        for dataset in datasets:
            correlations = [record["spearman_rho"] for record in records if record["method"] == method and record["dataset"] == dataset]
            row[f"{dataset}_rms_rho"] = float(np.sqrt(np.mean(np.square(correlations))))
        row["average_rms_rho"] = float(np.mean([row[f"{dataset}_rms_rho"] for dataset in datasets]))
        summary.append(row)
    summary.sort(key=lambda row: row["average_rms_rho"], reverse=True)
    root = Path(out_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    write_csv(root / "held_out_correlations.csv", records)
    write_csv(root / "held_out_performance.csv", summary)
    return records, summary
