"""Write every fitted model's map, weights, fit details, iterations, validation, and test results."""

import csv
import copy
import json
import numpy as np
from pathlib import Path
from . import visualization
from .anatomy_maps import AnatomyMaps
from .history import OptimizationHistory
from ..pipeline import basis_damage, prediction
from ..pipeline.fitting import NETWORK
from ..neural_network.anatomy import REPRESENTATIONS
from calvin_utils.neuroimaging_utils.io.exporters import NeuroimageFileOutporter


def write_csv(path, records):
    records = list(records)
    if not records:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)


def export_map(values, output_type, out_dir, mask_path):
    exporter = NeuroimageFileOutporter(output_ftype=output_type, mask_path=mask_path)
    exporter.validate_for_output()
    exporter.save_map(map_data=np.asarray(values).reshape(-1), file_name="optimized_map", out_dir=str(out_dir))


def weight_records(result):
    """One row per basis weight; the network has one weight per basis and algorithm."""
    if np.ndim(result["weights"]) == 1:
        return [{"map": name, "weight": float(weight), "selected": bool(weight != 0), "optimizer_method": result["method"]} for name, weight in zip(result["map_names"], result["weights"])]
    return [{"map": name, "algorithm": method, "weight": float(weight)} for name, row in zip(result["map_names"], result["weights"]) for method, weight in zip(result["methods"], row)]


def evaluation(directory, values):
    if values is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    write_csv(directory / "predictions.csv", values["predictions"])
    write_csv(directory / "summary.csv", ({"dataset": name, **summary} for name, summary in values["summary"].items()))
    write_csv(directory / "fold_weights.csv", values.get("weights", ()))


def algorithm_details(result, final):
    """Penalty calibration, plus the SDP's recovery diagnostics and RMS rho's multistart record."""
    fit = result["fit"]
    write_csv(final / "elastic_net_calibration.csv", ({key: fit[key] for key in ("regularization_strength", "elastic_net_alpha", "reference_merit", "reference_null_merit", "reference_merit_gain", "l1_reference", "l2_reference", "l1_strength", "l2_strength")},))
    write_csv(final / "dataset_signs.csv", ({"dataset": name, "sign": sign} for name, sign in (fit.get("dataset_signs") or {}).items()))
    if result["method"] == "sparse_semidefinite":
        (final / "sdp_recovery_diagnostic.json").write_text(json.dumps({key: fit[key] for key in ("objective", "status", "solver", "solve_time", "n_iterations", "leading_eigenvalue", "positive_eigenvalue_sum", "leading_eigenvalue_fraction", "numerical_rank", "spectral_mass_threshold", "leading_vector_recovery_acceptable", "recovery_recommendation")}, indent=2) + "\n")
        np.savez_compressed(final / "sdp_lifted_solution.npz", theta=fit["theta"], eigenvalues=fit["eigenvalues"], recovered_weights=result["weights"])
    if result["method"] == "rms_rho":
        write_csv(final / "rms_multistart_summary.csv", fit["starts"])
        write_csv(final / "rms_multistart_weights.csv", ({name: float(weight) for name, weight in zip(result["map_names"], values)} for values in fit["start_weights"]))


def network_details(result, final):
    """Representation maps, every network parameter, the integration weights, the training loss, and the model description."""
    parameters = result["network_parameters"]
    np.save(final / "anatomical_representation_maps.npy", np.asarray(result["representation_maps"], dtype=np.float32))
    np.savez_compressed(final / "neural_network_parameters.npz", **{key: value for key, value in parameters.items() if key != "mode"})
    write_csv(final / "integration_weights.csv", ({"representation": representation, "algorithm": method, "algorithm_bias": float(parameters["algorithm_bias"][group, column]), "algorithm_mix": float(parameters["algorithm_mix"][group, column]), "representation_bias": float(parameters["representation_bias"][group]), "representation_mix": float(parameters["representation_mix"][group]), "representation_baseline": float(parameters["representation_baseline"][group])} for group, representation in enumerate(REPRESENTATIONS) for column, method in enumerate(result["methods"])))
    write_csv(final / "training_history.csv", result["fit"]["history"])
    (final / "model.json").write_text(json.dumps({"mode": result["mode"], "methods": result["methods"], "representations": REPRESENTATIONS, "basis_representations": result["basis_representations"], "dataset_signs": result["dataset_signs"], "algorithm_orientations": result["algorithm_orientations"].tolist(), "prediction_scale": result["fit"]["prediction_scale"], "optimizer": result["fit"]["optimizer"], "batch_mode": result["fit"]["batch_mode"], "device": result["fit"]["device"], "epochs": result["fit"]["epochs"], "loss": result["fit"]["loss"]}, indent=2) + "\n")


def algorithm_history(result, output_type, mask_path):
    """The solver's weights and objective at every iteration, replayed through the bases."""
    trajectory = result["fit"].get("history", [(result["weights"], result["fit"]["objective"])])
    return OptimizationHistory(weights=np.stack([item[0] for item in trajectory]), losses=np.asarray([item[1] for item in trajectory]), maps=basis_damage.stack(result["library"], result["scales"]), map_names=result["map_names"], output_type=output_type or "", mask_path=str(Path(mask_path).expanduser().resolve()) if mask_path else "", objective_label=result["method"])


def network_history(result, damage, max_frames, output_type, mask_path):
    """The network's map at up to max_frames - 1 evenly spaced training epochs, then the saved map."""
    from ..neural_network import training
    trajectory = result["fit"]["parameter_trajectory"]
    indices = visualization._frame_indices(len(trajectory), max(int(max_frames) - 1, 1))
    frames, mixes = training.replay(copy.deepcopy(result["model"]), trajectory, {name: prediction.normalize(data["patients"]) for name, data in damage["datasets"].items()}, result["dataset_weights"], indices)
    losses = [result["fit"]["history"][index]["loss"] for index in indices]
    return OptimizationHistory(weights=np.stack([*mixes, result["network_parameters"]["representation_mix"]]), losses=np.asarray([*losses, result["fit"]["loss"]]), maps=None, frames=np.stack([*frames, result["symptom_map"]]).astype(np.float32), steps=np.asarray([index + 1 for index in indices] + [len(trajectory)]), map_names=REPRESENTATIONS, output_type=output_type or "", mask_path=str(Path(mask_path).expanduser().resolve()) if mask_path else "", objective_label="pairwise softplus loss")


def iterations(result, damage, final, output_type, mask_path, store_history, store_map_stack, render_gif, gif_fps, gif_max_frames, gif_view):
    """store_history: the algorithm's iteration record (optimization_history.npz). store_map_stack: the 4D map stack, for algorithms and the network. render_gif: the algorithm's weight dashboard GIF."""
    network = result["method"] == NETWORK
    render_gif = render_gif and not network
    if not (store_history or store_map_stack or render_gif):
        return
    record = network_history(result, damage, gif_max_frames, output_type, mask_path) if network else algorithm_history(result, output_type, mask_path)
    stack_path = visualization.optimization_stack_path(final / "optimization.gif", output_type or "")
    if store_history and not network:
        record.save(final / "optimization_history.npz")
    if store_map_stack and stack_path is not None:
        visualization.export_optimization_map_stack(record, stack_path, output_type=output_type, mask_path=mask_path, max_frames=None if network else gif_max_frames)
    if render_gif:
        visualization.render_optimization_history(record, final / "optimization.gif", fps=gif_fps, max_frames=gif_max_frames, view=gif_view, write_map_stack=False)


def final_model(result, damage, final, output_type, mask_path, store_history, store_map_stack, render_gif, gif_fps, gif_max_frames, gif_view):
    """The map, weights, fit details and iterations of a fit on every patient."""
    final.mkdir(parents=True, exist_ok=True)
    if output_type is not None:
        export_map(result["symptom_map"], output_type, final, mask_path)
    write_csv(final / "optimized_weights.csv", weight_records(result))
    if result["method"] == NETWORK:
        network_details(result, final)
    else:
        algorithm_details(result, final)
    AnatomyMaps(final, Path(result["library"]["shards"][0]["path"]).parent).run()
    iterations(result, damage, final, output_type, mask_path, store_history, store_map_stack, render_gif, gif_fps, gif_max_frames, gif_view)


def save_model(result, damage, root, output_type, mask_path, store_history, store_map_stack, render_gif, gif_fps, gif_max_frames, gif_view):
    """Everything one model produced, under root: final_model/ for a fit on every patient, then its evaluations."""
    if result["symptom_map"] is not None:
        final_model(result, damage, root / "final_model", output_type, mask_path, store_history, store_map_stack, render_gif, gif_fps, gif_max_frames, gif_view)
    evaluation(root / "apparent_evaluation", result["apparent"])
    evaluation(root / "cross_validation", result["cross_validation"])
    evaluation(root / "test_evaluation", result["test"])
    return root


def main(results, damage, out_dir, winner, winner_only, output_type, mask_path, store_history, store_map_stack, render_gif, gif_fps, gif_max_frames, gif_view):
    """Save every model in its own directory, or only the winner directly in out_dir; return each saved model's directory."""
    if out_dir is None:
        return {}
    selected = {winner: results[winner]} if winner_only else results
    return {model: save_model(result, damage, Path(out_dir).expanduser() / model if len(selected) > 1 else Path(out_dir).expanduser(), output_type, mask_path, store_history, store_map_stack, render_gif, gif_fps, gif_max_frames, gif_view) for model, result in selected.items()}


def target_validation(comparisons, out_dir):
    """Save held-out correlations and RMS-rho summaries across optimized targets."""
    if out_dir is None:
        return None
    records = [{"method": method, "target": target, "dataset": dataset, **performance} for target, comparison in comparisons.items() for method, summary in comparison["summaries"].items() for dataset, performance in summary.items()]
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
