"""Filesystem output for one completed optimization run."""

import csv
import json
import numpy as np
from pathlib import Path
from . import visualization
from ..pipeline import basis_damage
from .history import OptimizationHistory
from calvin_utils.neuroimaging_utils.io.exporters import NeuroimageFileOutporter


def export_map(values, output_type, out_dir, mask_path):
    exporter = NeuroimageFileOutporter(output_ftype=output_type, mask_path=mask_path)
    exporter.validate_for_output()
    exporter.save_map(map_data=np.asarray(values).reshape(-1), file_name="optimized_map", out_dir=str(out_dir))


def weights(result):
    return [{"map": name, "weight": float(weight), "selected": bool(weight != 0), "optimizer_method": result["method"]} for name, weight in zip(result["map_names"], result["weights"])]


def write_csv(path, records):
    records = list(records)
    if not records:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)


def history(result, path, output_type, mask_path):
    trajectory = result["fit"].get("history", [(result["weights"], result["fit"]["objective"])])
    record = OptimizationHistory(weights=np.stack([item[0] for item in trajectory]), losses=np.asarray([item[1] for item in trajectory]), maps=basis_damage.stack(result["library"], result["scales"]), map_names=result["map_names"], output_type=output_type, mask_path=str(Path(mask_path).expanduser().resolve()) if mask_path else "", objective_label=result["method"])
    return record.save(path)


def diagnostics(result, out_dir):
    if result["method"] != "sparse_semidefinite":
        return
    fit = result["fit"]
    keys = ("objective", "status", "solver", "solve_time", "n_iterations", "leading_eigenvalue", "positive_eigenvalue_sum", "leading_eigenvalue_fraction", "numerical_rank", "spectral_mass_threshold", "leading_vector_recovery_acceptable", "recovery_recommendation")
    (out_dir / "sdp_recovery_diagnostic.json").write_text(json.dumps({key: fit[key] for key in keys}, indent=2) + "\n")
    np.savez_compressed(out_dir / "sdp_lifted_solution.npz", theta=fit["theta"], eigenvalues=fit["eigenvalues"], recovered_weights=result["weights"])


def validation(result, root, key):
    values = result[key]
    if values is None:
        return
    directory = root / key
    directory.mkdir(parents=True, exist_ok=True)
    write_csv(directory / "predictions.csv", values["predictions"])
    write_csv(directory / "summary.csv", ({"dataset": name, **summary} for name, summary in values["summary"].items()))
    write_csv(directory / "fold_weights.csv", values["weights"])


def main(result, out_dir, *, output_type=None, mask_path=None, store_history=False, render_gif=False, gif_fps=5, gif_max_frames=60, gif_view="auto"):
    """Save one completed optimization result.
    result: dict; fitted weights, map, diagnostics, predictions, and optional validation results.
    out_dir: str | Path; root output directory.
    output_type: str | None; neuroimage output type, or no map export when omitted.
    mask_path: str | Path | None; mask or ordered feature reference used for output.
    store_history: bool; whether to save coefficient and objective history.
    render_gif: bool; whether to render optimization history as a GIF.
    gif_fps: int; GIF frames per second.
    gif_max_frames: int; maximum rendered history frames.
    gif_view: str; visualization view passed to the history renderer.
    """
    root = Path(out_dir).expanduser()
    final = root / "final_model"
    final.mkdir(parents=True, exist_ok=True)
    if output_type is not None:
        export_map(result["symptom_map"], output_type, final, mask_path)
    weight_table = result["weight_table"]
    write_csv(final / "optimized_weights.csv", weight_table)
    write_csv(final / "apparent_performance.csv", ({"dataset": name, **values} for name, values in result["performance"].items()))
    calibration_keys = ("regularization_strength", "elastic_net_alpha", "reference_merit", "reference_null_merit", "reference_merit_gain", "l1_reference", "l2_reference", "l1_strength", "l2_strength")
    write_csv(final / "elastic_net_calibration.csv", ({key: result["fit"][key] for key in calibration_keys},))
    if result["fit"].get("dataset_signs") is not None:
        write_csv(final / "dataset_signs.csv", ({"dataset": name, "sign": sign} for name, sign in result["fit"]["dataset_signs"].items()))
    diagnostics(result, final)
    if result["method"] == "rms_rho":
        write_csv(final / "rms_multistart_summary.csv", result["fit"]["starts"])
        write_csv(final / "rms_multistart_weights.csv", ({name: float(weight) for name, weight in zip(result["map_names"], values)} for values in result["fit"]["start_weights"]))
    validation(result, root, "within_dataset_validation")
    validation(result, root, "cross_dataset_validation")
    if result.get("test") is not None:
        test_dir = root / "test_evaluation"
        test_dir.mkdir(parents=True, exist_ok=True)
        write_csv(test_dir / "test_predictions.csv", result["test"]["predictions"])
        write_csv(test_dir / "test_summary.csv", ({"dataset": name, **values} for name, values in result["test"]["summary"].items()))
    if store_history or render_gif:
        history_path = history(result, final / "optimization_history.npz", output_type or "", mask_path)
        if render_gif:
            visualization.render_optimization_history(history_path, final / "optimization.gif", fps=gif_fps, max_frames=gif_max_frames, view=gif_view)
    return weight_table


def target_validation(comparisons, out_dir):
    """Save held-out correlations and RMS-rho summaries across optimized targets."""
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
