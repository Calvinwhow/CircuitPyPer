"""Run the established regression pipeline on fitted basis maps and weights."""

import numpy as np
import pandas as pd
from pathlib import Path
from calvin_utils.neuroimaging_utils.io.exporters import NeuroimageFileOutporter
from ..pipeline import basis_damage


def _regression_classes():
    """Load optional inference dependencies only when inference is requested."""
    try:
        from calvin_utils.permutation_analysis_utils.voxelwise_regression import VoxelwiseRegression
        from calvin_utils.permutation_analysis_utils.voxelwise_regression_prep import RegressionPrep
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Basis-map inference requires the regression dependencies.") from exc
    return RegressionPrep, VoxelwiseRegression


def save_basis_maps(basis_maps, out_dir, output_type, mask_path):
    exporter = NeuroimageFileOutporter(output_ftype=output_type, mask_path=mask_path)
    exporter.validate_for_output()
    stems = [f"basis_{index:04d}" for index in range(len(basis_maps))]
    for stem, values in zip(stems, basis_maps):
        exporter.save_map(values, stem, str(out_dir))
    native = [path for path in out_dir.iterdir() if exporter.is_native_map_file(path)]
    return [str(next(path for path in native if exporter.io.native_map_stem(path) == stem)) for stem in stems]


def main(basis_maps, weights, out_dir, *, output_type, mask_path, n_permutations=1000):
    """Regress normalized basis maps on frozen optimized weights, then permute.
    basis_maps: np.ndarray with shape (B, V); normalized basis maps.
    weights: np.ndarray with shape (B,); frozen optimized coefficients.
    out_dir: str | Path; destination for basis images and regression outputs.
    output_type: str; neuroimage output type accepted by NeuroimageFileOutporter.
    mask_path: str | Path | None; mask or ordered feature reference used for output.
    n_permutations: int; number of regression-label permutations.
    """
    RegressionPrep, VoxelwiseRegression = _regression_classes()
    out_dir = Path(out_dir).expanduser()
    map_dir = out_dir / "basis_maps"
    regression_dir = out_dir / "regression"
    map_dir.mkdir(parents=True, exist_ok=True)
    regression_dir.mkdir(parents=True, exist_ok=True)
    paths = save_basis_maps(np.asarray(basis_maps), map_dir, output_type, mask_path)
    preparer = RegressionPrep(design_matrix=pd.DataFrame({"optimized_weight": weights}), contrast_matrix=np.asarray([[1.0]]), outcome_df=pd.DataFrame({"basis_map": paths}), out_dir=str(out_dir), neuroimaging_variables=["basis_map"], mask_path=mask_path, data_transform_method=None, formula="basis_map ~ optimized_weight")
    _, json_path = preparer.run()
    regression = VoxelwiseRegression(json_path=json_path, mask_path=None, out_dir=str(regression_dir), regression_type="linear", n_permutations=n_permutations)
    regression.run()
    return {"dataset": json_path, "regression_dir": str(regression_dir)}


def network_weights(symptom_map, stacked, map_names, out_dir):
    """One weight per basis for the network: its final map's least-squares weights on the bases plus a voxel-uniform constant (the baselines' extent term).
    While the network's ReLUs pass every voxel this reproduces its map exactly (R^2 = 1); below 1, R^2 measures what the thresholds added. Writes network_basis_weights.csv.
    """
    design = np.vstack([stacked, np.ones((1, stacked.shape[1]), dtype=stacked.dtype)]).T.astype(float)
    target = np.asarray(symptom_map, dtype=float)
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    residual = target - design @ coefficients
    r2 = float(1 - residual @ residual / np.sum((target - target.mean()) ** 2))
    print(f"Network map as a weighted combination of the bases plus a constant: R^2 = {r2:.6f}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"map": map_names, "weight": coefficients[:-1], "constant": coefficients[-1], "projection_r2": r2}).to_csv(out_dir / "network_basis_weights.csv", index=False)
    return coefficients[:-1]


def infer(result, model_dir, output_type, mask_path, n_permutations):
    """Regress the normalized bases on the winning model's weights and permute; the network supplies its least-squares basis weights."""
    if not n_permutations:
        return None
    if result["symptom_map"] is None:
        print("Skipping basis-map regression: it regresses the map fitted on every patient (cross_validation=None).", flush=True)
        return None
    if model_dir is None or output_type is None:
        raise ValueError("Basis-map regression requires out_dir and output_type.")
    stacked = basis_damage.stack(result["library"], result["scales"])
    out_dir = Path(model_dir) / "final_model" / "basis_map_regression"
    weights = result["weights"] if np.ndim(result["weights"]) == 1 else network_weights(result["symptom_map"], stacked, result["map_names"], out_dir)
    return main(stacked, weights, out_dir, output_type=output_type, mask_path=mask_path, n_permutations=n_permutations)
