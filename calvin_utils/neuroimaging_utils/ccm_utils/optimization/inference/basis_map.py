"""Run the established regression pipeline on fitted basis maps and weights."""

import numpy as np
import pandas as pd
from pathlib import Path
from calvin_utils.neuroimaging_utils.io.exporters import NeuroimageFileOutporter

try:
    from calvin_utils.permutation_analysis_utils.voxelwise_regression import VoxelwiseRegression
    from calvin_utils.permutation_analysis_utils.voxelwise_regression_prep import RegressionPrep
except ModuleNotFoundError:
    RegressionPrep = None
    VoxelwiseRegression = None


def save_basis_maps(basis_maps, out_dir, output_type, mask_path):
    exporter = NeuroimageFileOutporter(output_ftype=output_type, mask_path=mask_path)
    exporter.validate_for_output()
    stems = [f"basis_{index:04d}" for index in range(len(basis_maps))]
    for stem, values in zip(stems, basis_maps):
        exporter.save_map(values, stem, str(out_dir))
    native = [path for path in out_dir.iterdir() if exporter.is_native_map_file(path)]
    return [next(path for path in native if exporter.io.native_map_stem(path) == stem) for stem in stems]


def main(basis_maps, weights, out_dir, *, output_type, mask_path, n_permutations=1000):
    """Regress normalized basis maps on frozen optimized weights, then permute.
    basis_maps: np.ndarray with shape (B, V); normalized basis maps.
    weights: np.ndarray with shape (B,); frozen optimized coefficients.
    out_dir: str | Path; destination for basis images and regression outputs.
    output_type: str; neuroimage output type accepted by NeuroimageFileOutporter.
    mask_path: str | Path | None; mask or ordered feature reference used for output.
    n_permutations: int; number of regression-label permutations.
    """
    if RegressionPrep is None or VoxelwiseRegression is None:
        raise ModuleNotFoundError("Basis-map inference requires the regression dependencies.")
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
