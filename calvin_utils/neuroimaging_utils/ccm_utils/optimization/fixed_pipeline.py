"""I/O and final fitting for optimization of existing component maps."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from calvin_utils.file_utils.import_functions import GiiNiiFileImport
from calvin_utils.neuroimaging_utils.ccm_utils.npy_utils import DataLoader
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.convergent_map_optimizer import LocalizationOptimizer
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.history import OptimizationHistory
from calvin_utils.neuroimaging_utils.output_functions import NeuroimageFileOutporter


def load_map_files(map_files, *, mask_path, map_output_type=None):
    """Import native component maps into one ordered feature space."""
    if len(map_files) < 2:
        print("WARNING: Provide at least two component maps in MAP_FILES to run a proper optimization.")
    paths = {name: Path(path).expanduser().resolve() for name, path in map_files.items()}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Component map {name!r} does not exist: {path}")
    importer = GiiNiiFileImport(import_path=None, mask_path=mask_path)
    detected = {name: importer._identify_file_type(path) for name, path in paths.items()}
    output_type = map_output_type
    if output_type is None:
        if "npy" in detected.values():
            raise ValueError("Set MAP_OUTPUT_TYPE when MAP_FILES contains bare .npy arrays.")
        types = {"surface" if kind in {"gii", "freesurfer"} else kind
                 for kind in detected.values()}
        if len(types) != 1:
            raise ValueError(f"Component maps have mixed image types: {detected}")
        output_type = types.pop()
    output_type = {"nifti": "nii", "gii": "surface",
                   "freesurfer": "surface"}.get(output_type, output_type)
    if output_type not in {"nii", "surface", "fiber", "nii_timeseries"}:
        raise ValueError(f"Unknown MAP_OUTPUT_TYPE: {output_type!r}")
    for name, kind in detected.items():
        native_type = {"gii": "surface", "freesurfer": "surface"}.get(kind, kind)
        if native_type not in {"npy", output_type}:
            raise ValueError(
                f"Component map {name!r} is {kind}, but output type is {output_type}."
            )
    exporter = NeuroimageFileOutporter(output_ftype=output_type, mask_path=mask_path)
    exporter.validate_for_output()
    maps = {}
    for name, path in paths.items():
        values = (np.load(path, mmap_mode="r") if detected[name] == "npy"
                  else exporter.load_map_values(path))
        maps[name] = np.asarray(values, dtype=float).reshape(-1)
    return maps, output_type


def infer_map_sample_sizes(map_files):
    """Read each map's contributing N from its nearest regression manifest."""
    sizes = {}
    for name, value in map_files.items():
        path = Path(value).expanduser().resolve()
        for directory in path.parents:
            manifest_path = directory / 'dataset_dict.json'
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text())
            payload = manifest.get('neuroimaging_regression')
            if payload and 'design_matrix' in payload:
                array_path = Path(payload['design_matrix']).expanduser()
                if not array_path.is_absolute():
                    array_path = manifest_path.parent / array_path
                sizes[name] = int(np.load(array_path, mmap_mode='r').shape[0])
                break
            if name in manifest and 'niftis' in manifest[name]:
                array_path = Path(manifest[name]['niftis']).expanduser()
                if not array_path.is_absolute():
                    array_path = manifest_path.parent / array_path
                sizes[name] = int(np.load(array_path, mmap_mode='r').shape[0])
                break
    return sizes


def make_dataset_loader(out_dir, *, dataset_specs=None, manifest_path=None,
                        mask_path=None, subject_col=None,
                        data_transform_method='standardize'):
    """Use either a CSV specification dictionary or a DataLoader JSON."""
    if bool(dataset_specs) == bool(manifest_path):
        raise ValueError("Set a CSV dataset dictionary or a manifest path, not both.")
    if dataset_specs:
        return DataLoader.from_csv_dict(
            dataset_specs, out_dir, mask_path=mask_path,
            subject_col=subject_col,
            data_transform_method=data_transform_method,
        )
    return DataLoader(Path(manifest_path).expanduser())


def make_scoring_loader(out_dir, *, scoring_datasets=None,
                        scoring_manifest_path=None, mask_path=None,
                        subject_col=None, data_transform_method='standardize'):
    """Compatibility name for :func:`make_dataset_loader`."""
    return make_dataset_loader(
        out_dir, dataset_specs=scoring_datasets,
        manifest_path=scoring_manifest_path, mask_path=mask_path,
        subject_col=subject_col,
        data_transform_method=data_transform_method,
    )


def optimize_maps(corr_map_dict, scoring_loader, output_type, *, data_mode,
                  weight_init_mode, history_path=None, map_sample_sizes=None,
                  mask_path=None, max_iters=500,
                  random_state=None, return_optimizer=False):
    """Fit final weights on every scoring patient and optionally save replay data."""
    optimizer = LocalizationOptimizer(
        corr_map_dict, scoring_loader, data_mode=data_mode,
        mode=weight_init_mode, map_sample_sizes=map_sample_sizes,
        random_state=random_state,
    )
    optimizer.engine.convergence_monitor.max_iterations = max_iters
    optimized_map, _ = optimizer.optimize(store_iters=history_path is not None)
    if history_path is not None:
        history = OptimizationHistory(
            weights=np.asarray(optimizer.engine.iter_weights).reshape(
                -1, optimizer.MAPS.shape[0]
            ),
            losses=np.asarray(optimizer.engine.iter_losses),
            maps=optimizer.MAPS,
            map_names=tuple(optimizer.corr_map_names),
            output_type=output_type,
            mask_path=str(Path(mask_path).expanduser().resolve()) if mask_path else "",
            objective_label="Training RMS Spearman rho (final weights)",
        )
        history.save(history_path)
        print(f"Saved optimization history to: {history_path}")
    return (optimized_map, optimizer) if return_optimizer else optimized_map


def export_maps(optimized_map, output_type, out_dir, *, mask_path):
    """Save the final weighted map using native image or fiber I/O."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    exporter = NeuroimageFileOutporter(
        output_ftype=output_type, mask_path=mask_path
    )
    exporter.validate_for_output()
    exporter.save_map(
        map_data=np.asarray(optimized_map).reshape(-1),
        file_name="optimized_map", out_dir=str(out_dir),
    )
    print(f"Saved {output_type} map to: {out_dir}")
