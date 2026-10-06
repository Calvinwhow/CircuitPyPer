"""Export volumes, surfaces, and fibers through a common interface."""

import os
import fnmatch
import re
from pathlib import Path
import numpy as np
if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [np.bool_, np.bytes_, np.str_, np.object_],
    }
if not hasattr(np, "maximum_sctype"):
    def _maximum_sctype(t):
        dtype = np.dtype(t)
        if np.issubdtype(dtype, np.complexfloating):
            return np.complex128
        if np.issubdtype(dtype, np.floating):
            return np.float64
        if np.issubdtype(dtype, np.unsignedinteger):
            return np.uint64
        if np.issubdtype(dtype, np.integer):
            return np.int64
        return dtype.type

    np.maximum_sctype = _maximum_sctype

class NeuroimageFileOutporter:
    def __init__(self, output_ftype, mask_path=None):
        output_ftype = {
            "nifti": "nii",
            "gii": "surface",
            "freesurfer": "surface",
        }.get(output_ftype, output_ftype)
        self.output_ftype = output_ftype
        self.mask_path = mask_path

        if output_ftype == "nii":
            from calvin_utils.neuroimaging_utils.nifti_utils.volume_io import NiftiIO

            self.io = NiftiIO(mask_path=mask_path)
        elif output_ftype == "surface":
            from calvin_utils.neuroimaging_utils.surface_utils.surface_io import SurfaceIO

            self.io = SurfaceIO(mask_path=mask_path)
        elif output_ftype == "fiber":
            from calvin_utils.neuroimaging_utils.tract_utils.fiber_io import FiberIO

            self.io = FiberIO(mask_path=mask_path)
        elif output_ftype == "nii_timeseries":
            from calvin_utils.neuroimaging_utils.nifti_utils.volume_io import VolumetricTimeSeriesIO

            self.io = VolumetricTimeSeriesIO(mask_path=mask_path)
        else:
            raise ValueError(f"Unknown output_ftype: {output_ftype}")

    def save_map(self, map_data, file_name, out_dir, visualize=False):
        """
        Save one statistical vector using the underlying I/O class.
        """
        fake_target = os.path.join(out_dir, file_name)
        self.io.save_files(
            arr=np.asarray(map_data),
            file_paths=[fake_target],
            dry_run=False,
            file_suffix=""
        )
        if visualize and hasattr(self.io, "_map_to_image") and hasattr(self.io, "_visualize_map"):
            img = self.io._map_to_image(np.asarray(map_data))
            self.io._visualize_map(img, title=os.path.basename(file_name))

    def validate_for_output(self):
        """Ask the selected backend to validate its output configuration."""
        self.io.validate_for_output()

    def prepare_map_for_evaluation(self, map_data):
        """Return one fitted map as a backend-defined 1D evaluation vector."""
        result = np.asarray(self.io.prepare_map_for_evaluation(map_data), dtype=np.float32)
        if result.ndim != 1:
            raise ValueError(f"Expected a 1D evaluation map, got shape {result.shape}")
        return result

    def prepare_evaluation_data(self, file_paths):
        """Return subject evaluation maps as ``(subjects, locations)``."""
        result = np.asarray(self.io.prepare_evaluation_data(file_paths), dtype=np.float32)
        if result.ndim != 2:
            raise ValueError(f"Expected 2D evaluation data, got shape {result.shape}")
        return result

    def evaluation_size(self, model_size):
        return int(self.io.evaluation_size(model_size))

    def is_native_map_file(self, path):
        return bool(self.io.is_native_map_file(path))

    def load_map_values(self, path):
        """Return native saved-map values as a format-independent 1D array."""
        result = np.asarray(self.io.load_map_values(path), dtype=float).reshape(-1)
        return result

    def load_named_maps(self, directory, names):
        """Load named native maps without exposing backend file conventions."""
        native_files = [
            path for path in Path(directory).iterdir()
            if path.is_file() and self.is_native_map_file(path)
        ]

        def natural_key(path):
            stem = self.io.native_map_stem(path)
            return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", stem)]

        loaded = []
        for name in names:
            matches = sorted(
                (
                    path for path in native_files
                    if fnmatch.fnmatch(self.io.native_map_stem(path), name)
                ),
                key=natural_key,
            )
            # Fiber output writes an ordinary map plus a ``*_symmetric``
            # presentation variant. Broad parameter patterns such as
            # ``beta_predictor_[0-9]*`` must not load both as independent model
            # coefficients. Keep the ordinary set unless symmetry was asked
            # for explicitly; direct ``*_symmetric`` patterns still work.
            if "_symmetric" not in name:
                ordinary = [
                    path for path in matches
                    if not self.io.native_map_stem(path).endswith("_symmetric")
                ]
                if ordinary:
                    matches = ordinary
            if not matches:
                raise FileNotFoundError(
                    f"No native map matching '{name}' was found in {directory}."
                )
            arrays = [self.load_map_values(path) for path in matches]
            loaded.append(np.column_stack(arrays) if "*" in name else arrays[0])
        return loaded

    def view_map(self, map_data, file_name):
        """
        View one statistical vector using the underlying I/O class.
        """
        if hasattr(self.io, "_map_to_image"):
            from nilearn import plotting

            img = self.io._map_to_image(np.asarray(map_data))
            return plotting.view_img(img, title=os.path.basename(file_name))
        return None
