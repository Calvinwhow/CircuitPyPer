from __future__ import annotations

import json
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
import nibabel as nib
from nibabel.streamlines import Tractogram
from scipy.io import savemat


class FiberResultVisualizer:
    """
    Export fiberwise regression results to visualization-friendly files.

    This class is intended for outputs produced by:

        from calvin_utils.permutation_analysis_utils.voxelwise_regression import VoxelwiseRegression

    When ``RegressionPrep`` used fiber inputs, ``VoxelwiseRegression`` writes
    statistical maps through ``FiberIO``. Those files are currently saved as
    object-array ``*.fib.npy`` files containing one array per fiber:

        fiber_i.shape == (n_vertices_i, 4)
        columns: x, y, z, statistic_value

    Expected regression output names include:

        contrast_tval_0.fib.npy
        contrast_tval_FWE_0.fib.npy
        contrast_pval_FWE_0.fib.npy
        beta_predictor_0.fib.npy
        R2_vals.fib.npy

    This exporter can also take:

    1. ``values_path`` as a plain ``.npy`` vector of shape ``(n_fibers,)`` plus
       ``fiber_atlas_path`` pointing to a canonical atlas ``.npz`` with a
       ``fibers`` key.
    2. ``values`` directly as a numpy-like vector plus ``fiber_atlas_path``.
    3. ``fiber_atlas_path`` alone, when the goal is to export a fiber atlas for
       visual inspection rather than visualize regression values.

    Main output
    -----------
    Lead-DBS discriminative fiber MAT:

        <out_dir>/<output_name>_discfibers.mat

    containing:

        fibcell : MATLAB cell array, one ``N_i x 3`` fiber per cell, in MNI mm
        vals    : ``n_fibers x 1`` statistic values

    This is the format used by Lead-DBS helper functions such as
    ``ea_discfibers2trk`` and ``ea_discfibers2nifti``.

    Optional output
    ---------------
    Lead-DBS FTR-style MAT:

        <out_dir>/<output_name>_ftr.mat

    containing:

        ea_fibformat = "1.1"
        fourindex = 1
        fibers = concatenated ``N x 4`` matrix, columns x/y/z/local_fiber_id
        idx = per-fiber vertex counts
        vals = per-fiber statistic values
        voxmm = "mm"

    In MATLAB/Lead-DBS this can be converted to TrackVis with:

        ea_ftr2trk('/path/to/result_ftr.mat')

    DSI Studio / MRtrix-readable TCK:

        <out_dir>/<output_name>.tck

    This is the most useful output for direct DSI Studio visual inspection.
    It stores selected fibers in MNI/world coordinates. TCK does not preserve
    per-fiber statistic values; use the JSON or MAT sidecars for values.
    Set ``tck_suffix=".tt.tck"`` if you want DSI-style naming.

    Simple MATLAB track cell MAT:

        <out_dir>/<output_name>_track.mat

    containing:

        track : MATLAB cell array, one ``N_i x 3`` fiber per cell
        vals  : ``n_fibers x 1`` values, if available

    Thresholding
    ------------
    By default all finite fibers are exported. Use ``sign``, ``min_abs_value``,
    or ``top_percent`` to reduce the displayed fibers.

    ``sign``:
        "both"     keep positive and negative finite values
        "positive" keep values > 0
        "negative" keep values < 0

    ``min_abs_value``:
        Keep fibers with absolute statistic >= this value.

    ``top_percent``:
        Keep the top percent of fibers by absolute statistic after sign/value
        filtering. For example, ``top_percent=5`` exports the strongest 5%.

    ``symmetric``:
        If True, duplicate the selected fibers after thresholding and mirror
        them across the MNI sagittal plane by default. This creates a bilateral
        visualization from a unilateral atlas or result. The default mirror is
        left-right, ``x -> -x`` around ``x=0``.

    ``run()`` returns a dictionary with output paths and exported counts.
    """

    def __init__(
        self,
        *,
        values_path: str | Path | None = None,
        fiber_atlas_path: str | Path | None = None,
        values=None,
        out_dir: str | Path,
        output_name: str | None = None,
        sign: str = "both",
        min_abs_value: float | None = None,
        top_percent: float | None = None,
        save_discfibers_mat: bool = True,
        save_ftr_mat: bool = True,
        save_tck: bool = False,
        save_track_mat: bool = False,
        tck_suffix: str = ".tck",
        symmetric: bool = False,
        mirror_axis: str = "x",
        mirror_origin: float = 0.0,
    ):
        self.values_path = Path(values_path) if values_path is not None else None
        self.fiber_atlas_path = Path(fiber_atlas_path) if fiber_atlas_path is not None else None
        self.values = None if values is None else np.asarray(values, dtype=np.float32).flatten()
        self.out_dir = Path(out_dir)
        self.output_name = output_name
        self.sign = sign
        self.min_abs_value = min_abs_value
        self.top_percent = top_percent
        self.save_discfibers_mat = bool(save_discfibers_mat)
        self.save_ftr_mat = bool(save_ftr_mat)
        self.save_tck = bool(save_tck)
        self.save_track_mat = bool(save_track_mat)
        self.tck_suffix = tck_suffix
        self.symmetric = bool(symmetric)
        self.mirror_axis = mirror_axis
        self.mirror_origin = float(mirror_origin)

        self.fibers = None
        self.selected_fibers = None
        self.selected_values = None
        self.n_selected_before_symmetry = None
        self.keep_mask = None
        self.discfibers_mat_path = None
        self.ftr_mat_path = None
        self.tck_path = None
        self.track_mat_path = None
        self.metadata_path = None
        self.result = None

    def run(self):
        self.validate_inputs()
        self.prepare_output_dir()
        self.resolve_output_name()
        self.load_fibers_and_values()
        self.validate_fiber_value_alignment()
        self.apply_selection()
        self.apply_symmetry()
        self.save_discfibers_mat_file()
        self.save_ftr_mat_file()
        self.save_tck_file()
        self.save_track_mat_file()
        self.save_metadata()
        self.package_result()
        return self.get_result()

    def validate_inputs(self):
        if self.values_path is None and self.values is None and self.fiber_atlas_path is None:
            raise ValueError("Provide values_path, values, or fiber_atlas_path.")
        if self.values_path is not None and not self.values_path.exists():
            raise FileNotFoundError(f"values_path does not exist: {self.values_path}")
        if self.fiber_atlas_path is not None and not self.fiber_atlas_path.exists():
            raise FileNotFoundError(f"fiber_atlas_path does not exist: {self.fiber_atlas_path}")
        if self.sign not in {"both", "positive", "pos", "negative", "neg"}:
            raise ValueError("sign must be one of: both, positive, pos, negative, neg.")
        if self.top_percent is not None and not (0 < float(self.top_percent) <= 100):
            raise ValueError("top_percent must be in (0, 100].")
        if not str(self.tck_suffix).endswith(".tck"):
            raise ValueError("tck_suffix must end with '.tck'.")
        if self.mirror_axis not in {"x", "y", "z", 0, 1, 2}:
            raise ValueError("mirror_axis must be one of: x, y, z, 0, 1, 2.")

    def prepare_output_dir(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def resolve_output_name(self):
        if self.output_name is not None:
            return
        if self.values_path is not None:
            name = self.values_path.name
            self.output_name = name[:-8] if name.endswith(".fib.npy") else self.values_path.stem
        elif self.fiber_atlas_path is not None:
            name = self.fiber_atlas_path.name
            self.output_name = name[:-4] if name.endswith(".npz") else self.fiber_atlas_path.stem
        else:
            self.output_name = "fiber_result"

    def load_fibers_and_values(self):
        if self._values_path_is_fib_npy():
            self.fibers, loaded_values = self._load_fib_npy(self.values_path)
            self.values = loaded_values if self.values is None else self.values
            return

        if self.fiber_atlas_path is None and self._values_path_is_atlas():
            self.fiber_atlas_path = self.values_path
            self.values_path = None

        if self.fiber_atlas_path is None:
            raise ValueError("fiber_atlas_path is required when values_path is not a .fib.npy geometry file.")
        self.fibers = self._load_atlas_fibers(self.fiber_atlas_path)
        if self.values is None:
            self.values = (
                np.ones(len(self.fibers), dtype=np.float32)
                if self.values_path is None
                else self._load_value_vector(self.values_path)
            )

    def validate_fiber_value_alignment(self):
        if len(self.fibers) != self.values.shape[0]:
            raise ValueError(
                f"Fiber/value length mismatch: {len(self.fibers)} fibers but {self.values.shape[0]} values."
            )

    def apply_selection(self):
        vals = np.asarray(self.values, dtype=np.float32).flatten()
        keep = np.isfinite(vals)

        if self.sign in {"positive", "pos"}:
            keep &= vals > 0
        elif self.sign in {"negative", "neg"}:
            keep &= vals < 0

        if self.min_abs_value is not None:
            keep &= np.abs(vals) >= float(self.min_abs_value)

        if self.top_percent is not None and np.any(keep):
            kept_abs = np.abs(vals[keep])
            percentile = 100.0 - float(self.top_percent)
            cutoff = np.nanpercentile(kept_abs, percentile)
            keep &= np.abs(vals) >= cutoff

        self.keep_mask = keep
        self.selected_fibers = [fiber for fiber, selected in zip(self.fibers, keep) if selected]
        self.selected_values = vals[keep].astype(np.float32)
        self.n_selected_before_symmetry = int(len(self.selected_fibers))

    def apply_symmetry(self):
        if not self.symmetric:
            return
        mirrored_fibers = [self._mirror_fiber(fiber) for fiber in self.selected_fibers]
        self.selected_fibers = self.selected_fibers + mirrored_fibers
        self.selected_values = np.concatenate([self.selected_values, self.selected_values]).astype(np.float32)

    def save_discfibers_mat_file(self):
        if not self.save_discfibers_mat:
            return
        self.discfibers_mat_path = self.out_dir / f"{self.output_name}_discfibers.mat"
        savemat(
            self.discfibers_mat_path,
            {
                "fibcell": self._matlab_cell_array(self.selected_fibers),
                "vals": self.selected_values[:, None],
                "isdiscfibers": np.asarray(1),
            },
            do_compression=True,
        )

    def save_ftr_mat_file(self):
        if not self.save_ftr_mat:
            return
        self.ftr_mat_path = self.out_dir / f"{self.output_name}_ftr.mat"
        savemat(
            self.ftr_mat_path,
            {
                "ea_fibformat": "1.1",
                "fourindex": np.asarray(1),
                "fibers": self._lead_fiber_matrix(self.selected_fibers),
                "idx": self._fiber_lengths(self.selected_fibers)[:, None],
                "vals": self.selected_values[:, None],
                "voxmm": "mm",
            },
            do_compression=True,
        )

    def save_tck_file(self):
        if not self.save_tck:
            return
        self.tck_path = self.out_dir / f"{self.output_name}{self.tck_suffix}"
        tractogram = Tractogram(
            streamlines=[np.asarray(fiber[:, :3], dtype=np.float32) for fiber in self.selected_fibers],
            affine_to_rasmm=np.eye(4),
        )
        nib.streamlines.save(tractogram, str(self.tck_path))

    def save_track_mat_file(self):
        if not self.save_track_mat:
            return
        self.track_mat_path = self.out_dir / f"{self.output_name}_track.mat"
        savemat(
            self.track_mat_path,
            {
                "track": self._matlab_cell_array(self.selected_fibers),
                "vals": self.selected_values[:, None],
                "voxmm": "mm",
                "coordinate_space": "MNI/world",
            },
            do_compression=True,
        )

    def save_metadata(self):
        self.metadata_path = self.out_dir / f"{self.output_name}_metadata.json"
        self.metadata_path.write_text(json.dumps(self._metadata(), indent=2))

    def package_result(self):
        self.result = {
            "discfibers_mat": str(self.discfibers_mat_path) if self.discfibers_mat_path is not None else None,
            "ftr_mat": str(self.ftr_mat_path) if self.ftr_mat_path is not None else None,
            "tck": str(self.tck_path) if self.tck_path is not None else None,
            "track_mat": str(self.track_mat_path) if self.track_mat_path is not None else None,
            "metadata": str(self.metadata_path),
            "n_input_fibers": int(len(self.fibers)),
            "n_selected_before_symmetry": self.n_selected_before_symmetry,
            "n_exported_fibers": int(len(self.selected_fibers)),
            "symmetric": self.symmetric,
            "values": self.selected_values,
            "keep_mask": self.keep_mask,
        }

    def get_result(self):
        return self.result

    def _metadata(self):
        return {
            "values_path": str(self.values_path) if self.values_path is not None else None,
            "fiber_atlas_path": str(self.fiber_atlas_path) if self.fiber_atlas_path is not None else None,
            "output_name": self.output_name,
            "sign": self.sign,
            "min_abs_value": self.min_abs_value,
            "top_percent": self.top_percent,
            "n_input_fibers": int(len(self.fibers)),
            "n_selected_before_symmetry": self.n_selected_before_symmetry,
            "n_exported_fibers": int(len(self.selected_fibers)),
            "symmetric": self.symmetric,
            "mirror_axis": self.mirror_axis,
            "mirror_origin": self.mirror_origin,
            "discfibers_mat": str(self.discfibers_mat_path) if self.discfibers_mat_path is not None else None,
            "ftr_mat": str(self.ftr_mat_path) if self.ftr_mat_path is not None else None,
            "tck": str(self.tck_path) if self.tck_path is not None else None,
            "track_mat": str(self.track_mat_path) if self.track_mat_path is not None else None,
        }

    def _values_path_is_fib_npy(self):
        return self.values_path is not None and self.values_path.name.endswith(".fib.npy")

    def _mirror_fiber(self, fiber):
        mirrored = np.asarray(fiber, dtype=np.float32).copy()
        axis = self._axis_index(self.mirror_axis)
        mirrored[:, axis] = (2.0 * self.mirror_origin) - mirrored[:, axis]
        return mirrored

    @staticmethod
    def _axis_index(axis):
        if axis in {0, 1, 2}:
            return int(axis)
        return {"x": 0, "y": 1, "z": 2}[axis]

    def _values_path_is_atlas(self):
        if self.values_path is None:
            return False
        suffixes = [suffix.lower() for suffix in self.values_path.suffixes]
        if suffixes[-1:] == [".npz"]:
            obj = np.load(self.values_path, allow_pickle=True)
            return "fibers" in obj
        if suffixes[-1:] == [".npy"]:
            obj = np.load(self.values_path, allow_pickle=True)
            return isinstance(obj, np.ndarray) and obj.dtype == object
        return False

    @staticmethod
    def _load_fib_npy(path: Path):
        obj = np.load(path, allow_pickle=True)
        fibers = [np.asarray(fiber, dtype=np.float32) for fiber in obj.tolist()]
        values = []
        xyz_fibers = []
        for fiber in fibers:
            if fiber.ndim != 2 or fiber.shape[1] not in (3, 4):
                raise ValueError(f"Invalid fiber shape in {path}: {fiber.shape}")
            xyz_fibers.append(fiber[:, :3].astype(np.float32, copy=False))
            if fiber.shape[1] == 4:
                values.append(float(np.nanmedian(fiber[:, 3])))
            else:
                values.append(np.nan)
        return xyz_fibers, np.asarray(values, dtype=np.float32)

    @staticmethod
    def _load_atlas_fibers(path: Path):
        suffixes = [suffix.lower() for suffix in path.suffixes]
        if suffixes[-1:] == [".npz"]:
            obj = np.load(path, allow_pickle=True)
            if "fibers" not in obj:
                raise ValueError(f"NPZ atlas missing 'fibers' key: {path}")
            fibers = obj["fibers"].tolist()
        elif suffixes[-1:] == [".npy"]:
            obj = np.load(path, allow_pickle=True)
            fibers = obj.tolist()
        else:
            raise ValueError(f"Unsupported fiber atlas format: {path}")
        return [np.asarray(fiber, dtype=np.float32)[:, :3] for fiber in fibers]

    @staticmethod
    def _load_value_vector(path: Path):
        arr = np.load(path, allow_pickle=True)
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D value vector in {path}, got shape {arr.shape}")
        return arr.astype(np.float32)

    @staticmethod
    def _matlab_cell_array(fibers):
        cell = np.empty((len(fibers), 1), dtype=object)
        for i, fiber in enumerate(fibers):
            cell[i, 0] = np.asarray(fiber[:, :3], dtype=np.float32)
        return cell

    @staticmethod
    def _fiber_lengths(fibers):
        return np.asarray([fiber.shape[0] for fiber in fibers], dtype=np.int64)

    @classmethod
    def _lead_fiber_matrix(cls, fibers):
        chunks = []
        for local_id, fiber in enumerate(fibers, start=1):
            local_col = np.full((fiber.shape[0], 1), local_id, dtype=np.float32)
            chunks.append(np.hstack([fiber[:, :3], local_col]))
        if not chunks:
            return np.empty((0, 4), dtype=np.float32)
        return np.vstack(chunks).astype(np.float32, copy=False)
