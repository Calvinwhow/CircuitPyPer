import os
import json
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
from pathlib import Path
from tqdm import tqdm


class FiberVoxelIndexer:
    """
    Build a voxel index for a canonical fiber library on a reference NIfTI grid.

    Core representation:
        self.fibers[fiber_idx]            -> ndarray, shape (n_vertices, 3) or (n_vertices, 4)
        self.fiber_voxel_indices[fiber_idx] -> 1D ndarray of unique linear voxel indices touched by that fiber

    The index is built in the voxel lattice of reference_nifti_path.
    """

    def __init__(self, reference_nifti_path, fiber_mask=None, step_size_vox=0.5):
        self.reference_nifti_path = reference_nifti_path
        self.reference_img = nib.load(reference_nifti_path)
        self.reference_affine = self.reference_img.affine
        self.reference_shape = self.reference_img.shape[:3]
        self.inv_affine = np.linalg.inv(self.reference_affine)

        self.fiber_mask = fiber_mask
        self.step_size_vox = float(step_size_vox)

        self.fibers = None
        self.fiber_voxel_indices = None

    @staticmethod
    def _identify_fiber_file_type(path):
        p = Path(path)
        suffixes = [s.lower() for s in p.suffixes]

        if suffixes[-1:] == ['.trk']:
            return 'trk'
        if suffixes[-1:] == ['.tck']:
            return 'tck'
        if suffixes[-1:] == ['.trx']:
            return 'trx'
        if suffixes[-1:] == ['.npy']:
            return 'npy'
        if suffixes[-1:] == ['.npz']:
            return 'npz'
        if suffixes[-1:] == ['.json']:
            return 'json'

        return 'unknown'

    @staticmethod
    def _parse_tck_header(path):
        header = {}
        with open(path, "rb") as f:
            while True:
                line = f.readline()
                if line == b"":
                    raise ValueError(f"TCK header ended before END marker: {path}")
                text = line.decode("utf-8", errors="replace").strip()
                if text == "END":
                    break
                if ":" in text:
                    key, value = text.split(":", 1)
                    header[key.strip().lower()] = value.strip()

        file_field = header.get("file", "")
        parts = file_field.split()
        if len(parts) != 2 or parts[0] != ".":
            raise ValueError(f"Only inline TCK data are supported, got file field: {file_field}")
        header["data_offset"] = int(parts[1])
        return header

    @classmethod
    def _load_tck_tolerant(cls, path):
        """
        Load MRtrix TCK streamlines while tolerating files that omit the final
        ``inf inf inf`` marker. DSI Studio can write TCK-like files with valid
        NaN streamline delimiters but without nibabel's expected EOF row.
        """
        header = cls._parse_tck_header(path)
        datatype = header.get("datatype", "Float32LE").lower()
        dtype_map = {
            "float32le": "<f4",
            "float32be": ">f4",
            "float64le": "<f8",
            "float64be": ">f8",
        }
        if datatype not in dtype_map:
            raise ValueError(f"Unsupported TCK datatype '{header.get('datatype')}' in {path}")

        raw = np.fromfile(path, dtype=np.dtype(dtype_map[datatype]), offset=header["data_offset"])
        n_values = (raw.size // 3) * 3
        if n_values == 0:
            return []
        if n_values != raw.size:
            raw = raw[:n_values]
        points = raw.reshape(-1, 3)

        separators = np.where(~np.isfinite(points).all(axis=1))[0]
        fibers = []
        start = 0
        for stop in separators:
            row = points[stop]
            if np.isinf(row).all():
                break
            if stop > start:
                fibers.append(points[start:stop].astype(np.float32, copy=True))
            start = stop + 1

        if start < points.shape[0]:
            tail = points[start:]
            tail = tail[np.isfinite(tail).all(axis=1)]
            if tail.shape[0] > 0:
                fibers.append(tail.astype(np.float32, copy=True))

        fibers = cls._maybe_apply_known_dsi_tck_transform(fibers, header)
        return fibers

    @staticmethod
    def _maybe_apply_known_dsi_tck_transform(fibers, header):
        if not fibers:
            return fibers

        dim_text = header.get("dim", "")
        try:
            dim = tuple(int(part.strip()) for part in dim_text.split(","))
        except ValueError:
            return fibers

        if dim != (157, 189, 136):
            return fibers

        sample = np.concatenate([fiber[: min(10, len(fiber)), :3] for fiber in fibers[: min(100, len(fibers))]])
        if sample.size == 0:
            return fibers

        lower_ok = np.nanmin(sample, axis=0) >= -1.0
        upper_ok = np.nanmax(sample, axis=0) <= (np.asarray(dim, dtype=np.float32) + 1.0)
        if not (np.all(lower_ok) and np.all(upper_ok)):
            return fibers

        affine = np.asarray(
            [
                [-1.0, 0.0, 0.0, 78.0],
                [0.0, -1.0, 0.0, 76.0],
                [0.0, 0.0, 1.0, -50.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return [nib.affines.apply_affine(affine, fiber[:, :3]).astype(np.float32) for fiber in fibers]

    @classmethod
    def from_fiber_file(cls, fiber_file_path, reference_nifti_path, fiber_mask=None, step_size_vox=0.5):
        obj = cls(
            reference_nifti_path=reference_nifti_path,
            fiber_mask=fiber_mask,
            step_size_vox=step_size_vox,
        )
        fibers = obj.load_fibers(fiber_file_path)
        obj.build_index(fibers)
        return obj

    def load_fibers(self, fiber_file_path):
        """
        Return list of fibers, each shaped (n_vertices, 3) or (n_vertices, 4).
        For streamline files, only xyz are loaded.
        """
        return list(self.iter_fibers(fiber_file_path))

    def iter_fibers(self, fiber_file_path):
        """Yield fibers without materializing tractogram geometry when possible.

        TRK/TCK/TRX files use nibabel's lazy tractogram loader. Object-array
        NPY/NPZ and JSON containers are inherently materialized by their storage
        formats, but are yielded one fiber at a time without creating a second
        full Python list.
        """
        ftype = self._identify_fiber_file_type(fiber_file_path)
        fiber_mask = None
        if self.fiber_mask is not None:
            fiber_mask = np.asarray(self.fiber_mask).astype(bool).reshape(-1)

        if ftype in {'trk', 'trx'}:
            tractogram = nib.streamlines.load(
                fiber_file_path, lazy_load=True
            ).tractogram
            source = tractogram.streamlines

        elif ftype == 'tck':
            try:
                tractogram = nib.streamlines.load(
                    fiber_file_path, lazy_load=True
                ).tractogram
                source = tractogram.streamlines
            except Exception:
                source = self._load_tck_tolerant(fiber_file_path)

        elif ftype == 'npy':
            obj = np.load(fiber_file_path, allow_pickle=True)
            if isinstance(obj, np.ndarray) and obj.dtype == object:
                source = obj
            else:
                raise ValueError(f"Expected object-array of fibers in {fiber_file_path}")

        elif ftype == 'npz':
            obj = np.load(fiber_file_path, allow_pickle=True)
            if 'fibers' not in obj:
                raise ValueError(f"NPZ fiber file missing 'fibers' key: {fiber_file_path}")
            source = obj['fibers']

        elif ftype == 'json':
            with open(fiber_file_path, 'r') as f:
                obj = json.load(f)
            if 'fibers' not in obj:
                raise ValueError(f"JSON fiber file missing 'fibers' key: {fiber_file_path}")
            source = obj['fibers']

        else:
            raise RuntimeError(f"Unsupported fiber file type: {fiber_file_path}")

        count = 0
        for index, value in enumerate(source):
            count = index + 1
            if fiber_mask is not None and index >= fiber_mask.size:
                raise ValueError(
                    f"fiber_mask length ({fiber_mask.size}) is shorter than the "
                    "number of fibers."
                )
            fiber = np.asarray(value, dtype=np.float32)
            if fiber.ndim != 2 or fiber.shape[1] not in (3, 4):
                raise ValueError(
                    f"Fiber {index} has invalid shape {fiber.shape}. "
                    f"Expected (n_vertices, 3) or (n_vertices, 4)."
                )
            if fiber_mask is None or fiber_mask[index]:
                yield fiber

        if fiber_mask is not None and fiber_mask.size != count:
            raise ValueError(
                f"fiber_mask length ({fiber_mask.size}) does not match number "
                f"of fibers ({count})."
            )

    def load_selected_fibers(
        self,
        fiber_file_path,
        selected_indices,
        expected_count=None,
    ):
        """Load only selected atlas indices when the source format permits it."""
        selected_indices = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
        if selected_indices.size == 0:
            return []
        if np.any(selected_indices < 0) or np.any(np.diff(selected_indices) <= 0):
            raise ValueError("selected_indices must be sorted, unique, and non-negative.")

        ftype = self._identify_fiber_file_type(fiber_file_path)
        source = None
        if ftype == "npy":
            source = np.load(fiber_file_path, allow_pickle=True)
            if source.dtype != object:
                raise ValueError(
                    f"Expected object-array of fibers in {fiber_file_path}"
                )
        elif ftype == "npz":
            archive = np.load(fiber_file_path, allow_pickle=True)
            if "fibers" not in archive:
                raise ValueError(
                    f"NPZ fiber file missing 'fibers' key: {fiber_file_path}"
                )
            source = archive["fibers"]
        elif ftype == "json":
            with open(fiber_file_path) as fiber_file:
                payload = json.load(fiber_file)
            if "fibers" not in payload:
                raise ValueError(
                    f"JSON fiber file missing 'fibers' key: {fiber_file_path}"
                )
            source = payload["fibers"]

        if source is not None:
            total = len(source)
            if expected_count is not None and total != int(expected_count):
                raise ValueError(
                    f"Fiber atlas contains {total} fibers but the values vector "
                    f"contains {expected_count}."
                )
            if selected_indices[-1] >= total:
                raise ValueError("Selected fiber indices exceed the atlas fiber count.")
            selected = [
                np.asarray(source[index], dtype=np.float32)
                for index in selected_indices
            ]
            for index, fiber in zip(selected_indices, selected):
                if fiber.ndim != 2 or fiber.shape[1] not in (3, 4):
                    raise ValueError(
                        f"Fiber {index} has invalid shape {fiber.shape}."
                    )
            return selected

        selected = []
        wanted_position = 0
        total = 0
        for index, fiber in enumerate(self.iter_fibers(fiber_file_path)):
            total = index + 1
            if wanted_position < selected_indices.size and index == selected_indices[wanted_position]:
                selected.append(fiber)
                wanted_position += 1

        if expected_count is not None and total != int(expected_count):
            raise ValueError(
                f"Fiber atlas contains {total} fibers but the values vector "
                f"contains {expected_count}."
            )
        if wanted_position != selected_indices.size:
            raise ValueError("Selected fiber indices exceed the atlas fiber count.")
        return selected

    def world_to_voxel(self, xyz_world):
        """
        xyz_world: ndarray (..., 3) in world/MNI coordinates
        returns voxel coordinates in floating point index space
        """
        xyz_world = np.asarray(xyz_world, dtype=np.float32)
        orig_shape = xyz_world.shape
        flat = xyz_world.reshape(-1, 3)
        hom = np.concatenate([flat, np.ones((flat.shape[0], 1), dtype=np.float32)], axis=1)
        vox = hom @ self.inv_affine.T
        return vox[:, :3].reshape(orig_shape)

    def _segment_voxel_indices(self, p0_world, p1_world):
        """
        Rasterize a single segment by dense sampling in voxel space.
        Returns unique linear voxel indices crossed by the segment.
        """
        p0_vox = self.world_to_voxel(np.asarray(p0_world))[0:3]
        p1_vox = self.world_to_voxel(np.asarray(p1_world))[0:3]

        return self._voxel_segment_indices(p0_vox, p1_vox)

    def _voxel_segment_indices(self, p0_vox, p1_vox):
        """Rasterize a segment whose endpoints are already in voxel space."""
        delta = p1_vox - p0_vox
        dist = float(np.linalg.norm(delta))

        if dist == 0:
            pts = np.asarray([p0_vox], dtype=np.float32)
        else:
            n_steps = max(2, int(np.ceil(dist / self.step_size_vox)) + 1)
            t = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)[:, None]
            pts = p0_vox[None, :] + t * delta[None, :]

        ijk = np.round(pts).astype(np.int32)

        in_bounds = (
            (ijk[:, 0] >= 0) & (ijk[:, 0] < self.reference_shape[0]) &
            (ijk[:, 1] >= 0) & (ijk[:, 1] < self.reference_shape[1]) &
            (ijk[:, 2] >= 0) & (ijk[:, 2] < self.reference_shape[2])
        )

        ijk = ijk[in_bounds]
        if ijk.shape[0] == 0:
            return np.empty(0, dtype=np.int64)

        lin = np.ravel_multi_index(
            (ijk[:, 0], ijk[:, 1], ijk[:, 2]),
            dims=self.reference_shape
        )

        return np.unique(lin)

    def _fiber_voxel_index(self, fiber_xyz):
        """
        Convert one fiber polyline into the unique set of voxel indices it traverses.
        """
        fiber_xyz = np.asarray(fiber_xyz, dtype=np.float32)
        if fiber_xyz.shape[0] == 0:
            return np.empty(0, dtype=np.int64)
        if fiber_xyz.shape[0] == 1:
            p = self.world_to_voxel(fiber_xyz[:, :3])[0]
            p = np.round(p).astype(np.int32)
            if (
                0 <= p[0] < self.reference_shape[0] and
                0 <= p[1] < self.reference_shape[1] and
                0 <= p[2] < self.reference_shape[2]
            ):
                return np.asarray([np.ravel_multi_index((p[0], p[1], p[2]), self.reference_shape)], dtype=np.int64)
            return np.empty(0, dtype=np.int64)

        xyz = fiber_xyz[:, :3]
        voxel_xyz = self.world_to_voxel(xyz)
        starts = voxel_xyz[:-1]
        deltas = voxel_xyz[1:] - starts
        distances = np.linalg.norm(deltas, axis=1)
        steps_per_segment = np.where(
            distances == 0,
            1,
            np.maximum(2, np.ceil(distances / self.step_size_vox).astype(np.int64) + 1),
        ).astype(np.int64)
        total_steps = int(steps_per_segment.sum())
        if total_steps == 0:
            return np.empty(0, dtype=np.int64)

        segment_ids = np.repeat(
            np.arange(starts.shape[0], dtype=np.int64),
            steps_per_segment,
        )
        segment_offsets = np.repeat(
            np.cumsum(steps_per_segment) - steps_per_segment,
            steps_per_segment,
        )
        local_steps = np.arange(total_steps, dtype=np.int64) - segment_offsets
        denominators = np.maximum(steps_per_segment[segment_ids] - 1, 1)
        fractions = (local_steps / denominators).astype(np.float32)[:, None]
        points = starts[segment_ids] + fractions * deltas[segment_ids]
        ijk = np.round(points).astype(np.int32)
        in_bounds = (
            (ijk[:, 0] >= 0) & (ijk[:, 0] < self.reference_shape[0]) &
            (ijk[:, 1] >= 0) & (ijk[:, 1] < self.reference_shape[1]) &
            (ijk[:, 2] >= 0) & (ijk[:, 2] < self.reference_shape[2])
        )
        ijk = ijk[in_bounds]
        if ijk.shape[0] == 0:
            return np.empty(0, dtype=np.int64)
        return np.unique(
            np.ravel_multi_index(
                (ijk[:, 0], ijk[:, 1], ijk[:, 2]),
                dims=self.reference_shape,
            )
        )

    def voxel_indices_for_fiber(self, fiber):
        """Return unique reference-grid voxel indices for one world-space fiber."""
        return self._fiber_voxel_index(np.asarray(fiber, dtype=np.float32)[:, :3])

    def build_index(self, fibers):
        """
        fibers: list of arrays, each (n_vertices, 3) or (n_vertices, 4)
        """
        self.fibers = [np.asarray(f, dtype=np.float32) for f in fibers]
        self.fiber_voxel_indices = []

        for fiber in tqdm(self.fibers, desc='Indexing fibers into voxel space'):
            self.fiber_voxel_indices.append(self._fiber_voxel_index(fiber[:, :3]))

        return self

    def query_image_hits(self, voxel_data):
        """
        voxel_data must already be on the reference grid.

        Returns:
            fiber_hit_flags: bool array, shape (n_fibers,)
            hit_linear_indices_per_fiber: list of 1D arrays
        """
        flat = voxel_data.reshape(-1)
        active = flat != 0

        n_fibers = len(self.fiber_voxel_indices)
        hit_flags = np.zeros(n_fibers, dtype=bool)
        hit_linear_indices_per_fiber = [None] * n_fibers

        for i, lin_idx in enumerate(self.fiber_voxel_indices):
            if lin_idx.size == 0:
                hit_linear_indices_per_fiber[i] = np.empty(0, dtype=np.int64)
                continue

            local_hits = lin_idx[active[lin_idx]]
            if local_hits.size > 0:
                hit_flags[i] = True
            hit_linear_indices_per_fiber[i] = local_hits

        return hit_flags, hit_linear_indices_per_fiber
