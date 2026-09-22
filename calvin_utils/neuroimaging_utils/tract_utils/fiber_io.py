import os
import json
import hashlib
import numpy as np
import nibabel as nib
from datetime import datetime, timezone
from pathlib import Path
from tqdm import tqdm
from calvin_utils.neuroimaging_utils.tract_utils.fiber_converter import FiberFormatConverter
from calvin_utils.neuroimaging_utils.tract_utils.fiber_result_visualizer import FiberResultVisualizer
from calvin_utils.neuroimaging_utils.tract_utils.tract_density import TractDensity
PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_FIBER_MASK = None
DEFAULT_MNI_MASK = os.path.join(PACKAGE_ROOT, "resources", "MNI152_T1_2mm_brain_mask.nii")

class FiberIO:
    """Fiber-space I/O using a canonical, ordered streamline atlas.

    Core assumptions
    ----------------
    1. Each patient file represents magnitudes over the SAME ordered fiber set.
    2. The reference mask/library stores the canonical polylines.
    3. Regression operates on per-fiber magnitudes only, not raw polyline vertices.

    Canonical regression output
    ---------------------------
    ``save_files`` writes an atlas-referenced pair::

        <stem>.fib.npy  float32 vector with one value per atlas fiber
        <stem>.fib.json required descriptor for the values vector

    ``<stem>.fib.npy`` and ``<stem>.fib.json`` are a logical pair
    and must be copied, renamed, or archived together. The descriptor records
    both relative and absolute paths to the values file, so the JSON can be
    used as the entry point. The vector contains no
    geometry. Its element ``i`` applies to fiber ``i`` in the canonical atlas
    named by the descriptor. The descriptor stores absolute and relative atlas
    paths, counts, dtype/shape, ordering, mask/fill behavior, summary values,
    the values-file SHA-256, and the source geometry file when one exists.

    Consumers should prefer the descriptor's relative atlas path when it still
    resolves and otherwise use its absolute path. Descriptor-aware consumers
    validate the vector hash, atlas file size, and fiber/value counts before
    reconstructing streamlines. Legacy ``*.fib.values.npy``/``*.fib.desc.json``
    pairs and geometry-bearing ``*.fib.npy`` files remain readable, but new
    output is written only with the compact names above.

    Masking conventions
    -------------------
    ``import_fiber_to_numpy_array`` returns ``(n_fibers_kept, n_files)``.
    Masks operate on fiber indices, never vertices. Before output, masked
    vectors are expanded to the full atlas order using ``fill_value``; both the
    geometry-bearing and lightweight outputs therefore have exactly one entry
    per canonical atlas fiber.
    """

    output_ftype = "fiber"

    def __init__(self, mask_path=None, threshold=0):
        self.mask_path = DEFAULT_FIBER_MASK if mask_path == 'default' else mask_path
        self.threshold = threshold
        self._reference_fibers = None
        self._fiber_mask = None
        self._density_projector_cache = None
        self._mirror_neighbor_cache = {}

    def validate_for_output(self):
        if self.mask_path is None:
            raise ValueError(
                "Fiber output requires the canonical .npz fiber atlas as mask_path."
            )
        # Resolve now so configuration errors are reported before a regression runs.
        self.reference_fibers

    @property
    def reference_fibers(self):
        if self._reference_fibers is None and self.mask_path is not None:
            self._reference_fibers = self._load_reference_fibers(self.mask_path)
        return self._reference_fibers

    @property
    def fiber_mask(self):
        """
        Boolean mask over reference fibers.
        If no explicit mask is encoded, all reference fibers are kept.
        """
        if self.mask_path is None:
            return None
        if self._fiber_mask is None:
            self._fiber_mask = self._resolve_fiber_mask(self.mask_path, self.threshold)
        return self._fiber_mask

    def _identify_fiber_file_type(self, path):
        p = Path(path)
        suffixes = [s.lower() for s in p.suffixes]

        if suffixes[-1:] == ['.npy']:
            return 'npy'
        if suffixes[-1:] == ['.npz']:
            return 'npz'
        if suffixes[-1:] == ['.json']:
            return 'json'
        if suffixes[-1:] == ['.fibfilt']:
            return 'fibfilt'

        return 'unknown'

    def _load_reference_fibers(self, mask_path):
        """
        Load canonical fiber library.

        Expected returned structure:
        fibers = list of arrays
        fibers[i].shape == (n_vertices_i, 3) or (n_vertices_i, 4)

        For now:
        - .npy may contain an object array/list of fibers
        - .npz may contain key 'fibers'
        - .json may contain {"fibers": [[[x,y,z], ...], ...]}
        """
        ftype = self._identify_fiber_file_type(mask_path)

        if ftype == 'npy':
            obj = np.load(mask_path, allow_pickle=True)
            if isinstance(obj, np.ndarray) and obj.dtype == object:
                fibers = obj.tolist()
            else:
                raise ValueError(f"Expected object-array fiber library in {mask_path}")
        elif ftype == 'npz':
            obj = np.load(mask_path, allow_pickle=True)
            if 'fibers' not in obj:
                raise ValueError(f"NPZ fiber library missing 'fibers' key: {mask_path}")
            fibers = obj['fibers'].tolist()
        elif ftype == 'json':
            with open(mask_path, 'r') as f:
                obj = json.load(f)
            if 'fibers' not in obj:
                raise ValueError(f"JSON fiber library missing 'fibers' key: {mask_path}")
            fibers = obj['fibers']
        elif ftype == 'fibfilt':
            raise NotImplementedError("Add .fibfilt reference fiber parsing here.")
        else:
            raise RuntimeError(f"Unknown or unsupported fiber mask file type: {mask_path}")

        fibers = [np.asarray(f, dtype=np.float32) for f in fibers]

        if len(fibers) == 0:
            raise ValueError("Reference fiber library is empty.")

        for i, fiber in enumerate(fibers):
            if fiber.ndim != 2 or fiber.shape[1] not in (3, 4):
                raise ValueError(
                    f"Fiber {i} has invalid shape {fiber.shape}. "
                    f"Expected (n_vertices, 3) or (n_vertices, 4)."
                )

        return fibers

    def _resolve_fiber_mask(self, mask_path=None, threshold=None):
        """
        Resolve a boolean mask over the canonical reference fibers.

        Supported patterns for now:
        1. Reference library only -> keep all fibers
        2. NPZ/JSON may optionally contain per-fiber mask vector
        """
        mask_path = self.mask_path if mask_path is None else mask_path
        threshold = self.threshold if threshold is None else threshold

        if mask_path is None:
            return None

        ftype = self._identify_fiber_file_type(mask_path)

        if ftype == 'npz':
            obj = np.load(mask_path, allow_pickle=True)
            if 'fiber_mask' in obj:
                mask = np.asarray(obj['fiber_mask']).flatten() > threshold
                if self.reference_fibers is not None and mask.shape[0] != len(self.reference_fibers):
                    raise ValueError("fiber_mask length does not match reference fiber count.")
                return mask

        if ftype == 'json':
            with open(mask_path, 'r') as f:
                obj = json.load(f)
            if 'fiber_mask' in obj:
                mask = np.asarray(obj['fiber_mask']).flatten() > threshold
                if self.reference_fibers is not None and mask.shape[0] != len(self.reference_fibers):
                    raise ValueError("fiber_mask length does not match reference fiber count.")
                return mask

        return np.ones(len(self.reference_fibers), dtype=bool)

    def _load_single_fiber_values(self, file_path):
        """
        Load one patient’s per-fiber magnitude vector.

        Accepted forms for now:
        - .npy: 1D vector length n_fibers
        - .npz: key 'values' or 'fiber_values'
        - .json: key 'values' or 'fiber_values'
        - .fibfilt: add parser later
        """
        path = Path(file_path)
        lower_name = path.name.lower()
        # The atlas the caller supplied is the one to validate against. The
        # descriptor's recorded paths are tried after it, so a result still
        # opens on the machine that wrote it, and on one where the atlas lives
        # elsewhere. Every integrity check -- checksum, size -- still applies.
        atlas_override = self.mask_path if self.mask_path not in (None, "default") else None

        if lower_name.endswith((".fib.json", ".fib.desc.json")):
            path = FiberFormatConverter.values_from_description(path)
            FiberFormatConverter.atlas_from_values_description(path, atlas_override=atlas_override)
            lower_name = path.name.lower()

        ftype = self._identify_fiber_file_type(path)

        if ftype == 'npy':
            arr = np.load(path, allow_pickle=True)
            if arr.dtype == object:
                return self._extract_geometry_values(arr, path)
            if arr.ndim != 1:
                raise ValueError(f"Expected 1D fiber value vector in {path}, got shape {arr.shape}")
            if lower_name.endswith((".fib.npy", ".values.npy")):
                description_path = FiberFormatConverter.values_description_path(path)
                if lower_name.endswith(".values.npy") or description_path.is_file():
                    FiberFormatConverter.atlas_from_values_description(
                        path, atlas_override=atlas_override)
            return arr.astype(np.float32)

        if ftype == 'npz':
            obj = np.load(file_path, allow_pickle=True)
            key = 'values' if 'values' in obj else 'fiber_values' if 'fiber_values' in obj else None
            if key is None:
                raise ValueError(f"NPZ file missing 'values' or 'fiber_values' key: {file_path}")
            arr = np.asarray(obj[key]).flatten()
            return arr.astype(np.float32)

        if ftype == 'json':
            with open(file_path, 'r') as f:
                obj = json.load(f)
            key = 'values' if 'values' in obj else 'fiber_values' if 'fiber_values' in obj else None
            if key is None:
                raise ValueError(f"JSON file missing 'values' or 'fiber_values' key: {file_path}")
            arr = np.asarray(obj[key]).flatten()
            return arr.astype(np.float32)

        if ftype == 'fibfilt':
            raise NotImplementedError("Add .fibfilt patient-value parsing here.")

        raise RuntimeError(f"Unknown or unsupported fiber file type: {file_path}")

    @staticmethod
    def _extract_geometry_values(arr, source):
        """Reduce a geometry-bearing native fiber map to one value per fiber."""
        if arr.ndim == 3 and arr.shape[-1] >= 4:
            fiber_items = arr
        elif arr.ndim == 1:
            if arr.size == 0:
                return np.asarray([], dtype=np.float32)
            if np.asarray(arr[0]).ndim == 0:
                return np.asarray(arr, dtype=np.float32)
            fiber_items = arr
        else:
            raise ValueError(
                f"Cannot identify fibers in {source}; object-array shape={arr.shape}"
            )

        values = []
        for item in fiber_items:
            fiber = np.asarray(item, dtype=np.float32)
            if fiber.ndim != 2 or fiber.shape[1] < 4:
                raise ValueError(
                    f"Cannot identify fiber values in {source}; item shape={fiber.shape}"
                )
            statistic = fiber[:, 3]
            finite = statistic[np.isfinite(statistic)]
            values.append(float(np.median(finite)) if finite.size else np.nan)
        return np.asarray(values, dtype=np.float32)

    @staticmethod
    def mask_array(arr, mask):
        """
        Fiber-level masking.
        arr shape:
        - 1D: (n_fibers,)
        - 2D: (n_fibers, n_files)
        """
        if mask is None:
            mask_indices = np.arange(arr.shape[0])
            return None, mask_indices, arr

        if arr.ndim == 1:
            masked_arr = arr[mask]
        else:
            masked_arr = arr[mask, :]

        mask_indices = np.where(mask)[0]
        return mask.astype(int), mask_indices, masked_arr

    @staticmethod
    def unmask_array(arr, mask, fill_value=0):
        """
        Restore masked fiber values back to the full reference fiber set.
        """
        if mask is None:
            return arr

        n_fibers = mask.shape[0]

        if arr.ndim == 1:
            out = np.full(n_fibers, fill_value, dtype=arr.dtype)
            out[mask] = arr
        else:
            out = np.full((n_fibers, arr.shape[1]), fill_value, dtype=arr.dtype)
            out[mask, :] = arr

        return out

    
    ### Reading ###
    def save_as_trk(self, arr, file_paths, reference_trk_path, file_suffix=None, mask=None, fill_value=0):
        if arr.ndim == 1:
            arr = arr[:, None]

        converter = FiberFormatConverter(reference_trk_path)

        for i, file_path in enumerate(file_paths):
            out_dir = os.path.dirname(file_path)
            base = os.path.splitext(os.path.basename(file_path))[0]
            out_name = base + (file_suffix if file_suffix is not None else "")

            fiber_list = self.assign_values_to_fibers(
                arr[:, i],
                mask=mask,
                fill_value=fill_value,
            )

            out_path = os.path.join(out_dir, f"{out_name}.trk")
            converter.convert_fibers_to_reference_format(fiber_list, out_path)
            
    def import_fiber_to_numpy_array(self, file_paths):
        """
        Import many patient files as a matrix of shape (n_fibers_kept, n_files).
        """
        data_list = []
        expected_len = len(self.reference_fibers) if self.reference_fibers is not None else None

        for file_path in tqdm(file_paths, desc='Importing fiber files'):
            data = self._load_single_fiber_values(file_path)

            if expected_len is None:
                expected_len = data.shape[0]
            elif data.shape[0] != expected_len:
                raise ValueError(
                    f"Fiber length mismatch. Expected {expected_len} fibers but got "
                    f"{data.shape[0]} in file: {file_path}"
                )

            data_list.append(data)

        arr = np.column_stack(data_list)

        mask = self.fiber_mask
        _, _, arr = self.mask_array(arr, mask)

        return arr

    def prepare_map_for_evaluation(self, map_data):
        """Convert one fiberwise map to a masked MNI density vector."""
        projector = self._get_density_projector()
        values = np.asarray(map_data, dtype=np.float32).reshape(-1)

        if values.shape[0] != projector["n_fibers"]:
            mask = self.fiber_mask
            if mask is not None and values.shape[0] == int(mask.sum()):
                values = self.unmask_array(values, mask, fill_value=0)

        if values.shape[0] != projector["n_fibers"]:
            raise ValueError(
                f"Fiber value length {values.shape[0]} does not match atlas "
                f"fiber count {projector['n_fibers']}."
            )

        weights = values[projector["fiber_indices"]]
        density = np.bincount(
            projector["voxel_indices"],
            weights=weights,
            minlength=projector["n_mask_voxels"],
        )
        return density.astype(np.float32, copy=False)

    def prepare_evaluation_data(self, file_paths):
        """Return fiber profiles or MNI lesion maps in one evaluation space."""
        file_paths = [str(path) for path in file_paths]
        if not file_paths:
            raise ValueError("No evaluation files were provided.")

        if all(
            path.lower().endswith(
                (".fib.npy", ".fib.json", ".values.npy", ".fib.desc.json")
            )
            for path in file_paths
        ):
            return np.vstack([
                self.prepare_map_for_evaluation(self.load_map_values(path))
                for path in file_paths
            ])

        if all(path.lower().endswith((".nii", ".nii.gz")) for path in file_paths):
            from calvin_utils.neuroimaging_utils.nifti_utils.volume_io import NiftiIO

            volume_io = NiftiIO(mask_path=DEFAULT_MNI_MASK)
            return volume_io.prepare_evaluation_data(file_paths)

        raise ValueError(
            "Fiber evaluation requires all subject files to be either fiber "
            "profiles (*.fib.npy/*.fib.json, or legacy "
            "*.fib.values.npy/*.fib.desc.json) "
            "or MNI lesion maps (*.nii or *.nii.gz)."
        )

    def evaluation_size(self, model_size=None):
        """Return the number of MNI-mask voxels used for fiber evaluation."""
        return int(self._get_density_projector()["n_mask_voxels"])

    @staticmethod
    def is_native_map_file(path):
        """Identify one native result without counting its JSON sidecar.

        New ``*.fib.npy``/``*.fib.json`` pairs use the NumPy file as the native
        map. A legacy values pair is accepted when no new pair with the same
        stem is present. Legacy self-contained ``*.fib.npy`` also remains valid.
        """
        path = Path(path)
        lower_name = path.name.lower()
        if lower_name.endswith(".values.npy"):
            if not FiberIO.values_description_path(path).is_file():
                return False
            if lower_name.endswith(".fib.values.npy"):
                new_values_path = path.with_name(f"{path.name[:-15]}.fib.npy")
                if (
                    new_values_path.is_file()
                    and FiberIO.values_description_path(new_values_path).is_file()
                ):
                    return False
            return True
        if lower_name.endswith(".fib.npy"):
            if FiberIO.values_description_path(path).is_file():
                return True
            values_path = FiberIO.values_companion_path(path)
            legacy_values_path = path.with_name(f"{path.name[:-8]}.values.npy")
            complete_pair = any(
                candidate.is_file()
                and FiberIO.values_description_path(candidate).is_file()
                for candidate in (values_path, legacy_values_path)
            )
            return not complete_pair
        return False

    def load_map_values(self, path):
        """Load one scalar per fiber from a vector or geometry-bearing result.

        A descriptor entry point or numeric one-dimensional ``.npy`` file
        is returned as ``float32``. A legacy object-array ``*.fib.npy`` is
        reduced to the median of column four for each fiber. This method reads
        values only; callers that need geometry must separately provide the
        canonical atlas or use a descriptor-aware converter.
        """
        path = Path(path)
        if path.name.lower().endswith((".fib.json", ".fib.desc.json")):
            path = FiberFormatConverter.values_from_description(path)
        description_path = FiberFormatConverter.values_description_path(path)
        if path.name.lower().endswith(".values.npy") or description_path.is_file():
            # The descriptor is part of the native format, not optional
            # documentation. Resolve it here to validate provenance and the
            # vector checksum even when this caller only needs magnitudes.
            FiberFormatConverter.atlas_from_values_description(path)

        arr = np.load(str(path), allow_pickle=True)
        if arr.dtype != object:
            if arr.ndim != 1:
                raise ValueError(f"Expected a 1D fiber vector in {path}, got {arr.shape}")
            values = arr.astype(np.float32, copy=False)
        else:
            values = self._extract_geometry_values(arr, path)

        mask = self.fiber_mask
        if mask is None or values.shape[0] == int(mask.sum()):
            return values
        if values.shape[0] == mask.shape[0]:
            return values[mask]
        raise ValueError(
            f"Fiber value length {values.shape[0]} in {path} does not match the "
            f"masked ({int(mask.sum())}) or full ({mask.shape[0]}) atlas size."
        )

    @staticmethod
    def native_map_stem(path):
        name = os.path.basename(str(path))
        lower_name = name.lower()
        if lower_name.endswith(".fib.values.npy"):
            return name[:-15]
        if lower_name.endswith(".values.npy"):
            return name[:-11]
        if lower_name.endswith(".fib.npy"):
            return name[:-8]
        if lower_name.endswith(".fib.json"):
            return name[:-9]
        if lower_name.endswith(".fib.desc.json"):
            return name[:-14]
        return os.path.splitext(name)[0]

    def _get_density_projector(self):
        """Build and cache the fiber-to-masked-MNI projection used by CV."""
        if self._density_projector_cache is not None:
            return self._density_projector_cache
        if self.mask_path is None:
            raise ValueError("Fiber evaluation requires a canonical .npz fiber atlas.")

        fibers = [np.asarray(fiber, dtype=np.float32)[:, :3] for fiber in self.reference_fibers]
        mask_img = nib.load(DEFAULT_MNI_MASK)
        mask_data = mask_img.get_fdata().reshape(-1) > 0
        mask_shape = mask_img.shape[:3]
        inv_affine = np.linalg.inv(mask_img.affine)

        full_to_mask = np.full(mask_data.shape[0], -1, dtype=np.int64)
        full_to_mask[mask_data] = np.arange(int(mask_data.sum()), dtype=np.int64)

        voxel_chunks = []
        fiber_chunks = []
        for fiber_idx, fiber in enumerate(tqdm(fibers, desc="Building fiber density projector", unit="fiber")):
            # Preserve the established CV behavior: evaluate the atlas and its
            # left-right mirror without materializing either as a file.
            for xyz in (fiber, self._mirror_fiber_xyz(fiber)):
                ijk = self._world_to_ijk_for_density(xyz, inv_affine)
                in_bounds = (
                    (ijk[:, 0] >= 0) & (ijk[:, 0] < mask_shape[0])
                    & (ijk[:, 1] >= 0) & (ijk[:, 1] < mask_shape[1])
                    & (ijk[:, 2] >= 0) & (ijk[:, 2] < mask_shape[2])
                )
                ijk = ijk[in_bounds]
                if ijk.shape[0] == 0:
                    continue
                full_idx = np.ravel_multi_index((ijk[:, 0], ijk[:, 1], ijk[:, 2]), mask_shape)
                mask_idx = full_to_mask[full_idx]
                mask_idx = mask_idx[mask_idx >= 0]
                if mask_idx.shape[0] == 0:
                    continue
                voxel_chunks.append(mask_idx.astype(np.int64, copy=False))
                fiber_chunks.append(np.full(mask_idx.shape[0], fiber_idx, dtype=np.int64))

        self._density_projector_cache = {
            "n_fibers": len(fibers),
            "n_mask_voxels": int(mask_data.sum()),
            "voxel_indices": np.concatenate(voxel_chunks) if voxel_chunks else np.empty(0, dtype=np.int64),
            "fiber_indices": np.concatenate(fiber_chunks) if fiber_chunks else np.empty(0, dtype=np.int64),
        }
        return self._density_projector_cache

    @staticmethod
    def _world_to_ijk_for_density(xyz, inv_affine):
        xyz = np.asarray(xyz, dtype=np.float32)
        hom = np.c_[xyz, np.ones(xyz.shape[0], dtype=np.float32)]
        return np.rint((hom @ inv_affine.T)[:, :3]).astype(np.int64)

    @staticmethod
    def _mirror_fiber_xyz(fiber):
        mirrored = np.asarray(fiber, dtype=np.float32).copy()
        mirrored[:, 0] = -mirrored[:, 0]
        return mirrored

    @staticmethod
    def _fiber_mirror_landmarks(fiber, mirrored=False):
        """Return a small, direction-independent signature for mirror matching."""
        xyz = np.asarray(fiber, dtype=np.float32)[:, :3]
        if xyz.shape[0] == 0 or not np.all(np.isfinite(xyz)):
            raise ValueError("Fiber mirror matching requires finite, nonempty geometry.")
        landmarks = np.vstack((xyz[0], xyz[-1], xyz.mean(axis=0)))
        if mirrored:
            landmarks[:, 0] *= -1.0
        # Streamline vertex order is arbitrary. Put the endpoints in a stable
        # order so the same trajectory has the same signature in either order.
        if tuple(landmarks[1]) < tuple(landmarks[0]):
            landmarks[[0, 1]] = landmarks[[1, 0]]
        return landmarks.reshape(-1)

    def _approximate_mirror_neighbors(self, tolerance_mm=5.0):
        """Map each atlas fiber to its nearest approximate left-right mirror.

        Matching uses only two endpoints and the centroid, so its memory cost is
        fixed at nine floats per fiber rather than the atlas's full vertices.
        The mapping is deliberately allowed to be many-to-one: this atlas is a
        random tractography sample rather than an explicitly paired atlas, and
        demanding a unique partner leaves most fibers untouched. Symmetric
        values are formed from the original vector in one vectorized pass, so
        many-to-one neighbors never overwrite or collapse one another.
        """
        tolerance_mm = float(tolerance_mm)
        if tolerance_mm <= 0:
            raise ValueError("symmetry_tolerance_mm must be positive.")
        cached = self._mirror_neighbor_cache.get(tolerance_mm)
        if cached is not None:
            return cached

        from scipy.spatial import cKDTree

        fibers = self.reference_fibers
        landmarks = np.asarray(
            [self._fiber_mirror_landmarks(fiber) for fiber in fibers],
            dtype=np.float32,
        )
        mirrored = np.asarray(
            [self._fiber_mirror_landmarks(fiber, mirrored=True) for fiber in fibers],
            dtype=np.float32,
        )
        distances, nearest = cKDTree(landmarks).query(
            mirrored,
            k=1,
            workers=-1,
        )
        nearest = np.asarray(nearest, dtype=np.int64)
        # The Euclidean query distance spans three 3-D landmarks; report and
        # threshold it as an RMS landmark displacement in millimetres.
        distances = np.asarray(distances, dtype=np.float64) / np.sqrt(3.0)
        indices = np.arange(len(fibers), dtype=np.int64)
        accepted = np.isfinite(distances) & (distances <= tolerance_mm)
        neighbors = np.full(len(fibers), -1, dtype=np.int64)
        neighbors[accepted] = nearest[accepted]
        reciprocal = accepted & (neighbors[neighbors.clip(min=0)] == indices)
        self_matches = accepted & (neighbors == indices)
        metadata = {
            "symmetric": True,
            "method": "nearest mirrored endpoint-centroid interpolation",
            "mirror_axis": "x",
            "mirror_origin": 0.0,
            "tolerance_mm": tolerance_mm,
            "reducer": "mean of original and nearest mirrored-neighbor value",
            "unmatched_policy": "preserve original value",
            "mapped_fiber_count": int(np.count_nonzero(accepted)),
            "reciprocal_fiber_count": int(np.count_nonzero(reciprocal)),
            "reciprocal_pair_count": int(
                np.count_nonzero(reciprocal & (indices < neighbors))
            ),
            "self_mirror_count": int(np.count_nonzero(self_matches)),
            "unmatched_fiber_count": int(np.count_nonzero(~accepted)),
            "mapped_fraction": float(np.mean(accepted)),
            "mean_mirror_distance_mm": (
                float(np.mean(distances[accepted])) if np.any(accepted) else None
            ),
            "max_mirror_distance_mm": (
                float(np.max(distances[accepted])) if np.any(accepted) else None
            ),
        }
        result = (neighbors, metadata)
        self._mirror_neighbor_cache[tolerance_mm] = result
        return result

    def _approximate_symmetric_values(
        self,
        values,
        tolerance_mm=5.0,
    ):
        """Average each value with its nearest mirrored-neighbor value."""
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.shape[0] != len(self.reference_fibers):
            raise ValueError(
                f"Fiber value length ({values.shape[0]}) does not match "
                f"reference atlas length ({len(self.reference_fibers)})."
            )
        neighbors, metadata = self._approximate_mirror_neighbors(tolerance_mm)
        mapped = np.flatnonzero(neighbors >= 0)
        symmetric = values.copy()
        symmetric[mapped] = (
            values[mapped].astype(np.float64)
            + values[neighbors[mapped]].astype(np.float64)
        ) / 2.0
        return symmetric.astype(np.float32, copy=False), dict(metadata)

    def _map_to_image(self, map_data):
        """Return an in-memory NIfTI representation for generic viewers."""
        from calvin_utils.neuroimaging_utils.nifti_utils.volume_io import NiftiIO

        density = self.prepare_map_for_evaluation(map_data)
        return NiftiIO(mask_path=DEFAULT_MNI_MASK)._map_to_image(density)

    def assign_values_to_fibers(self, values, mask=None, fill_value=0):
        """
        Project one scalar per fiber back onto fiber vertices.

        Returns a list of arrays shaped (n_vertices_i, 4):
        [x, y, z, magnitude]
        """
        if self.reference_fibers is None:
            raise ValueError("Reference fibers are required for reconstruction.")

        if mask is None:
            mask = self.fiber_mask

        values = np.asarray(values).flatten()

        if mask is not None and values.shape[0] != int(mask.sum()):
            raise ValueError(
                f"Masked value vector length ({values.shape[0]}) does not match number of kept fibers ({int(mask.sum())})."
            )

        full_values = self.unmask_array(values, mask, fill_value=fill_value)

        out_fibers = []
        for fiber, val in zip(self.reference_fibers, full_values):
            xyz = fiber[:, :3]
            m = np.full((xyz.shape[0], 1), val, dtype=np.float32)
            out_fibers.append(np.concatenate([xyz, m], axis=1))

        return out_fibers

    @staticmethod
    def values_companion_path(fiber_path):
        """Return the values path used by the legacy split-name format."""
        fiber_path = Path(fiber_path)
        name = fiber_path.name
        if not name.lower().endswith(".fib.npy"):
            raise ValueError(f"Expected a .fib.npy path, got: {fiber_path}")
        return fiber_path.with_name(f"{name[:-4]}.values.npy")

    @staticmethod
    def values_description_path(values_path):
        """Map a fiber values file to its required descriptor path.

        New ``map.fib.npy`` maps to ``map.fib.json``. Legacy
        ``map.fib.values.npy`` and ``map.values.npy`` still map to
        ``map.fib.desc.json``.
        """
        values_path = Path(values_path)
        name = values_path.name
        lower_name = name.lower()
        if lower_name.endswith(".fib.npy"):
            return values_path.with_name(f"{name[:-4]}.json")
        if not lower_name.endswith(".values.npy"):
            raise ValueError(
                f"Expected a .fib.npy or .values.npy path, got: {values_path}"
            )
        base = name[:-11]
        if not base.lower().endswith(".fib"):
            base += ".fib"
        return values_path.with_name(f"{base}.desc.json")

    @classmethod
    def write_values_description(
        cls,
        values_path,
        fiber_atlas_path,
        *,
        source_geometry_path=None,
        input_value_count=None,
        kept_fiber_count=None,
        fill_value=0,
        symmetry=None,
    ):
        """Write the descriptor paired with an existing fiber values file.

        Parameters
        ----------
        values_path : path-like
            Existing one-dimensional ``*.fib.npy`` vector. New descriptors are
            written only for the compact format; legacy split-name pairs are
            read-only compatibility inputs.
        fiber_atlas_path : path-like
            Canonical fiber atlas whose fiber order defines the vector indices.
        source_geometry_path : path-like, optional
            Geometry-bearing ``*.fib.npy`` from which the values were derived.
        input_value_count, kept_fiber_count : int, optional
            Counts before expansion and after masking. They document whether a
            masked regression vector was expanded to full atlas order.
        fill_value : float, default=0
            Value assigned to excluded fibers during full-atlas expansion.
        symmetry : dict, optional
            Description of any atlas-level value symmetrization.

        Returns
        -------
        pathlib.Path
            The written ``*.fib.json`` path.

        Notes
        -----
        This method is intended for backfilling metadata around an already
        existing vector. Normal regression output should use ``save_files``,
        which creates the vector and descriptor together.
        """
        values_path = Path(values_path).expanduser().resolve()
        fiber_atlas_path = Path(fiber_atlas_path).expanduser().resolve()
        if not values_path.name.lower().endswith(".fib.npy"):
            raise ValueError(
                "New fiber output must use <stem>.fib.npy; writing legacy "
                ".fib.values.npy/.fib.desc.json pairs is no longer supported."
            )
        if not values_path.is_file():
            raise FileNotFoundError(values_path)
        if not fiber_atlas_path.is_file():
            raise FileNotFoundError(fiber_atlas_path)

        values = np.load(values_path, mmap_mode="r")
        if values.ndim != 1:
            raise ValueError(f"Expected a 1D value vector in {values_path}, got {values.shape}")
        finite = np.asarray(values[np.isfinite(values)], dtype=np.float32)
        value_count = int(values.shape[0])
        input_value_count = value_count if input_value_count is None else int(input_value_count)
        kept_fiber_count = value_count if kept_fiber_count is None else int(kept_fiber_count)
        atlas_stat = fiber_atlas_path.stat()
        digest = hashlib.sha256()
        with values_path.open("rb") as values_file:
            for chunk in iter(lambda: values_file.read(1024 * 1024), b""):
                digest.update(chunk)
        values_relative_path = os.path.relpath(values_path, values_path.parent)
        description = {
            "schema": "calvin_utils.fiber_values",
            "schema_version": 2,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "values_file": values_path.name,
            "values_path": str(values_path),
            "values_relative_path": values_relative_path,
            "fiber_atlas": {
                "path": str(fiber_atlas_path),
                "relative_path": os.path.relpath(fiber_atlas_path, values_path.parent),
                "fiber_count": value_count,
                "size_bytes": int(atlas_stat.st_size),
                "mtime_ns": int(atlas_stat.st_mtime_ns),
            },
            "values": {
                "path": str(values_path),
                "relative_path": values_relative_path,
                "size_bytes": int(values_path.stat().st_size),
                "dtype": str(values.dtype),
                "shape": [value_count],
                "semantics": "one scalar per fiber",
                "ordering": "values[i] corresponds to fiber_atlas fiber i",
                "sha256": digest.hexdigest(),
                "finite_count": int(finite.size),
                "nonfinite_count": value_count - int(finite.size),
                "nonzero_count": int(np.count_nonzero(finite)),
                "minimum": float(np.min(finite)) if finite.size else None,
                "maximum": float(np.max(finite)) if finite.size else None,
            },
            "mask": {
                "input_value_count": input_value_count,
                "kept_fiber_count": kept_fiber_count,
                "full_fiber_count": value_count,
                "fill_value": float(fill_value),
                "expanded_to_full_atlas": input_value_count != value_count,
            },
            "symmetry": (
                dict(symmetry) if symmetry is not None else {"symmetric": False}
            ),
            "source_geometry_file": (
                str(Path(source_geometry_path).expanduser().resolve())
                if source_geometry_path is not None
                else None
            ),
        }
        description_path = cls.values_description_path(values_path)
        description_path.write_text(json.dumps(description, indent=2) + "\n", encoding="utf-8")
        return description_path

    @classmethod
    def write_values_companion(cls, fiber_path, fiber_atlas_path, output_path=None):
        """Backfill a new-format value/descriptor pair from ``*.fib.npy``.

        Column four is reduced to one median value per fiber and saved as
        ``float32``. ``fiber_atlas_path`` is mandatory because a value vector
        without its geometry provenance is ambiguous. Numeric inputs are paired
        in place. Geometry-bearing inputs are preserved and, unless an explicit
        output is supplied, migrate to ``<stem>_values.fib.npy``. The sibling
        ``*.fib.json`` is always written in the same operation.
        """
        fiber_path = Path(fiber_path).expanduser()
        stored = np.load(fiber_path, allow_pickle=True)
        values = (
            cls._extract_geometry_values(stored, fiber_path)
            if stored.dtype == object
            else np.asarray(stored, dtype=np.float32).reshape(-1)
        )
        if output_path is None:
            output_path = (
                fiber_path
                if stored.dtype != object
                else fiber_path.with_name(f"{fiber_path.name[:-8]}_values.fib.npy")
            )
        else:
            output_path = Path(output_path).expanduser()
        if not output_path.name.lower().endswith(".fib.npy"):
            raise ValueError(f"New fiber values output must end with .fib.npy: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, np.asarray(values, dtype=np.float32))
        cls.write_values_description(
            output_path,
            fiber_atlas_path,
            source_geometry_path=fiber_path if stored.dtype == object else None,
        )
        return output_path
    
    ### Writing ###
    def save_files(
        self,
        arr,
        file_paths,
        dry_run=True,
        file_suffix=None,
        mask=None,
        fill_value=0,
        convert_to_nifti=True,
        convert_to_leaddbs=False,
        symmetric=True,
        symmetry_tolerance_mm=5.0,
        sign="both",
        save_values=True,
        save_geometry=False,
    ):
        """Save fiber statistics and their visualization/export companions.

        ``arr`` may be ``(n_fibers_kept,)`` or
        ``(n_fibers_kept, n_files)``. Every output column is restored to full
        canonical-atlas order before it is written.

        Output contract
        ---------------
        ``*.fib.npy`` and ``*.fib.json`` are always written as the native
        regression result. The NumPy file is a one-dimensional float32 vector;
        geometry remains in the canonical atlas recorded by the JSON. The old
        split names and geometry-bearing outputs are no longer written.

        The ordinary pair is always preserved. With ``symmetric=True`` (the
        default), an additional ``*_symmetric.fib.npy``/``*.fib.json`` pair is
        written. Each value is averaged with the value on its nearest approximate
        mirrored fiber; fibers without a match within ``symmetry_tolerance_mm``
        retain their original value.

        With ``convert_to_nifti=True``, signed tract-density NIfTIs are produced
        for the ordinary and symmetric pairs. With ``convert_to_leaddbs=True``,
        positive and/or negative FTR MAT files are produced according to
        ``sign``. Symmetric companions consume the already-symmetric values and
        therefore do not mirror their geometry a second time.

        The ``mask_path`` supplied when constructing ``FiberIO`` is recorded as
        the canonical atlas. Consequently, a writable fiber result requires a
        real atlas path; the descriptor is not allowed to guess its geometry.
        """
        if arr.ndim == 1:
            arr = arr[:, None]
        if not save_values:
            raise ValueError("Fiber output always writes the .fib.npy/.fib.json pair.")
        if save_geometry:
            raise ValueError(
                "save_geometry is no longer supported; .fib.npy is now the compact "
                "value vector and .fib.json records its atlas."
            )

        for i, file_path in tqdm(list(enumerate(file_paths)), desc='Saving fiber files'):
            out_dir = os.path.dirname(file_path)
            base = os.path.splitext(os.path.basename(file_path))[0]
            out_name = base + (file_suffix if file_suffix is not None else '')
            values_path = os.path.join(out_dir, f"{out_name}.fib.npy")
            description_path = self.values_description_path(values_path)
            os.makedirs(out_dir, exist_ok=True)

            effective_mask = self.fiber_mask if mask is None else np.asarray(mask, dtype=bool)
            if self.reference_fibers is None:
                raise ValueError("Reference fibers are required for reconstruction.")
            values = np.asarray(arr[:, i], dtype=np.float32).reshape(-1)
            if effective_mask is not None and values.shape[0] != int(effective_mask.sum()):
                raise ValueError(
                    f"Masked value vector length ({values.shape[0]}) does not match "
                    f"number of kept fibers ({int(effective_mask.sum())})."
                )
            full_values = np.asarray(
                self.unmask_array(values, effective_mask, fill_value=fill_value),
                dtype=np.float32,
            )
            if full_values.shape[0] != len(self.reference_fibers):
                raise ValueError(
                    f"Fiber value length ({full_values.shape[0]}) does not match "
                    f"reference atlas length ({len(self.reference_fibers)})."
                )
            if dry_run:
                print(f"Saving values to: {values_path}")
                print(f"Saving values description to: {description_path}")
            else:
                np.save(values_path, full_values)
                description_path = self.write_values_description(
                    values_path,
                    self.mask_path,
                    input_value_count=values.shape[0],
                    kept_fiber_count=(
                        int(effective_mask.sum())
                        if effective_mask is not None
                        else values.shape[0]
                    ),
                    fill_value=fill_value,
                )

            output_variants = [(out_name, description_path)]
            if symmetric:
                symmetric_name = f"{out_name}_symmetric"
                symmetric_values_path = os.path.join(
                    out_dir, f"{symmetric_name}.fib.npy"
                )
                symmetric_description_path = self.values_description_path(
                    symmetric_values_path
                )
                if dry_run:
                    print(f"Saving symmetric values to: {symmetric_values_path}")
                    print(
                        "Saving symmetric values description to: "
                        f"{symmetric_description_path}"
                    )
                else:
                    symmetric_values, symmetry_metadata = (
                        self._approximate_symmetric_values(
                            full_values,
                            tolerance_mm=symmetry_tolerance_mm,
                        )
                    )
                    np.save(symmetric_values_path, symmetric_values)
                    symmetric_description_path = self.write_values_description(
                        symmetric_values_path,
                        self.mask_path,
                        input_value_count=values.shape[0],
                        kept_fiber_count=(
                            int(effective_mask.sum())
                            if effective_mask is not None
                            else values.shape[0]
                        ),
                        fill_value=fill_value,
                        symmetry=symmetry_metadata,
                    )
                    print(
                        "Approximate mirror matching: "
                        f"{symmetry_metadata['mapped_fiber_count']}/"
                        f"{len(self.reference_fibers)} fibers mapped."
                    )
                output_variants.append(
                    (symmetric_name, symmetric_description_path)
                )

            print(f"Saving positive/negative fibers: {sign}")
            if convert_to_nifti:
                print(f"Saving volumetric version of fibers (.nii.gz).")
                if not dry_run:
                    for variant_name, variant_description_path in output_variants:
                        TractDensity(
                            fiber_path=variant_description_path,
                            reference_nifti_path=DEFAULT_MNI_MASK,
                            out_path=os.path.join(
                                out_dir, f"{variant_name}.nii.gz"
                            ),
                            fiberset=sign,
                            symmetric=False,
                            threshold=None,
                        ).run()

            if convert_to_leaddbs:
                lead_signs = (
                    ("positive", "negative")
                    if sign == "both"
                    else ("positive",)
                    if sign in {"positive", "pos"}
                    else ("negative",)
                )
                print(
                    "Saving Lead-DBS compatible fibers independently for: "
                    + ", ".join(lead_signs)
                )
                if not dry_run:
                    for variant_name, variant_description_path in output_variants:
                        for lead_sign in lead_signs:
                            FiberResultVisualizer(
                                values_path=variant_description_path,
                                out_dir=os.path.dirname(values_path),
                                output_name=f"{variant_name}_{lead_sign}",
                                sign=lead_sign,
                                symmetric=False,
                                min_abs_value=None,
                                top_percent=None,
                            ).run()
