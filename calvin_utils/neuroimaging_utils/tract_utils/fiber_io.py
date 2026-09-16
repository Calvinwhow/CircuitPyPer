import os
import json
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from calvin_utils.neuroimaging_utils.tract_utils.fiber_converter import FiberFormatConverter
from calvin_utils.neuroimaging_utils.tract_utils.fiber_result_visualizer import FiberResultVisualizer
from calvin_utils.neuroimaging_utils.tract_utils.tract_density import TractDensity
PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_FIBER_MASK = None
DEFAULT_MNI_MASK = os.path.join(PACKAGE_ROOT, "resources", "MNI152_T1_2mm_brain_mask.nii")

class FiberIO:
    """
    Fiber-space I/O using a canonical ordered fiber library.
    This is specifically for importing Fiber Connectivity files.
    
    Core assumptions
    ----------------
    1. Each patient file represents magnitudes over the SAME ordered fiber set.
    2. The reference mask/library stores the canonical polylines.
    3. Regression operates on per-fiber magnitudes only, not raw polyline vertices.

    Conventions
    -----------
    - import_fiber_to_numpy_array(file_paths) returns shape (n_fibers_kept, n_files)
    - mask unit is fiber index, not vertex index
    - unmasking restores values to the full reference fiber set
    - writing assigns one scalar magnitude to every vertex of a fiber
    """

    output_ftype = "fiber"

    def __init__(self, mask_path=None, threshold=0):
        self.mask_path = DEFAULT_FIBER_MASK if mask_path == 'default' else mask_path
        self.threshold = threshold
        self._reference_fibers = None
        self._fiber_mask = None
        self._density_projector_cache = None

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
        ftype = self._identify_fiber_file_type(file_path)

        if ftype == 'npy':
            arr = np.load(file_path, allow_pickle=True)
            if arr.dtype == object:
                return self._extract_geometry_values(arr, file_path)
            if arr.ndim != 1:
                raise ValueError(f"Expected 1D fiber value vector in {file_path}, got shape {arr.shape}")
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

        if all(path.lower().endswith(".fib.npy") for path in file_paths):
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
            "profiles (*.fib.npy) or MNI lesion maps (*.nii or *.nii.gz)."
        )

    def evaluation_size(self, model_size=None):
        """Return the number of MNI-mask voxels used for fiber evaluation."""
        return int(self._get_density_projector()["n_mask_voxels"])

    @staticmethod
    def is_native_map_file(path):
        return str(path).lower().endswith(".fib.npy")

    def load_map_values(self, path):
        """Load one value per fiber from a vector or geometry-bearing result."""
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
        return name[:-8] if name.lower().endswith(".fib.npy") else os.path.splitext(name)[0]

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
        convert_to_leaddbs=True,
        symmetric=False,
        sign="both",
    ):
        """
        Save per-file fiber statistics back to geometry-aware outputs.

        arr shape:
        - 1D: (n_fibers_kept,)
        - 2D: (n_fibers_kept, n_files)

        For now this writes .npy object arrays of [(x,y,z,m), ...] fibers.
        By default, it also writes a faithful volumetric companion containing
        both signed tails without thresholding or implicit mirroring. Lead-DBS
        output is split into ``*_positive_ftr.mat`` and
        ``*_negative_ftr.mat`` so the two tails can be loaded independently.
        Explicit ``sign`` and ``symmetric`` values remain available for
        display-specific exports. Add .fibfilt writer later.
        """
        if arr.ndim == 1:
            arr = arr[:, None]

        for i, file_path in tqdm(list(enumerate(file_paths)), desc='Saving fiber files'):
            out_dir = os.path.dirname(file_path)
            base = os.path.splitext(os.path.basename(file_path))[0]
            out_name = base + (file_suffix if file_suffix is not None else '')
            out_path = os.path.join(out_dir, f"{out_name}.fib.npy")
            os.makedirs(out_dir, exist_ok=True)
            fiber_list = self.assign_values_to_fibers(
                arr[:, i],
                mask=mask,
                fill_value=fill_value,
            )


            if dry_run:
                print(f"Saving to: {out_path}")
            else:
                np.save(out_path, np.array(fiber_list, dtype=object), allow_pickle=True)

            print(f"Saving positive/negative fibers: {sign}")
            print(f"Saving symmetric version of fibers: {symmetric}")
            if convert_to_nifti:
                print(f"Saving volumetric version of fibers (.nii.gz).")
                if not dry_run:
                    TractDensity(
                        fiber_path=out_path,
                        reference_nifti_path=DEFAULT_MNI_MASK,
                        out_path=os.path.join(out_dir, f"{out_name}.nii.gz"),
                        fiberset=sign,
                        symmetric=symmetric,
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
                    for lead_sign in lead_signs:
                        FiberResultVisualizer(
                            values_path=out_path,
                            out_dir=os.path.dirname(out_path),
                            output_name=f"{out_name}_{lead_sign}",
                            sign=lead_sign,
                            symmetric=symmetric,
                            min_abs_value=None,
                            top_percent=None,
                        ).run()
