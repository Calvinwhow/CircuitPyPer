import os
import json
import numpy as np
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

    def __init__(self, mask_path=None, threshold=0):
        self.mask_path = DEFAULT_FIBER_MASK if mask_path == 'default' else mask_path
        self.threshold = threshold
        self._reference_fibers = None
        self._fiber_mask = None

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
    def save_files(self, arr, file_paths, dry_run=True, file_suffix=None, mask=None, fill_value=0, convert_to_nifti=True, convert_to_leaddbs=True, symmetric=True, sign = "positive"):
        """
        Save per-file fiber statistics back to geometry-aware outputs.

        arr shape:
        - 1D: (n_fibers_kept,)
        - 2D: (n_fibers_kept, n_files)

        For now this writes .npy object arrays of [(x,y,z,m), ...] fibers.
        Add .fibfilt writer later.
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
                TractDensity(
                    fiber_path=out_path,
                    reference_nifti_path=DEFAULT_MNI_MASK,
                    out_path=os.path.join(out_dir, f"{out_name}.nii.gz"),
                    fiberset=sign,
                    symmetric=symmetric,
                    threshold=None).run()
            
            if convert_to_leaddbs:
                print(f"Saving lead-dbs compatible version of fibers (.mat).")
                FiberResultVisualizer(
                    values_path=out_path,
                    out_dir=os.path.dirname(out_path),
                    sign=sign,
                    symmetric=symmetric,
                    min_abs_value=None,
                    top_percent=None).run()
