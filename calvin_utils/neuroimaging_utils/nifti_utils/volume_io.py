import os
import numpy as np
import nibabel as nib
from nilearn import image, plotting
from tqdm import tqdm
from calvin_utils.neuroimaging_utils.ccm_utils.bounding_box import NiftiBoundingBox
from calvin_utils.neuroimaging_utils.nifti_utils.generate_nifti import view_and_save_nifti

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_MASK = os.path.join(PACKAGE_ROOT, "resources", "MNI152_T1_2mm_brain_mask.nii")

class NiftiIO:
    def __init__(self, mask_path='default', threshold=0):
        self.mask_path = mask_path
        self.threshold = threshold
        self._affines = set()
        self.bbox = None
        self.bbox_mask = None
        self.bbox_4d = None

    @property
    def resolved_mask_path(self):
        if self.mask_path == 'default':
            return DEFAULT_MASK
        return self.mask_path

    @property
    def mask(self):
        if self.mask_path == 'default':
            return nib.load(DEFAULT_MASK).get_fdata().flatten()
        if self.mask_path is None:
            return None
        return nib.load(self.mask_path).get_fdata().flatten()

    @property
    def affines(self):
        return self._affines

    @affines.setter
    def affines(self, value):
        self._affines.add(value)

    def _visualize_map(self, img, title=None):
        """
        Open a NIfTI image in the browser via nilearn.

        Parameters
        ----------
        img : nibabel.Nifti1Image
            In-memory image to visualize.
        title : str, optional
            Viewer title.
        """
        try:
            plotting.view_img(img, title=title).open_in_browser()
        except Exception:
            pass

    def _map_to_image(self, map_data: np.ndarray, fill_value=0):
        """
        Convert a masked or full-length vector map into an in-memory NIfTI image.
        """
        data_vec = np.asarray(map_data)
        if data_vec.ndim == 2:
            if data_vec.shape[1] == 1:
                data_vec = data_vec[:, 0]
            elif data_vec.shape[0] == 1:
                data_vec = data_vec[0, :]
            else:
                raise ValueError(f"Expected a single map, got shape {data_vec.shape}")
        elif data_vec.ndim != 1:
            raise ValueError(f"Expected 1D or single-map 2D array, got ndim={data_vec.ndim}")

        mask_path = self.resolved_mask_path
        if mask_path is None:
            raise ValueError("Cannot build a volumetric NIfTI image without a mask_path.")

        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()
        mask_indices = mask_data.flatten() > self.threshold
        full_size = int(mask_indices.shape[0])
        n_masked = int(mask_indices.sum())

        if data_vec.shape[0] == n_masked:
            full_vec = np.full(full_size, fill_value, dtype=data_vec.dtype)
            full_vec[mask_indices] = data_vec
        elif data_vec.shape[0] == full_size:
            full_vec = data_vec
        else:
            raise ValueError(
                f"Map length {data_vec.shape[0]} does not match masked ({n_masked}) or full ({full_size}) voxel counts."
            )

        vol3d = full_vec.reshape(mask_img.shape).astype(np.float32)
        return nib.Nifti1Image(vol3d, affine=mask_img.affine)

    def _load_single_nifti(self, file_path):
        nifti_img = image.load_img(file_path)
        self.affines = tuple(nifti_img.affine.flatten())
        return nifti_img.get_fdata().flatten()

    def align_imported_matrices(self, file_paths):
        self.bbox = NiftiBoundingBox(file_paths)
        self.bbox.generate_bounding_box()
        self.bbox.add_niftis_to_bounding_box()
        self.bbox.collapse_bbox_to_3d()
        self.bbox_mask = self.bbox.collapsed_bbox_to_mask()
        self.bbox_4d = self.bbox._stacked_data
        return np.column_stack([
            self.bbox_4d[:, :, :, i].flatten()
            for i in range(self.bbox_4d.shape[3])
        ])

    @staticmethod
    def mask_array(arr: np.ndarray, mask_path: str = None, threshold: float = 0):
        if mask_path is None:
            mask = np.ones(arr.shape[0], dtype=bool)
            mask_indices = np.arange(arr.shape[0])
            masked_arr = arr
        else:
            mask = nib.load(mask_path).get_fdata().flatten()
            mask_indices = mask > threshold
            if arr.ndim == 1:
                masked_arr = arr[mask_indices]
            else:
                masked_arr = arr[mask_indices, :]
        return mask, mask_indices, masked_arr

    @staticmethod
    def unmask_array(arr: np.ndarray, mask_path: str = None, threshold: float = 0, fill_value=0):
        if mask_path is None:
            return arr

        mask = nib.load(mask_path).get_fdata().flatten()
        mask_indices = mask > threshold

        if arr.ndim == 1:
            unmasked_arr = np.full(mask.shape, fill_value, dtype=arr.dtype)
            unmasked_arr[mask_indices] = arr
        else:
            unmasked_arr = np.full((mask.shape[0], arr.shape[1]), fill_value, dtype=arr.dtype)
            unmasked_arr[mask_indices, :] = arr

        return unmasked_arr

    def import_nifti_to_numpy_array(self, file_paths):
        cols = [self._load_single_nifti(fp) for fp in file_paths]

        if len(self.affines) > 1:
            arr = self.align_imported_matrices(file_paths)
        else:
            arr = np.column_stack(cols)

        if self.mask_path == 'default':
            _, _, arr = self.mask_array(arr, DEFAULT_MASK, self.threshold)
        elif self.mask_path is not None:
            _, _, arr = self.mask_array(arr, self.mask_path, self.threshold)

        return arr

    def save_files(self, arr: np.ndarray, file_paths, dry_run=True, file_suffix=None, fill_value=0):
        """
        Save one NIfTI map per output file.

        Accepted arr shapes:
        - (n_voxels,) for a single file
        - (n_voxels, n_files)
        - (1, n_voxels) for a single file
        - (n_voxels, 1) for a single file

        If a mask is provided, arr is treated as masked when its length equals the number
        of voxels in the mask above threshold; otherwise it must match the full volume size.
        """
        arr = np.asarray(arr)

        if arr.ndim == 1:
            arr = arr[:, None]
        elif arr.ndim == 2 and len(file_paths) == 1:
            if arr.shape[1] == 1:
                pass
            elif arr.shape[0] == 1:
                arr = arr.T
            else:
                raise ValueError(f"Ambiguous 2D arr shape for single file: {arr.shape}")
        elif arr.ndim != 2:
            raise ValueError(f"Unsupported arr.ndim={arr.ndim}")

        if arr.shape[1] != len(file_paths):
            raise ValueError(
                f"Number of files ({len(file_paths)}) does not match arr.shape[1] ({arr.shape[1]})"
            )

        mask_path = self.resolved_mask_path
        mask_img = None
        mask_indices = None
        full_size = None
        n_masked = None

        if mask_path is not None:
            mask_img = nib.load(mask_path)
            mask_data = mask_img.get_fdata()
            mask_indices = mask_data.flatten() > self.threshold
            full_size = int(mask_indices.shape[0])
            n_masked = int(mask_indices.sum())

        for i, file_path in tqdm(list(enumerate(file_paths)), desc='Saving files'):
            out_dir = os.path.dirname(file_path)
            base = os.path.splitext(os.path.basename(file_path))[0]
            out_name = base + (file_suffix if file_suffix is not None else '')
            out_path = os.path.join(out_dir, f"{out_name}.nii.gz")

            data_vec = arr[:, i]

            if mask_img is None:
                if dry_run:
                    print(f"Saving to: {out_path}")
                else:
                    os.makedirs(out_dir, exist_ok=True)
                    view_and_save_nifti(data_vec, out_dir=out_dir, output_name=out_name, silent=True)
                continue

            if data_vec.shape[0] == n_masked:
                full_vec = np.full(full_size, fill_value, dtype=data_vec.dtype)
                full_vec[mask_indices] = data_vec
            elif data_vec.shape[0] == full_size:
                full_vec = data_vec
            else:
                raise ValueError(
                    f"Map length {data_vec.shape[0]} does not match masked ({n_masked}) or full ({full_size}) voxel counts."
                )

            vol3d = full_vec.reshape(mask_img.shape).astype(np.float32)

            if dry_run:
                print(f"Saving to: {out_path}")
            else:
                os.makedirs(out_dir, exist_ok=True)
                img = nib.Nifti1Image(vol3d, affine=mask_img.affine)
                nib.save(img, out_path)

class VolumetricTimeSeriesIO:
    def __init__(self, mask_path='default', threshold=0):
        self.mask_path = mask_path
        self.threshold = threshold
        self._affines = set()
        self._shapes = set()
        self._n_timepoints = set()

    @property
    def resolved_mask_path(self):
        if self.mask_path == 'default':
            return DEFAULT_MASK
        return self.mask_path

    @property
    def mask(self):
        mask_path = self.resolved_mask_path
        if mask_path is None:
            return None
        return nib.load(mask_path).get_fdata().flatten()

    @property
    def affines(self):
        return self._affines

    @affines.setter
    def affines(self, value):
        self._affines.add(value)

    @property
    def shapes(self):
        return self._shapes

    @shapes.setter
    def shapes(self, value):
        self._shapes.add(value)

    def _load_single_nifti(self, file_path):
        img = image.load_img(file_path)
        data = img.get_fdata()

        if data.ndim != 4:
            raise ValueError(f"Expected 4D NIfTI, got shape {data.shape} in {file_path}")

        self.affines = tuple(img.affine.flatten())
        self.shapes = tuple(data.shape[:3])
        self._n_timepoints.add(int(data.shape[3]))
        return img

    def _get_reference_img(self, imgs):
        mask_path = self.resolved_mask_path
        if mask_path is not None:
            return nib.load(mask_path)
        return image.index_img(imgs[0], 0)

    def _img_to_vox_by_time(self, img):
        data = img.get_fdata()
        x, y, z, t = data.shape
        return data.reshape(x * y * z, t)

    @staticmethod
    def mask_array(arr: np.ndarray, mask_path: str = None, threshold: float = 0):
        if mask_path is None:
            mask = np.ones(arr.shape[0], dtype=bool)
            mask_indices = np.arange(arr.shape[0])
            return mask, mask_indices, arr

        mask = nib.load(mask_path).get_fdata().flatten()
        mask_indices = mask > threshold

        if arr.ndim == 1:
            masked_arr = arr[mask_indices]
        elif arr.ndim == 2:
            masked_arr = arr[mask_indices, :]
        elif arr.ndim == 3:
            masked_arr = arr[mask_indices, :, :]
        else:
            raise ValueError(f"Unsupported arr.ndim={arr.ndim}")

        return mask, mask_indices, masked_arr

    @staticmethod
    def unmask_array(arr: np.ndarray, mask_path: str = None, threshold: float = 0, fill_value=0):
        if mask_path is None:
            return arr

        mask = nib.load(mask_path).get_fdata().flatten()
        mask_indices = mask > threshold
        n_vox = mask.shape[0]

        if arr.ndim == 1:
            out = np.full(n_vox, fill_value, dtype=arr.dtype)
            out[mask_indices] = arr
        elif arr.ndim == 2:
            out = np.full((n_vox, arr.shape[1]), fill_value, dtype=arr.dtype)
            out[mask_indices, :] = arr
        elif arr.ndim == 3:
            out = np.full((n_vox, arr.shape[1], arr.shape[2]), fill_value, dtype=arr.dtype)
            out[mask_indices, :, :] = arr
        else:
            raise ValueError(f"Unsupported arr.ndim={arr.ndim}")

        return out

    def import_nifti_to_timeseries_array(self, file_paths):
        imgs = [self._load_single_nifti(fp) for fp in file_paths]
        ref_img = self._get_reference_img(imgs)

        data_list = []
        for img in tqdm(imgs, desc="Importing 4D NIfTI files"):
            if img.shape[:3] != ref_img.shape[:3] or not np.allclose(img.affine, ref_img.affine):
                img = image.resample_to_img(img, ref_img, interpolation="continuous")
            vox_by_time = self._img_to_vox_by_time(img)
            data_list.append(vox_by_time)

        arr = np.stack(data_list, axis=2)  # (n_voxels, n_time, n_files)

        mask_path = self.resolved_mask_path
        if mask_path is not None:
            _, _, arr = self.mask_array(arr, mask_path, self.threshold)

        return arr

    def import_nifti_to_numpy_array(self, file_paths, flatten_time=True):
        arr = self.import_nifti_to_timeseries_array(file_paths)
        if not flatten_time:
            return arr
        n_vox, n_time, n_files = arr.shape
        return arr.reshape(n_vox * n_time, n_files)

    def save_files(self, arr: np.ndarray, file_paths, dry_run=True, file_suffix=None, fill_value=0):
        """
        Writes masked flattened time series back to 4D NIfTI.

        Accepted arr shapes:
        - (masked_voxels * timepoints,) 
        - (masked_voxels * timepoints, n_files)
        - (masked_voxels, timepoints)
        - (masked_voxels, timepoints, n_files)
        """
        mask_path = self.resolved_mask_path
        if mask_path is None:
            raise ValueError("mask_path is required to write 4D volumetric data.")

        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()
        mask_indices = mask_data.flatten() > self.threshold
        n_masked_vox = int(mask_indices.sum())
        spatial_shape = mask_data.shape

        arr = np.asarray(arr)

        if arr.ndim == 1:
            if arr.shape[0] % n_masked_vox != 0:
                raise ValueError(
                    f"Flat array length {arr.shape[0]} is not divisible by n_masked_vox={n_masked_vox}"
                )
            n_time = arr.shape[0] // n_masked_vox
            arr = arr.reshape(n_masked_vox, n_time, 1)

        elif arr.ndim == 2:
            if len(file_paths) == 1:
                if arr.shape[0] == n_masked_vox:
                    arr = arr[:, :, None]
                elif arr.shape[0] % n_masked_vox == 0:
                    n_time = arr.shape[0] // n_masked_vox
                    arr = arr.reshape(n_masked_vox, n_time, arr.shape[1])
                else:
                    raise ValueError(
                        f"2D array first dimension {arr.shape[0]} is incompatible with n_masked_vox={n_masked_vox}"
                    )
            else:
                if arr.shape[0] % n_masked_vox != 0:
                    raise ValueError(
                        f"2D array first dimension {arr.shape[0]} is not divisible by n_masked_vox={n_masked_vox}"
                    )
                n_time = arr.shape[0] // n_masked_vox
                arr = arr.reshape(n_masked_vox, n_time, arr.shape[1])

        elif arr.ndim == 3:
            if arr.shape[0] != n_masked_vox:
                raise ValueError(
                    f"Expected first dimension {n_masked_vox}, got {arr.shape[0]}"
                )
        else:
            raise ValueError(f"Unsupported arr.ndim={arr.ndim}")

        if arr.shape[2] != len(file_paths):
            raise ValueError(
                f"Number of files ({len(file_paths)}) does not match arr.shape[2] ({arr.shape[2]})"
            )

        for i, file_path in tqdm(list(enumerate(file_paths)), desc="Saving 4D files"):
            out_dir = os.path.dirname(file_path)
            base = os.path.splitext(os.path.basename(file_path))[0]
            out_name = base + (file_suffix if file_suffix is not None else '')
            out_path = os.path.join(out_dir, f"{out_name}.nii.gz")

            masked_vox_by_time = arr[:, :, i]                         # (masked_vox, time)
            full_vox_by_time = self.unmask_array(
                masked_vox_by_time,
                mask_path=mask_path,
                threshold=self.threshold,
                fill_value=fill_value
            )                                                         # (full_vox, time)

            vol4d = full_vox_by_time.reshape(*spatial_shape, full_vox_by_time.shape[1])

            if dry_run:
                print(f"Saving to: {out_path}")
            else:
                img = nib.Nifti1Image(vol4d.astype(np.float32), affine=mask_img.affine)
                nib.save(img, out_path)
