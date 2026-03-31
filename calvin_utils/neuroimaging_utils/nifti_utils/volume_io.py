import os
import numpy as np
import nibabel as nib
from nilearn import image
from tqdm import tqdm
from calvin_utils.neuroimaging_utils.ccm_utils.bounding_box import NiftiBoundingBox
from calvin_utils.neuroimaging_utils.nifti_utils.generate_nifti import view_and_save_nifti

DEFAULT_MASK = "circuit_pyper/resources/MNI152_T1_2mm_brain_mask.nii"


class NiftiIO:
    def __init__(self, mask_path='default', threshold=0):
        self.mask_path = mask_path
        self.threshold = threshold
        self._affines = set()
        self.bbox = None
        self.bbox_mask = None
        self.bbox_4d = None

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

    @staticmethod
    def save_files(arr: np.ndarray, file_paths, dry_run=True, file_suffix=None):
        for i, file_path in tqdm(enumerate(file_paths), desc='Saving files'):
            out_dir = os.path.dirname(file_path)
            nifti_name = os.path.splitext(os.path.basename(file_path))[0] + (file_suffix if file_suffix is not None else '')

            if dry_run:
                print(f"Saving to: {os.path.join(out_dir, nifti_name)}")
            else:
                view_and_save_nifti(arr[:, i], out_dir=out_dir, output_name=nifti_name, silent=True)