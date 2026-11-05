import os

import nibabel as nib
import numpy as np
from nilearn import image
from tqdm import tqdm

from calvin_utils.resource_paths import lazy_resource_path


DEFAULT_MASK = lazy_resource_path("MNI152_T1_2mm_brain_mask.nii")


class NiftiIO:
    """Read and write masked volumes through one flattened location axis."""

    output_ftype = "nii"

    def __init__(self, mask_path="default", threshold=0):
        self.mask_path = mask_path
        self.threshold = threshold
        self._affines = set()

    def validate_for_output(self):
        return None

    @property
    def resolved_mask_path(self):
        return DEFAULT_MASK if self.mask_path == "default" else self.mask_path

    @property
    def mask(self):
        return None if self.resolved_mask_path is None else nib.load(self.resolved_mask_path).get_fdata().reshape(-1)

    @property
    def affines(self):
        return self._affines

    @affines.setter
    def affines(self, value):
        self._affines.add(value)

    def _visualize_map(self, img, title=None):
        from nilearn import plotting
        plotting.view_img(img, title=title).open_in_browser()

    @staticmethod
    def mask_array(arr, mask_path=None, threshold=0):
        mask = np.ones(arr.shape[0], dtype=bool) if mask_path is None else nib.load(mask_path).get_fdata().reshape(-1) > threshold
        return mask, mask, arr[mask, ...]

    @staticmethod
    def unmask_array(arr, mask_path=None, threshold=0, fill_value=0):
        if mask_path is None: return arr
        mask = nib.load(mask_path).get_fdata().reshape(-1) > threshold
        restored = np.full((mask.size, *np.asarray(arr).shape[1:]), fill_value, dtype=np.asarray(arr).dtype)
        restored[mask, ...] = arr
        return restored

    def _load_single_nifti(self, file_path):
        img = image.load_img(file_path)
        self.affines = tuple(img.affine.reshape(-1))
        return img

    def _reference_image(self, imgs):
        return nib.load(self.resolved_mask_path) if self.resolved_mask_path is not None else image.index_img(imgs[0], 0) if len(imgs[0].shape) == 4 else imgs[0]

    @staticmethod
    def _matches_reference(img, reference):
        return img.shape[:3] == reference.shape[:3] and np.allclose(img.affine, reference.affine)

    def _align_image(self, img, reference):
        return img if self._matches_reference(img, reference) else image.resample_to_img(img, reference, interpolation="continuous")

    @staticmethod
    def _voxel_frames(img):
        data = img.get_fdata(dtype=np.float32)
        return data.reshape(int(np.prod(data.shape[:3])), -1)

    def _masked_frames(self, img, reference):
        frames = self._voxel_frames(self._align_image(img, reference))
        return self.mask_array(frames, self.resolved_mask_path, self.threshold)[2]

    def import_nifti_to_numpy_array(self, file_paths):
        imgs = [self._load_single_nifti(path) for path in file_paths]
        reference = self._reference_image(imgs)
        return np.column_stack([self._masked_frames(img, reference).reshape(-1) for img in imgs])

    def _map_to_image(self, map_data, fill_value=0):
        mask_img = nib.load(self.resolved_mask_path)
        full_voxels = int(np.prod(mask_img.shape))
        n_voxels = int(np.count_nonzero(mask_img.get_fdata().reshape(-1) > self.threshold))
        locations = n_voxels if np.asarray(map_data).size % n_voxels == 0 else full_voxels
        frames = np.asarray(map_data).reshape(locations, -1)
        volume = (self.unmask_array(frames, self.resolved_mask_path, self.threshold, fill_value) if locations == n_voxels else frames).reshape(*mask_img.shape, frames.shape[1])
        return nib.Nifti1Image(volume[..., 0] if frames.shape[1] == 1 else volume, mask_img.affine)

    def prepare_map_for_evaluation(self, map_data):
        return np.asarray(map_data, dtype=np.float32).reshape(-1)

    def prepare_evaluation_data(self, file_paths):
        return self.import_nifti_to_numpy_array(list(file_paths)).T.astype(np.float32, copy=False)

    @staticmethod
    def evaluation_size(model_size):
        return int(model_size)

    @staticmethod
    def is_native_map_file(path):
        return str(path).lower().endswith((".nii", ".nii.gz"))

    def load_map_values(self, path):
        return self.import_nifti_to_numpy_array([str(path)])[:, 0]

    @staticmethod
    def native_map_stem(path):
        name = os.path.basename(str(path))
        return name[:-7] if name.lower().endswith(".nii.gz") else os.path.splitext(name)[0]

    @staticmethod
    def _output_path(file_path, file_suffix):
        name = os.path.basename(file_path)
        stem = name[:-7] if name.lower().endswith(".nii.gz") else os.path.splitext(name)[0]
        return os.path.join(os.path.dirname(file_path), f"{stem}{file_suffix or ''}.nii.gz")

    def save_files(self, arr, file_paths, dry_run=True, file_suffix=None, fill_value=0):
        values = np.asarray(arr)
        values = values.reshape(-1, 1) if len(file_paths) == 1 else values
        for index, file_path in tqdm(list(enumerate(file_paths)), desc="Saving files"):
            out_path = self._output_path(file_path, file_suffix)
            if dry_run:
                print(f"Saving to: {out_path}")
            elif self.resolved_mask_path is None:
                from calvin_utils.neuroimaging_utils.nifti_utils.generate_nifti import view_and_save_nifti
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                view_and_save_nifti(values[:, index], out_dir=os.path.dirname(out_path), output_name=self.native_map_stem(out_path), silent=True)
            else:
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                nib.save(self._map_to_image(values[:, index], fill_value), out_path)
