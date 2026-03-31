import os
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from nibabel.freesurfer.io import read_morph_data, read_annot

DEFAULT_SURFACE_MASK = None


class SurfaceIO:
    def __init__(self, mask_path=None, threshold=0):
        self.mask_path = mask_path
        self.threshold = threshold

    def _identify_surface_file_type(self, path):
        p = Path(path)
        suffixes = [s.lower() for s in p.suffixes]
        basename = p.name.lower()

        if suffixes[-2:] == ['.gii', '.gz'] or suffixes[-1:] == ['.gii']:
            return 'gii'

        freesurfer_scalar_suffixes = {
            'lh.thickness', 'rh.thickness',
            'lh.curv', 'rh.curv',
            'lh.sulc', 'rh.sulc',
            'lh.area', 'rh.area',
            'lh.volume', 'rh.volume'
        }

        freesurfer_mesh_suffixes = {
            'lh.white', 'rh.white',
            'lh.pial', 'rh.pial',
            'lh.inflated', 'rh.inflated',
            'lh.sphere', 'rh.sphere',
            'lh.orig', 'rh.orig'
        }

        freesurfer_annot_suffixes = {
            'lh.annot', 'rh.annot'
        }

        if basename in freesurfer_scalar_suffixes or any(basename.endswith("." + s.split(".", 1)[1]) for s in freesurfer_scalar_suffixes):
            return 'freesurfer_scalar'

        if basename in freesurfer_mesh_suffixes or any(basename.endswith("." + s.split(".", 1)[1]) for s in freesurfer_mesh_suffixes):
            return 'freesurfer_mesh'

        if basename in freesurfer_annot_suffixes or basename.endswith('.annot'):
            return 'freesurfer_annot'

        return 'unknown'

    def _load_single_gifti(self, file_path):
        img = nib.load(file_path)

        scalar_arrays = []
        for darray in img.darrays:
            data = np.asarray(darray.data)
            if data.ndim == 1:
                scalar_arrays.append(data)
            elif data.ndim == 2 and data.shape[1] == 1:
                scalar_arrays.append(data[:, 0])

        if len(scalar_arrays) == 0:
            raise ValueError(f"No 1D scalar data found in GIFTI file: {file_path}")

        if len(scalar_arrays) > 1:
            raise ValueError(f"Multiple scalar arrays found in GIFTI file: {file_path}. This importer expects one scalar map per file.")

        return scalar_arrays[0].flatten()

    def _load_single_freesurfer_scalar(self, file_path):
        return read_morph_data(file_path).flatten()

    def _load_single_freesurfer_annot(self, file_path):
        labels, _, _ = read_annot(file_path)
        return labels.flatten()

    def _load_single_surface_file(self, file_path):
        ftype = self._identify_surface_file_type(file_path)

        if ftype == 'gii':
            return self._load_single_gifti(file_path)

        if ftype == 'freesurfer_scalar':
            return self._load_single_freesurfer_scalar(file_path)

        if ftype == 'freesurfer_annot':
            return self._load_single_freesurfer_annot(file_path)

        if ftype == 'freesurfer_mesh':
            raise RuntimeError(
                f"SurfaceIO received a FreeSurfer mesh file, not a scalar surface file: {file_path}. "
                f"Mesh files like white/pial/inflated are not importable into a vertex-by-file matrix."
            )

        raise RuntimeError(f"Unknown or unsupported surface file type: {file_path}")

    def _resolve_mask_vector(self, mask_path=None, threshold=None):
        mask_path = self.mask_path if mask_path is None else mask_path
        threshold = self.threshold if threshold is None else threshold

        if mask_path is None:
            return None

        ftype = self._identify_surface_file_type(mask_path)

        if ftype == 'gii':
            mask = self._load_single_gifti(mask_path)
        elif ftype == 'freesurfer_scalar':
            mask = self._load_single_freesurfer_scalar(mask_path)
        elif ftype == 'freesurfer_annot':
            mask = self._load_single_freesurfer_annot(mask_path)
        else:
            raise RuntimeError(f"Unsupported surface mask file type: {mask_path}")

        return mask > threshold

    @staticmethod
    def mask_array(arr, mask):
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
        if mask is None:
            return arr

        n_vertices = mask.shape[0]

        if arr.ndim == 1:
            out = np.full(n_vertices, fill_value, dtype=arr.dtype)
            out[mask] = arr
        else:
            out = np.full((n_vertices, arr.shape[1]), fill_value, dtype=arr.dtype)
            out[mask, :] = arr

        return out

    def import_surface_to_numpy_array(self, file_paths):
        data_list = []
        expected_len = None

        for file_path in tqdm(file_paths, desc='Importing surface files'):
            data = self._load_single_surface_file(file_path)

            if expected_len is None:
                expected_len = data.shape[0]
            elif data.shape[0] != expected_len:
                raise ValueError(
                    f"Surface length mismatch. Expected {expected_len} vertices but got {data.shape[0]} in file: {file_path}"
                )

            data_list.append(data)

        arr = np.column_stack(data_list)

        mask = self._resolve_mask_vector()
        _, _, arr = self.mask_array(arr, mask)

        return arr