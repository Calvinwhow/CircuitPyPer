import os
import numpy as np
import nibabel as nib

from calvin_utils.neuroimaging_utils.registration_utils.apply_warp.schema import WarpedImage


class ApplyWarp:
    """
    An image placed in a warped frame by an affine warp: its voxels untouched, the warp put in front of its affine.

    A file in a scan's native space, warped by that scan's registration (e.g. SkullStripResult.transform),
    lands on the warped scan exactly where it sat on the native one. Nothing is resampled.
    """

    def __init__(self, nifti_path, warp):
        """
        nifti_path (str): path to the image to warp (NIfTI file), in the same native space the warp starts from.
        warp (np.ndarray, shape (4, 4), float): the affine warp, from native world mm to the warped frame's world mm.
        """
        self.nifti_path = nifti_path
        self.warp = warp

    ###PROPERTY METHODS###
    @property
    def nifti_path(self):
        return self._nifti_path

    @nifti_path.setter
    def nifti_path(self, path):
        path = str(path)
        if not os.path.exists(path):
            raise ValueError(f"File does not exist: {path}")
        self._nifti_path = path

    @property
    def warp(self):
        return self._warp

    @warp.setter
    def warp(self, matrix):
        matrix = np.asarray(matrix, float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(f"warp must be a finite (4, 4) matrix, got shape {matrix.shape}")
        if abs(np.linalg.det(matrix[:3, :3])) < 1e-9:
            raise ValueError("warp is singular")
        self._warp = matrix

    ###BUSINESS LOGIC###
    def run(self):
        """
        Returns (WarpedImage): the image's own voxels (data) and its voxel-to-world matrix in the warped frame (affine = warp @ file affine).
        """
        img = nib.load(self.nifti_path)
        return WarpedImage(data=np.asanyarray(img.dataobj), affine=self.warp @ np.asarray(img.affine, float))
