import os
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates


class BrainMaskTransfer:
    """
    A template's brain mask brought onto a scan's own voxel grid, through a registration.

    Every scan voxel is carried into template world mm by the registration matrix and reads the template
    mask there (linearly interpolated, inside where it reaches 0.5). The mask comes back on exactly the
    scan's grid; the scan is not touched.
    """

    def __init__(self, scan_shape, scan_affine, template_mask_path, affine_matrix):
        """
        scan_shape (np.ndarray, shape (3,), int): the scan's voxel grid size (I, J, K).
        scan_affine (np.ndarray, shape (4, 4), float): the scan's voxel-to-world matrix.
        template_mask_path (str): path to the template's brain mask (NIfTI file).
        affine_matrix (np.ndarray, shape (4, 4), float): the registration, from the scan's world mm to the template's world mm.
        """
        self.scan_shape = scan_shape
        self.scan_affine = scan_affine
        template_mask_img = self._load_and_validate_image(template_mask_path)
        self.template_mask_data = self._volume(template_mask_img)
        self.template_mask_affine = template_mask_img.affine
        self.affine_matrix = affine_matrix

    ###INTERNAL METHODS###
    def _load_and_validate_image(self, path):
        if not os.path.exists(path):
            raise ValueError(f"File does not exist: {path}")
        img = nib.load(path)
        if not isinstance(img.affine, np.ndarray) or img.affine.shape != (4, 4):
            raise ValueError(f"Affine matrix is corrupt or invalid in file: {path}")
        return img

    def _volume(self, img):
        """The image's first 3D volume as float32."""
        data = np.asarray(img.dataobj, dtype=np.float32)
        return data.reshape(data.shape[:3] + (-1,))[..., 0]

    def _matrix(self, matrix, name):
        matrix = np.asarray(matrix, float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(f"{name} must be a finite (4, 4) matrix, got shape {matrix.shape}")
        return matrix

    def _scan_to_template_voxels(self):
        """(4, 4) matrix from scan voxel indices to template voxel indices."""
        return np.linalg.inv(self.template_mask_affine) @ self.affine_matrix @ self.scan_affine

    def _mask_slice(self, to_template, k):
        """The mask on scan slice k (third voxel axis), shape (I, J), True inside the brain."""
        nx, ny = int(self.scan_shape[0]), int(self.scan_shape[1])
        i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        ijk = np.stack([i.ravel(), j.ravel(), np.full(i.size, k)], axis=1).astype(float)
        template_ijk = ijk @ to_template[:3, :3].T + to_template[:3, 3]
        values = map_coordinates(self.template_mask_data, template_ijk.T, order=1, mode="constant", cval=0.0)
        return values.reshape(nx, ny) >= 0.5

    ###PROPERTY METHODS###
    @property
    def scan_shape(self):
        return self._scan_shape

    @scan_shape.setter
    def scan_shape(self, shape):
        shape = np.asarray(shape, int).ravel()
        if shape.shape != (3,) or not (shape > 0).all():
            raise ValueError(f"scan_shape must be three positive sizes, got {shape}")
        self._scan_shape = shape

    @property
    def scan_affine(self):
        return self._scan_affine

    @scan_affine.setter
    def scan_affine(self, matrix):
        self._scan_affine = self._matrix(matrix, "scan_affine")

    @property
    def template_mask_data(self):
        """np.ndarray, shape (I, J, K), float32: the template's brain mask."""
        return self._template_mask_data

    @template_mask_data.setter
    def template_mask_data(self, data):
        self._template_mask_data = np.asarray(data, np.float32)

    @property
    def template_mask_affine(self):
        """np.ndarray, shape (4, 4), float: the template mask's voxel-to-world matrix."""
        return self._template_mask_affine

    @template_mask_affine.setter
    def template_mask_affine(self, matrix):
        self._template_mask_affine = self._matrix(matrix, "template_mask_affine")

    @property
    def affine_matrix(self):
        return self._affine_matrix

    @affine_matrix.setter
    def affine_matrix(self, matrix):
        self._affine_matrix = self._matrix(matrix, "affine_matrix")

    ###BUSINESS LOGIC###
    def run(self):
        """
        Returns (np.ndarray, shape (I, J, K), bool): the brain mask on the scan's grid, True inside.
        """
        to_template = self._scan_to_template_voxels()
        return np.stack([self._mask_slice(to_template, k) for k in range(int(self.scan_shape[2]))], axis=2)
