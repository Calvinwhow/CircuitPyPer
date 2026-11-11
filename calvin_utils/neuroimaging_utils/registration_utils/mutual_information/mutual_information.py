import os
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates

from calvin_utils.neuroimaging_utils.registration_utils.mutual_information.intensity_mutual_information_utils import IntensityMutualInformation


class MutualInformation:
    """
    How well a scan, carried through a candidate matrix, matches a template inside the template's brain mask.

    A fixed random sample of template voxels inside the brain mask is drawn once. Each run carries those
    points into the scan through a template-to-scan matrix, reads the scan there (linearly interpolated,
    0 outside the scan) and scores the scan's intensities against the template's (IntensityMutualInformation).
    Only the brain is compared, so a registration maximising this score is not pulled by skull, scalp or neck.
    """

    def __init__(self, moving_path, template_path, template_mask_path, n_samples=50000, bins=32, seed=0):
        """
        moving_path (str): path to the scan (NIfTI file).
        template_path (str): path to the template (NIfTI file), e.g. an MNI T1.
        template_mask_path (str): path to the template's brain mask (NIfTI file), on the template's grid.
        n_samples (int): number of template points inside the brain mask the score is computed at, at least 1000.
        bins (int): number of histogram bins per image.
        seed (int): seed for drawing the sample points, so a score is repeatable.
        """
        moving_img = self._load_and_validate_image(moving_path)
        template_img = self._load_and_validate_image(template_path)
        template_mask_img = self._load_and_validate_image(template_mask_path)
        if template_mask_img.shape[:3] != template_img.shape[:3] or not np.allclose(template_mask_img.affine, template_img.affine, atol=1e-3):
            raise ValueError("the template mask must be on the template's grid (same shape and affine)")
        self.moving_data = self._volume(moving_img)
        self.moving_affine = moving_img.affine
        self.template_data = self._volume(template_img)
        self.template_affine = template_img.affine
        self.template_mask = self._volume(template_mask_img) > 0.5
        self.n_samples = n_samples
        self.bins = bins
        self.seed = seed
        self.sample_ijk = self._draw_sample_ijk()
        self.intensity_mutual_information = IntensityMutualInformation(self.template_data[tuple(self.sample_ijk.T)], np.percentile(self.moving_data[self.moving_data > 0], [1, 99]), bins=self.bins)

    ###INTERNAL METHODS###
    def _load_and_validate_image(self, path):
        if not os.path.exists(path):
            raise ValueError(f"File does not exist: {path}")
        img = nib.load(path)
        if not isinstance(img.affine, np.ndarray) or img.affine.shape != (4, 4):
            raise ValueError(f"Affine matrix is corrupt or invalid in file: {path}")
        return img

    def _volume(self, img):
        """The image's first 3D volume as float32, non-finite values set to 0."""
        data = np.asarray(img.dataobj, dtype=np.float32)
        data = data.reshape(data.shape[:3] + (-1,))[..., 0]
        return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    def _voxels_to_world(self, affine, ijk):
        """(N, 3) voxel indices -> (N, 3) world mm."""
        return ijk @ affine[:3, :3].T + affine[:3, 3]

    def _draw_sample_ijk(self):
        """(n_samples, 3) template voxel indices inside the brain mask, drawn with the seed."""
        brain_ijk = np.argwhere(self.template_mask)
        if len(brain_ijk) == 0:
            raise ValueError("the template mask is empty")
        chosen = np.random.default_rng(self.seed).choice(len(brain_ijk), size=min(self.n_samples, len(brain_ijk)), replace=False)
        return brain_ijk[chosen]

    def _moving_values(self, template_to_moving):
        """The scan's intensities at the sample points under a template-to-scan matrix, shape (N,); 0 outside the scan."""
        to_voxels = np.linalg.inv(self.moving_affine) @ template_to_moving
        ijk = self._voxels_to_world(to_voxels, self.sample_points)
        return map_coordinates(self.moving_data, ijk.T, order=1, mode="constant", cval=0.0)

    ###PROPERTY METHODS###
    @property
    def moving_data(self):
        """np.ndarray, shape (I, J, K), float32: the scan's first volume."""
        return self._moving_data

    @moving_data.setter
    def moving_data(self, data):
        self._moving_data = np.asarray(data, np.float32)

    @property
    def moving_affine(self):
        """np.ndarray, shape (4, 4), float: the scan's voxel-to-world matrix."""
        return self._moving_affine

    @moving_affine.setter
    def moving_affine(self, affine):
        self._moving_affine = np.asarray(affine, float)

    @property
    def template_data(self):
        """np.ndarray, shape (I, J, K), float32: the template's first volume."""
        return self._template_data

    @template_data.setter
    def template_data(self, data):
        self._template_data = np.asarray(data, np.float32)

    @property
    def template_affine(self):
        """np.ndarray, shape (4, 4), float: the template's voxel-to-world matrix."""
        return self._template_affine

    @template_affine.setter
    def template_affine(self, affine):
        self._template_affine = np.asarray(affine, float)

    @property
    def template_mask(self):
        """np.ndarray, shape (I, J, K), bool: the template's brain mask, True inside."""
        return self._template_mask

    @template_mask.setter
    def template_mask(self, mask):
        self._template_mask = np.asarray(mask, bool)

    @property
    def n_samples(self):
        return self._n_samples

    @n_samples.setter
    def n_samples(self, value):
        if int(value) < 1000:
            raise ValueError(f"n_samples must be at least 1000, got {value}")
        self._n_samples = int(value)

    @property
    def bins(self):
        return self._bins

    @bins.setter
    def bins(self, value):
        self._bins = int(value)

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = int(value)

    @property
    def sample_ijk(self):
        """np.ndarray, shape (N, 3), int: template voxel indices the score is computed at."""
        return self._sample_ijk

    @sample_ijk.setter
    def sample_ijk(self, ijk):
        ijk = np.asarray(ijk)
        if ijk.ndim != 2 or ijk.shape[1] != 3:
            raise ValueError(f"sample_ijk must have shape (N, 3), got {ijk.shape}")
        self._sample_ijk = ijk
        self._sample_points = self._voxels_to_world(self.template_affine, ijk.astype(float))

    @property
    def sample_points(self):
        """np.ndarray, shape (N, 3), float: the sample points in template world mm."""
        return self._sample_points

    @property
    def intensity_mutual_information(self):
        """IntensityMutualInformation: the template's intensities at the sample points, binned once."""
        return self._intensity_mutual_information

    @intensity_mutual_information.setter
    def intensity_mutual_information(self, scorer):
        if not isinstance(scorer, IntensityMutualInformation):
            raise ValueError("intensity_mutual_information must be an IntensityMutualInformation")
        self._intensity_mutual_information = scorer

    ###BUSINESS LOGIC###
    def run(self, template_to_moving):
        """
        template_to_moving (np.ndarray, shape (4, 4), float): candidate matrix from template world mm to scan world mm.
        Returns (float): mutual information of the template and the scan at the sample points, in nats.
        """
        template_to_moving = np.asarray(template_to_moving, float)
        if template_to_moving.shape != (4, 4):
            raise ValueError(f"template_to_moving must have shape (4, 4), got {template_to_moving.shape}")
        return self.intensity_mutual_information.run(self._moving_values(template_to_moving))
