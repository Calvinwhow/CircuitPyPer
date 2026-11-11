import os
import numpy as np
import nibabel as nib

from calvin_utils.resource_paths import resource_path
from calvin_utils.neuroimaging_utils.registration_utils.affine_registration.affine_registration import AffineRegistration
from calvin_utils.neuroimaging_utils.registration_utils.skull_strip.schema import SkullStripResult
from calvin_utils.neuroimaging_utils.registration_utils.skull_strip.brain_mask_transfer_utils import BrainMaskTransfer
from calvin_utils.neuroimaging_utils.registration_utils.skull_strip.rigid_from_affine_utils import RigidFromAffine


class SkullStrip:
    """
    A scan skull-stripped with the MNI brain mask, returned in the frame of the registration asked for.

    AffineRegistration fits the scan to the MNI template (rigid, then full affine), and BrainMaskTransfer
    brings the MNI brain mask onto the scan's own grid through that fit; the mask always uses the full
    affine, which fits each brain's size. The scan's voxels inside the mask are kept, 0 outside, and never
    resampled. ``affine`` chooses the registration returned: the full affine (the brain stretched to MNI),
    or its simplification to rotation and translation about the brain's centre (RigidFromAffine), which
    keeps the brain's own size and shape.
    """

    def __init__(self, nifti_path, template_path=None, template_mask_path=None, affine=False, n_samples=50000, bins=32, seed=0):
        """
        nifti_path (str): path to the scan to skull-strip (NIfTI file), e.g. a patient's T1.
        template_path (str or None): path to the MNI template (NIfTI file); None is the packaged MNI152_T1_2mm_brain.nii.
        template_mask_path (str or None): path to the template's brain mask (NIfTI file); None is the packaged MNI152_T1_2mm_brain_mask_filled.nii.
        affine (bool): True returns the full affine registration (brain stretched to MNI); False returns its rigid simplification (brain's own size and shape).
        n_samples (int): number of MNI brain points the registration's mutual information is scored at, at least 1000.
        bins (int): number of histogram bins per image for the mutual information.
        seed (int): seed for drawing those sample points, so a run is repeatable.
        """
        self.nifti_path = nifti_path
        self.template_path = template_path
        self.template_mask_path = template_mask_path
        self.affine = affine
        self.affine_registration = AffineRegistration(self.nifti_path, self.template_path, self.template_mask_path, n_samples=n_samples, bins=bins, seed=seed)

    ###INTERNAL METHODS###
    def _existing_path(self, path):
        path = str(path)
        if not os.path.exists(path):
            raise ValueError(f"File does not exist: {path}")
        return path

    def _brain_center(self, mask, scan_affine):
        """Centre of the brain mask, world mm in the scan's own space, shape (3,)."""
        ijk = np.argwhere(mask)
        if len(ijk) == 0:
            raise ValueError("the brain mask came back empty: the registration failed to overlap the template")
        return scan_affine[:3, :3] @ ijk.mean(axis=0) + scan_affine[:3, 3]

    def _masked(self, data, mask):
        """The scan's voxels inside ``mask`` (I, J, K), 0 outside; a 4D scan is masked in every volume; dtype kept."""
        inside = mask.reshape(mask.shape + (1,) * (data.ndim - 3))
        return np.where(inside, data, 0).astype(data.dtype)

    ###PROPERTY METHODS###
    @property
    def nifti_path(self):
        return self._nifti_path

    @nifti_path.setter
    def nifti_path(self, path):
        self._nifti_path = self._existing_path(path)

    @property
    def template_path(self):
        return self._template_path

    @template_path.setter
    def template_path(self, path):
        self._template_path = self._existing_path(path if path is not None else resource_path("MNI152_T1_2mm_brain.nii"))

    @property
    def template_mask_path(self):
        return self._template_mask_path

    @template_mask_path.setter
    def template_mask_path(self, path):
        self._template_mask_path = self._existing_path(path if path is not None else resource_path("MNI152_T1_2mm_brain_mask_filled.nii"))

    @property
    def affine(self):
        return self._affine

    @affine.setter
    def affine(self, value):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"affine must be True or False, got {value!r}")
        self._affine = bool(value)

    @property
    def affine_registration(self):
        """AffineRegistration: the scan's registration to the MNI template."""
        return self._affine_registration

    @affine_registration.setter
    def affine_registration(self, registration):
        if not isinstance(registration, AffineRegistration):
            raise ValueError("affine_registration must be an AffineRegistration")
        self._affine_registration = registration

    ###BUSINESS LOGIC###
    def run(self):
        """
        Returns (SkullStripResult): the stripped brain (own voxels), its affine in the registration's frame, the brain mask and the registration returned.
        """
        affine_matrix = self.affine_registration.run()
        scan = nib.load(self.nifti_path)
        scan_affine = np.asarray(scan.affine, float)
        data = np.asanyarray(scan.dataobj)
        mask = BrainMaskTransfer(np.array(scan.shape[:3]), scan_affine, self.template_mask_path, affine_matrix).run()
        transform = affine_matrix if self.affine else RigidFromAffine(affine_matrix, self._brain_center(mask, scan_affine)).run()
        return SkullStripResult(brain=self._masked(data, mask), brain_affine=transform @ scan_affine, brain_mask=mask, transform=transform)
