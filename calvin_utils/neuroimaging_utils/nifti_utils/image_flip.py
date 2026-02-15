import os
import numpy as np
import nibabel as nib
from nibabel import orientations

class NiftiHorizontalFlipper:
    """
    Flip NIfTI volumes left–right (sagittal mirror) while keeping medial
    voxels medial and producing a valid affine.

    Parameters
    ----------
    nifti_path : str
        One *.nii / *.nii.gz file.
    flip_affine : bool
        If True, the affine is flipped, translating right to left. 
        ONLY DO THIS IF YOUR LESIONS ARE NOT ALREADY BOUNDED IN AN MNI BOUNDING BOX
    """

    def __init__(self, nifti_path, flip_affine=False):
        self.flip_affine = flip_affine
        if isinstance(nifti_path, str):
            self.img = self._load(nifti_path)
        self._prep_outdir(nifti_path)

    # helpers
    def _load(self, path):
        return nib.load(path)
    
    def _prep_outdir(self, nifti_path, prefix="flipped_"):
        out_dir = os.path.dirname(nifti_path)
        os.makedirs(out_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(nifti_path))[0].replace(".nii", "")
        self.out_path = os.path.join(out_dir, f"{prefix}{base_name}.nii.gz")

    def _flip_img(self, nii):
        """
        Flip image in voxel-space, then adjust affine so MNI world-space coordinates are mirrored across x = 0.
        """
        data      = nii.get_fdata(dtype=np.float32)
        hdr       = nii.header.copy()
        affine    = nii.affine.copy()

        # Ensure we know which voxel axis is the world X‐axis
        axcodes = orientations.aff2axcodes(affine)           # e.g. ('R', 'A', 'S')
        x_axis   = axcodes.index('R') if 'R' in axcodes else axcodes.index('L')

        # 1) flip voxel order along that axis
        data = np.flip(data, axis=x_axis)

        # 2) update affine: negate the X axis and translate by (n-1)*voxel_size
        voxel_size = np.abs(affine[x_axis, x_axis])
        if self.flip_affine:
            affine[x_axis, x_axis] *= -1
            affine[x_axis, 3] += (data.shape[x_axis] - 1) * voxel_size
        return nib.Nifti1Image(data, affine, hdr)

    # public API
    def run(self):
        """Mirror every NIfTI in `self.nifti_paths` and write to `out_dir`"""
        flipped_img = self._flip_img(self.img)
        nib.save(flipped_img, self.out_path)