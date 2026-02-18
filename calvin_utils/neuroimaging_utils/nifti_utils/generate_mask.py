from typing import Iterable, List, Tuple

import numpy as np
import nibabel as nib

from calvin_utils.neuroimaging_utils.ccm_utils.bounding_box import NiftiBoundingBox
from calvin_utils.neuroimaging_utils.nifti_utils.utils_mask import normalize_paths


class GenerateMask:
    """
    Generate an in-memory composite mask that encompasses all provided NIfTI files.

    This is intended to provide a shared spatial reference when no external mask
    is provided. The mask can be reused across downstream operations without
    persisting it to disk.

    Parameters
    ----------
    nifti_paths : Iterable[str]
        List of NIfTI file paths to encompass.
    threshold : float or None, default None
        Optional threshold applied to the collapsed data before binarization.
        If None, no additional thresholding is applied and any nonzero voxel is
        included in the mask.
    verbose : bool, default True
        Whether to print progress messages.
    """

    def __init__(
        self,
        nifti_paths: Iterable[str],
        *,
        threshold: float | None = None,
        verbose: bool = True,
    ):
        self.nifti_paths = nifti_paths
        self.threshold = threshold
        self.verbose = verbose

        self._bbox = None
        self._mask_data = None
        self._mask_img = None
        self._mask_indices = None
        self._collapsed_data = None
        self.force_reorient = False

    # ---- properties ----
    @property
    def nifti_paths(self) -> List[str]:
        return self._nifti_paths

    @nifti_paths.setter
    def nifti_paths(self, value: Iterable[str]):
        paths = normalize_paths(value)
        if not paths:
            raise ValueError("nifti_paths must contain at least one valid path.")
        self._nifti_paths = paths

    @property
    def threshold(self) -> float | None:
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        if value is None:
            self._threshold = None
            return
        self._threshold = float(value)

    @property
    def verbose(self) -> bool:
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = bool(value)

    @property
    def mask_data(self) -> np.ndarray | None:
        return self._mask_data

    @property
    def mask_img(self) -> nib.Nifti1Image | None:
        return self._mask_img

    @property
    def mask_indices(self) -> np.ndarray | None:
        return self._mask_indices

    @property
    def collapsed_data(self) -> np.ndarray | None:
        return self._collapsed_data

    # ---- main API ----
    def run(self) -> Tuple[nib.Nifti1Image, np.ndarray]:
        """
        Generate the composite mask in memory.

        Returns
        -------
        (mask_img, mask_indices)
            mask_img is a NIfTI image in the bounding-box space.
            mask_indices is a boolean 1D array (flattened) for fast indexing.
        """
        self._build_bbox()
        self._collapse_data()
        self._build_mask()
        self._build_mask_img()
        return self._mask_img, self._mask_indices

    # ---- internal methods ----
    def _build_bbox(self):
        if self.verbose:
            print(f"Generating bounding box for {len(self.nifti_paths)} files...")
        bbox = NiftiBoundingBox(self.nifti_paths)
        bbox.force_reorient = self.force_reorient
        bbox.generate_bounding_box()
        bbox.add_niftis_to_bounding_box()
        self._bbox = bbox

    def _collapse_data(self):
        if self._bbox is None:
            raise ValueError("Bounding box not initialized.")
        self._bbox.collapse_bbox_to_3d()
        self._collapsed_data = self._bbox._collapsed_data

    def _build_mask(self):
        if self._collapsed_data is None:
            raise ValueError("Collapsed data not available.")
        if self.threshold is None:
            mask = self._collapsed_data > 0
        else:
            mask = self._collapsed_data > self.threshold
        self._mask_data = mask.astype(np.uint8)
        self._mask_indices = self._mask_data.flatten().astype(bool)

    def _build_mask_img(self):
        if self._mask_data is None or self._bbox is None:
            raise ValueError("Mask data or bounding box missing.")
        affine = self._bbox.bounding_box_affine
        header = nib.Nifti1Header()
        header.set_xyzt_units('mm')
        header.set_sform(affine, code=1)
        header.set_qform(affine, code=1)
        header.set_data_dtype(np.uint8)
        header['descrip'] = 'Composite bounding-box mask'
        self._mask_img = nib.Nifti1Image(self._mask_data, affine=affine, header=header)
