"""Data structures that ApplyWarp takes or returns."""

from dataclasses import dataclass

import numpy as np


@dataclass
class WarpedImage:
    """
    An image's own voxels, placed in a warped frame.

    data (np.ndarray, shape (I, J, K) or (I, J, K, T), the file's dtype): the image's voxels, unchanged and never resampled.
    affine (np.ndarray, shape (4, 4), float): voxel-to-world matrix in the warped frame, = warp @ the file's affine.
    """

    data: np.ndarray
    affine: np.ndarray
