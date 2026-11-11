"""Data structures that SkullStrip takes or returns."""

from dataclasses import dataclass

import numpy as np


@dataclass
class SkullStripResult:
    """
    A scan, skull-stripped, in the frame of the registration it was asked for.

    brain (np.ndarray, shape (I, J, K) or (I, J, K, T), the scan's dtype): the scan's own voxels inside the brain mask, 0 outside; never resampled.
    brain_affine (np.ndarray, shape (4, 4), float): voxel-to-world matrix of brain in the registration's frame, = transform @ the scan's affine.
    brain_mask (np.ndarray, shape (I, J, K), bool): the brain mask on the scan's own grid, True inside.
    transform (np.ndarray, shape (4, 4), float): the registration returned, from the scan's world mm to the MNI frame's world mm (rigid or full affine).
    """

    brain: np.ndarray
    brain_affine: np.ndarray
    brain_mask: np.ndarray
    transform: np.ndarray
