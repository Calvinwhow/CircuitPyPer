"""Data structures that RunRegistration takes or returns."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RegistrationOutputs:
    """
    Where one scan's outputs are written; each field's name is also its column in the scan CSV.

    brain_path (Path): the skull-stripped scan in the registration's frame, its own voxels (NIfTI) -- what a viewer receives.
    brain_mask_path (Path): the brain mask on the scan's own grid, in the scan's own space (NIfTI, uint8, 1 inside).
    transform_path (Path): the 4x4 registration returned (rigid or full affine) as text, scan world mm -> MNI frame world mm.
    """

    brain_path: Path
    brain_mask_path: Path
    transform_path: Path
