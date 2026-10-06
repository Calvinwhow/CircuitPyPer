#!/usr/bin/env python3
"""Register target fiber sets and save their inverted voxelwise maps."""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from calvin_utils.neuroimaging_utils.tract_utils.inverted_connectivity import (
    InvertedFiberConnectivity,
)


# =============================================================================
# CONFIG
# =============================================================================

CONNECTOME_PATH = ""
REFERENCE_NIFTI_PATH = ""
MASK_PATH = None
TARGET_FIBER_PATHS = []
OUT_DIR = ""

SAMPLE_INTERVAL_MM = 1.0
MAX_LENGTH_MM = None            # None uses the connectome's observed maximum.
STEP_SIZE_VOX = 0.5
FIBER_BATCH_SIZE = 2048
CACHE_DIR = None

SIMILARITY = "padded_cosine"   # Also: overlap_cosine, coverage_weighted_cosine.
VOXEL_REDUCTION = "max"        # 'max', 'mean', or 'sum'
OUTPUT_SUFFIX = None


def main():
    if not CONNECTOME_PATH or not REFERENCE_NIFTI_PATH:
        raise ValueError("Set CONNECTOME_PATH and REFERENCE_NIFTI_PATH in CONFIG.")
    if not TARGET_FIBER_PATHS or not OUT_DIR:
        raise ValueError("Set TARGET_FIBER_PATHS and OUT_DIR in CONFIG.")

    mapper = InvertedFiberConnectivity(
        connectome_path=CONNECTOME_PATH,
        reference_nifti_path=REFERENCE_NIFTI_PATH,
        mask_path=MASK_PATH,
        sample_interval_mm=SAMPLE_INTERVAL_MM,
        max_length_mm=MAX_LENGTH_MM,
        step_size_vox=STEP_SIZE_VOX,
        fiber_batch_size=FIBER_BATCH_SIZE,
        cache_dir=CACHE_DIR,
    )
    saved_paths = mapper.save_inverted_profiles_from_fibers(
        target_fiber_paths=TARGET_FIBER_PATHS,
        out_dir=Path(OUT_DIR).expanduser(),
        similarity=SIMILARITY,
        voxel_reduction=VOXEL_REDUCTION,
        suffix=OUTPUT_SUFFIX,
    )
    print(f"Saved {len(saved_paths)} inverted fiber map(s).")


if __name__ == "__main__":
    main()
