#!/usr/bin/env python3
"""Stream a connectome once and save inverted target-trajectory maps."""

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

VOXEL_SEEDED_CONNECTOME_PATH = ""  # Sampled-fiber directory from script 00.
REFERENCE_NIFTI_PATH = ""
MASK_PATH = None
TARGET_FIBER_PATHS = []
OUT_DIR = ""

# Target fiber-value selection. The paired .fib.json identifies the ordered
# atlas whose fibers are selected by these values.
TARGET_WEIGHTING = "binary"     # 'binary' or raw-value 'weighted'
TARGET_MIN_ABS_VALUE = None      # None keeps every eligible positive value.
TARGET_TOP_PERCENT = None        # Binary default: upper 5% of raw values.
TARGET_FIBER_ATLAS_PATH = None   # Only needed for a relocated/bare legacy vector.
TARGET_BUNDLE_REDUCTION = "average_trajectory"  # Or: 'pairwise_mean'.

SAMPLE_INTERVAL_MM = 1.0
MAX_LENGTH_MM = None            # None leaves trajectories untruncated.
MEMORY_BUDGET_MB = 256          # Temporary broadcast-array budget.
MAX_FIBER_BATCH_SIZE = 16_384

SIMILARITY = "valid_sample_cosine"  # Ignore missing slots outside the shorter fiber.
OUTPUT_SUFFIX = None


def main():
    if not VOXEL_SEEDED_CONNECTOME_PATH or not REFERENCE_NIFTI_PATH:
        raise ValueError(
            "Set VOXEL_SEEDED_CONNECTOME_PATH and REFERENCE_NIFTI_PATH in CONFIG."
        )
    if not TARGET_FIBER_PATHS or not OUT_DIR:
        raise ValueError("Set TARGET_FIBER_PATHS and OUT_DIR in CONFIG.")

    mapper = InvertedFiberConnectivity(
        voxel_seeded_connectome_path=VOXEL_SEEDED_CONNECTOME_PATH,
        reference_nifti_path=REFERENCE_NIFTI_PATH,
        mask_path=MASK_PATH,
        sample_interval_mm=SAMPLE_INTERVAL_MM,
        max_length_mm=MAX_LENGTH_MM,
        memory_budget_mb=MEMORY_BUDGET_MB,
        max_fiber_batch_size=MAX_FIBER_BATCH_SIZE,
    )
    saved_paths = mapper.save_inverted_profiles_from_fibers(
        target_fiber_paths=TARGET_FIBER_PATHS,
        out_dir=Path(OUT_DIR).expanduser(),
        similarity=SIMILARITY,
        suffix=OUTPUT_SUFFIX,
        target_weighting=TARGET_WEIGHTING,
        target_min_abs_value=TARGET_MIN_ABS_VALUE,
        target_top_percent=TARGET_TOP_PERCENT,
        target_fiber_atlas_path=TARGET_FIBER_ATLAS_PATH,
        target_bundle_reduction=TARGET_BUNDLE_REDUCTION,
    )
    print(f"Saved {len(saved_paths)} inverted fiber map(s).")


if __name__ == "__main__":
    main()
