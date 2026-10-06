#!/usr/bin/env python3
"""Track and store an independent structural fiber bundle from every voxel."""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from calvin_utils.neuroimaging_utils.tract_utils.voxel_seeded_connectome import (
    VoxelSeededConnectomeBuilder,
)


# =============================================================================
# CONFIG
# =============================================================================

RECONSTRUCTION_PATH = ""  # DSI Studio .fib or .fib.gz reconstruction.
REFERENCE_NIFTI_PATH = ""
MASK_PATH = ""
OUTPUT_PATH = ""          # New directory; recommended suffix: .sampled_fibers

FIBERS_PER_VOXEL = 8
MAX_ATTEMPTS_PER_VOXEL = 64
RANDOM_SEED = 42
FA_THRESHOLD = 0.03
ANGLE_THRESHOLD = 60.0
TRACKING_STEP_SIZE = 1.0
MAX_STEPS = 250
MIN_LENGTH_MM = 10.0

# Inversion-ready representation. Inversion must use the same values.
SAMPLE_INTERVAL_MM = 1.0
MAX_LENGTH_MM = None
MAX_BUFFER_POINTS = 2_097_152  # About 24 MiB of buffered float32 XYZ data.


def main():
    required = {
        "RECONSTRUCTION_PATH": RECONSTRUCTION_PATH,
        "REFERENCE_NIFTI_PATH": REFERENCE_NIFTI_PATH,
        "MASK_PATH": MASK_PATH,
        "OUTPUT_PATH": OUTPUT_PATH,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"Set these CONFIG values: {', '.join(missing)}")

    builder = VoxelSeededConnectomeBuilder(
        reconstruction_path=RECONSTRUCTION_PATH,
        reference_nifti_path=REFERENCE_NIFTI_PATH,
        mask_path=MASK_PATH,
        fibers_per_voxel=FIBERS_PER_VOXEL,
        max_attempts_per_voxel=MAX_ATTEMPTS_PER_VOXEL,
        random_seed=RANDOM_SEED,
        fa_threshold=FA_THRESHOLD,
        angle_threshold=ANGLE_THRESHOLD,
        step_size=TRACKING_STEP_SIZE,
        max_steps=MAX_STEPS,
        min_length=MIN_LENGTH_MM,
    )
    store = builder.build_sampled(
        OUTPUT_PATH,
        sample_interval_mm=SAMPLE_INTERVAL_MM,
        max_length_mm=MAX_LENGTH_MM,
        max_buffer_points=MAX_BUFFER_POINTS,
    )
    print(
        f"Saved {store.n_fibers} sampled fibers ({store.n_points} XYZ samples) "
        f"in memory-mapped length buckets at {store.path}."
    )


if __name__ == "__main__":
    main()
