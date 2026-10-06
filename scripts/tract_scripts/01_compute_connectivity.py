#!/usr/bin/env python3
"""Generate configured NIfTI-to-fiber connectivity profiles."""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from calvin_utils.neuroimaging_utils.tract_utils.compute_connectivity import (
    FiberConnectivity,
)


# =============================================================================
# CONFIG
# =============================================================================

FIBER_ATLAS_PATH = ""
REFERENCE_NIFTI_PATH = ""
NIFTI_PATHS = []
OUT_DIR = ""

MODE = "binary"                # 'binary', 'max', 'mean', or 'sum'
BINARIZE = True
THRESHOLD = 0
STEP_SIZE_VOX = 0.5
FIBER_MASK = None

OUTPUT_SUFFIX = "_fiber_connectivity"
SAVE_MATRIX = False
MATRIX_NAME = "fiber_connectivity_matrix.npy"


def main():
    if not FIBER_ATLAS_PATH or not REFERENCE_NIFTI_PATH:
        raise ValueError(
            "Set FIBER_ATLAS_PATH and REFERENCE_NIFTI_PATH in CONFIG."
        )
    if not NIFTI_PATHS or not OUT_DIR:
        raise ValueError("Set NIFTI_PATHS and OUT_DIR in CONFIG.")

    mapper = FiberConnectivity(fiber_indexer=None, mode=MODE)
    mapper.generate_and_save_from_paths(
        fiber_file_path=FIBER_ATLAS_PATH,
        reference_nifti_path=REFERENCE_NIFTI_PATH,
        nifti_paths=NIFTI_PATHS,
        out_dir=Path(OUT_DIR).expanduser(),
        mode=MODE,
        binarize=BINARIZE,
        threshold=THRESHOLD,
        suffix=OUTPUT_SUFFIX,
        step_size_vox=STEP_SIZE_VOX,
        fiber_mask=FIBER_MASK,
        save_matrix=SAVE_MATRIX,
        matrix_name=MATRIX_NAME,
    )
    print(f"Saved fiber-connectivity profiles to: {OUT_DIR}")


if __name__ == "__main__":
    main()
