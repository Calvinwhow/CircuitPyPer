#!/usr/bin/env python3
"""Generate functional-connectivity NIfTIs from configured input maps."""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from calvin_utils.neuroimaging_utils.fmri_utils.compute_connectivity import (
    DEFAULT_CHUNK_INDEX,
    DEFAULT_CONNECTOME_DIR,
    DEFAULT_MASK,
    FunctionalConnectivity,
)


# =============================================================================
# CONFIG
# =============================================================================

NIFTI_PATHS = []
OUT_DIR = ""

CONNECTOME_DIR = DEFAULT_CONNECTOME_DIR
MASK_PATH = DEFAULT_MASK
CHUNK_INDEX_PATH = DEFAULT_CHUNK_INDEX
CHUNK_PATTERN = "{chunk}_AvgR.npy"
ROW_BATCH_MB = 256

BINARIZE = False
THRESHOLD = None
OUTPUT_SUFFIX = "_functional_connectivity"


def main():
    if not NIFTI_PATHS or not OUT_DIR:
        raise ValueError("Set NIFTI_PATHS and OUT_DIR in CONFIG.")

    mapper = FunctionalConnectivity(
        connectome_dir=CONNECTOME_DIR,
        mask_path=MASK_PATH,
        chunk_index_path=CHUNK_INDEX_PATH,
        chunk_pattern=CHUNK_PATTERN,
        row_batch_mb=ROW_BATCH_MB,
    )
    saved_paths = mapper.save_profiles_from_niftis(
        nifti_paths=NIFTI_PATHS,
        out_dir=Path(OUT_DIR).expanduser(),
        binarize=BINARIZE,
        threshold=THRESHOLD,
        suffix=OUTPUT_SUFFIX,
    )
    print(f"Saved {len(saved_paths)} functional-connectivity map(s).")


if __name__ == "__main__":
    main()
