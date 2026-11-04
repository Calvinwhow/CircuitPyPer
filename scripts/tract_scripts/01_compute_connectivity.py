#!/usr/bin/env python3
"""Generate fiber-connectivity profiles from the lesion paths in a CSV."""

import sys
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path: sys.path.insert(0, str(PROJECT_DIR))

from calvin_utils.neuroimaging_utils.tract_utils.compute_connectivity import FiberConnectivity
from calvin_utils.resource_paths import lazy_resource_path


# =============================================================================
# CONFIG
# =============================================================================

CSV_PATH = Path("/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/optimized_master_list_filtered.csv")
NIFTI_COLUMN = "Nifti_File_Path"
OUTPUT_COLUMN = "zarr5GB"

FIBER_ATLAS_PATH = Path("/Volumes/OneTouch/resources/zarr5GB.npz")  # .npz fiber atlas from 00_seed_atlas.py
REFERENCE_NIFTI_PATH = lazy_resource_path("MNI152_T1_2mm_brain_mask_dil.nii.gz")  # defines the voxel grid
MODE = "max"  # 'binary', 'max', 'mean', or 'sum'
STEP_SIZE_VOX = 0.5
FIBER_MASK = None

BINARIZE = False
THRESHOLD = 0       # Only applied if Binarize is True
OUTPUT_SUFFIX = ""


def main():
    df = pd.read_csv(CSV_PATH)
    valid_rows = df[NIFTI_COLUMN].notna() & df[NIFTI_COLUMN].astype(str).str.strip().ne("")
    nifti_paths = df.loc[valid_rows, NIFTI_COLUMN].map(Path).tolist()
    mapper = FiberConnectivity.from_atlas(FIBER_ATLAS_PATH, REFERENCE_NIFTI_PATH, mode=MODE, step_size_vox=STEP_SIZE_VOX, fiber_mask=FIBER_MASK)
    saved_paths = mapper.save_profiles_from_niftis(nifti_paths, binarize=BINARIZE, threshold=THRESHOLD, suffix=OUTPUT_SUFFIX)
    df.loc[valid_rows, OUTPUT_COLUMN] = [str(path) for path in saved_paths]
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved {len(saved_paths)} fiber-connectivity profiles and updated {CSV_PATH}")


if __name__ == "__main__": main()
