#!/usr/bin/env python3
"""Generate connectivity maps from the lesion paths in a CSV."""

import sys
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path: sys.path.insert(0, str(PROJECT_DIR))

from calvin_utils.neuroimaging_utils.fmri_utils.compute_connectivity import DEFAULT_CHUNK_INDEX, DEFAULT_MASK, FunctionalConnectivity


# =============================================================================
# CONFIG
# =============================================================================



CSV_PATH = Path("/Volumes/OneTouch/01x_Dhand_RCPStrokes/metadata/master.csv")
NIFTI_COLUMN = "Nifti_File_Path"
OUTPUT_COLUMN = "conn_path"

CONNECTOME_ROOT = Path("/Volumes/HowExp2/resources/connectome_rs_fMRI/GSP1000")
CONNECTOME_DIR = CONNECTOME_ROOT / "T_fp8"
MASK_PATH = DEFAULT_MASK
CHUNK_INDEX_PATH = DEFAULT_CHUNK_INDEX
CHUNK_PATTERN = "{chunk}_T.npy"
COMBO_DIR = CONNECTOME_ROOT / "Combo_fp8"
STD_MASK_PATH = CONNECTOME_ROOT / "GSP1000_std_weighted_mask.nii.gz"
NORM_MASK_PATH = CONNECTOME_ROOT / "GSP1000_norm_weighted_mask.nii.gz"
ROW_BATCH_MB = 256

BINARIZE = False
THRESHOLD = None
OUTPUT_SUFFIX = ""


def main():
    df = pd.read_csv(CSV_PATH)
    valid_rows = df[NIFTI_COLUMN].notna() & df[NIFTI_COLUMN].astype(str).str.strip().ne("")
    nifti_paths = df.loc[valid_rows, NIFTI_COLUMN].map(Path).tolist()
    mapper = FunctionalConnectivity(connectome_dir=CONNECTOME_DIR, mask_path=MASK_PATH, chunk_index_path=CHUNK_INDEX_PATH, chunk_pattern=CHUNK_PATTERN, row_batch_mb=ROW_BATCH_MB, combo_dir=COMBO_DIR, std_mask_path=STD_MASK_PATH, norm_mask_path=NORM_MASK_PATH)
    saved_paths = mapper.save_profiles_from_niftis(nifti_paths=nifti_paths, binarize=BINARIZE, threshold=THRESHOLD, suffix=OUTPUT_SUFFIX)
    df.loc[valid_rows, OUTPUT_COLUMN] = [str(path) for path in saved_paths]
    df.to_csv(CSV_PATH, index=False)
    print(f"Saved {len(saved_paths)} connectivity maps and updated {CSV_PATH}")


if __name__ == "__main__": main()
