#!/usr/bin/env python3
"""Skull-strip each patient's scan and register it to MNI, at its own resolution, for viewing.

The public entry to calvin_utils.neuroimaging_utils.registration_utils: runs its orchestrator, run_registration.RunRegistration. Per scan (SkullStrip):
    1. Register the scan to the MNI template by mutual information: rigid first (RigidRegistration), then full affine from there (AffineRegistration).
    2. Bring the MNI brain mask back onto the scan's own voxel grid through the full affine fit (BrainMaskTransfer).
    3. Keep the scan's voxels inside the mask, zero outside, and put the registration chosen by AFFINE in front of the scan's affine:
       AFFINE = False: the rigid simplification of the fit, rotation and translation only (RigidFromAffine) -- the brain keeps its own size and shape and faces the MNI axes.
       AFFINE = True:  the full affine fit -- the brain is stretched to MNI's size and shape, so MNI tracts and maps sit closer on it.
    The voxels are never resampled in either frame: the brain keeps its full quality.
    MNI tracts and maps still sit close but not exactly on the anatomy; that needs a nonlinear warp, which this does not do.

Input:
    INPUT_CSV, one row per scan, with NIFTI_COLUMN holding each scan's NIfTI path (a T1 works best; any contrast registers by mutual information).

Outputs, written beside each scan, BIDS-style (for sub-01_T1w.nii.gz with AFFINE = False; MNIaffine in place of MNIrigid when True), and their paths written back into INPUT_CSV under these column names:
    brain_path         sub-01_space-MNIrigid_desc-brain_T1w.nii.gz    the skull-stripped scan in the registration's frame: what the viewer receives
    brain_mask_path    sub-01_desc-brain_mask.nii.gz                  the brain mask on the scan's own grid, in the scan's own space
    transform_path     sub-01_from-native_to-MNIrigid_xfm.txt         4x4 registration returned, scan world mm -> MNI frame world mm

Run:
    Set the CONFIG below and run the script, or pass the same settings on the command line; an argument left out uses its CONFIG value:
    python scripts/registration_pipeline.py --input_csv /data/patients.csv --nifti_column t1_path --affine true

Notes:
    About 20 s per 1 mm T1. The packaged mask is the MNI 2 mm brain mask with its enclosed holes (the ventricles) filled, so the stripped brain is solid.
"""

import sys
import argparse
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from calvin_utils.neuroimaging_utils.registration_utils.run_registration import RunRegistration


# =============================================================================
# CONFIG
# =============================================================================

INPUT_CSV = ""                 # Required. Path to a CSV with one row per scan; the output paths are written back into it.
NIFTI_COLUMN = "nifti_path"    # Name of the CSV column holding each scan's NIfTI path (.nii or .nii.gz), e.g. a patient's T1.
TEMPLATE_PATH = None           # Path to the MNI template NIfTI the scans are registered to; None uses the packaged MNI152_T1_2mm_brain.nii.
TEMPLATE_MASK_PATH = None      # Path to that template's brain mask NIfTI, on the template's grid; None uses the packaged MNI152_T1_2mm_brain_mask_filled.nii.
AFFINE = False                 # Registration the brain is returned in: False keeps its own size and shape (rigid); True stretches it to MNI (full affine).
N_SAMPLES = 50000              # Number of MNI brain points the mutual information is scored at (at least 1000); more is steadier and slower.
BINS = 32                      # Number of intensity histogram bins per image for the mutual information.
SEED = 0                       # Seed for drawing the sample points; the same seed gives the same result.


def boolean(text):
    """A command-line true/false."""
    if text.lower() in ("true", "1", "yes"):
        return True
    if text.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {text!r}")


def parse_arguments():
    """Command-line settings; each one left out is None, and main falls back to its CONFIG value."""
    parser = argparse.ArgumentParser(description="Skull-strip each scan in a CSV and register it to MNI at its own resolution; outputs are written beside each scan and their paths back into the CSV.")
    parser.add_argument("--input_csv", default=None, help="str, a file path: CSV with one row per scan; the three output paths are written back into it as columns. Default: INPUT_CSV.")
    parser.add_argument("--nifti_column", default=None, help="str, a column name: the CSV column holding each scan's NIfTI path (.nii or .nii.gz), one 3D or 4D scan per row. Default: NIFTI_COLUMN.")
    parser.add_argument("--template_path", default=None, help="str, a file path: the 3D MNI template NIfTI the scans are registered to. Default: TEMPLATE_PATH, or the packaged MNI152_T1_2mm_brain.nii when that is None.")
    parser.add_argument("--template_mask_path", default=None, help="str, a file path: the 3D brain mask NIfTI of that template, on the template's voxel grid, 1 inside the brain. Default: TEMPLATE_MASK_PATH, or the packaged MNI152_T1_2mm_brain_mask_filled.nii when that is None.")
    parser.add_argument("--affine", type=boolean, default=None, help="bool, true or false: the registration each brain is returned in -- false keeps its own size and shape (rigid), true stretches it to MNI (full affine). Default: AFFINE.")
    parser.add_argument("--n_samples", type=int, default=None, help="int, at least 1000: number of MNI brain points the mutual information is scored at; more is steadier and slower. Default: N_SAMPLES.")
    parser.add_argument("--bins", type=int, default=None, help="int, at least 2: number of intensity histogram bins per image for the mutual information. Default: BINS.")
    parser.add_argument("--seed", type=int, default=None, help="int: seed for drawing the sample points; the same seed gives the same result. Default: SEED.")
    return parser.parse_args()


def setting(argument, config_value):
    """The command-line value when given, otherwise the CONFIG value."""
    return config_value if argument is None else argument


def main():
    """Read the settings, then skull-strip and register every scan in the CSV, save beside each, and write the output paths back."""
    args = parse_arguments()
    input_csv = setting(args.input_csv, INPUT_CSV)
    if not input_csv:
        raise ValueError("Set INPUT_CSV, or pass --input_csv.")
    output_paths = RunRegistration(input_csv, setting(args.nifti_column, NIFTI_COLUMN), setting(args.template_path, TEMPLATE_PATH), setting(args.template_mask_path, TEMPLATE_MASK_PATH), affine=setting(args.affine, AFFINE), n_samples=setting(args.n_samples, N_SAMPLES), bins=setting(args.bins, BINS), seed=setting(args.seed, SEED)).run()
    print(f"registered {len(output_paths)} scan(s); output paths are in {input_csv}")


if __name__ == "__main__":
    main()
