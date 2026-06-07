#!/usr/bin/env python3
"""
Plain, edit-the-variables-at-the-top runner for converting LONI-style DICOM
downloads into a simple BIDS-style directory layout.

This is intentionally not a CLI. Change the values in the CONFIG section, then
run this file directly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


# =============================================================================
# CONFIG
# =============================================================================


# The raw folder is the input. It will not be modified.
RAW_ROOT = "/Volumes/HowExp2/datasets/01f_PPMI_Parkinson_Atrophy/BIDS/PPMI"

# New BIDS-style folders will be written here.
BIDS_ROOT = "/Volumes/HowExp2/datasets/01f_PPMI_Parkinson_Atrophy/BIDS"

# Describe where DICOM folders live under RAW_ROOT.
# PPMI example:
#   RAW_ROOT/3000/sag_3D_FSPGR_BRAVO_straight/2011-02-01_08_05_22.0/I224562/*.dcm
#   Relative parts are:
#       0: 3000
#       1: sag_3D_FSPGR_BRAVO_straight
#       2: 2011-02-01_08_05_22.0
#       3: I224562
# Set this to the raw-folder pattern that lands on the directory containing
# DICOM files. Use one "*" per folder level below RAW_ROOT. For LONI folders
# like subject/scan/date/image_id/*.dcm, use "*/*/*/*".
DICOM_DIR_GLOB = "*/*/*/*"

# Usually leave as "*.dcm". Change only if the DICOM files use a different
# extension or casing in a specific download.
DICOM_FILE_GLOB = "*.dcm"

# Set to the relative path part containing the subject ID. For LONI/PPMI above,
# part 0 is "3000", so SUBJECT_PART_INDEX = 0. Do not leave this as None.
SUBJECT_PART_INDEX = 0

# Set this when the raw folder has a meaningful session/visit folder that should
# become the BIDS session label. Leave as None when using SESSION_LABEL_OVERRIDE.
# For LONI/PPMI above, part 2 is a scan timestamp, so we leave this as None and
# force "ses-baseline" below.
SESSION_PART_INDEX = None
SESSION_LABEL_OVERRIDE = "ses-baseline" # Leave as None only when SESSION_PART_INDEX points to a raw folder part

# Output layout:
#   <BIDS_ROOT>/sub-<subject>/<ses-session>/<SESSION_OUTPUT_FOLDER>/<OUTPUT_NIFTI_NAME>
# Usually leave these prefixes alone for BIDS. Change only if adapting this
# script for a non-BIDS folder convention.
SUBJECT_PREFIX = "sub-"
SESSION_PREFIX = "ses-"

# Set this to the BIDS modality folder under each session. Use "anat" for T1/T2
# structural MRI, "func" for fMRI, "dwi" for diffusion, etc.
SESSION_OUTPUT_FOLDER = "anat"

# Set this to the final NIfTI filename to write inside SESSION_OUTPUT_FOLDER.
# For BIDS-valid naming you may prefer something like
# "sub-<id>_ses-<id>_T1w.nii.gz", but this script keeps it simple/editable.
OUTPUT_NIFTI_NAME = "t1.nii.gz"

# Conversion command. Usually "dcm2niix"; older installs may use "dcm2nii".
DCM2NII_CMD = "dcm2niix"

# Leave as None to process every subject found under RAW_ROOT.
# Example: SUBJECTS = ["3000", "3001"]
SUBJECTS = None

# Optional filters for selecting scan folders. These are matched against the
# DICOM directory path relative to RAW_ROOT. Leave empty to accept all.
# Examples: REQUIRED_PATH_PATTERNS = ["T1"], ["MPRAGE", "FSPGR"]
REQUIRED_PATH_PATTERNS = []

# If a subject/session has multiple DICOM folders after filtering, use this one.
# Keep 0 for the first sorted match. Use DRY_RUN=True to inspect the matches.
DICOM_DIR_INDEX = 0

# Keep False to skip subject/session outputs whose NIfTI already exists.
OVERWRITE = False

# Set True to print what would happen without running dcm2nii.
DRY_RUN = False


CIRCUIT_PYPER_DIR = Path(__file__).resolve().parents[1]
if str(CIRCUIT_PYPER_DIR) not in sys.path:
    sys.path.insert(0, str(CIRCUIT_PYPER_DIR))

from calvin_utils.neuroimaging_utils.dcm_utils.dcm_to_nii import run_dcm2nii_over_bids


# =============================================================================
# PATH PARSING
# =============================================================================


def get_relative_parts(path):
    """Return path parts relative to RAW_ROOT."""
    return path.relative_to(Path(RAW_ROOT).expanduser().resolve()).parts


def get_part(path, part_index, label):
    """Get one configured raw path part."""
    parts = get_relative_parts(path)
    if part_index is None:
        return None
    try:
        return parts[part_index]
    except IndexError as exc:
        raise IndexError(
            f"{label}_PART_INDEX={part_index} is outside relative path {parts}"
        ) from exc


def strip_prefix(value, prefix):
    """Remove a BIDS prefix if the raw value already has one."""
    value = str(value)
    if value.startswith(prefix):
        return value[len(prefix):]
    return value


def make_bids_label(value, prefix):
    """Add a BIDS prefix unless it is already present."""
    value = str(value)
    if value.startswith(prefix):
        return value
    return f"{prefix}{value}"


def get_subject_id(dicom_dir):
    """Extract the subject ID from one DICOM directory path."""
    subject_id = get_part(dicom_dir, SUBJECT_PART_INDEX, "SUBJECT")
    return strip_prefix(subject_id, SUBJECT_PREFIX)


def get_session_label(dicom_dir):
    """Extract or override the session label for one DICOM directory path."""
    if SESSION_LABEL_OVERRIDE:
        return make_bids_label(SESSION_LABEL_OVERRIDE, SESSION_PREFIX)

    session_id = get_part(dicom_dir, SESSION_PART_INDEX, "SESSION")
    if session_id is None:
        raise ValueError("SESSION_PART_INDEX must be set when SESSION_LABEL_OVERRIDE is None")
    return make_bids_label(strip_prefix(session_id, SESSION_PREFIX), SESSION_PREFIX)


def get_dicom_group(dicom_dir):
    """Return the configured DICOM group identifier for logging and grouping."""
    return str(dicom_dir)


def get_output_dir(subject_id, session_label):
    """Build the output directory under one BIDS session."""
    return (
        Path(BIDS_ROOT)
        / make_bids_label(subject_id, SUBJECT_PREFIX)
        / session_label
        / SESSION_OUTPUT_FOLDER
    )


# =============================================================================
# FIND FILES
# =============================================================================


def subject_is_selected(subject_id):
    """Return True if this subject should be converted."""
    if SUBJECTS is None:
        return True
    return str(subject_id) in {str(subject) for subject in SUBJECTS}


def path_is_allowed(dicom_dir):
    """Filter DICOM directories by requested path patterns."""
    if not REQUIRED_PATH_PATTERNS:
        return True
    relative_path = str(dicom_dir.relative_to(Path(RAW_ROOT).expanduser().resolve()))
    return any(pattern in relative_path for pattern in REQUIRED_PATH_PATTERNS)


def has_real_dicoms(dicom_dir):
    """Ignore macOS AppleDouble files that can also end in .dcm."""
    return any(
        path.is_file() and not path.name.startswith("._")
        for path in dicom_dir.glob(DICOM_FILE_GLOB)
    )


def find_dicom_dirs():
    """Find all candidate DICOM directories under RAW_ROOT."""
    raw_root = Path(RAW_ROOT).expanduser().resolve()
    dicom_dirs = []
    for dicom_dir in sorted(raw_root.glob(DICOM_DIR_GLOB)):
        if not dicom_dir.is_dir():
            continue
        if not path_is_allowed(dicom_dir):
            continue
        if not has_real_dicoms(dicom_dir):
            continue

        subject_id = get_subject_id(dicom_dir)
        if not subject_is_selected(subject_id):
            continue

        dicom_dirs.append(dicom_dir)
    return dicom_dirs


def group_dicom_dirs(dicom_dirs):
    """Group candidate DICOM folders by BIDS subject/session output."""
    grouped = {}
    for dicom_dir in dicom_dirs:
        subject_id = get_subject_id(dicom_dir)
        session_label = get_session_label(dicom_dir)
        key = (subject_id, session_label)
        grouped.setdefault(key, []).append(dicom_dir)
    return grouped


# =============================================================================
# RUN CONVERSION
# =============================================================================


def convert_group(subject_id, session_label, dicom_dirs):
    """Convert one selected DICOM directory to one BIDS NIfTI."""
    output_dir = get_output_dir(subject_id, session_label)
    output_path = output_dir / OUTPUT_NIFTI_NAME

    if DICOM_DIR_INDEX >= len(dicom_dirs):
        print(
            f"{subject_id}/{session_label} has {len(dicom_dirs)} DICOM dirs, "
            f"but DICOM_DIR_INDEX={DICOM_DIR_INDEX}"
        )
        return False

    dicom_dir = dicom_dirs[DICOM_DIR_INDEX]

    if output_path.exists() and not OVERWRITE:
        print(f"Skipping {subject_id}/{session_label}; output exists: {output_path}")
        return True

    if len(dicom_dirs) > 1:
        print(f"{subject_id}/{session_label} has multiple DICOM dirs; using: {dicom_dir}")

    print(f"Subject/session: {subject_id}/{session_label}")
    print(f"  DICOM group: {get_dicom_group(dicom_dir)}")
    print(f"  DICOM dir:   {dicom_dir}")
    print(f"  NIfTI:       {output_path}")

    if DRY_RUN:
        return True

    output_dir.mkdir(parents=True, exist_ok=True)
    converted_path = run_dcm2nii_over_bids(
        dcm_dir=dicom_dir,
        out_dir=output_dir,
        dcm2nii_cmd=DCM2NII_CMD,
        output_name=OUTPUT_NIFTI_NAME,
        overwrite=OVERWRITE,
    )
    return converted_path is not None


def run_loni_dcm_to_bids_conversion():
    """Orchestrates conversion over every selected raw DICOM folder."""
    raw_root = Path(RAW_ROOT).expanduser().resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(f"RAW_ROOT does not exist: {raw_root}")

    dicom_dirs = find_dicom_dirs()
    grouped = group_dicom_dirs(dicom_dirs)

    if not grouped:
        print("No DICOM folders selected.")
        return

    print(f"Raw root: {raw_root}")
    print(f"BIDS output root: {Path(BIDS_ROOT).expanduser().resolve()}")
    print(f"DICOM folders found: {len(dicom_dirs)}")
    print(f"Subject/session outputs: {len(grouped)}")
    print(f"DRY_RUN={DRY_RUN}, OVERWRITE={OVERWRITE}")

    n_success = 0
    n_failed = 0
    for (subject_id, session_label), group_dirs in sorted(grouped.items()):
        try:
            if convert_group(subject_id, session_label, group_dirs):
                n_success += 1
            else:
                n_failed += 1
        except Exception as exc:
            n_failed += 1
            print(f"Error converting {subject_id}/{session_label}: {exc}")

    print("Done.")
    print(f"Successful/skipped: {n_success}")
    print(f"Failed/missing: {n_failed}")


if __name__ == "__main__":
    os.makedirs(BIDS_ROOT, exist_ok=True)
    run_loni_dcm_to_bids_conversion()
