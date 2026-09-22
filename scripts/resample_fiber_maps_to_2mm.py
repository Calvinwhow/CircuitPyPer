#!/usr/bin/env python3

from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_from_to
from glob import glob

# =============================================================================
# CONFIG
# =============================================================================
REFERENCE = Path(
    "/Users/cu135/Software_Local/calvin_utils_project/circuit_pyper/resources/MNI152_T1_2mm_brain_mask.nii"
)

PATTERNS = Path("/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimization/symptom_on_lhs/fiber_regressions-HigherIsBetter/*/regression/contrast_tval*.nii.gz")


def make_2mm(source, reference):
    if source.name.endswith(".nii.gz"):
        out_path = source.with_name(source.name[:-7] + "_2mm.nii.gz")
    else:
        out_path = source.with_name(source.stem + "_2mm.nii")

    if out_path.exists():
        return

    img = nib.load(source)
    resampled = resample_from_to(img, reference, order=1)
    nib.save(resampled, out_path)

    print(out_path)


reference = nib.load(REFERENCE)

for source in glob(PATTERNS):
    make_2mm(source, reference)