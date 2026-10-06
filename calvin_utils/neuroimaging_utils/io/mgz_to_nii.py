"""MGZ-to-NIfTI conversion."""

from glob import glob
import os
import nibabel as nib

def mgz_to_nii(glob_path: str) -> None:
    for path in glob(glob_path):
        if not path.lower().endswith('.mgz'):
            continue
        try:
            img = nib.load(path)
            out_path = os.path.splitext(path)[0] + '.nii.gz'
            nib.save(img, out_path)
            print(f'Converted: {path} -> {out_path}')
        except Exception as e:
            print(f'Error converting {path}: {e}')
