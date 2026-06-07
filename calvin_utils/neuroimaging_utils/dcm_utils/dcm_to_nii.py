from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


def run_dcm2nii_over_bids(
    dcm_dir: str | Path,
    out_dir: str | Path,
    dcm2nii_cmd: str = "dcm2niix",
    output_name: str = "T1.nii.gz",
    overwrite: bool = False,
) -> Path | None:
    """
    Convert one DICOM directory to one gzipped NIfTI in an output directory.

    This is intentionally small so scripts can handle their own directory layout.
    The DICOM source directory is left unchanged.
    """
    dcm_dir = Path(dcm_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_path = out_dir / output_name

    if out_path.exists() and not overwrite:
        print(f"Skipping existing output: {out_path}")
        return out_path

    if not dcm_dir.is_dir():
        print(f"Missing DICOM directory: {dcm_dir}")
        return None

    real_dicoms = [
        path
        for path in dcm_dir.glob("*.dcm")
        if path.is_file() and not path.name.startswith("._")
    ]
    if not real_dicoms:
        print(f"No DICOM files found in: {dcm_dir}")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="dcm2nii_") as tmp:
        tmp_dir = Path(tmp)
        subprocess.run(
            [dcm2nii_cmd, "-b", "y", "-z", "y", "-o", str(tmp_dir), str(dcm_dir)],
            check=True,
        )

        nifti_files = sorted(tmp_dir.glob("*.nii.gz")) + sorted(tmp_dir.glob("*.nii"))
        if not nifti_files:
            print(f"dcm2nii produced no NIfTI files for: {dcm_dir}")
            return None

        if len(nifti_files) > 1:
            print(f"Multiple NIfTI files produced for {dcm_dir}; using {nifti_files[0].name}")

        if out_path.exists() and overwrite:
            out_path.unlink()
        shutil.move(str(nifti_files[0]), str(out_path))

        for sidecar in tmp_dir.glob("*.json"):
            sidecar_out = out_path.with_suffix("").with_suffix(".json")
            if sidecar_out.exists() and overwrite:
                sidecar_out.unlink()
            if not sidecar_out.exists():
                shutil.move(str(sidecar), str(sidecar_out))
            break

    return out_path
