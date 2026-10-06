"""Combine bilateral Lead-DBS stimulation volumes without import-time work."""

from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

from calvin_utils.neuroimaging_utils.io.importers import GiiNiiFileImport
from calvin_utils.neuroimaging_utils.nifti_utils.generate_nifti import (
    view_and_save_nifti,
)


DEFAULT_RIGHT_PATTERN = (
    "stimulations/MNI152NLin2009bAsym/gs_2023Aysu/"
    "*sim-binary_model-simbio_hemi-R.nii"
)
DEFAULT_LEFT_PATTERN = (
    "stimulations/MNI152NLin2009bAsym/gs_2023Aysu/"
    "*sim-binary_model-simbio_hemi-L.nii"
)


def combine_bilateral_niftis(
    root_dir: str | Path,
    *,
    right_pattern: str = DEFAULT_RIGHT_PATTERN,
    left_pattern: str = DEFAULT_LEFT_PATTERN,
    output_name: str = "sim-efield_model-simbio_hemi-bl.nii",
) -> list[Path]:
    """Sum matching left/right images for each subject directory."""
    root = Path(root_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Lead-DBS root directory does not exist: {root}")

    written = []
    for subject_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        right = glob(str(subject_dir / right_pattern))
        left = glob(str(subject_dir / left_pattern))
        if not right or not left:
            continue

        input_dir = Path(right[0]).parent
        importer = GiiNiiFileImport(
            str(input_dir), file_pattern="*sim-binary_model-simbio_hemi-[LR].nii"
        )
        imports = importer.run()
        if imports.shape[1] != 2:
            raise ValueError(
                f"Expected two hemisphere images for {subject_dir.name}, "
                f"found {imports.shape[1]}"
            )
        output_path = input_dir / f"{subject_dir.name}_{output_name}"
        view_and_save_nifti(
            imports.iloc[:, 0] + imports.iloc[:, 1],
            out_dir=str(input_dir),
            output_file=output_path.name,
        )
        written.append(output_path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root_dir", help="Lead-DBS derivatives directory")
    parser.add_argument("--right-pattern", default=DEFAULT_RIGHT_PATTERN)
    parser.add_argument("--left-pattern", default=DEFAULT_LEFT_PATTERN)
    parser.add_argument("--output-name", default="sim-efield_model-simbio_hemi-bl.nii")
    args = parser.parse_args(argv)
    outputs = combine_bilateral_niftis(
        args.root_dir,
        right_pattern=args.right_pattern,
        left_pattern=args.left_pattern,
        output_name=args.output_name,
    )
    print(f"Wrote {len(outputs)} bilateral image(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
