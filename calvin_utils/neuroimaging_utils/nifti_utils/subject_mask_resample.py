import os
import glob
import re
from typing import Dict, List

import numpy as np
import pandas as pd
import nibabel as nib

from calvin_utils.neuroimaging_utils.ccm_utils.bounding_box import NiftiBoundingBox
from calvin_utils.neuroimaging_utils.nifti_utils.image_resample import ImageDownSampler


class SubjectMaskResampler:
    """
    Discover per-subject files via glob, build a shared bounding-box mask,
    resample all subject files to that mask, and save a combined file per subject.
    """

    def __init__(
        self,
        glob_pattern: str,
        output_dir: str,
        *,
        subject_extractor: str | None = None,
        combined_suffix: str = "_combined.nii.gz",
        resampled_subdir: str = "resampled",
        keep_resampled: bool = True,
        combine_method: str = "sum",
        binarize: bool = False,
        binarize_threshold: float = 0.0,
    ):
        self.glob_pattern = glob_pattern
        self.output_dir = output_dir
        self.subject_extractor = subject_extractor
        self.combined_suffix = combined_suffix
        self.resampled_subdir = resampled_subdir
        self.keep_resampled = keep_resampled
        self.combine_method = combine_method
        self.binarize = binarize
        self.binarize_threshold = binarize_threshold

        self._subject_files = {}
        self._mask_path = None

    # ---- properties ----
    @property
    def glob_pattern(self):
        return self._glob_pattern

    @glob_pattern.setter
    def glob_pattern(self, value):
        if not isinstance(value, str) or not value:
            raise ValueError("glob_pattern must be a non-empty string")
        self._glob_pattern = value

    @property
    def output_dir(self):
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value):
        if not isinstance(value, str) or not value:
            raise ValueError("output_dir must be a non-empty string")
        os.makedirs(value, exist_ok=True)
        self._output_dir = value

    @property
    def subject_extractor(self):
        return self._subject_extractor

    @subject_extractor.setter
    def subject_extractor(self, value):
        if value is None:
            self._subject_extractor = None
            return
        if not isinstance(value, str) or not value:
            raise ValueError("subject_extractor must be a non-empty string or None")
        self._subject_extractor = value

    @property
    def combined_suffix(self):
        return self._combined_suffix

    @combined_suffix.setter
    def combined_suffix(self, value):
        if not isinstance(value, str) or not value:
            raise ValueError("combined_suffix must be a non-empty string")
        self._combined_suffix = value

    @property
    def resampled_subdir(self):
        return self._resampled_subdir

    @resampled_subdir.setter
    def resampled_subdir(self, value):
        if not isinstance(value, str) or not value:
            raise ValueError("resampled_subdir must be a non-empty string")
        self._resampled_subdir = value

    @property
    def keep_resampled(self):
        return self._keep_resampled

    @keep_resampled.setter
    def keep_resampled(self, value):
        self._keep_resampled = bool(value)

    @property
    def combine_method(self):
        return self._combine_method

    @combine_method.setter
    def combine_method(self, value):
        if value not in {"sum", "max"}:
            raise ValueError("combine_method must be 'sum' or 'max'")
        self._combine_method = value

    @property
    def binarize(self):
        return self._binarize

    @binarize.setter
    def binarize(self, value):
        self._binarize = bool(value)

    @property
    def binarize_threshold(self):
        return self._binarize_threshold

    @binarize_threshold.setter
    def binarize_threshold(self, value):
        self._binarize_threshold = float(value)

    @property
    def subject_files(self):
        return self._subject_files

    @property
    def mask_path(self):
        return self._mask_path

    # ---- main API ----
    def run(self) -> Dict[str, str]:
        self._discover_subject_files()
        self._create_bounding_mask()
        outputs = self._resample_and_combine_subjects()
        self._save_manifest(outputs)
        return outputs

    # ---- internal methods ----
    def _discover_subject_files(self):
        files = self._glob_all_files()
        if not files:
            raise ValueError("No files found. Check glob_pattern.")
        subject_files = self._group_files_by_subject(files)
        if not subject_files and self.subject_extractor:
            subject_files = self._group_files_by_subject(files, force_default=True)
        if not subject_files:
            example = files[0] if files else ""
            raise ValueError(
                "No subject groups created. Check subject_extractor or paths. "
                f"Example path: {example}"
            )
        self._subject_files = subject_files

    def _create_bounding_mask(self):
        all_files = [f for files in self._subject_files.values() for f in files]
        bbox = NiftiBoundingBox(all_files)
        bbox.gen_mask(self.output_dir)
        self._mask_path = os.path.join(self.output_dir, "mask.nii.gz")

    def _resample_and_combine_subjects(self) -> Dict[str, str]:
        outputs = {}
        resampled_dir = os.path.join(self.output_dir, self.resampled_subdir)
        os.makedirs(resampled_dir, exist_ok=True)
        mask_img = nib.load(self._mask_path)
        target_shape = mask_img.shape

        for subject_id, files in self._subject_files.items():
            combined = np.zeros(target_shape, dtype=np.float32)
            for fpath in files:
                resampled_path = self._resample_to_mask(fpath, resampled_dir, subject_id)
                data = nib.load(resampled_path).get_fdata()
                if data.shape != target_shape:
                    raise ValueError(f"Resampled shape mismatch for {resampled_path}")
                if self.combine_method == "sum":
                    combined += data
                else:
                    combined = np.maximum(combined, data)

            if self.binarize:
                combined = (combined > self.binarize_threshold).astype(np.float32)

            out_path = os.path.join(self.output_dir, f"{subject_id}{self.combined_suffix}")
            out_img = nib.Nifti1Image(combined, mask_img.affine, mask_img.header)
            nib.save(out_img, out_path)
            outputs[subject_id] = out_path

        if not self.keep_resampled:
            self._cleanup_resampled(resampled_dir)

        return outputs

    # ---- helper methods ----
    def _glob_all_files(self) -> List[str]:
        return sorted(glob.glob(self.glob_pattern, recursive=True))

    def _group_files_by_subject(self, files: List[str], force_default: bool = False) -> Dict[str, List[str]]:
        subject_files = {}
        if self.subject_extractor and not force_default:
            pattern = re.compile(self.subject_extractor)
            for fpath in files:
                match = pattern.search(fpath)
                if not match:
                    continue
                subject_id = match.group(1) if match.groups() else match.group(0)
                subject_files.setdefault(subject_id, []).append(fpath)
        else:
            for fpath in files:
                subject_id = os.path.basename(os.path.dirname(fpath))
                subject_files.setdefault(subject_id, []).append(fpath)
        return subject_files

    def _resample_to_mask(self, file_path: str, resampled_dir: str, subject_id: str) -> str:
        base = os.path.basename(file_path)
        target_base = os.path.join(resampled_dir, f"{subject_id}__{base}")
        downsampler = ImageDownSampler(file_path, self._mask_path)
        downsampler.output_path = target_base
        downsampler.resample_img()
        return downsampler.output_path

    def _cleanup_resampled(self, resampled_dir: str):
        for f in glob.glob(os.path.join(resampled_dir, "*.nii*")):
            try:
                os.remove(f)
            except OSError:
                pass

    def _save_manifest(self, outputs: Dict[str, str]):
        rows = []
        for subject_id, files in self._subject_files.items():
            rows.append({
                "subject": subject_id,
                "n_files": len(files),
                "files": ";".join(files),
                "combined_path": outputs.get(subject_id, ""),
            })
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(self.output_dir, "subjects_manifest.csv"), index=False)
