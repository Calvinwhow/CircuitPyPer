import os
import re
from typing import Iterable, List

import pandas as pd

from calvin_utils.ccm_utils.bounding_box import NiftiBoundingBox


class MergeVTAColumns:
    """
    Merge multiple unilateral VTA columns per subject into a single VTA,
    then create a group mask across all merged VTAs.

    Steps:
    1) For each row/subject, combine the NIfTI paths from vta_cols using NiftiBoundingBox.
    2) Save merged VTA per subject.
    3) Build a group mask from all merged VTAs.
    4) Write merged VTA paths back into the original CSV.
    5) Save group mask alongside the CSV.
    """

    def __init__(
        self,
        csv_path: str,
        *,
        vta_cols: Iterable[str],
        subject_col: str = "subject",
        output_col: str = "merged_vta_path",
        output_dir: str | None = None,
        output_subdir: str = "merged_vtas",
        output_suffix: str = "_merged_vta.nii.gz",
        mask_name: str = "merged_vta_mask.nii.gz",
        overwrite: bool = False,
        verbose: bool = True,
    ):
        self.csv_path = csv_path
        self.vta_cols = self._coerce_cols(vta_cols)
        self.subject_col = subject_col
        self.output_col = output_col
        self.output_dir = output_dir
        self.output_subdir = output_subdir
        self.output_suffix = output_suffix
        self.mask_name = mask_name
        self.overwrite = overwrite
        self.verbose = verbose

        self._df = None
        self._merged_paths = []

    @staticmethod
    def _coerce_cols(vta_cols: Iterable[str]) -> List[str]:
        if isinstance(vta_cols, str):
            vta_cols = [c.strip() for c in vta_cols.split(",") if c.strip()]
        cols = list(vta_cols)
        if not cols:
            raise ValueError("vta_cols must contain at least one column name.")
        return cols

    @property
    def resolved_output_dir(self) -> str:
        if self.output_dir:
            return os.path.abspath(os.path.expanduser(self.output_dir))
        return os.path.dirname(os.path.abspath(self.csv_path))

    def run(self):
        self._load_csv()
        self._validate_columns()
        self._merge_subjects()
        self._write_csv()
        mask_path = self._create_group_mask()
        return self._df, mask_path

    # ---- internal: load/validate ----
    def _load_csv(self):
        self._df = pd.read_csv(self.csv_path)

    def _validate_columns(self):
        required = set(self.vta_cols + [self.subject_col])
        missing = required - set(self._df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    # ---- internal: merging ----
    def _merge_subjects(self):
        out_dir = os.path.join(self.resolved_output_dir, self.output_subdir)
        os.makedirs(out_dir, exist_ok=True)

        merged_paths = []
        for idx, row in self._df.iterrows():
            subject_id = row[self.subject_col]
            paths = self._collect_paths(row)
            if not paths:
                merged_paths.append("")
                continue

            out_path = os.path.join(out_dir, f"{self._safe_subject(subject_id)}{self.output_suffix}")
            if os.path.exists(out_path) and not self.overwrite:
                merged_paths.append(out_path)
                continue

            self._merge_paths(paths, out_path)
            merged_paths.append(out_path)

            if self.verbose:
                print(f"Merged {len(paths)} VTAs for {subject_id} -> {out_path}")

        self._merged_paths = merged_paths
        self._df[self.output_col] = merged_paths

    def _collect_paths(self, row) -> List[str]:
        paths = []
        for col in self.vta_cols:
            val = row.get(col)
            if pd.isna(val):
                continue
            if isinstance(val, str):
                val = val.strip()
                if not val:
                    continue
                if ";" in val:
                    parts = [v.strip() for v in val.split(";") if v.strip()]
                    paths.extend(parts)
                else:
                    paths.append(val)
            elif isinstance(val, (list, tuple)):
                paths.extend([v for v in val if isinstance(v, str) and v.strip()])
        return [os.path.expanduser(p) for p in paths]

    @staticmethod
    def _safe_subject(subject_id) -> str:
        safe = str(subject_id).strip()
        safe = safe.replace(os.path.sep, "_")
        safe = re.sub(r"\s+", "_", safe)
        if not safe:
            safe = "subject"
        return safe

    @staticmethod
    def _merge_paths(paths: List[str], out_path: str):
        bbox = NiftiBoundingBox(paths)
        bbox.generate_bounding_box()
        bbox.add_niftis_to_bounding_box()
        bbox.collapse_bbox_to_3d()
        bbox.save_nifti(bbox._collapsed_data, out_path)

    # ---- internal: output ----
    def _write_csv(self):
        self._df.to_csv(self.csv_path, index=False)

    def _create_group_mask(self):
        paths = [p for p in self._merged_paths if isinstance(p, str) and p]
        if not paths:
            if self.verbose:
                print("No merged VTAs found. Skipping group mask creation.")
            return None

        out_path = os.path.join(self.resolved_output_dir, self.mask_name)
        if os.path.exists(out_path) and not self.overwrite:
            return out_path

        bbox = NiftiBoundingBox(paths)
        bbox.generate_bounding_box()
        bbox.add_niftis_to_bounding_box()
        bbox.collapse_bbox_to_3d()
        mask = bbox.collapsed_bbox_to_mask()
        bbox.save_nifti(mask, out_path)
        if self.verbose:
            print(f"Saved group mask -> {out_path}")
        return out_path
