from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional, Sequence, Union, List

import numpy as np
import nibabel as nib
import pandas as pd
from nilearn import plotting

from calvin_utils.nifti_utils.model_vta import ModelVTA


class SphereROIBuilder:
    """Build spherical ROIs in MNI space (or via *ModelVTA* when a brain mask is unavailable)."""

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        mask_path: Optional[Union[str, Path]],
        out_dir: Union[str, Path],
        radius: float,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.brain_img = nib.load(str(mask_path)) if mask_path else None
        self.radius = float(radius)

        if self.brain_img is not None:
            self._mni_coords_brain = self._voxelwise_mni_coords()
            # treat any positive value as brain (handles float masks)
            self._inside_brain = self.brain_img.get_fdata().ravel() > 0
        else:
            self._mni_coords_brain = None
            self._inside_brain = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def generate_from_single_coord(
        self,
        centre_xyz: Sequence[float] | str,
        subject: str,
        session: str | None = None,
        project_on_brain: bool = True,
        suffix: str = "",
    ) -> str:
        centre = self._normalize_coord_entry(centre_xyz)[0]
        return self._build_and_save(centre, subject, session, project_on_brain, suffix)

    def generate_from_separate_coord_cols(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        z_col: str,
        sub_col: str,
        session_col: str | None = None,
        project_on_brain: bool = True,
    ) -> pd.DataFrame:
        for i, row in df.iterrows():
            centre = np.asarray([row[x_col], row[y_col], row[z_col]], float)
            df.at[i, "roi_path"] = self._build_and_save(
                centre,
                row[sub_col],
                row[session_col] if session_col else None,
                project_on_brain,
            )
        return df

    def generate_from_coordlist_column(
        self,
        df: pd.DataFrame,
        coord_col: str,
        sub_col: str,
        session_col: str | None = None,
        project_on_brain: bool = True,
        save_individual_roi: bool = False,
    ) -> pd.DataFrame:
        """Handle one‑or‑many coordinates per row, automatically using *ModelVTA* when no mask is present."""
        for i, row in df.iterrows():
            coords = self._normalize_coord_entry(row[coord_col])

            # ---------------------------------------------------------- #
            # Case 1 – have a brain mask → can build a composite volume
            # ---------------------------------------------------------- #
            if self.brain_img is not None:
                composite = None
                for idx, c in enumerate(coords, 1):
                    if save_individual_roi:
                        self._build_and_save(
                            c,
                            row[sub_col],
                            row[session_col] if session_col else None,
                            project_on_brain,
                            suffix=f"_{idx:02d}" if len(coords) > 1 else "",
                        )
                    mask = self._build_sphere(c, project_on_brain)
                    composite = mask if composite is None else composite + mask

                df.at[i, "roi_path"] = self._save_built_sphere(
                    composite,
                    row[sub_col],
                    row[session_col] if session_col else None,
                    suffix="_composite",
                )
            # ---------------------------------------------------------- #
            # Case 2 – no brain mask → fall back to ModelVTA, one file
            # per coordinate (optionally skip individual if requested)
            # ---------------------------------------------------------- #
            else:
                paths: List[str] = []
                for idx, c in enumerate(coords, 1):
                    p = self._build_and_save(
                        c,
                        row[sub_col],
                        row[session_col] if session_col else None,
                        project_on_brain=False,  # projection meaningless w/o mask
                        suffix=f"_{idx:02d}" if save_individual_roi and len(coords) > 1 else "",
                    )
                    paths.append(p)

                # Store first path if single, else semi‑colon list
                df.at[i, "roi_path"] = paths[0] if len(paths) == 1 else ";".join(paths)

        return df

    # ------------------------------------------------------------------ #
    # Validation helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_coord_entry(entry) -> np.ndarray:
        """Robustly coerce *entry* to an (n, 3) float array.

        Accepts a single triplet or list‑of‑triplets, either as a Python object
        or as a string representation (e.g. "[34,‑18,52]" or
        "[[12,34,56], [-7,8,9]]").  Gives a clear error if **any** sub‑list is
        missing a value.
        """
        if isinstance(entry, str):
            entry = ast.literal_eval(entry.strip())

        # Allow bare scalars separated by comma/space ("x,y,z")
        if isinstance(entry, (int, float)):
            raise ValueError("Coordinate must have 3 values, got a scalar.")

        # Make nested list structure explicit
        if isinstance(entry, (list, tuple, np.ndarray)) and not any(
            isinstance(x, (list, tuple, np.ndarray)) for x in entry
        ):
            entry = [entry]  # single triplet → wrap

        # Validate each triplet length
        cleaned: list[list[float]] = []
        for triplet in entry:
            if len(triplet) != 3:
                raise ValueError(
                    f"Every coordinate requires 3 elements; got {triplet} (len={len(triplet)})"
                )
            cleaned.append([float(v) for v in triplet])

        return np.asarray(cleaned, dtype=float)

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #
    def _voxelwise_mni_coords(self) -> np.ndarray:
        data = self.brain_img.get_fdata()
        ijk = (
            np.stack(np.meshgrid(*[np.arange(s) for s in data.shape], indexing="ij"), 0)
            .reshape(3, -1)
            .T
        )
        return (self.brain_img.affine @ np.c_[ijk, np.ones(len(ijk))].T).T[:, :3]

    @staticmethod
    def _distances(p: np.ndarray, coords: np.ndarray) -> np.ndarray:
        return np.linalg.norm(coords - p, axis=1)

    def _nearest_brain_voxel(self, xyz: np.ndarray) -> np.ndarray:
        inside = self._mni_coords_brain[self._inside_brain]
        return inside[np.argmin(self._distances(xyz, inside))]

    def _build_sphere_mask(self, centre: np.ndarray) -> np.ndarray:
        d = self._distances(centre, self._mni_coords_brain)
        mask = (d <= self.radius).astype(int).reshape(self.brain_img.shape)
        mask[self.brain_img.get_fdata() <= 0] = 0
        return mask

    def _build_sphere(self, centre: np.ndarray, project_on_brain: bool) -> np.ndarray:
        if self.brain_img is None:
            raise RuntimeError("Cannot build sphere mask without brain_img.")
        if project_on_brain:
            centre = self._nearest_brain_voxel(centre)
        return self._build_sphere_mask(centre)

    # ------------------------------------------------------------------ #
    # I/O helpers
    # ------------------------------------------------------------------ #
    def _prep_bids(self, sub: str, ses: str | None, suffix: str) -> tuple[Path, str]:
        ses_part = f"ses-{ses}" if ses else "ses-01"
        roi_dir = self.out_dir / f"sub-{sub.replace(' ', '_')}" / ses_part / "roi"
        roi_dir.mkdir(parents=True, exist_ok=True)
        fname = (
            f"sub-{sub}_MNI152_T1_{self.radius}mm_sphere_roi{suffix}.nii.gz".replace(
                " ", "_"
            )
        )
        return roi_dir, fname

    def _save_mask(self, mask: np.ndarray, path: Path):
        nib.save(nib.Nifti1Image(mask, self.brain_img.affine, self.brain_img.header), path)

    # ------------------------------------------------------------------ #
    # Core save/build
    # ------------------------------------------------------------------ #
    def _save_built_sphere(
        self,
        sphere: np.ndarray,
        sub: str,
        ses: str | None,
        suffix: str = "",
    ) -> str:
        d, fname = self._prep_bids(sub, ses, suffix)
        self._save_mask(sphere, d / fname)
        return str(d / fname)

    def _build_and_save(
        self,
        centre_xyz: np.ndarray,
        sub: str,
        ses: str | None,
        project_on_brain: bool,
        suffix: str = "",
    ) -> str:
        d, fname = self._prep_bids(sub, ses, suffix)
        if self.brain_img is None:
            ModelVTA(center_coord=centre_xyz, output_path=d).run(
                radius_mm=self.radius, filename=fname
            )
        else:
            mask = self._build_sphere(centre_xyz, project_on_brain)
            self._save_mask(mask, d / fname)
        return str(d / fname)
