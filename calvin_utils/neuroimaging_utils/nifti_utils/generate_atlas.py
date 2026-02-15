import re
import pathlib
import warnings
import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img

def split_atlas(atlas_nii: str, labels_txt: str, out_dir: str | pathlib.Path) -> None:
    """
    Parameters
    ----------
    atlas_nii   : str | Path  to integer valued NIfTI, e.g. AAL.nii.gz
    labels_txt  : str | Path  to tabular text/LUT file: <index> <region_name> <other>
    out_dir     : str | Path  to directory where masks will be written
    """
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load once
    atlas_img = nib.load(atlas_nii)
    atlas = atlas_img.get_fdata()
    affine, header = atlas_img.affine, atlas_img.header

    # iterate over label file
    with open(labels_txt) as fp:
        for line in fp:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # split on any whitespace; ignore extra columns
            idx_str, name = re.split(r"\s+", line, maxsplit=1)
            idx = int(idx_str)
            if idx == 0:
                continue  # skip background

            mask = (atlas == idx).astype(np.uint8)
            if not mask.any():
                continue  # atlas lacks this label

            # safe filename
            fname = re.sub(r"[^0-9A-Za-z]+", "_", name) + ".nii.gz"
            nib.save(nib.Nifti1Image(mask, affine, header), out_dir / fname)
            print("✔", fname)


class AtlasAggregator:
    """
    Build a resampled 4D atlas with flexible grouping rules and export ROIs.
    """

    DEFAULT_MASK = (
        pathlib.Path(__file__).resolve().parents[2]
        / "resources"
        / "MNI152_T1_2mm_brain_mask.nii"
    )

    def __init__(
        self,
        labels_txt: str | pathlib.Path,
        atlas_nii: str | pathlib.Path,
        output_dir: str | pathlib.Path,
        mask_path: str | pathlib.Path | None = None,
        index_base: int = 1,
        index_col: int = 0,
        name_col: int = 1,
        has_header: bool = False,
    ) -> None:
        self.labels_txt = pathlib.Path(labels_txt)
        self.atlas_nii = pathlib.Path(atlas_nii)
        self.output_dir = pathlib.Path(output_dir)
        self.mask_path = pathlib.Path(mask_path) if mask_path else self.DEFAULT_MASK
        self.index_base = index_base
        self.index_col = index_col
        self.name_col = name_col
        self.has_header = has_header

    def load_labels(self) -> list[dict]:
        labels = []
        with open(self.labels_txt) as fp:
            for line_idx, line in enumerate(fp):
                if self.has_header and line_idx == 0:
                    continue
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = re.split(r"\s+", line)
                if self.index_col >= len(parts) or self.name_col >= len(parts):
                    continue
                try:
                    idx = int(parts[self.index_col])
                except ValueError:
                    raise ValueError(
                        f"Invalid index value '{parts[self.index_col]}' in {self.labels_txt}"
                    )
                name = parts[self.name_col]
                labels.append({"index": idx, "name": name})
        return labels

    def resample_atlas(self, interpolation: str = "nearest") -> nib.Nifti1Image:
        atlas_img = nib.load(self.atlas_nii)
        mask_img = nib.load(self.mask_path)
        return resample_to_img(atlas_img, mask_img, interpolation=interpolation)

    def ensure_4d_atlas(
        self,
        atlas_img: nib.Nifti1Image,
        labels: list[dict],
        output_dir: str | pathlib.Path | None = None,
    ) -> nib.Nifti1Image:
        data = atlas_img.get_fdata()
        if data.ndim == 4:
            return atlas_img
        if data.ndim != 3:
            raise ValueError("Atlas must be 3D (integer labels) or 4D (parcel volumes).")

        if not labels:
            raise ValueError("3D atlas conversion requires a labels .txt with indices.")
        data_int = np.rint(data)
        if not np.allclose(data, data_int):
            warnings.warn(
                "3D atlas contains non-integer values; rounding to nearest integer.",
                RuntimeWarning,
            )
        data_int = data_int.astype(int)
        label_indices = sorted({label["index"] for label in labels})
        max_idx = max(label_indices)
        out_data = np.zeros(data.shape + (max_idx - self.index_base + 1,), dtype=np.float32)
        for label in labels:
            idx = label["index"]
            vol_idx = idx - self.index_base
            if vol_idx < 0 or vol_idx >= out_data.shape[3]:
                continue
            out_data[:, :, :, vol_idx] = (data_int == idx).astype(np.float32)

        out_img = nib.Nifti1Image(out_data, atlas_img.affine, atlas_img.header)
        output_dir = pathlib.Path(output_dir) if output_dir else self.atlas_nii.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{self._atlas_base_name()}_4d.nii.gz"
        nib.save(out_img, out_path)
        return out_img

    def group_key_from_parenthetical(self, name: str) -> str | None:
        """Use the first parenthetical substring as the group key.

        Choose this when the grouping label is inside parentheses, e.g.
        "Region (Group)" or "Parcel (Hemisphere)" and you want to merge on
        the parenthetical text. Returns None if no parentheses are found.
        """
        match = re.search(r"\(([^)]+)\)", name)
        return match.group(1).strip() if match else None

    def group_key_strip_side(self, name: str) -> str:
        """Strip a trailing left/right word from the end of the name.

        Choose this when laterality appears as a space-separated suffix
        like "Dentate left" or "Dentate right". This does not handle
        underscore-separated or prefixed laterality tokens.
        """
        return re.sub(r"\s*\b(left|right)\b\s*$", "", name, flags=re.IGNORECASE).strip()

    def _is_lateral_token(self, token: str) -> bool:
        return token.lower() in {"l", "r", "left", "right"}

    def group_key_drop_last_1_underscores(self, name: str) -> str:
        """Drop the last underscore token if it is a laterality token.

        Choose this when names are like "Dentate_L" or "Dentate_left".
        If the last token is not laterality, the name is returned unchanged.
        """
        parts = name.split("_")
        if len(parts) <= 1:
            return name
        last = parts[-1]
        if not self._is_lateral_token(last):
            warnings.warn(
                f"Expected laterality token at end of '{name}', leaving unchanged.",
                RuntimeWarning,
            )
            return name
        return "_".join(parts[:-1]).strip()

    def group_key_drop_first_1_underscores(self, name: str) -> str:
        """Drop the first underscore token if it is a laterality token.

        Choose this when names are like "LEFT_dentate" or "R_dentate".
        If the first token is not laterality, the name is returned unchanged.
        """
        parts = name.split("_")
        if len(parts) <= 1:
            return name
        first = parts[0]
        if not self._is_lateral_token(first):
            warnings.warn(
                f"Expected laterality token at start of '{name}', leaving unchanged.",
                RuntimeWarning,
            )
            return name
        return "_".join(parts[1:]).strip()

    def group_key_drop_last_2_underscores(self, name: str) -> str:
        """Drop the last two underscore tokens.

        Choose this when laterality and another suffix are in the last two
        underscore tokens, e.g. "Region_Subregion_L" (drops "Subregion_L").
        """
        parts = name.split("_")
        if len(parts) <= 2:
            return self.group_key_drop_last_1_underscores(name)
        return "_".join(parts[:-2]).strip()

    def group_key_drop_last_3_underscores(self, name: str) -> str:
        """Drop the last three underscore tokens.

        Choose this when the group key is the leading token(s) and the
        last three underscore segments are metadata to discard.
        """
        parts = name.split("_")
        if len(parts) <= 3:
            return self.group_key_drop_last_2_underscores(name)
        return "_".join(parts[:-3]).strip()

    def extract_side(self, name: str) -> str | None:
        match = re.search(r"\b(left|right)\b", name, flags=re.IGNORECASE)
        return match.group(1).lower() if match else None

    def build_group_map(
        self,
        labels: list[dict],
        group_key_fn=None,
        side_fn=None,
        make_bilateral: bool = True,
        name_fn=None,
    ) -> list[dict]:
        group_map = {}
        side_fn = side_fn or self.extract_side
        for label in labels:
            name = label["name"]
            group_key = group_key_fn(name) if group_key_fn else name
            if not group_key:
                group_key = name
            side = side_fn(name) if side_fn else None

            if make_bilateral:
                group_id = group_key
            else:
                group_id = f"{group_key}__{side}" if side else group_key

            if name_fn:
                group_name = name_fn(name, group_key, side, make_bilateral)
            else:
                if make_bilateral:
                    group_name = group_key
                else:
                    group_name = f"{group_key} {side}" if side else group_key

            if group_id not in group_map:
                group_map[group_id] = {"name": group_name, "indices": []}
            group_map[group_id]["indices"].append(label["index"])

        return list(group_map.values())

    def combine_parcels(
        self,
        atlas_img: nib.Nifti1Image,
        group_map: list[dict],
        binarize: bool = False,
    ) -> nib.Nifti1Image:
        data = atlas_img.get_fdata()
        if data.ndim != 4:
            raise ValueError("Atlas must be 4D with parcels in the 4th dimension.")

        n_vols = data.shape[3]
        combined = np.zeros(data.shape[:3] + (len(group_map),), dtype=np.float32)

        for out_idx, group in enumerate(group_map):
            group_data = np.zeros(data.shape[:3], dtype=np.float32)
            for idx in group["indices"]:
                vol_idx = idx - self.index_base
                if vol_idx < 0 or vol_idx >= n_vols:
                    continue
                group_data += data[:, :, :, vol_idx]
            if binarize:
                group_data = (group_data > 0).astype(np.uint8)
            combined[:, :, :, out_idx] = group_data

        return nib.Nifti1Image(combined, atlas_img.affine, atlas_img.header)

    def save_outputs(
        self,
        atlas_img: nib.Nifti1Image,
        group_map: list[dict],
        output_dir: str | pathlib.Path | None = None,
        atlas_name: str | None = None,
        save_rois: bool = True,
        save_coverage: bool = False,
    ) -> tuple[pathlib.Path, pathlib.Path]:
        output_dir = pathlib.Path(output_dir) if output_dir else self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = atlas_name or self._atlas_base_name()
        atlas_path = output_dir / f"{base_name}_resampled_grouped.nii.gz"
        labels_path = output_dir / f"{base_name}_resampled_grouped.txt"

        nib.save(atlas_img, atlas_path)

        with open(labels_path, "w") as fp:
            for i, group in enumerate(group_map, start=1):
                fp.write(f"{i} {group['name']}\n")

        if save_rois:
            roi_dir = output_dir / f"{base_name}_rois"
            roi_dir.mkdir(parents=True, exist_ok=True)
            data = atlas_img.get_fdata()
            for i, group in enumerate(group_map):
                roi = data[:, :, :, i]
                safe_name = self._safe_name(group["name"])
                roi_path = roi_dir / f"{safe_name}.nii.gz"
                nib.save(nib.Nifti1Image(roi, atlas_img.affine, atlas_img.header), roi_path)

        if save_coverage:
            self.save_coverage(atlas_img, output_dir=output_dir, atlas_name=base_name)

        return atlas_path, labels_path

    def run(
        self,
        group_key_fn=None,
        make_bilateral: bool = True,
        binarize: bool = False,
        resample: bool = True,
        save_rois: bool = True,
        save_coverage: bool = False,
    ) -> tuple[pathlib.Path, pathlib.Path]:
        labels = self.load_labels()
        atlas_img = self.resample_atlas() if resample else nib.load(self.atlas_nii)
        atlas_img = self.ensure_4d_atlas(atlas_img, labels)
        group_map = self.build_group_map(
            labels=labels,
            group_key_fn=group_key_fn,
            make_bilateral=make_bilateral,
        )
        new_img = self.combine_parcels(atlas_img, group_map, binarize=binarize)
        return self.save_outputs(
            new_img,
            group_map,
            save_rois=save_rois,
            save_coverage=save_coverage,
        )

    def save_coverage(
        self,
        atlas_img: nib.Nifti1Image,
        output_dir: str | pathlib.Path | None = None,
        atlas_name: str | None = None,
    ) -> pathlib.Path:
        output_dir = pathlib.Path(output_dir) if output_dir else self.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        base_name = atlas_name or self._atlas_base_name()
        coverage_path = output_dir / f"{base_name}_coverage.nii.gz"
        data = atlas_img.get_fdata()
        coverage = np.sum(data, axis=3)
        coverage = (coverage > 0).astype(np.uint8)
        nib.save(nib.Nifti1Image(coverage, atlas_img.affine, atlas_img.header), coverage_path)
        return coverage_path

    def _atlas_base_name(self) -> str:
        name = self.atlas_nii.name
        if name.endswith(".nii.gz"):
            return name[:-7]
        return pathlib.Path(name).stem

    def _safe_name(self, name: str) -> str:
        return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_") or "roi"


if __name__ == "__main__":
    split_atlas(
        atlas_nii="AAL.nii.gz",      # path to the combined atlas NIfTI file (integer valued)
        labels_txt="AAL.txt",        # path to a .txt file with labels, e.g. AAL.txt
        out_dir="aal_masks"          # path to output dir. will be created if absent
    )
