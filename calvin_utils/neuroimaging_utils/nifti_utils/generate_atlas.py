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
        self.labels_txt = self._coerce_path(labels_txt, "labels_txt")
        self.atlas_nii = self._coerce_path(atlas_nii, "atlas_nii")
        self.output_dir = self._coerce_path(output_dir, "output_dir")
        self.mask_path = self._coerce_path(mask_path, "mask_path") if mask_path else self.DEFAULT_MASK
        self.index_base = index_base
        self.index_col = index_col
        self.name_col = name_col
        self.has_header = has_header

    @staticmethod
    def _coerce_path(value: str | pathlib.Path, parameter_name: str) -> pathlib.Path:
        try:
            return pathlib.Path(value)
        except TypeError as exc:
            raise TypeError(
                f"{parameter_name} must be a path-like value, got "
                f"{type(value).__name__}. If you are rerunning a notebook cell, "
                "make sure the atlas NIfTI path variable was not overwritten by "
                "an AtlasAggregator instance."
            ) from exc

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


class CustomYabplotAtlasBuilder:
    """
    Build yabplot-compatible custom cortical and subcortical atlases from named
    NIfTI parcel masks.

    This is intended for directories that already contain one binary NIfTI per
    parcel, with meaningful filenames. Region names are derived from filenames
    after removing ``.nii`` / ``.nii.gz`` and trailing underscores.

    Outputs
    -------
    ``out_dir/source_volumes``
        Combined labeled NIfTI volumes and a Workbench-style label text file.
    ``out_dir/cortical``
        A custom cortical atlas for ``yabplot.plot_cortical``. Contains one
        fsLR32k vertex-label CSV and one LUT text file.
    ``out_dir/subcortical``
        A custom subcortical atlas for ``yabplot.plot_subcortical``. Contains
        one smoothed ``.vtk`` mesh per selected subcortical parcel plus
        ``atlas_LUT.txt``.

    Notes
    -----
    The cortical atlas path uses nearest-neighbor sampling from the combined
    labeled NIfTI volume to yabplot's fsLR32k surface vertices. It does not use
    Connectome Workbench ribbon-constrained projection. If Workbench is
    available and you need a stricter cortical projection, use
    ``yabplot.build_cortical_atlas`` with the generated
    ``source_volumes/*_wb_labels.txt``.
    """

    DEFAULT_SUBCORTICAL_KEYWORDS = (
        "Hippocampus",
        "Amygdala",
        "Caudate",
        "Putamen",
        "Pallidum",
        "Thalamus",
        "Thal",
        "Cerebelum",
        "Cerebellum",
        "Vermis",
        "N_Acc",
        "VTA",
        "SN",
        "Red_N",
        "LC",
        "Raphe",
    )

    def __init__(
        self,
        parcel_dir: str | pathlib.Path,
        out_dir: str | pathlib.Path,
        atlas_name: str = "custom",
        subcortical_keywords: tuple[str, ...] | list[str] | None = None,
        smooth_i: int = 20,
        smooth_f: float = 0.7,
        bmesh: str = "midthickness",
    ) -> None:
        self.parcel_dir = pathlib.Path(parcel_dir).expanduser()
        self.out_dir = pathlib.Path(out_dir).expanduser()
        self.atlas_name = atlas_name
        self.subcortical_keywords = tuple(
            subcortical_keywords or self.DEFAULT_SUBCORTICAL_KEYWORDS
        )
        self.smooth_i = smooth_i
        self.smooth_f = smooth_f
        self.bmesh = bmesh

        self.source_dir = self.out_dir / "source_volumes"
        self.cortical_dir = self.out_dir / "cortical"
        self.subcortical_dir = self.out_dir / "subcortical"
        self.labels = {}
        self.cortical_labels = {}
        self.subcortical_labels = {}

    def run(self) -> dict:
        parcel_files = self._find_parcel_files()
        self._make_dirs()
        volumes = self._build_labeled_volumes(parcel_files)
        self._build_cortical_atlas(volumes["cortical"])
        self._build_subcortical_atlas(parcel_files)
        return {
            "out_dir": self.out_dir,
            "source_dir": self.source_dir,
            "cortical_dir": self.cortical_dir,
            "subcortical_dir": self.subcortical_dir,
            "n_parcels": len(parcel_files),
            "n_cortical": len(self.cortical_labels),
            "n_subcortical": len(self.subcortical_labels),
        }

    def _make_dirs(self) -> None:
        for directory in (self.source_dir, self.cortical_dir, self.subcortical_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def _find_parcel_files(self) -> list[pathlib.Path]:
        files = sorted(
            path
            for path in self.parcel_dir.iterdir()
            if path.name.endswith((".nii", ".nii.gz")) and not path.name.startswith("._")
        )
        if not files:
            raise FileNotFoundError(f"No NIfTI parcel files found in {self.parcel_dir}")

        by_name = {}
        for path in files:
            by_name[self._clean_region_name(path)] = path
        return [by_name[name] for name in sorted(by_name)]

    def _build_labeled_volumes(self, parcel_files: list[pathlib.Path]) -> dict[str, pathlib.Path]:
        first = nib.load(str(parcel_files[0]))
        shape = first.shape
        affine = first.affine
        header = first.header.copy()

        full = np.zeros(shape, dtype=np.int16)
        cortical = np.zeros(shape, dtype=np.int16)
        subcortical = np.zeros(shape, dtype=np.int16)
        overlap_counts = []

        for rid, path in enumerate(parcel_files, start=1):
            img = nib.load(str(path))
            if img.shape != shape or not np.allclose(img.affine, affine):
                raise ValueError(f"Parcel is not in the same image space: {path}")

            name = self._clean_region_name(path)
            mask = img.get_fdata() > 0
            overlap_counts.append(int(np.count_nonzero((full > 0) & mask)))

            self.labels[rid] = name
            full[mask] = rid
            if self._is_subcortical(name):
                self.subcortical_labels[rid] = name
                subcortical[mask] = rid
            else:
                self.cortical_labels[rid] = name
                cortical[mask] = rid

        paths = {
            "full": self.source_dir / f"{self.atlas_name}_all_labels.nii.gz",
            "cortical": self.source_dir / f"{self.atlas_name}_cortical_labels.nii.gz",
            "subcortical": self.source_dir / f"{self.atlas_name}_subcortical_labels.nii.gz",
        }
        nib.save(nib.Nifti1Image(full, affine, header), paths["full"])
        nib.save(nib.Nifti1Image(cortical, affine, header), paths["cortical"])
        nib.save(nib.Nifti1Image(subcortical, affine, header), paths["subcortical"])
        self._write_workbench_label_file()

        total_overlap = sum(overlap_counts)
        if total_overlap:
            print(f"[warning] parcel masks overlap in {total_overlap} voxel assignments.")
            print("[warning] combined label volumes keep the later parcel label at overlaps.")

        return paths

    def _build_cortical_atlas(self, cortical_volume: pathlib.Path) -> None:
        import yabplot as yab

        lh_data, rh_data = yab.project_vol2surf(
            str(cortical_volume),
            bmesh=self.bmesh,
            mask_medial_wall=True,
            interpolation="nearest",
        )
        labels = np.nan_to_num(
            np.concatenate([lh_data, rh_data]),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(int)
        survivors = set(np.unique(labels)) - {0}

        csv_path = self.cortical_dir / f"{self.atlas_name}_conte69.csv"
        lut_path = self.cortical_dir / f"{self.atlas_name}_LUT.txt"
        np.savetxt(csv_path, labels, fmt="%i")

        with open(lut_path, "w") as file:
            for rid, name in self.cortical_labels.items():
                if rid not in survivors:
                    print(f"[warning] cortical parcel lost on surface: {name} ({rid})")
                    continue
                r, g, b = self._rgb_for_id(rid)
                file.write(f"{rid}  {name}  {r}  {g}  {b}  0\n")

    def _build_subcortical_atlas(self, parcel_files: list[pathlib.Path]) -> None:
        import pyvista as pv
        from skimage import measure

        for old_mesh in self.subcortical_dir.glob("*.vtk"):
            old_mesh.unlink()

        written = []
        rid_out = 1
        for path in parcel_files:
            name = self._clean_region_name(path)
            if not self._is_subcortical(name):
                continue

            img = nib.load(str(path))
            mask = (img.get_fdata() > 0).astype(np.uint8)
            if mask.sum() == 0:
                print(f"[warning] empty subcortical parcel skipped: {name}")
                continue

            try:
                verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
            except ValueError as exc:
                print(f"[warning] subcortical parcel skipped: {name}: {exc}")
                continue

            verts_mni = nib.affines.apply_affine(img.affine, verts)
            faces_pv = np.column_stack((np.full(len(faces), 3), faces)).astype(
                np.int64
            ).ravel()
            mesh = pv.PolyData(verts_mni, faces_pv)
            mesh = mesh.smooth(n_iter=self.smooth_i, relaxation_factor=self.smooth_f)
            mesh.compute_normals(inplace=True)

            if mesh.n_points < 4 or abs(mesh.volume) < 0.01:
                print(f"[warning] tiny subcortical mesh skipped: {name}")
                continue

            mesh.save(self.subcortical_dir / f"{name}.vtk")
            written.append((rid_out, name))
            rid_out += 1

        with open(self.subcortical_dir / "atlas_LUT.txt", "w") as file:
            for rid, name in written:
                file.write(f"{rid} {name}\n")

    def _write_workbench_label_file(self) -> None:
        path = self.source_dir / f"{self.atlas_name}_wb_labels.txt"
        with open(path, "w") as file:
            for rid, name in self.labels.items():
                r, g, b = self._rgb_for_id(rid)
                file.write(f"{name}\n{rid} {r} {g} {b} 255\n")

    def _is_subcortical(self, name: str) -> bool:
        return any(keyword in name for keyword in self.subcortical_keywords)

    @staticmethod
    def _clean_region_name(path: str | pathlib.Path) -> str:
        name = pathlib.Path(path).name
        for suffix in (".nii.gz", ".nii"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        name = re.sub(r"_+$", "", name)
        return name.replace(" ", "_").replace("/", "-")

    @staticmethod
    def _rgb_for_id(rid: int) -> list[int]:
        rng = np.random.default_rng(rid)
        return rng.integers(50, 255, 3).tolist()


if __name__ == "__main__":
    split_atlas(
        atlas_nii="AAL.nii.gz",      # path to the combined atlas NIfTI file (integer valued)
        labels_txt="AAL.txt",        # path to a .txt file with labels, e.g. AAL.txt
        out_dir="aal_masks"          # path to output dir. will be created if absent
    )
