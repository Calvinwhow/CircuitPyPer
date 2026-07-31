from __future__ import annotations

import json
from pathlib import Path

import numpy as np

if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [np.bool_, np.bytes_, np.str_, np.object_],
    }
if not hasattr(np, "maximum_sctype"):
    def _maximum_sctype(t):
        dtype = np.dtype(t)
        if np.issubdtype(dtype, np.complexfloating):
            return np.complex128
        if np.issubdtype(dtype, np.floating):
            return np.float64
        if np.issubdtype(dtype, np.unsignedinteger):
            return np.uint64
        if np.issubdtype(dtype, np.integer):
            return np.int64
        return dtype.type

    np.maximum_sctype = _maximum_sctype

import nibabel as nib
from tqdm import tqdm

from calvin_utils.neuroimaging_utils.tract_utils.fiber_intersection import FiberVoxelIndexer


class TractDensity:
    """
    Convert fibers into a Lead-DBS-style tract-density NIfTI.

    This mirrors the important behavior of Lead-DBS
    ``ea_discfibers2nifti.m``:

    1. Load fibers and one scalar value per fiber.
    2. Optionally keep positive, negative, or both tails using a top-percent
       threshold.
    3. Convert fiber vertices from MNI/world mm into a reference NIfTI voxel
       grid.
    4. Add the fiber value to every rounded vertex voxel.
    5. For negative-only exports, multiply the final image by ``-1`` so the
       output is positive for display.

    Inputs
    ------
    fiber_path:
        Fiber geometry source. Supported inputs are ``.fib.npy`` regression
        outputs or Calvin internal ``.npz`` atlas files.

        A ``.fib.npy`` regression output is an object array where each fiber is
        ``N x 4`` and column 4 contains the fiber statistic.

        A ``.npz`` atlas should contain a ``fibers`` key. If ``values_path`` or
        ``values`` is not provided, every fiber receives value 1.

        External tract formats such as ``.tck``/``.tt.tck``/``.trk`` should be
        converted first with ``FiberAtlasConverter``.

    reference_nifti_path:
        NIfTI defining output shape and affine. Use the same MNI reference used
        by the rest of the fiber workflow.

    out_path:
        Output ``.nii`` or ``.nii.gz`` path.

    values_path / values:
        Optional one-value-per-fiber vector. Use this when converting an atlas
        plus separate regression values. Ignored for ``.fib.npy`` if ``values``
        is not explicitly provided.

    fiberset:
        ``"positive"``, ``"negative"``, or ``"both"``. Matches Lead-DBS
        naming; ``"pos"`` and ``"neg"`` are accepted aliases.

    threshold:
        Fraction or percent of the positive/negative tail to keep. Lead-DBS
        treats values above 1 as percentages, so ``5`` and ``0.05`` both mean
        top 5 percent. Set ``threshold=None`` to skip tail thresholding.

    unique_per_fiber:
        If False, match ``ea_discfibers2nifti`` exactly by counting every
        rounded vertex. If True, each fiber contributes at most once to each
        voxel.

    symmetric:
        If True, duplicate the selected fibers after thresholding and mirror
        them across the MNI sagittal plane by default. This creates a bilateral
        tract-density image from a unilateral atlas or result. The default
        mirror is left-right, ``x -> -x`` around ``x=0``.

    run() returns a dictionary with output paths, counts, and thresholds.
    """

    def __init__(
        self,
        *,
        fiber_path: str | Path,
        reference_nifti_path: str | Path,
        out_path: str | Path,
        values_path: str | Path | None = None,
        values=None,
        fiberset: str = "positive",
        threshold: float | None = 0.05,
        unique_per_fiber: bool = False,
        dtype=np.float32,
        show_progress: bool = True,
        symmetric: bool = False,
        mirror_axis: str = "x",
        mirror_origin: float = 0.0,
    ):
        self.fiber_path = Path(fiber_path)
        self.reference_nifti_path = Path(reference_nifti_path)
        self.out_path = Path(out_path)
        self.values_path = Path(values_path) if values_path is not None else None
        self.values = None if values is None else np.asarray(values, dtype=np.float32).flatten()
        self.fiberset = fiberset
        self.threshold = threshold
        self.unique_per_fiber = bool(unique_per_fiber)
        self.dtype = dtype
        self.show_progress = bool(show_progress)
        self.symmetric = bool(symmetric)
        self.mirror_axis = mirror_axis
        self.mirror_origin = float(mirror_origin)

        self.reference_img = None
        self.fibers = None
        self.selected_fibers = None
        self.n_selected_before_symmetry = None
        self.loaded_values = None
        self.selected_values = None
        self.keep_mask = None
        self.image = None
        self.metadata_path = None
        self.result = None
        self.positive_threshold = None
        self.negative_threshold = None

    def run(self):
        self.validate_inputs()
        self.load_reference()
        self.load_fibers_and_values()
        self.validate_fiber_value_alignment()
        self.apply_leaddbs_selection()
        self.apply_symmetry()
        self.build_density_image()
        self.save_nifti()
        self.save_metadata()
        self.package_result()
        return self.get_result()

    def validate_inputs(self):
        if not self.fiber_path.exists():
            raise FileNotFoundError(f"fiber_path does not exist: {self.fiber_path}")
        if not self.reference_nifti_path.exists():
            raise FileNotFoundError(f"reference_nifti_path does not exist: {self.reference_nifti_path}")
        if self.values_path is not None and not self.values_path.exists():
            raise FileNotFoundError(f"values_path does not exist: {self.values_path}")
        if not (self.fiber_path.name.endswith(".fib.npy") or self.fiber_path.suffix.lower() == ".npz"):
            raise ValueError(
                "TractDensity requires a Calvin .npz fiber atlas or a regression .fib.npy output. "
                "Convert external tract files first with FiberAtlasConverter."
            )
        if self.fiberset not in {"both", "positive", "pos", "negative", "neg"}:
            raise ValueError("fiberset must be one of: both, positive, pos, negative, neg.")
        if self.threshold is not None and float(self.threshold) <= 0:
            raise ValueError("threshold must be positive, or None.")
        if self.mirror_axis not in {"x", "y", "z", 0, 1, 2}:
            raise ValueError("mirror_axis must be one of: x, y, z, 0, 1, 2.")

    def load_reference(self):
        self.reference_img = nib.load(str(self.reference_nifti_path))

    def load_fibers_and_values(self):
        if self.fiber_path.name.endswith(".fib.npy"):
            self.fibers, fiber_values = self._load_fib_npy(self.fiber_path)
            self.loaded_values = fiber_values if self.values is None else self.values
            return

        indexer = FiberVoxelIndexer(reference_nifti_path=str(self.reference_nifti_path))
        self.fibers = [fiber[:, :3] for fiber in indexer.load_fibers(str(self.fiber_path))]

        if self.values is not None:
            self.loaded_values = self.values
        elif self.values_path is not None:
            self.loaded_values = self._load_values(self.values_path)
        else:
            self.loaded_values = np.ones(len(self.fibers), dtype=np.float32)

    def validate_fiber_value_alignment(self):
        if len(self.fibers) != self.loaded_values.shape[0]:
            raise ValueError(
                f"Fiber/value length mismatch: {len(self.fibers)} fibers but "
                f"{self.loaded_values.shape[0]} values."
            )

    def apply_leaddbs_selection(self):
        vals = np.asarray(self.loaded_values, dtype=np.float32).flatten()
        vals = vals.copy()
        vals[~np.isfinite(vals)] = 0

        if self.threshold is None:
            keep = vals != 0 if self.fiberset != "both" else np.isfinite(vals)
            if self.fiberset in {"positive", "pos"}:
                keep &= vals > 0
            elif self.fiberset in {"negative", "neg"}:
                keep &= vals < 0
            self.positive_threshold = None
            self.negative_threshold = None
        else:
            threshold = float(self.threshold)
            if threshold > 1:
                threshold = threshold / 100.0
            pos_vals = np.sort(vals[vals > 0])[::-1]
            neg_vals = np.sort(vals[vals < 0])

            posthresh = np.inf
            negthresh = -np.inf
            if self.fiberset in {"positive", "pos", "both"} and pos_vals.size > 0:
                posthresh = self._tail_cutoff(pos_vals, threshold)
            elif self.fiberset in {"positive", "pos"}:
                raise ValueError("No positive fibers found.")

            if self.fiberset in {"negative", "neg", "both"} and neg_vals.size > 0:
                negthresh = self._tail_cutoff(neg_vals, threshold)
            elif self.fiberset in {"negative", "neg"}:
                raise ValueError("No negative fibers found.")

            if self.fiberset in {"positive", "pos"}:
                keep = (vals >= posthresh) & (vals > 0)
            elif self.fiberset in {"negative", "neg"}:
                keep = (vals <= negthresh) & (vals < 0)
            else:
                keep = (vals >= posthresh) | (vals <= negthresh)
            self.positive_threshold = None if np.isinf(posthresh) else float(posthresh)
            self.negative_threshold = None if np.isinf(negthresh) else float(negthresh)

        self.keep_mask = keep
        self.selected_fibers = [fiber for fiber, selected in zip(self.fibers, keep) if selected]
        self.selected_values = vals[keep].astype(np.float32)
        self.n_selected_before_symmetry = int(len(self.selected_fibers))

    def apply_symmetry(self):
        if not self.symmetric:
            return
        mirrored_fibers = [self._mirror_fiber(fiber) for fiber in self.selected_fibers]
        self.selected_fibers = self.selected_fibers + mirrored_fibers
        self.selected_values = np.concatenate([self.selected_values, self.selected_values]).astype(np.float32)

    def build_density_image(self):
        shape = self.reference_img.shape[:3]
        density = np.zeros(shape, dtype=np.float64)
        inv_affine = np.linalg.inv(self.reference_img.affine)

        iterator = self._progress(
            zip(self.selected_fibers, self.selected_values),
            total=len(self.selected_fibers),
            desc="Rasterizing fibers",
            unit="fiber",
        )
        for fiber, value in iterator:
            ijk = self._world_to_ijk(fiber[:, :3], inv_affine)
            ijk = self._in_bounds_ijk(ijk, shape)
            if ijk.shape[0] == 0:
                continue
            if self.unique_per_fiber:
                ijk = np.unique(ijk, axis=0)
            density[ijk[:, 0], ijk[:, 1], ijk[:, 2]] += float(value)

        if self.fiberset in {"negative", "neg"}:
            density *= -1.0

        self.image = density.astype(self.dtype)

    def save_nifti(self):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        out_img = nib.Nifti1Image(self.image, self.reference_img.affine, self.reference_img.header.copy())
        out_img.set_data_dtype(np.dtype(self.dtype))
        nib.save(out_img, str(self.out_path))

    def save_metadata(self):
        suffix = "".join(self.out_path.suffixes)
        stem = self.out_path.name[: -len(suffix)] if suffix else self.out_path.stem
        self.metadata_path = self.out_path.with_name(f"{stem}_metadata.json")
        self.metadata_path.write_text(json.dumps(self._metadata(), indent=2))

    def package_result(self):
        self.result = {
            "nifti": str(self.out_path),
            "metadata": str(self.metadata_path),
            "n_input_fibers": int(len(self.fibers)),
            "n_selected_before_symmetry": self.n_selected_before_symmetry,
            "n_exported_fibers": int(len(self.selected_fibers)),
            "symmetric": self.symmetric,
            "positive_threshold": self.positive_threshold,
            "negative_threshold": self.negative_threshold,
            "keep_mask": self.keep_mask,
            "values": self.selected_values,
        }

    def get_result(self):
        return self.result

    def _metadata(self):
        nonzero_voxels = int(np.count_nonzero(self.image)) if self.image is not None else 0
        return {
            "fiber_path": str(self.fiber_path),
            "reference_nifti_path": str(self.reference_nifti_path),
            "out_path": str(self.out_path),
            "values_path": str(self.values_path) if self.values_path is not None else None,
            "fiberset": self.fiberset,
            "threshold": self.threshold,
            "positive_threshold": self.positive_threshold,
            "negative_threshold": self.negative_threshold,
            "unique_per_fiber": self.unique_per_fiber,
            "n_input_fibers": int(len(self.fibers)),
            "n_selected_before_symmetry": self.n_selected_before_symmetry,
            "n_exported_fibers": int(len(self.selected_fibers)),
            "symmetric": self.symmetric,
            "mirror_axis": self.mirror_axis,
            "mirror_origin": self.mirror_origin,
            "nonzero_voxels": nonzero_voxels,
            "image_sum": float(np.nansum(self.image)) if self.image is not None else None,
        }

    def _progress(self, iterable, **kwargs):
        if not self.show_progress:
            return iterable
        return tqdm(iterable, **kwargs)

    @staticmethod
    def _tail_cutoff(sorted_tail, threshold):
        idx = int(np.floor(sorted_tail.size * threshold + 0.5)) - 1
        idx = min(max(idx, 0), sorted_tail.size - 1)
        return float(sorted_tail[idx])

    @staticmethod
    def _world_to_ijk(xyz, inv_affine):
        xyz = np.asarray(xyz, dtype=np.float32)
        hom = np.c_[xyz, np.ones(xyz.shape[0], dtype=np.float32)]
        ijk_float = hom @ inv_affine.T
        return np.rint(ijk_float[:, :3]).astype(np.int64)

    @staticmethod
    def _in_bounds_ijk(ijk, shape):
        keep = (
            (ijk[:, 0] >= 0) & (ijk[:, 0] < shape[0])
            & (ijk[:, 1] >= 0) & (ijk[:, 1] < shape[1])
            & (ijk[:, 2] >= 0) & (ijk[:, 2] < shape[2])
        )
        return ijk[keep]

    def _mirror_fiber(self, fiber):
        mirrored = np.asarray(fiber, dtype=np.float32).copy()
        axis = self._axis_index(self.mirror_axis)
        mirrored[:, axis] = (2.0 * self.mirror_origin) - mirrored[:, axis]
        return mirrored

    @staticmethod
    def _axis_index(axis):
        if axis in {0, 1, 2}:
            return int(axis)
        return {"x": 0, "y": 1, "z": 2}[axis]

    @staticmethod
    def _load_fib_npy(path):
        obj = np.load(path, allow_pickle=True)
        fibers = [np.asarray(fiber, dtype=np.float32) for fiber in obj.tolist()]
        xyz_fibers = []
        values = []
        for fiber in fibers:
            if fiber.ndim != 2 or fiber.shape[1] not in (3, 4):
                raise ValueError(f"Invalid fiber shape in {path}: {fiber.shape}")
            xyz_fibers.append(fiber[:, :3].astype(np.float32, copy=False))
            if fiber.shape[1] == 4:
                values.append(float(np.nanmedian(fiber[:, 3])))
            else:
                values.append(1.0)
        return xyz_fibers, np.asarray(values, dtype=np.float32)

    @staticmethod
    def _load_values(path):
        arr = np.load(path, allow_pickle=True)
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D value vector in {path}, got shape {arr.shape}")
        return arr.astype(np.float32)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create a Lead-DBS-style tract-density NIfTI from fibers.")
    parser.add_argument("--fiber-path", required=True, help="Input .fib.npy, .npz, .npy, .tck/.tt.tck, .trk, etc.")
    parser.add_argument("--reference-nifti", required=True, help="Reference NIfTI defining output grid.")
    parser.add_argument("--out", required=True, help="Output .nii or .nii.gz path.")
    parser.add_argument("--values-path", help="Optional .npy vector with one value per fiber.")
    parser.add_argument("--fiberset", default="positive", choices=["both", "positive", "pos", "negative", "neg"])
    parser.add_argument("--threshold", type=float, default=0.05, help="Tail fraction or percent. Use -1 to disable.")
    parser.add_argument("--unique-per-fiber", action="store_true", help="Count each fiber at most once per voxel.")
    parser.add_argument("--symmetric", action="store_true", help="Duplicate selected fibers mirrored across the chosen axis.")
    parser.add_argument("--mirror-axis", default="x", choices=["x", "y", "z"], help="Axis to mirror for symmetric export.")
    parser.add_argument("--mirror-origin", type=float, default=0.0, help="Coordinate of mirror plane.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    args = parser.parse_args()

    threshold = None if args.threshold < 0 else args.threshold
    result = TractDensity(
        fiber_path=args.fiber_path,
        reference_nifti_path=args.reference_nifti,
        out_path=args.out,
        values_path=args.values_path,
        fiberset=args.fiberset,
        threshold=threshold,
        unique_per_fiber=args.unique_per_fiber,
        show_progress=not args.no_progress,
        symmetric=args.symmetric,
        mirror_axis=args.mirror_axis,
        mirror_origin=args.mirror_origin,
    ).run()
    print(json.dumps({k: v for k, v in result.items() if k not in {"keep_mask", "values"}}, indent=2))
