import os
import json
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
from pathlib import Path
from tqdm import tqdm
from calvin_utils.neuroimaging_utils.tract_utils.fiber_intersection import FiberVoxelIndexer
from calvin_utils.neuroimaging_utils.tract_utils.fiber_io import FiberIO

class FiberConnectivity:
    """
    Generate individualized per-fiber connectivity profiles from patient NIfTIs.

    Output format:
        One ``.fib.npy``/``.fib.json`` pair per patient. The NumPy file is a
        1D float32 vector of shape ``(n_fibers,)`` and the JSON records its
        canonical atlas.

    That is the format expected by FiberIO._load_single_fiber_values(...).
    """

    def __init__(self, fiber_indexer, mode='max', fiber_atlas_path=None):
        """        
        mode:
            'binary' -> 1 if any hit, else 0
            'max'    -> max voxel value hit by that fiber
            'mean'   -> mean voxel value hit by that fiber
            'sum'    -> sum of voxel values hit by that fiber
        """
        self.fiber_indexer = fiber_indexer
        self.reference_shape = (
            fiber_indexer.reference_shape if fiber_indexer is not None else None
        )
        self.reference_affine = (
            fiber_indexer.reference_affine if fiber_indexer is not None else None
        )
        self.reference_img = (
            fiber_indexer.reference_img if fiber_indexer is not None else None
        )
        self.mode = mode
        self.fiber_atlas_path = (
            Path(fiber_atlas_path) if fiber_atlas_path is not None else None
        )

    def _load_patient_nifti(self, nifti_path):
        img = nib.load(nifti_path)

        if img.shape[:3] != self.reference_shape or not np.allclose(img.affine, self.reference_affine):
            from nilearn import image

            img = image.resample_to_img(
                source_img=img,
                target_img=self.reference_img,
                interpolation='continuous'
            )

        data = img.get_fdata(dtype=np.float32)
        return img, data

    @staticmethod
    def _binarize_data(data, threshold):
        if threshold is None:
            raise ValueError("Threshold error. Binarize=True, but threshold=None. Set threshold to a value for binarization.")
        return (data > threshold).astype(np.uint8)
    
    def _reduce_hits_to_fiber_values(self, voxel_data, hit_flags, hit_linear_indices_per_fiber):
        """

        """
        flat = np.asarray(voxel_data).reshape(-1)
        n_fibers = len(hit_flags)

        if self.mode == "binary":
            return hit_flags.astype(np.float32)

        if self.mode not in {"max", "mean", "sum"}:
            raise ValueError(f"Unsupported mode: {self.mode}")

        out = np.zeros(n_fibers, dtype=np.float32)

        lengths = np.array([
            0 if lin_hits is None else lin_hits.size
            for lin_hits in hit_linear_indices_per_fiber
        ], dtype=np.int64)

        if lengths.sum() == 0:
            return out

        fiber_ids = np.repeat(np.arange(n_fibers), lengths)

        all_hits = np.concatenate([
            lin_hits for lin_hits in hit_linear_indices_per_fiber
            if lin_hits is not None and lin_hits.size > 0
        ])

        vals = flat[all_hits].astype(np.float32)

        valid = ~np.isnan(vals)
        fiber_ids = fiber_ids[valid]
        vals = vals[valid]

        if vals.size == 0:
            return out

        if self.mode == "sum":
            return np.bincount(
                fiber_ids,
                weights=vals,
                minlength=n_fibers
            ).astype(np.float32)

        if self.mode == "mean":
            sums = np.bincount(fiber_ids, weights=vals, minlength=n_fibers)
            counts = np.bincount(fiber_ids, minlength=n_fibers)

            np.divide(sums, counts, out=out, where=counts > 0)
            return out.astype(np.float32)

        if self.mode == "max":
            order = np.argsort(fiber_ids, kind="stable")
            fiber_ids = fiber_ids[order]
            vals = vals[order]

            starts = np.r_[0, np.flatnonzero(np.diff(fiber_ids)) + 1]
            max_vals = np.maximum.reduceat(vals, starts)

            out[fiber_ids[starts]] = max_vals
            return out
    # def _reduce_hits_to_fiber_values(self, voxel_data, hit_flags, hit_linear_indices_per_fiber, mode):
    #     """
    #     mode:
    #         'binary' -> 1 if any hit, else 0
    #         'max'    -> max voxel value hit by that fiber
    #         'mean'   -> mean voxel value hit by that fiber
    #         'sum'    -> sum of voxel values hit by that fiber
    #     """
    #     flat = voxel_data.reshape(-1)
    #     out = np.zeros(len(hit_flags), dtype=np.float32)

    #     if mode == 'binary':
    #         out[hit_flags] = 1.0
    #         return out

    #     for i, lin_hits in enumerate(hit_linear_indices_per_fiber):
    #         if lin_hits is None or lin_hits.size == 0:
    #             continue

    #         vals = flat[lin_hits]
    #         if vals.size == 0 or np.all(np.isnan(vals)):
    #             out[i] = np.nan
    #         elif mode == 'max':
    #             out[i] = np.nanmax(vals)
    #         elif mode == 'mean':
    #             out[i] = np.nanmean(vals)
    #         elif mode == 'sum':
    #             out[i] = np.nansum(vals)
    #         else:
    #             raise ValueError(f"Unsupported mode: {mode}")

    #     return out

    def generate_connectivity_profile(self, nifti_path, binarize, threshold):
        """
        Return one patient's fiber vector of shape (n_fibers,).
        """
        _, data = self._load_patient_nifti(nifti_path)

        if binarize:
            query_data = self._binarize_data(data, threshold)
            value_data = data if self.mode != 'binary' else query_data
        else:
            query_data = data
            value_data = data

        hit_flags, hit_linear_indices_per_fiber = self.fiber_indexer.query_image_hits(query_data)

        fiber_values = self._reduce_hits_to_fiber_values(
            voxel_data=value_data,
            hit_flags=hit_flags,
            hit_linear_indices_per_fiber=hit_linear_indices_per_fiber
            )

        return fiber_values.astype(np.float32)

    def generate_profiles_from_niftis(self, nifti_paths, binarize=False, threshold=None):
        """
        Return matrix shape (n_patients, n_fibers).
        binarize: whether to binarize input niftis using a threshold. Must set threshold to a value. 
        threshold: values below this are removed from the seed nifti. Used if binarize=True
        """
        print(f"Thresholding input niftis: threshold={threshold}")
        profiles = []
        for nifti_path in tqdm(nifti_paths, desc='Generating fiber connectivity profiles'):
            vec = self.generate_connectivity_profile(
                nifti_path=nifti_path,
                binarize=binarize,
                threshold=threshold
            )
            profiles.append(vec)

        return np.vstack(profiles).astype(np.float32)

    @classmethod
    def from_atlas(cls, fiber_atlas_path, reference_nifti_path, mode="binary", step_size_vox=0.5, fiber_mask=None):
        """Index one fiber atlas in one reference voxel space and return a mapper ready to profile NIfTIs."""
        indexer = FiberVoxelIndexer.from_fiber_file(fiber_file_path=os.fspath(fiber_atlas_path), reference_nifti_path=os.fspath(reference_nifti_path), fiber_mask=fiber_mask, step_size_vox=step_size_vox)
        return cls(indexer, mode=mode, fiber_atlas_path=fiber_atlas_path)

    @property
    def connectome_label(self):
        """Return <atlas>-<mode> for output filenames, e.g. HCP842_tracts-binary."""
        return f"{self.fiber_atlas_path.name.split('.')[0]}-{self.mode}"

    @staticmethod
    def _safe_stem(path):
        path = Path(path)
        return path.name[:-7] if path.name.endswith(".nii.gz") else path.stem

    @staticmethod
    def _profile_output_path(nifti_path, suffix, out_dir=None):
        """Return <stem><suffix>.fib.npy in out_dir, defaulting to a sibling ``fibers`` directory (sub-x/roi/ -> sub-x/fibers/)."""
        nifti_path = Path(nifti_path).expanduser()
        destination = Path(out_dir).expanduser() if out_dir is not None else nifti_path.parent.parent / "fibers"
        return destination / f"{FiberConnectivity._safe_stem(nifti_path)}{suffix}.fib.npy"

    def save_profile(self, fiber_values, out_path, fiber_atlas_path=None):
        atlas_path = (
            Path(fiber_atlas_path)
            if fiber_atlas_path is not None
            else self.fiber_atlas_path
        )
        if atlas_path is None:
            raise ValueError(
                "Saving a fiber profile requires fiber_atlas_path so its "
                ".fib.json descriptor can be written."
            )
        out_path = Path(out_path)
        if not out_path.name.lower().endswith(".fib.npy"):
            raise ValueError(f"Fiber profile output must end with .fib.npy: {out_path}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fiber_values = np.asarray(fiber_values, dtype=np.float32).flatten()
        np.save(out_path, fiber_values)
        return FiberIO.write_values_description(out_path, atlas_path)

    def save_profiles_from_niftis(self, nifti_paths, out_dir=None, binarize=True, threshold=0, suffix=""):
        """Write one .fib.npy/.fib.json pair per NIfTI, named <stem>_<atlas>-<mode><suffix>, and return the .fib.npy paths."""
        if self.fiber_atlas_path is None: raise ValueError("Saving fiber profiles requires fiber_atlas_path so each .fib.json can identify its geometry.")
        saved_paths = []
        for nifti_path in tqdm(nifti_paths, desc="Saving fiber connectivity profiles"):
            values = self.generate_connectivity_profile(nifti_path=nifti_path, binarize=binarize, threshold=threshold)
            out_path = self._profile_output_path(nifti_path, f"_{self.connectome_label}{suffix}", out_dir=out_dir)
            self.save_profile(values, out_path)
            saved_paths.append(out_path)
        return saved_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate fiber connectivity profiles from NIfTI files.")
    parser.add_argument("--fiber_file", required=True, help="Path to tract file (.trk/.tck/.npy/etc)")
    parser.add_argument("--reference_nifti", required=True, help="Reference NIfTI defining voxel space")
    parser.add_argument("--niftis", nargs="+", required=True, help="List of patient NIfTI files")
    parser.add_argument("--out_dir", default=None, help="Output directory; defaults to a sibling fibers/ directory beside each NIfTI's folder")
    parser.add_argument("--mode", default="binary", choices=["binary", "max", "mean", "sum"], help="Aggregation mode")
    parser.add_argument("--threshold", type=float, default=0.0, help="Threshold for binarization")
    parser.add_argument("--no_binarize", action="store_true", help="Disable binarization")
    parser.add_argument("--step_size_vox", type=float, default=0.5, help="Voxel step size for fiber rasterization")
    parser.add_argument("--fiber_mask", default=None, help="Optional fiber mask (same ordering as tract file)")
    args = parser.parse_args()

    mapper = FiberConnectivity.from_atlas(args.fiber_file, args.reference_nifti, mode=args.mode, step_size_vox=args.step_size_vox, fiber_mask=args.fiber_mask)
    mapper.save_profiles_from_niftis(args.niftis, out_dir=args.out_dir, binarize=not args.no_binarize, threshold=args.threshold)
