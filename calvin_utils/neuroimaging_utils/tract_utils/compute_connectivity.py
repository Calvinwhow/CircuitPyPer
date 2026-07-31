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

class FiberConnectivity:
    """
    Generate individualized per-fiber connectivity profiles from patient NIfTIs.

    Output format:
        One .npy file per patient, containing a 1D float32 vector of shape (n_fibers,)

    That is the format expected by FiberIO._load_single_fiber_values(...).
    """

    def __init__(self, fiber_indexer, mode='max'):
        """        
        mode:
            'binary' -> 1 if any hit, else 0
            'max'    -> max voxel value hit by that fiber
            'mean'   -> mean voxel value hit by that fiber
            'sum'    -> sum of voxel values hit by that fiber
        """
        self.fiber_indexer = fiber_indexer
        self.reference_shape = fiber_indexer.reference_shape
        self.reference_affine = fiber_indexer.reference_affine
        self.reference_img = fiber_indexer.reference_img
        self.mode = mode

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

    @staticmethod
    def _safe_stem(path, bids_style=True):
        """If bids_style, save to a neighbouring directory the nifti was found in. If not, saves beside the nifti"""
        p = Path(path)

        if p.name.endswith(".nii.gz"):  # Fixes bad handling of double suffix
            stem = p.name[:-7]
        else:
            stem = p.stem

        if bids_style:
            return p.parent / "fibers" / stem

        return stem

    def save_profile(self, fiber_values, out_path):
        fiber_values = np.asarray(fiber_values, dtype=np.float32).flatten()
        np.save(out_path, fiber_values)

    def save_profiles_from_niftis(
        self,
        nifti_paths,
        out_dir,
        binarize=True,
        threshold=0,
        suffix='_fiber_connectivity',
        save_matrix=True,
        matrix_name='cohort_fiber_connectivity_matrix.npy',
    ):
        """
        Write one .npy vector per patient, each shape (n_fibers,).

        Optional:
            save_matrix=True also writes a stacked matrix shape (n_patients, n_fibers)
        """
        os.makedirs(out_dir, exist_ok=True)

        matrix = []

        for nifti_path in tqdm(nifti_paths, desc='Saving fiber connectivity profiles'):
            vec = self.generate_connectivity_profile(
                nifti_path=nifti_path,
                binarize=binarize,
                threshold=threshold
            )
            matrix.append(vec)

            stem = self._safe_stem(nifti_path)
            out_path = os.path.join(out_dir, f"{stem}{suffix}.npy")
            self.save_profile(vec, out_path)

        if save_matrix:
            mat = np.vstack(matrix).astype(np.float32)
            np.save(os.path.join(out_dir, matrix_name), mat)

    def generate_and_save_from_paths(
        self,
        fiber_file_path,
        reference_nifti_path,
        nifti_paths,
        out_dir,
        binarize=True,
        threshold=0,
        suffix='_fiber_connectivity',
        step_size_vox=0.5,
        fiber_mask=None,
        save_matrix=False,
        matrix_name='fiber_connectivity_matrix.npy',
    ):
        """
        Convenience wrapper that rebuilds the indexer from tract files and writes patient .npy files.
        """
        self.fiber_indexer = FiberVoxelIndexer.from_fiber_file(
            fiber_file_path=fiber_file_path,
            reference_nifti_path=reference_nifti_path,
            fiber_mask=fiber_mask,
            step_size_vox=step_size_vox,
        )
        self.reference_shape = self.fiber_indexer.reference_shape
        self.reference_affine = self.fiber_indexer.reference_affine
        self.reference_img = self.fiber_indexer.reference_img

        self.save_profiles_from_niftis(
            nifti_paths=nifti_paths,
            out_dir=out_dir,
            binarize=binarize,
            threshold=threshold,
            suffix=suffix,
            save_matrix=save_matrix,
            matrix_name=matrix_name,
        )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate fiber connectivity profiles from NIfTI files.")

    parser.add_argument("--fiber_file", required=True, help="Path to tract file (.trk/.tck/.npy/etc)")
    parser.add_argument("--reference_nifti", required=True, help="Reference NIfTI defining voxel space")
    parser.add_argument("--niftis", nargs="+", required=True, help="List of patient NIfTI files")
    parser.add_argument("--out_dir", required=True, help="Output directory for .npy fiber profiles")

    parser.add_argument("--mode", default="binary", choices=["binary", "max", "mean", "sum"], help="Aggregation mode")
    parser.add_argument("--threshold", type=float, default=0.0, help="Threshold for binarization")
    parser.add_argument("--no_binarize", action="store_true", help="Disable binarization")
    parser.add_argument("--step_size_vox", type=float, default=0.5, help="Voxel step size for fiber rasterization")

    parser.add_argument("--fiber_mask", default=None, help="Optional fiber mask (same ordering as tract file)")
    parser.add_argument("--save_matrix", action="store_true", help="Also save full (n_patients, n_fibers) matrix")
    parser.add_argument("--matrix_name", default="fiber_connectivity_matrix.npy", help="Matrix filename")

    args = parser.parse_args()

    mapper = FiberConnectivity(fiber_indexer=None)

    mapper.generate_and_save_from_paths(
        fiber_file_path=args.fiber_file,
        reference_nifti_path=args.reference_nifti,
        nifti_paths=args.niftis,
        out_dir=args.out_dir,
        mode=args.mode,
        binarize=not args.no_binarize,
        threshold=args.threshold,
        step_size_vox=args.step_size_vox,
        fiber_mask=args.fiber_mask,
        save_matrix=args.save_matrix,
        matrix_name=args.matrix_name,
    )
