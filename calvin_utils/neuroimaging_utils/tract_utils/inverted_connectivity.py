"""Orientation-invariant cosine matching between streamline geometries."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy import sparse
from tqdm import tqdm

from calvin_utils.neuroimaging_utils.nifti_utils.volume_io import NiftiIO
from calvin_utils.neuroimaging_utils.tract_utils.fiber_geometry_registration import (
    GeodesicFiberRegistration,
)
from calvin_utils.neuroimaging_utils.tract_utils.fiber_intersection import (
    FiberVoxelIndexer,
)


class InvertedFiberConnectivity:
    """Match target trajectories to a connectome and project them to voxels.

    Every streamline is sampled at a fixed millimeter interval. The shorter
    sequence is slid along the longer sequence in both endpoint orientations;
    the placement minimizing pointwise XYZ L2 distance defines the arrays used
    for cosine similarity.
    """

    def __init__(
        self,
        connectome_path,
        reference_nifti_path,
        mask_path=None,
        sample_interval_mm=1.0,
        max_length_mm=None,
        step_size_vox=0.5,
        fiber_batch_size=2048,
        cache_dir=None,
    ):
        self.connectome_path = Path(connectome_path)
        self.reference_nifti_path = Path(reference_nifti_path)
        self.mask_path = Path(mask_path or reference_nifti_path)
        self.sample_interval_mm = float(sample_interval_mm)
        self.requested_max_length_mm = (
            None if max_length_mm is None else float(max_length_mm)
        )
        self.step_size_vox = float(step_size_vox)
        self.fiber_batch_size = int(fiber_batch_size)

        if self.sample_interval_mm <= 0:
            raise ValueError("sample_interval_mm must be greater than zero.")
        if (
            self.requested_max_length_mm is not None
            and self.requested_max_length_mm <= 0
        ):
            raise ValueError("max_length_mm must be greater than zero.")
        if self.fiber_batch_size < 1:
            raise ValueError("fiber_batch_size must be at least one.")

        if cache_dir is None:
            name = self._safe_stem(self.connectome_path)
            cache_dir = self.connectome_path.parent / f".{name}_trajectory_cache"
        self.cache_dir = Path(cache_dir)
        self.incidence_cache_path = self.cache_dir / "voxel_fiber_incidence.npz"
        self.forward_cache_path = self.cache_dir / "fibers_forward.npy"
        self.reverse_cache_path = self.cache_dir / "fibers_reverse.npy"
        self.length_cache_path = self.cache_dir / "fiber_lengths_mm.npy"
        self.count_cache_path = self.cache_dir / "fiber_sample_counts.npy"
        self.geometry_manifest_path = self.cache_dir / "geometry_manifest.json"

        self.geometry_registration = GeodesicFiberRegistration(
            sample_interval_mm=self.sample_interval_mm,
            max_length_mm=self.requested_max_length_mm,
        )

        self.reference_indexer = FiberVoxelIndexer(
            reference_nifti_path=str(self.reference_nifti_path),
            step_size_vox=self.step_size_vox,
        )
        self._mask_indices = None
        self._full_to_mask = None
        self._connectome_fibers = None
        self._incidence = None
        self._geometry_manifest = None
        self._validate_spatial_inputs()

    @staticmethod
    def _safe_stem(path):
        name = Path(path).name
        for suffix in (".nii.gz", ".fib.gz", ".fib.npy", ".fib.json"):
            if name.lower().endswith(suffix):
                return name[: -len(suffix)]
        return Path(name).stem

    def _validate_spatial_inputs(self):
        import nibabel as nib

        reference = self.reference_indexer.reference_img
        mask = nib.load(str(self.mask_path))
        if mask.shape[:3] != reference.shape[:3] or not np.allclose(
            mask.affine, reference.affine
        ):
            raise ValueError(
                "mask_path and reference_nifti_path must have matching shape "
                "and affine."
            )

        mask_flat = mask.get_fdata().reshape(-1) > 0
        if not np.any(mask_flat):
            raise ValueError("The structural-connectivity mask is empty.")
        self._mask_indices = np.flatnonzero(mask_flat)
        self._full_to_mask = np.full(mask_flat.size, -1, dtype=np.int64)
        self._full_to_mask[self._mask_indices] = np.arange(
            self._mask_indices.size, dtype=np.int64
        )

    @property
    def n_voxels(self):
        return int(self._mask_indices.size)

    def _source_signature(self):
        stat = self.connectome_path.stat()
        return {
            "path": str(self.connectome_path.resolve()),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    def _load_connectome(self):
        if self._connectome_fibers is None:
            self._connectome_fibers = self.reference_indexer.load_fibers(
                str(self.connectome_path)
            )
        return self._connectome_fibers

    @staticmethod
    def _atomic_save_json(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temp_path = Path(temporary.name)
            json.dump(payload, temporary, indent=2)
        os.replace(temp_path, path)

    def _read_geometry_manifest(self):
        if not self.geometry_manifest_path.is_file():
            return None
        try:
            with open(self.geometry_manifest_path) as manifest_file:
                manifest = json.load(manifest_file)
        except (OSError, ValueError, TypeError):
            return None

        if (
            manifest.get("format") != "sliding_geodesic_cosine_v2"
            or manifest.get("connectome") != self._source_signature()
            or not np.isclose(
                manifest.get("sample_interval_mm", -1), self.sample_interval_mm
            )
            or not self.forward_cache_path.is_file()
            or not self.reverse_cache_path.is_file()
            or not self.length_cache_path.is_file()
            or not self.count_cache_path.is_file()
        ):
            return None
        if self.requested_max_length_mm is not None and not np.isclose(
            manifest.get("max_length_mm", -1), self.requested_max_length_mm
        ):
            return None

        expected_shape = (
            int(manifest["n_fibers"]),
            int(manifest["max_samples"]),
            3,
        )
        try:
            forward = np.load(
                self.forward_cache_path, mmap_mode="r", allow_pickle=False
            )
            reverse = np.load(
                self.reverse_cache_path, mmap_mode="r", allow_pickle=False
            )
            lengths = np.load(
                self.length_cache_path, mmap_mode="r", allow_pickle=False
            )
            counts = np.load(
                self.count_cache_path, mmap_mode="r", allow_pickle=False
            )
        except (OSError, ValueError):
            return None
        if (
            forward.shape != expected_shape
            or reverse.shape != expected_shape
            or lengths.shape != (expected_shape[0],)
            or counts.shape != (expected_shape[0],)
        ):
            return None
        return manifest

    @property
    def has_geometry_cache(self):
        if self._geometry_manifest is None:
            self._geometry_manifest = self._read_geometry_manifest()
            if self._geometry_manifest is not None:
                self.geometry_registration.max_length_mm = float(
                    self._geometry_manifest["max_length_mm"]
                )
        return self._geometry_manifest is not None

    def prepare_geometry_cache(self, force=False):
        """Precompute regular-mm forward and endpoint-inverted fiber arrays."""
        if self.has_geometry_cache and not force:
            return self.geometry_manifest_path

        fibers = self._load_connectome()
        lengths = np.fromiter(
            (self.geometry_registration.geodesic_length(fiber) for fiber in fibers),
            dtype=np.float32,
            count=len(fibers),
        )
        observed_max = float(lengths.max()) if lengths.size else 0.0
        max_length_mm = (
            observed_max
            if self.requested_max_length_mm is None
            else self.requested_max_length_mm
        )
        self.geometry_registration.max_length_mm = max_length_mm
        max_samples = int(np.floor(max_length_mm / self.sample_interval_mm)) + 1
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        temporary_paths = [
            self.cache_dir / ".fibers_forward.tmp.npy",
            self.cache_dir / ".fibers_reverse.tmp.npy",
            self.cache_dir / ".fiber_lengths.tmp.npy",
            self.cache_dir / ".fiber_sample_counts.tmp.npy",
        ]
        try:
            forward = np.lib.format.open_memmap(
                temporary_paths[0],
                mode="w+",
                dtype=np.float32,
                shape=(len(fibers), max_samples, 3),
            )
            reverse = np.lib.format.open_memmap(
                temporary_paths[1],
                mode="w+",
                dtype=np.float32,
                shape=(len(fibers), max_samples, 3),
            )
            counts = np.empty(len(fibers), dtype=np.int32)
            for index, fiber in enumerate(
                tqdm(fibers, desc="Sampling connectome fiber geometry")
            ):
                forward_samples = self.geometry_registration.sample(
                    fiber, reverse=False
                )
                reverse_samples = self.geometry_registration.sample(
                    fiber, reverse=True
                )
                count = min(len(forward_samples), max_samples)
                counts[index] = count
                forward[index] = self.geometry_registration.pad(
                    forward_samples[:count], max_samples
                )
                reverse[index] = self.geometry_registration.pad(
                    reverse_samples[:count], max_samples
                )
            forward.flush()
            reverse.flush()
            del forward, reverse
            np.save(temporary_paths[2], lengths, allow_pickle=False)
            np.save(temporary_paths[3], counts, allow_pickle=False)
            os.replace(temporary_paths[0], self.forward_cache_path)
            os.replace(temporary_paths[1], self.reverse_cache_path)
            os.replace(temporary_paths[2], self.length_cache_path)
            os.replace(temporary_paths[3], self.count_cache_path)
        except Exception:
            for path in temporary_paths:
                path.unlink(missing_ok=True)
            raise

        manifest = {
            "format": "sliding_geodesic_cosine_v2",
            "connectome": self._source_signature(),
            "n_fibers": len(fibers),
            "sample_interval_mm": self.sample_interval_mm,
            "max_length_mm": max_length_mm,
            "observed_max_length_mm": observed_max,
            "max_samples": max_samples,
            "padding": "zero_tail",
            "registration": "minimum_l2_over_orientation_and_integer_offset",
        }
        self._atomic_save_json(self.geometry_manifest_path, manifest)
        self._geometry_manifest = manifest
        return self.geometry_manifest_path

    def _geometry_arrays(self):
        if not self.has_geometry_cache:
            self.prepare_geometry_cache()
        forward = np.load(
            self.forward_cache_path, mmap_mode="r", allow_pickle=False
        )
        reverse = np.load(
            self.reverse_cache_path, mmap_mode="r", allow_pickle=False
        )
        counts = np.load(
            self.count_cache_path, mmap_mode="r", allow_pickle=False
        )
        return forward, reverse, counts

    def _target_geometry(self, target_fiber_path):
        if not self.has_geometry_cache:
            self.prepare_geometry_cache()
        fibers = self.reference_indexer.load_fibers(str(target_fiber_path))
        if not fibers:
            raise ValueError(f"No target fibers found in {target_fiber_path}.")
        target = [
            self.geometry_registration.sample(fiber, reverse=False)
            for fiber in tqdm(fibers, desc="Sampling target fiber geometry")
        ]
        target = [samples for samples in target if len(samples) > 0]
        if not target:
            raise ValueError("All target fiber representations have zero norm.")
        return target

    @staticmethod
    def _alignment_score(alignment, similarity):
        if similarity == "padded_cosine":
            return alignment.padded_cosine
        if similarity == "overlap_cosine":
            return alignment.overlap_cosine
        if similarity == "coverage_weighted_cosine":
            return alignment.coverage_weighted_cosine
        raise ValueError(
            "similarity must be 'padded_cosine', 'overlap_cosine', or "
            "'coverage_weighted_cosine'."
        )

    def match_target_fibers(
        self,
        target_fiber_path,
        similarity="padded_cosine",
        return_alignments=False,
    ):
        """Register each connectome fiber to the best matching target fiber."""
        forward, reverse, counts = self._geometry_arrays()
        target = self._target_geometry(target_fiber_path)
        scores = np.full(forward.shape[0], -np.inf, dtype=np.float32)
        best_alignments = [] if return_alignments else None

        for index in tqdm(
            range(forward.shape[0]), desc="Registering target trajectories"
        ):
            count = int(counts[index])
            if count < 1:
                if return_alignments:
                    best_alignments.append(None)
                continue
            candidate_forward = np.asarray(forward[index, :count])
            candidate_reverse = np.asarray(reverse[index, :count])
            best_score = -np.inf
            best_alignment = None
            for target_samples in target:
                alignment = self.geometry_registration.align_sampled(
                    candidate_forward=candidate_forward,
                    candidate_reverse=candidate_reverse,
                    target=target_samples,
                )
                score = self._alignment_score(alignment, similarity)
                if score > best_score:
                    best_score = score
                    best_alignment = alignment
            scores[index] = best_score
            if return_alignments:
                best_alignments.append(best_alignment)
        if return_alignments:
            return scores, best_alignments
        return scores

    def _masked_voxels(self, full_voxel_indices):
        mapped = self._full_to_mask[np.asarray(full_voxel_indices, dtype=np.int64)]
        return mapped[mapped >= 0]

    @staticmethod
    def _atomic_save_sparse(path, matrix):
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp.npz",
            delete=False,
        ) as temporary:
            temp_path = Path(temporary.name)
        try:
            sparse.save_npz(temp_path, matrix, compressed=False)
            os.replace(temp_path, path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    @property
    def incidence(self):
        """Sparse ``(masked voxels, connectome fibers)`` projection matrix."""
        if self._incidence is not None:
            return self._incidence
        if self.incidence_cache_path.is_file():
            incidence = sparse.load_npz(self.incidence_cache_path).tocsr()
            if incidence.shape[0] != self.n_voxels:
                raise ValueError(
                    f"Cached incidence has {incidence.shape[0]} voxels; "
                    f"expected {self.n_voxels}."
                )
            if self.has_geometry_cache and incidence.shape[1] != int(
                self._geometry_manifest["n_fibers"]
            ):
                raise ValueError("Cached incidence and fiber geometry disagree.")
            self._incidence = incidence.astype(np.float32, copy=False)
            return self._incidence

        fibers = self._load_connectome()
        self.reference_indexer.build_index(fibers)
        mapped_per_fiber = [
            self._masked_voxels(indices)
            for indices in self.reference_indexer.fiber_voxel_indices
        ]
        lengths = np.fromiter(
            (indices.size for indices in mapped_per_fiber),
            dtype=np.int64,
            count=len(mapped_per_fiber),
        )
        indptr = np.empty(lengths.size + 1, dtype=np.int64)
        indptr[0] = 0
        np.cumsum(lengths, out=indptr[1:])
        indices = (
            np.concatenate(mapped_per_fiber).astype(np.int32, copy=False)
            if indptr[-1]
            else np.empty(0, dtype=np.int32)
        )
        fiber_by_voxel = sparse.csr_matrix(
            (np.ones(indices.size, dtype=np.float32), indices, indptr),
            shape=(len(mapped_per_fiber), self.n_voxels),
        )
        incidence = fiber_by_voxel.T.tocsr()
        incidence.sum_duplicates()
        incidence.data.fill(1.0)
        self._incidence = incidence
        self._atomic_save_sparse(self.incidence_cache_path, incidence)
        return incidence

    def project_fiber_scores(self, fiber_scores, reduction="max"):
        """Aggregate connectome-fiber similarities at every masked voxel."""
        fiber_scores = np.asarray(fiber_scores, dtype=np.float32).reshape(-1)
        incidence = self.incidence
        if fiber_scores.size != incidence.shape[1]:
            raise ValueError(
                f"Expected {incidence.shape[1]} fiber scores; "
                f"got {fiber_scores.size}."
            )
        if reduction == "sum":
            return np.asarray(incidence @ fiber_scores).reshape(-1)
        if reduction == "mean":
            output = np.asarray(incidence @ fiber_scores).reshape(-1)
            counts = np.diff(incidence.indptr)
            np.divide(output, counts, out=output, where=counts > 0)
            return output.astype(np.float32, copy=False)
        if reduction != "max":
            raise ValueError("reduction must be 'max', 'mean', or 'sum'.")

        output = np.zeros(self.n_voxels, dtype=np.float32)
        active = np.flatnonzero(np.diff(incidence.indptr) > 0)
        if active.size:
            values = fiber_scores[incidence.indices]
            starts = incidence.indptr[:-1][active]
            output[active] = np.maximum.reduceat(values, starts)
        return output

    def generate_inverted_connectivity_profile(
        self,
        target_fiber_path,
        voxel_reduction="max",
        similarity="padded_cosine",
    ):
        """Return a voxel map of target-trajectory cosine similarity."""
        fiber_scores = self.match_target_fibers(
            target_fiber_path, similarity=similarity
        )
        return self.project_fiber_scores(fiber_scores, reduction=voxel_reduction)

    def save_profile(self, profile, out_path):
        """Save a masked voxelwise trajectory-similarity profile as NIfTI."""
        out_path = Path(out_path)
        if out_path.name.endswith(".nii.gz"):
            output_base = out_path.with_name(out_path.name[:-7])
        elif out_path.suffix == ".nii":
            output_base = out_path.with_suffix("")
        else:
            output_base = out_path
        output_base.parent.mkdir(parents=True, exist_ok=True)
        NiftiIO(mask_path=str(self.mask_path)).save_files(
            arr=np.asarray(profile, dtype=np.float32),
            file_paths=[str(output_base)],
            dry_run=False,
        )
        return output_base.with_name(f"{output_base.name}.nii.gz")

    def save_inverted_profiles_from_fibers(
        self,
        target_fiber_paths,
        out_dir,
        similarity="padded_cosine",
        voxel_reduction="max",
        suffix=None,
    ):
        """Register and save one voxelwise NIfTI per target fiber set."""
        target_fiber_paths = list(target_fiber_paths)
        out_dir = Path(out_dir)
        suffix = suffix or f"_{similarity}_fiber_inversion"
        self.prepare_geometry_cache()
        saved_paths = []

        for target_path in tqdm(
            target_fiber_paths,
            desc="Saving inverted fiber-connectivity profiles",
        ):
            profile = self.generate_inverted_connectivity_profile(
                target_path,
                voxel_reduction=voxel_reduction,
                similarity=similarity,
            )
            output_path = out_dir / (
                f"{self._safe_stem(target_path)}{suffix}.nii.gz"
            )
            saved_paths.append(self.save_profile(profile, output_path))
        return saved_paths


FiberTrajectorySimilarity = InvertedFiberConnectivity
