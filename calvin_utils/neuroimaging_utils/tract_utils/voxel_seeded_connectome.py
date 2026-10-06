"""Precompute and stream fibers generated independently from every seed voxel."""

import json
import os
import tempfile
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm

from calvin_utils.neuroimaging_utils.tract_utils.connectome_seed import (
    ConnectomeSeed,
)
from calvin_utils.neuroimaging_utils.tract_utils.fiber_geometry_registration import (
    GeodesicFiberRegistration,
)
from calvin_utils.neuroimaging_utils.tract_utils.sampled_voxel_fiber_store import (
    SampledVoxelFiberStore,
)


class VoxelSeededFiberStore:
    """Disk-backed ragged streamlines with explicit seed-voxel ownership."""

    FORMAT = "calvin_utils.voxel_seeded_fibers.v1"

    def __init__(self, path):
        self.path = Path(path).expanduser()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        with h5py.File(self.path, "r") as store:
            if store.attrs.get("format") != self.FORMAT:
                raise ValueError(
                    f"Not a voxel-seeded fiber store: {self.path}"
                )
            self.reference_shape = tuple(
                int(value) for value in store.attrs["reference_shape"]
            )
            self.reference_affine = np.asarray(
                store.attrs["reference_affine"], dtype=np.float64
            )
            self.n_fibers = int(store["seed_voxel_indices"].shape[0])
            self.n_points = int(store["points"].shape[0])

    def validate_reference(self, reference_img):
        if (
            tuple(reference_img.shape[:3]) != self.reference_shape
            or not np.allclose(reference_img.affine, self.reference_affine)
        ):
            raise ValueError(
                "The voxel-seeded connectome and reference NIfTI must have "
                "matching shape and affine."
            )

    def iter_seeded_fibers(self, fiber_batch_size=4096):
        """Yield ``(full_linear_seed_voxel, fiber_xyz)`` without loading all fibers."""
        fiber_batch_size = int(fiber_batch_size)
        if fiber_batch_size < 1:
            raise ValueError("fiber_batch_size must be at least one.")
        with h5py.File(self.path, "r") as store:
            points = store["points"]
            offsets = store["offsets"]
            seed_voxels = store["seed_voxel_indices"]
            for batch_start in range(0, seed_voxels.shape[0], fiber_batch_size):
                batch_stop = min(
                    batch_start + fiber_batch_size, seed_voxels.shape[0]
                )
                batch_offsets = np.asarray(
                    offsets[batch_start : batch_stop + 1], dtype=np.int64
                )
                point_start = int(batch_offsets[0])
                batch_points = np.asarray(
                    points[point_start : int(batch_offsets[-1])],
                    dtype=np.float32,
                )
                batch_seeds = np.asarray(
                    seed_voxels[batch_start:batch_stop], dtype=np.int64
                )
                batch_offsets -= point_start
                for local_index, seed_voxel in enumerate(batch_seeds):
                    start = int(batch_offsets[local_index])
                    stop = int(batch_offsets[local_index + 1])
                    yield int(seed_voxel), batch_points[start:stop]

    @classmethod
    def write(
        cls,
        path,
        seeded_fibers,
        reference_img,
        metadata=None,
        point_chunk_size=262_144,
    ):
        """Stream seeded fibers into one chunked HDF5 file atomically."""
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp.h5",
        )
        os.close(file_descriptor)
        temporary_path = Path(temporary_name)
        try:
            with h5py.File(temporary_path, "w") as store:
                store.attrs["format"] = cls.FORMAT
                store.attrs["reference_shape"] = reference_img.shape[:3]
                store.attrs["reference_affine"] = reference_img.affine
                store.attrs["metadata_json"] = json.dumps(metadata or {})
                points = store.create_dataset(
                    "points",
                    shape=(0, 3),
                    maxshape=(None, 3),
                    dtype=np.float32,
                    chunks=(int(point_chunk_size), 3),
                )
                offsets = store.create_dataset(
                    "offsets",
                    shape=(1,),
                    maxshape=(None,),
                    dtype=np.int64,
                    chunks=True,
                )
                offsets[0] = 0
                seed_voxels = store.create_dataset(
                    "seed_voxel_indices",
                    shape=(0,),
                    maxshape=(None,),
                    dtype=np.int64,
                    chunks=True,
                )

                point_count = 0
                fiber_count = 0
                buffered_points = []
                buffered_lengths = []
                buffered_seeds = []
                buffered_point_count = 0

                def flush():
                    nonlocal point_count, fiber_count, buffered_point_count
                    if not buffered_lengths:
                        return
                    point_batch = np.vstack(buffered_points).astype(
                        np.float32, copy=False
                    )
                    batch_fiber_count = len(buffered_lengths)
                    next_point_count = point_count + point_batch.shape[0]
                    next_fiber_count = fiber_count + batch_fiber_count
                    points.resize((next_point_count, 3))
                    points[point_count:next_point_count] = point_batch
                    seed_voxels.resize((next_fiber_count,))
                    seed_voxels[fiber_count:next_fiber_count] = buffered_seeds
                    offsets.resize((next_fiber_count + 1,))
                    offsets[fiber_count + 1 : next_fiber_count + 1] = (
                        point_count + np.cumsum(buffered_lengths, dtype=np.int64)
                    )
                    point_count = next_point_count
                    fiber_count = next_fiber_count
                    buffered_points.clear()
                    buffered_lengths.clear()
                    buffered_seeds.clear()
                    buffered_point_count = 0

                for seed_voxel, fiber in seeded_fibers:
                    fiber = np.asarray(fiber, dtype=np.float32)
                    if fiber.ndim != 2 or fiber.shape[1] < 3 or fiber.shape[0] < 2:
                        raise ValueError(
                            "Every seeded fiber must have shape (vertices, >=3)."
                        )
                    fiber = fiber[:, :3]
                    buffered_points.append(fiber)
                    buffered_lengths.append(fiber.shape[0])
                    buffered_seeds.append(int(seed_voxel))
                    buffered_point_count += fiber.shape[0]
                    if buffered_point_count >= point_chunk_size:
                        flush()

                flush()

                store.attrs["n_fibers"] = fiber_count
                store.attrs["n_points"] = point_count
            os.replace(temporary_path, path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        return cls(path)


class VoxelSeededConnectomeBuilder:
    """Track a fixed fiber bundle from every active reference-mask voxel."""

    def __init__(
        self,
        reconstruction_path,
        reference_nifti_path,
        mask_path,
        fibers_per_voxel=8,
        max_attempts_per_voxel=64,
        random_seed=42,
        fa_threshold=0.03,
        angle_threshold=60.0,
        step_size=1.0,
        max_steps=250,
        min_length=10.0,
    ):
        self.reconstruction_path = Path(reconstruction_path).expanduser()
        self.reference_nifti_path = Path(reference_nifti_path).expanduser()
        self.mask_path = Path(mask_path).expanduser()
        self.reference_img = nib.load(str(self.reference_nifti_path))
        self.mask_img = nib.load(str(self.mask_path))
        if (
            self.mask_img.shape[:3] != self.reference_img.shape[:3]
            or not np.allclose(self.mask_img.affine, self.reference_img.affine)
        ):
            raise ValueError("mask_path and reference_nifti_path must match.")
        suffixes = [suffix.lower() for suffix in self.reconstruction_path.suffixes]
        if suffixes[-1:] != [".fib"] and suffixes[-2:] != [".fib", ".gz"]:
            raise ValueError(
                "Voxelwise tracking requires a DSI Studio .fib or .fib.gz "
                "reconstruction, not a flat tractogram."
            )
        self.fibers_per_voxel = int(fibers_per_voxel)
        self.max_attempts_per_voxel = int(max_attempts_per_voxel)
        self.random_seed = int(random_seed)
        if self.fibers_per_voxel < 1:
            raise ValueError("fibers_per_voxel must be at least one.")
        if self.max_attempts_per_voxel < self.fibers_per_voxel:
            raise ValueError(
                "max_attempts_per_voxel must be at least fibers_per_voxel."
            )

        self.tracker = ConnectomeSeed(
            connectome_path=self.reconstruction_path,
            auto_run=False,
            load_connectome=False,
            random_seed=self.random_seed,
            fa_threshold=fa_threshold,
            angle_threshold=angle_threshold,
            step_size=step_size,
            max_steps=max_steps,
            min_length=min_length,
            show_progress=False,
        )
        self.fib = self.tracker._load_fib_fields(self.reconstruction_path)

    @property
    def active_voxel_indices(self):
        mask = self.mask_img.get_fdata()
        return np.flatnonzero(np.isfinite(mask.reshape(-1)) & (mask.reshape(-1) > 0))

    def iter_seeded_fibers(self):
        """Yield successful streamlines grouped by their originating voxel."""
        shape = self.reference_img.shape[:3]
        for full_index in tqdm(
            self.active_voxel_indices,
            desc="Tracking voxel-seeded structural connectome",
            unit="voxel",
        ):
            ijk = np.asarray(np.unravel_index(full_index, shape), dtype=np.float32)
            world = nib.affines.apply_affine(self.reference_img.affine, ijk)
            base = nib.affines.apply_affine(self.fib["inv_affine"], world)
            if not self.tracker._fib_in_bounds(base[None, :], self.fib["dim"])[0]:
                continue
            rng = np.random.default_rng(
                np.random.SeedSequence([self.random_seed, int(full_index)])
            )
            accepted = 0
            for _ in range(self.max_attempts_per_voxel):
                start = base + rng.uniform(-0.5, 0.5, size=3).astype(np.float32)
                fiber = self.tracker._track_fib_streamline(self.fib, start, rng)
                if fiber is None:
                    continue
                yield int(full_index), fiber
                accepted += 1
                if accepted >= self.fibers_per_voxel:
                    break

    def _metadata(self):
        return {
            "reconstruction_path": str(self.reconstruction_path.resolve()),
            "reference_nifti_path": str(self.reference_nifti_path.resolve()),
            "mask_path": str(self.mask_path.resolve()),
            "fibers_per_voxel": self.fibers_per_voxel,
            "max_attempts_per_voxel": self.max_attempts_per_voxel,
            "random_seed": self.random_seed,
            "fa_threshold": self.tracker.fa_threshold,
            "angle_threshold": self.tracker.angle_threshold,
            "step_size": self.tracker.step_size,
            "max_steps": self.tracker.max_steps,
            "min_length": self.tracker.min_length,
        }

    def build(self, output_path):
        """Write the legacy raw-trajectory HDF5 store."""
        return VoxelSeededFiberStore.write(
            output_path,
            self.iter_seeded_fibers(),
            self.reference_img,
            metadata=self._metadata(),
        )

    def build_sampled(
        self,
        output_path,
        sample_interval_mm=1.0,
        max_length_mm=None,
        max_buffer_points=2_097_152,
    ):
        """Track and directly write inversion-ready memory-mapped shards."""
        registration = GeodesicFiberRegistration(
            sample_interval_mm=sample_interval_mm,
            max_length_mm=max_length_mm,
        )
        metadata = self._metadata()
        metadata["sample_interval_mm"] = registration.sample_interval_mm
        metadata["max_length_mm"] = registration.max_length_mm
        return SampledVoxelFiberStore.write(
            output_path,
            self.iter_seeded_fibers(),
            self.reference_img,
            registration=registration,
            metadata=metadata,
            max_buffer_points=max_buffer_points,
        )


__all__ = [
    "SampledVoxelFiberStore",
    "VoxelSeededConnectomeBuilder",
    "VoxelSeededFiberStore",
]
