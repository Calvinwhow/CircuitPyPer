"""Memory-mapped, length-bucketed trajectories for repeated fiber inversion."""

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SampledFiberBatch:
    """One directly broadcastable block of equal-length sampled fibers."""

    trajectories: np.ndarray
    seed_voxels: np.ndarray
    full_energy: np.ndarray

    @property
    def sample_count(self):
        return int(self.trajectories.shape[1])


class SampledVoxelFiberStore:
    """Directory store of float32 trajectory shards grouped by sample count."""

    FORMAT = "calvin_utils.sampled_voxel_fibers.v1"
    MANIFEST_NAME = "manifest.json"

    def __init__(self, path):
        self.path = Path(path).expanduser()
        manifest_path = self.path / self.MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        with manifest_path.open() as manifest_file:
            manifest = json.load(manifest_file)
        if manifest.get("format") != self.FORMAT:
            raise ValueError(f"Not a sampled voxel-fiber store: {self.path}")

        self.reference_shape = tuple(int(value) for value in manifest["reference_shape"])
        self.reference_affine = np.asarray(
            manifest["reference_affine"], dtype=np.float64
        )
        self.sample_interval_mm = float(manifest["sample_interval_mm"])
        stored_max_length = manifest.get("max_length_mm")
        self.max_length_mm = (
            None if stored_max_length is None else float(stored_max_length)
        )
        self.n_fibers = int(manifest["n_fibers"])
        self.n_points = int(manifest["n_sampled_points"])
        self.metadata = manifest.get("metadata", {})
        self.buckets = tuple(
            sorted(manifest["buckets"], key=lambda bucket: int(bucket["sample_count"]))
        )

    @classmethod
    def is_store(cls, path):
        path = Path(path).expanduser()
        return path.is_dir() and (path / cls.MANIFEST_NAME).is_file()

    def validate_reference(self, reference_img):
        if (
            tuple(reference_img.shape[:3]) != self.reference_shape
            or not np.allclose(reference_img.affine, self.reference_affine)
        ):
            raise ValueError(
                "The sampled voxel-fiber store and reference NIfTI must have "
                "matching shape and affine."
            )

    def validate_sampling(self, registration):
        if not np.isclose(
            registration.sample_interval_mm, self.sample_interval_mm
        ) or registration.max_length_mm != self.max_length_mm:
            raise ValueError(
                "Inversion sampling must match the precomputed store: "
                f"sample_interval_mm={self.sample_interval_mm}, "
                f"max_length_mm={self.max_length_mm}."
            )

    def iter_batches(self, max_fibers_per_batch=16_384):
        """Yield memory-mapped, equal-length arrays without rebuilding fibers."""
        max_fibers_per_batch = int(max_fibers_per_batch)
        if max_fibers_per_batch < 1:
            raise ValueError("max_fibers_per_batch must be at least one.")

        for bucket in self.buckets:
            for shard in bucket["shards"]:
                trajectories = np.load(
                    self.path / shard["trajectories"],
                    mmap_mode="r",
                    allow_pickle=False,
                )
                seed_voxels = np.load(
                    self.path / shard["seed_voxels"],
                    mmap_mode="r",
                    allow_pickle=False,
                )
                full_energy = np.load(
                    self.path / shard["full_energy"],
                    mmap_mode="r",
                    allow_pickle=False,
                )
                if (
                    trajectories.ndim != 3
                    or trajectories.shape[2] != 3
                    or trajectories.dtype != np.float32
                    or seed_voxels.shape != (trajectories.shape[0],)
                    or full_energy.shape != (trajectories.shape[0],)
                ):
                    raise ValueError(f"Invalid sampled-fiber shard: {shard}")

                for start in range(0, trajectories.shape[0], max_fibers_per_batch):
                    stop = min(start + max_fibers_per_batch, trajectories.shape[0])
                    yield SampledFiberBatch(
                        trajectories=trajectories[start:stop],
                        seed_voxels=seed_voxels[start:stop],
                        full_energy=full_energy[start:stop],
                    )

    @classmethod
    def write(
        cls,
        path,
        seeded_fibers,
        reference_img,
        registration,
        metadata=None,
        max_buffer_points=2_097_152,
    ):
        """Sample once and atomically write length-bucketed NumPy shards."""
        path = Path(path).expanduser()
        if path.exists():
            raise FileExistsError(
                f"Output already exists: {path}. Choose a new precompute directory."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = Path(
            tempfile.mkdtemp(dir=path.parent, prefix=f".{path.name}.")
        )
        max_buffer_points = int(max_buffer_points)
        if max_buffer_points < 1:
            raise ValueError("max_buffer_points must be at least one.")

        buffers = {}
        buffered_points = 0
        shard_numbers = {}
        bucket_records = {}
        n_fibers = 0
        n_points = 0

        def flush_length(sample_count):
            nonlocal buffered_points, n_fibers, n_points
            bucket = buffers.pop(sample_count, None)
            if not bucket:
                return
            trajectories = np.stack(bucket["trajectories"]).astype(
                np.float32, copy=False
            )
            seed_voxels = np.asarray(bucket["seed_voxels"], dtype=np.int64)
            order = np.argsort(seed_voxels, kind="stable")
            trajectories = trajectories[order]
            seed_voxels = seed_voxels[order]
            full_energy = np.einsum(
                "bic,bic->b", trajectories, trajectories
            ).astype(np.float32, copy=False)

            shard_number = shard_numbers.get(sample_count, 0)
            prefix = f"length_{sample_count:06d}_shard_{shard_number:06d}"
            trajectory_name = f"{prefix}.trajectories.npy"
            seed_name = f"{prefix}.seed_voxels.npy"
            energy_name = f"{prefix}.full_energy.npy"
            np.save(temporary_path / trajectory_name, trajectories, allow_pickle=False)
            np.save(temporary_path / seed_name, seed_voxels, allow_pickle=False)
            np.save(temporary_path / energy_name, full_energy, allow_pickle=False)

            record = bucket_records.setdefault(
                sample_count,
                {"sample_count": sample_count, "n_fibers": 0, "shards": []},
            )
            record["n_fibers"] += int(trajectories.shape[0])
            record["shards"].append(
                {
                    "trajectories": trajectory_name,
                    "seed_voxels": seed_name,
                    "full_energy": energy_name,
                    "n_fibers": int(trajectories.shape[0]),
                }
            )
            shard_numbers[sample_count] = shard_number + 1
            points_written = int(trajectories.shape[0] * sample_count)
            buffered_points -= points_written
            n_fibers += int(trajectories.shape[0])
            n_points += points_written

        try:
            for seed_voxel, fiber in seeded_fibers:
                trajectory = registration.sample(fiber)
                if trajectory.shape[0] == 0:
                    continue
                sample_count = int(trajectory.shape[0])
                bucket = buffers.setdefault(
                    sample_count, {"trajectories": [], "seed_voxels": []}
                )
                bucket["trajectories"].append(trajectory)
                bucket["seed_voxels"].append(int(seed_voxel))
                buffered_points += sample_count

                if buffered_points >= max_buffer_points:
                    largest_length = max(
                        buffers,
                        key=lambda length: length * len(buffers[length]["trajectories"]),
                    )
                    flush_length(largest_length)

            for sample_count in sorted(tuple(buffers)):
                flush_length(sample_count)

            manifest = {
                "format": cls.FORMAT,
                "reference_shape": [int(value) for value in reference_img.shape[:3]],
                "reference_affine": np.asarray(reference_img.affine).tolist(),
                "sample_interval_mm": registration.sample_interval_mm,
                "max_length_mm": registration.max_length_mm,
                "n_fibers": n_fibers,
                "n_sampled_points": n_points,
                "metadata": metadata or {},
                "buckets": [
                    bucket_records[length] for length in sorted(bucket_records)
                ],
            }
            manifest_path = temporary_path / cls.MANIFEST_NAME
            with manifest_path.open("w") as manifest_file:
                json.dump(manifest, manifest_file, indent=2)
            os.replace(temporary_path, path)
        except Exception:
            shutil.rmtree(temporary_path, ignore_errors=True)
            raise
        return cls(path)


__all__ = ["SampledFiberBatch", "SampledVoxelFiberStore"]
