"""Single-pass matching of explicitly voxel-seeded structural fibers."""

import numpy as np
from tqdm import tqdm


class StreamingFiberInverter:
    """Broadcast target matching and average scores within each seed voxel.

    Every input fiber is paired with the voxel from which tractography generated
    it. Its similarity contributes only to that seed voxel, never to voxels
    traversed later by the streamline.
    """

    def __init__(
        self,
        registration,
        seed_voxel_mapper,
        n_voxels,
        fiber_batch_size=4096,
        seed_voxel_batch_mapper=None,
        memory_budget_mb=256,
        max_fiber_batch_size=16_384,
    ):
        self.registration = registration
        self.seed_voxel_mapper = seed_voxel_mapper
        self.n_voxels = int(n_voxels)
        self.fiber_batch_size = int(fiber_batch_size)
        self.seed_voxel_batch_mapper = seed_voxel_batch_mapper
        self.memory_budget_bytes = int(float(memory_budget_mb) * 1024**2)
        self.max_fiber_batch_size = int(max_fiber_batch_size)
        if self.fiber_batch_size < 1:
            raise ValueError("fiber_batch_size must be at least one.")
        if self.memory_budget_bytes < 1:
            raise ValueError("memory_budget_mb must be greater than zero.")
        if self.max_fiber_batch_size < 1:
            raise ValueError("max_fiber_batch_size must be at least one.")

    @staticmethod
    def _normalize_targets(target_trajectories, target_weights):
        target_entries = list(target_trajectories)
        if not target_entries:
            raise ValueError("At least one target trajectory set is required.")
        if target_weights is None:
            target_weights = [None] * len(target_entries)
        else:
            target_weights = list(target_weights)
            if len(target_weights) != len(target_entries):
                raise ValueError("target_weights must contain one entry per target.")

        targets = []
        for entry, weights in zip(target_entries, target_weights):
            if (
                isinstance(entry, np.ndarray)
                and entry.ndim == 2
                and entry.shape[1] == 3
            ):
                trajectories = [np.asarray(entry, dtype=np.float32)]
            else:
                trajectories = [
                    np.asarray(trajectory, dtype=np.float32)
                    for trajectory in entry
                ]
            if not trajectories:
                raise ValueError("Every target must contain at least one trajectory.")
            if weights is None:
                normalized_weights = np.full(
                    len(trajectories), 1.0 / len(trajectories), dtype=np.float32
                )
            else:
                normalized_weights = np.asarray(weights, dtype=np.float32).reshape(-1)
                if normalized_weights.size != len(trajectories):
                    raise ValueError(
                        "Each target weight vector must match its trajectory count."
                    )
                weight_sum = float(normalized_weights.sum())
                if not np.isfinite(weight_sum) or np.isclose(weight_sum, 0):
                    raise ValueError("Target weights must have a finite nonzero sum.")
                normalized_weights = normalized_weights / weight_sum
            targets.append((trajectories, normalized_weights))
        return targets

    def _map_seed_batch(self, seed_voxels):
        seed_voxels = np.asarray(seed_voxels, dtype=np.int64)
        if self.seed_voxel_batch_mapper is not None:
            return np.asarray(
                self.seed_voxel_batch_mapper(seed_voxels), dtype=np.int64
            )
        return np.fromiter(
            (self.seed_voxel_mapper(seed) for seed in seed_voxels),
            dtype=np.int64,
            count=seed_voxels.size,
        )

    def _adaptive_batch_size(self, sample_count, targets):
        """Bound temporary fiber-by-shift arrays by the configured memory budget."""
        bytes_per_fiber = 0
        for trajectories, _ in targets:
            for target in trajectories:
                target_count = int(target.shape[0])
                shifts = abs(sample_count - target_count) + 1
                endpoint_search = max(sample_count, target_count)
                estimate = np.dtype(np.float32).itemsize * (
                    2 * shifts
                    + 2 * endpoint_search
                    + 6 * sample_count
                    + 32
                )
                bytes_per_fiber = max(bytes_per_fiber, estimate)
        return max(
            1,
            min(
                self.max_fiber_batch_size,
                self.memory_budget_bytes // max(bytes_per_fiber, 1),
            ),
        )

    @staticmethod
    def _segment_layout(sorted_voxels):
        starts = np.concatenate(
            (
                np.zeros(1, dtype=np.int64),
                np.flatnonzero(np.diff(sorted_voxels)) + 1,
            )
        )
        unique_voxels = sorted_voxels[starts]
        counts = np.diff(
            np.concatenate((starts, np.asarray([sorted_voxels.size], dtype=np.int64)))
        )
        return starts, unique_voxels, counts

    def run(
        self,
        seeded_fibers,
        target_trajectories,
        target_weights=None,
        similarity="valid_sample_cosine",
    ):
        """Return voxel profiles without materializing pairwise similarities.

        Each target entry may be one average trajectory or a sequence of
        individual bundle trajectories. In the latter case, similarities are
        reduced to their weighted mean immediately for each connectome fiber.
        """
        targets = self._normalize_targets(target_trajectories, target_weights)
        profiles = np.zeros((len(targets), self.n_voxels), dtype=np.float32)
        counts = np.zeros(self.n_voxels, dtype=np.int64)

        seed_buffer = []
        fiber_buffer = []

        def process_batch():
            if not fiber_buffer:
                return
            sampled = []
            mapped_voxels = []
            for seed_voxel, fiber in zip(seed_buffer, fiber_buffer):
                masked_voxel = self.seed_voxel_mapper(seed_voxel)
                if masked_voxel < 0:
                    continue
                trajectory = self.registration.sample(fiber)
                if trajectory.shape[0] == 0:
                    continue
                sampled.append(trajectory)
                mapped_voxels.append(masked_voxel)

            if not sampled:
                seed_buffer.clear()
                fiber_buffer.clear()
                return

            mapped_voxels_array = np.asarray(mapped_voxels, dtype=np.int64)
            scores = np.zeros((len(targets), len(sampled)), dtype=np.float32)
            sample_lengths = np.asarray(
                [trajectory.shape[0] for trajectory in sampled], dtype=np.int64
            )
            for sample_length in np.unique(sample_lengths):
                group_indices = np.flatnonzero(sample_lengths == sample_length)
                trajectory_batch = np.stack(
                    [sampled[index] for index in group_indices]
                ).astype(np.float32, copy=False)
                for target_index, (trajectories, weights) in enumerate(targets):
                    group_scores = np.zeros(group_indices.size, dtype=np.float32)
                    for target, weight in zip(trajectories, weights):
                        group_scores += weight * self.registration.score_sampled_batch(
                            trajectory_batch,
                            target,
                            similarity=similarity,
                        )
                    scores[target_index, group_indices] = group_scores

            for target_index in range(len(targets)):
                np.add.at(
                    profiles[target_index],
                    mapped_voxels_array,
                    scores[target_index],
                )
            counts[:] += np.bincount(
                mapped_voxels_array, minlength=self.n_voxels
            )
            seed_buffer.clear()
            fiber_buffer.clear()

        for seed_voxel, fiber in tqdm(
            seeded_fibers,
            desc="Streaming voxel-seeded fiber inversion",
        ):
            seed_buffer.append(seed_voxel)
            fiber_buffer.append(fiber)
            if len(fiber_buffer) >= self.fiber_batch_size:
                process_batch()

        process_batch()

        np.divide(
            profiles,
            counts[None, :],
            out=profiles,
            where=counts[None, :] > 0,
        )
        return profiles.astype(np.float32, copy=False)

    def run_sampled_batches(
        self,
        sampled_batches,
        target_trajectories,
        target_weights=None,
        similarity="valid_sample_cosine",
    ):
        """Score inversion-ready memory maps with adaptive, segmented batches."""
        targets = self._normalize_targets(target_trajectories, target_weights)
        profiles = np.zeros((len(targets), self.n_voxels), dtype=np.float32)
        counts = np.zeros(self.n_voxels, dtype=np.int64)

        for stored_batch in tqdm(
            sampled_batches,
            desc="Memory-mapped voxel-seeded fiber inversion",
        ):
            trajectories = np.asarray(stored_batch.trajectories, dtype=np.float32)
            full_energy = np.asarray(stored_batch.full_energy, dtype=np.float32)
            mapped_voxels = self._map_seed_batch(stored_batch.seed_voxels)
            valid = mapped_voxels >= 0
            if not np.any(valid):
                continue
            if not np.all(valid):
                trajectories = trajectories[valid]
                full_energy = full_energy[valid]
                mapped_voxels = mapped_voxels[valid]

            if np.any(np.diff(mapped_voxels) < 0):
                order = np.argsort(mapped_voxels, kind="stable")
                trajectories = trajectories[order]
                full_energy = full_energy[order]
                mapped_voxels = mapped_voxels[order]

            batch_size = self._adaptive_batch_size(
                int(trajectories.shape[1]), targets
            )
            for start in range(0, trajectories.shape[0], batch_size):
                stop = min(start + batch_size, trajectories.shape[0])
                trajectory_batch = trajectories[start:stop]
                energy_batch = full_energy[start:stop]
                voxel_batch = mapped_voxels[start:stop]
                segment_starts, unique_voxels, segment_counts = self._segment_layout(
                    voxel_batch
                )

                for target_index, (target_set, weights) in enumerate(targets):
                    fiber_scores = np.zeros(
                        trajectory_batch.shape[0], dtype=np.float32
                    )
                    for target, weight in zip(target_set, weights):
                        fiber_scores += (
                            weight
                            * self.registration.score_sampled_batch(
                                trajectory_batch,
                                target,
                                similarity=similarity,
                                candidate_full_energy=energy_batch,
                            )
                        )
                    profiles[target_index, unique_voxels] += np.add.reduceat(
                        fiber_scores, segment_starts
                    )
                counts[unique_voxels] += segment_counts

        np.divide(
            profiles,
            counts[None, :],
            out=profiles,
            where=counts[None, :] > 0,
        )
        return profiles


__all__ = ["StreamingFiberInverter"]
