"""Fixed-interval streamline sampling and longest-vector registration."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FiberAlignment:
    """A shorter sampled fiber placed inside the longer sampled fiber.

    Uncovered positions in the longer vector are conceptually missing. They are
    never replaced by zero and never participate in the cosine similarity.
    """

    cosine_similarity: float
    rms_distance_mm: float
    orientation: str
    candidate_offset_samples: int
    target_offset_samples: int
    overlap_samples: int
    candidate_samples: int
    target_samples: int
    sample_interval_mm: float

    @property
    def offset_mm(self):
        return (
            self.candidate_offset_samples - self.target_offset_samples
        ) * self.sample_interval_mm


class GeodesicFiberRegistration:
    """Orient once, then shift the shorter fiber within the longer vector."""

    VALID_SIMILARITIES = {"valid_sample_cosine", "overlap_cosine"}

    def __init__(self, sample_interval_mm=1.0, max_length_mm=None):
        self.sample_interval_mm = float(sample_interval_mm)
        self.max_length_mm = (
            None if max_length_mm is None else float(max_length_mm)
        )
        if self.sample_interval_mm <= 0:
            raise ValueError("sample_interval_mm must be greater than zero.")
        if self.max_length_mm is not None and self.max_length_mm <= 0:
            raise ValueError("max_length_mm must be greater than zero.")

    @staticmethod
    def geodesic_length(fiber):
        xyz = np.asarray(fiber, dtype=np.float32)[:, :3]
        xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
        if xyz.shape[0] < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())

    def sample(self, fiber, reverse=False):
        """Sample a polyline once at regular physical intervals."""
        xyz = np.asarray(fiber, dtype=np.float32)[:, :3]
        xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
        if reverse:
            xyz = xyz[::-1]
        if xyz.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float32)
        if xyz.shape[0] == 1:
            return xyz.copy()

        segment_lengths = np.linalg.norm(np.diff(xyz, axis=0), axis=1).astype(
            np.float32, copy=False
        )
        cumulative = np.concatenate(
            (np.zeros(1, dtype=np.float32), np.cumsum(segment_lengths, dtype=np.float32))
        )
        unique = np.concatenate(([True], np.diff(cumulative) > 1e-7))
        xyz = xyz[unique]
        cumulative = cumulative[unique]
        if cumulative.size == 1:
            return xyz.copy()

        sampled_length = float(cumulative[-1])
        if self.max_length_mm is not None:
            sampled_length = min(sampled_length, self.max_length_mm)
        count = int(np.floor(sampled_length / self.sample_interval_mm)) + 1
        distances = (
            np.arange(count, dtype=np.float32) * np.float32(self.sample_interval_mm)
        )
        sampled = np.empty((count, 3), dtype=np.float32)
        for axis in range(3):
            sampled[:, axis] = np.interp(distances, cumulative, xyz[:, axis])
        return sampled

    def sample_both(self, fiber):
        """Compatibility helper; the inversion path samples only once."""
        forward = self.sample(fiber)
        return forward, forward[::-1]

    @staticmethod
    def _validate_samples(samples):
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 2 or samples.shape[1] != 3:
            raise ValueError(
                f"Sampled fiber must have shape (samples, 3); got {samples.shape}."
            )
        if samples.shape[0] == 0:
            raise ValueError("A sampled fiber must contain at least one point.")
        if not np.all(np.isfinite(samples)):
            raise ValueError("Sampled fiber coordinates must be finite.")
        return samples

    @staticmethod
    def _cosine_valid_samples(first, second):
        """Cosine over matched samples only; missing longer-vector slots are ignored."""
        first = np.asarray(first, dtype=np.float32).reshape(-1)
        second = np.asarray(second, dtype=np.float32).reshape(-1)
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        if denominator == 0:
            return 0.0
        return float(np.dot(first, second) / denominator)

    @staticmethod
    def _reverse_from_endpoint_order(shorter, longer):
        """Choose endpoint order from the endpoints' positions along the longer fiber."""
        start_error = np.einsum(
            "ic,ic->i", longer - shorter[0], longer - shorter[0]
        )
        stop_error = np.einsum(
            "ic,ic->i", longer - shorter[-1], longer - shorter[-1]
        )
        start_index = int(np.argmin(start_error))
        stop_index = int(np.argmin(stop_error))
        if start_index != stop_index:
            return start_index > stop_index

        direct = np.sum((shorter[0] - longer[0]) ** 2) + np.sum(
            (shorter[-1] - longer[-1]) ** 2
        )
        reversed_order = np.sum((shorter[-1] - longer[0]) ** 2) + np.sum(
            (shorter[0] - longer[-1]) ** 2
        )
        return bool(reversed_order < direct)

    @classmethod
    def _orient_candidate_once(cls, candidate, target):
        """Orient candidate using endpoint order, without a second shift search."""
        if candidate.shape[0] <= target.shape[0]:
            reverse = cls._reverse_from_endpoint_order(candidate, target)
        else:
            reverse = cls._reverse_from_endpoint_order(target, candidate)
        return (candidate[::-1] if reverse else candidate), reverse

    @staticmethod
    def _sliding_squared_error(shorter, longer):
        """Pointwise float32 SSE for every full placement in the longer vector."""
        shorter_energy = np.einsum("ij,ij->", shorter, shorter)
        point_energy = np.einsum("ij,ij->i", longer, longer)
        cumulative = np.concatenate(
            (np.zeros(1, dtype=np.float32), np.cumsum(point_energy, dtype=np.float32))
        )
        width = shorter.shape[0]
        window_energy = cumulative[width:] - cumulative[:-width]
        cross = np.zeros(longer.shape[0] - width + 1, dtype=np.float32)
        for axis in range(3):
            cross += np.correlate(
                longer[:, axis], shorter[:, axis], mode="valid"
            ).astype(np.float32, copy=False)
        return np.maximum(
            shorter_energy + window_energy - np.float32(2.0) * cross,
            np.float32(0.0),
        )

    def align_sampled(self, candidate, target):
        """Orient once, then place the entire shorter fiber in the longer one."""
        candidate = self._validate_samples(candidate)
        target = self._validate_samples(target)
        candidate, reversed_candidate = self._orient_candidate_once(candidate, target)

        if candidate.shape[0] <= target.shape[0]:
            shorter, longer = candidate, target
            candidate_is_shorter = True
        else:
            shorter, longer = target, candidate
            candidate_is_shorter = False

        errors = self._sliding_squared_error(shorter, longer)
        offset = int(np.argmin(errors))
        if candidate_is_shorter:
            candidate_offset = offset
            target_offset = 0
            candidate_overlap = candidate
            target_overlap = target[offset : offset + candidate.shape[0]]
        else:
            candidate_offset = 0
            target_offset = offset
            candidate_overlap = candidate[offset : offset + target.shape[0]]
            target_overlap = target

        difference = candidate_overlap - target_overlap
        squared_error = float(np.einsum("ij,ij->", difference, difference))
        overlap_samples = shorter.shape[0]
        return FiberAlignment(
            cosine_similarity=self._cosine_valid_samples(
                candidate_overlap, target_overlap
            ),
            rms_distance_mm=float(np.sqrt(squared_error / overlap_samples)),
            orientation="reverse" if reversed_candidate else "forward",
            candidate_offset_samples=candidate_offset,
            target_offset_samples=target_offset,
            overlap_samples=overlap_samples,
            candidate_samples=candidate.shape[0],
            target_samples=target.shape[0],
            sample_interval_mm=self.sample_interval_mm,
        )

    @staticmethod
    def _batch_reverse_mask(candidates, target):
        """Determine one orientation per candidate from endpoint geodesic order."""
        candidate_length = candidates.shape[1]
        target_length = target.shape[0]
        if candidate_length <= target_length:
            target_energy = np.einsum("ic,ic->i", target, target)
            start = candidates[:, 0]
            stop = candidates[:, -1]
            start_error = (
                np.einsum("bc,bc->b", start, start)[:, None]
                + target_energy[None, :]
                - np.float32(2.0) * (start @ target.T)
            )
            start_index = np.argmin(start_error, axis=1)
            del start_error
            stop_error = (
                np.einsum("bc,bc->b", stop, stop)[:, None]
                + target_energy[None, :]
                - np.float32(2.0) * (stop @ target.T)
            )
            stop_index = np.argmin(stop_error, axis=1)
        else:
            candidate_energy = np.einsum("bic,bic->bi", candidates, candidates)
            start_error = (
                candidate_energy
                + np.einsum("c,c->", target[0], target[0])
                - np.float32(2.0) * np.einsum("bic,c->bi", candidates, target[0])
            )
            start_index = np.argmin(start_error, axis=1)
            del start_error
            stop_error = (
                candidate_energy
                + np.einsum("c,c->", target[-1], target[-1])
                - np.float32(2.0) * np.einsum("bic,c->bi", candidates, target[-1])
            )
            stop_index = np.argmin(stop_error, axis=1)

        reverse = start_index > stop_index
        ties = start_index == stop_index
        if np.any(ties):
            direct = np.einsum(
                "bc,bc->b", candidates[:, 0] - target[0], candidates[:, 0] - target[0]
            ) + np.einsum(
                "bc,bc->b", candidates[:, -1] - target[-1], candidates[:, -1] - target[-1]
            )
            reversed_order = np.einsum(
                "bc,bc->b", candidates[:, -1] - target[0], candidates[:, -1] - target[0]
            ) + np.einsum(
                "bc,bc->b", candidates[:, 0] - target[-1], candidates[:, 0] - target[-1]
            )
            reverse[ties] = reversed_order[ties] < direct[ties]
        return reverse

    def score_sampled_batch(
        self,
        candidate_samples,
        target,
        similarity="valid_sample_cosine",
        candidate_full_energy=None,
    ):
        """Orient once and broadcast one shift search over equal-length fibers."""
        if similarity not in self.VALID_SIMILARITIES:
            raise ValueError(
                "similarity must be 'valid_sample_cosine' "
                "('overlap_cosine' is accepted as an alias)."
            )
        candidates = np.asarray(candidate_samples, dtype=np.float32)
        target = self._validate_samples(target)
        if candidates.ndim != 3 or candidates.shape[2] != 3:
            raise ValueError("candidate_samples must have shape (fibers, samples, 3).")
        if candidate_full_energy is not None:
            candidate_full_energy = np.asarray(
                candidate_full_energy, dtype=np.float32
            ).reshape(-1)
            if candidate_full_energy.shape[0] != candidates.shape[0]:
                raise ValueError(
                    "candidate_full_energy must contain one value per fiber."
                )

        reverse = self._batch_reverse_mask(candidates, target)
        if not np.any(reverse):
            oriented = candidates
        elif np.all(reverse):
            oriented = candidates[:, ::-1]
        else:
            oriented = candidates.copy()
            oriented[reverse] = candidates[reverse, ::-1]
        candidate_length = oriented.shape[1]
        target_length = target.shape[0]

        if candidate_length <= target_length:
            windows = np.lib.stride_tricks.sliding_window_view(
                target, candidate_length, axis=0
            ).transpose(0, 2, 1)
            candidate_energy = (
                candidate_full_energy
                if candidate_full_energy is not None
                else np.einsum("bic,bic->b", oriented, oriented)
            )
            window_energy = np.einsum("dic,dic->d", windows, windows)
            cross = np.einsum("bic,dic->bd", oriented, windows)
            cross *= np.float32(-2.0)
            cross += candidate_energy[:, None]
            cross += window_energy[None, :]
            errors = np.maximum(cross, np.float32(0.0), out=cross)
            offset = errors.argmin(axis=1)
            rows = np.arange(oriented.shape[0])
            selected_windows = windows[offset]
            numerator = np.einsum("bic,bic->b", oriented, selected_windows)
            first_energy = candidate_energy
            second_energy = window_energy[offset]
        else:
            windows = np.lib.stride_tricks.sliding_window_view(
                oriented, target_length, axis=1
            ).transpose(0, 1, 3, 2)
            window_energy = np.einsum("bdic,bdic->bd", windows, windows)
            target_energy = np.einsum("ic,ic->", target, target)
            cross = np.einsum("bdic,ic->bd", windows, target)
            cross *= np.float32(-2.0)
            cross += window_energy
            cross += target_energy
            errors = np.maximum(cross, np.float32(0.0), out=cross)
            offset = errors.argmin(axis=1)
            rows = np.arange(oriented.shape[0])
            selected_windows = windows[rows, offset]
            numerator = np.einsum("bic,ic->b", selected_windows, target)
            first_energy = window_energy[rows, offset]
            second_energy = np.full(
                oriented.shape[0], target_energy, dtype=np.float32
            )

        denominator = np.sqrt(first_energy * second_energy)
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=np.float32),
            where=denominator > 0,
        ).astype(np.float32, copy=False)

    def align(self, candidate_fiber, target_fiber):
        """Sample and register two raw streamline polylines."""
        return self.align_sampled(
            candidate=self.sample(candidate_fiber),
            target=self.sample(target_fiber),
        )


FiberGeometryRegistration = GeodesicFiberRegistration


__all__ = [
    "FiberAlignment",
    "FiberGeometryRegistration",
    "GeodesicFiberRegistration",
]
