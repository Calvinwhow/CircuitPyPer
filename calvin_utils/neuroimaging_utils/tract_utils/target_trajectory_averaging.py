"""Create one representative trajectory from a fiber-valued target bundle."""

from pathlib import Path

import numpy as np

from calvin_utils.neuroimaging_utils.tract_utils.fiber_converter import (
    FiberFormatConverter,
)


class TargetTrajectoryAverager:
    """Select, register, and average the trajectories in one target map.

    Fiber-value inputs use their descriptor to recover the canonical ordered
    atlas. Values first select which fibers belong to the target. The surviving
    trajectories are then either averaged equally (``binary``) or using their
    original signed scalar values (``weighted``).

    Fibers may have different vertex counts, endpoint order, and geodesic
    lengths. Each is sampled at a fixed millimeter interval and aligned to the
    longest selected fiber before averaging. Missing portions of shorter fibers
    are excluded from the local denominator rather than represented as zeros.
    """

    VALUE_SUFFIXES = (
        ".fib.npy",
        ".fib.json",
        ".fib.values.npy",
        ".fib.desc.json",
    )
    DEFAULT_BINARY_TOP_PERCENT = 5.0

    def __init__(self, registration, raw_fiber_loader, selected_fiber_loader=None):
        self.registration = registration
        self.raw_fiber_loader = raw_fiber_loader
        self.selected_fiber_loader = selected_fiber_loader

    @classmethod
    def is_fiber_values_path(cls, path):
        return Path(path).name.lower().endswith(cls.VALUE_SUFFIXES)

    @staticmethod
    def upper_tail_mask(values, top_percent):
        """Select the upper tail of raw values without taking magnitudes."""
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        top_percent = float(top_percent)
        if not 0 < top_percent <= 100:
            raise ValueError("top_percent must be in (0, 100].")
        cutoff = np.nanpercentile(values, 100 - top_percent)
        return values >= cutoff

    def load_target(
        self,
        target_path,
        sign="positive",
        min_abs_value=None,
        top_percent=None,
        fiber_atlas_path=None,
    ):
        """Return selected target fibers and their scalar values.

        Raw tractograms and fiber atlases have no selection values, so all of
        their fibers receive a value of one.
        """
        if self.is_fiber_values_path(target_path):
            return self._load_value_target(
                target_path,
                sign=sign,
                min_abs_value=min_abs_value,
                top_percent=top_percent,
                fiber_atlas_path=fiber_atlas_path,
            )

        fibers = self.raw_fiber_loader(str(target_path))
        return fibers, np.ones(len(fibers), dtype=np.float32)

    def _load_value_target(
        self,
        target_path,
        sign,
        min_abs_value,
        top_percent,
        fiber_atlas_path,
    ):
        """Select scalar indices before loading their atlas trajectories."""
        target_path = Path(target_path).expanduser()
        if target_path.name.lower().endswith((".fib.json", ".fib.desc.json")):
            values_path = FiberFormatConverter.values_from_description(target_path)
        else:
            values_path = target_path

        try:
            stored = np.load(values_path, mmap_mode="r", allow_pickle=False)
        except ValueError:
            # Legacy geometry-bearing .fib.npy files store values inside each
            # object and cannot select indices before decoding their geometry.
            fibers, values, _ = FiberFormatConverter.load_fib_npy(
                values_path,
                fiber_atlas_path=fiber_atlas_path,
                sign="both",
                min_abs_value=None,
                top_percent=None,
            )
            values = np.asarray(values, dtype=np.float32)
            keep = FiberFormatConverter._fiber_value_mask(
                values,
                sign=sign,
                min_abs_value=min_abs_value,
                top_percent=None,
            )
            if top_percent is not None:
                keep &= self.upper_tail_mask(values, top_percent)
            return (
                [fiber for fiber, selected in zip(fibers, keep) if selected],
                values[keep],
            )

        if stored.ndim != 1:
            raise ValueError(
                f"Expected a one-dimensional fiber value vector, got {stored.shape}."
            )
        values = np.asarray(stored, dtype=np.float32)
        keep = FiberFormatConverter._fiber_value_mask(
            values,
            sign=sign,
            min_abs_value=min_abs_value,
            top_percent=None,
        )
        if top_percent is not None:
            keep &= self.upper_tail_mask(values, top_percent)
            if not np.any(keep):
                raise ValueError(
                    "No positive fibers survive the raw upper-tail threshold."
                )
        selected_indices = np.flatnonzero(keep)
        selected_values = values[selected_indices]

        description_path = FiberFormatConverter.values_description_path(values_path)
        if values_path.name.lower().endswith(".values.npy") or description_path.is_file():
            atlas_path = FiberFormatConverter.atlas_from_values_description(
                values_path,
                atlas_override=fiber_atlas_path,
            )
        elif fiber_atlas_path is not None:
            atlas_path = Path(fiber_atlas_path).expanduser()
        else:
            raise ValueError(
                f"Fiber values require the paired descriptor: {description_path}."
            )

        if self.selected_fiber_loader is None:
            all_fibers = self.raw_fiber_loader(str(atlas_path))
            fibers = [all_fibers[index] for index in selected_indices]
            if len(all_fibers) != len(values):
                raise ValueError(
                    "Fiber atlas and values vector contain different fiber counts."
                )
        else:
            fibers = self.selected_fiber_loader(
                str(atlas_path),
                selected_indices,
                expected_count=len(values),
            )
        return fibers, selected_values

    @staticmethod
    def averaging_weights(values, weighting):
        """Convert selected fiber values into geometric averaging weights."""
        weighting = str(weighting).strip().lower()
        if weighting == "binary":
            return np.ones(len(values), dtype=np.float32)
        if weighting == "weighted":
            return np.asarray(values, dtype=np.float32)
        raise ValueError("weighting must be 'binary' or 'weighted'.")

    def average(self, fibers, weights=None):
        """Register fibers to the longest member and return their XYZ mean."""
        fibers = list(fibers)
        if not fibers:
            raise ValueError("At least one target fiber is required.")

        if weights is None:
            weights = np.ones(len(fibers), dtype=np.float32)
        else:
            weights = np.asarray(weights, dtype=np.float32).reshape(-1)
            if weights.size != len(fibers):
                raise ValueError("weights must contain one value per target fiber.")
            if not np.all(np.isfinite(weights)):
                raise ValueError("Target-fiber weights must be finite.")

        sampled = []
        sampled_weights = []
        for fiber, weight in zip(fibers, weights):
            trajectory = self.registration.sample(fiber)
            if trajectory.shape[0] == 0 or weight == 0:
                continue
            sampled.append(trajectory)
            sampled_weights.append(float(weight))

        if not sampled:
            raise ValueError("No non-empty, nonzero-weight target fibers remain.")

        anchor_index = int(np.argmax([len(trajectory) for trajectory in sampled]))
        anchor = sampled[anchor_index]
        coordinate_sum = np.zeros(anchor.shape, dtype=np.float32)
        weight_sum = np.zeros(anchor.shape[0], dtype=np.float32)

        for index, (trajectory, weight) in enumerate(zip(sampled, sampled_weights)):
            if index == anchor_index:
                oriented = trajectory
                offset = 0
            else:
                alignment = self.registration.align_sampled(
                    candidate=trajectory,
                    target=anchor,
                )
                oriented = (
                    trajectory[::-1]
                    if alignment.orientation == "reverse"
                    else trajectory
                )
                offset = alignment.candidate_offset_samples

            stop = min(offset + oriented.shape[0], anchor.shape[0])
            available = stop - offset
            coordinate_sum[offset:stop] += oriented[:available] * weight
            weight_sum[offset:stop] += weight

        if np.any(np.isclose(weight_sum, 0)):
            raise ValueError(
                "Signed target weights cancel to zero at one or more samples."
            )
        return (coordinate_sum / weight_sum[:, None]).astype(np.float32)

    def from_path(
        self,
        target_path,
        weighting="binary",
        sign="positive",
        min_abs_value=None,
        top_percent=None,
        fiber_atlas_path=None,
    ):
        """Load a target map and return its registered average trajectory.

        For a fiber-values target, binary averaging defaults to the positive
        members of the upper five percent of all raw values (the
        >=95th-percentile tail).
        Pass ``top_percent=100`` to retain every eligible nonzero fiber.
        Weighted averaging defaults to all eligible nonzero fibers.
        """
        normalized_weighting = str(weighting).strip().lower()
        if (
            normalized_weighting == "binary"
            and top_percent is None
            and self.is_fiber_values_path(target_path)
        ):
            top_percent = self.DEFAULT_BINARY_TOP_PERCENT

        fibers, values = self.load_target(
            target_path,
            sign=sign,
            min_abs_value=min_abs_value,
            top_percent=top_percent,
            fiber_atlas_path=fiber_atlas_path,
        )
        weights = self.averaging_weights(values, normalized_weighting)
        return self.average(fibers, weights=weights)

    def sampled_from_path(
        self,
        target_path,
        weighting="binary",
        sign="positive",
        min_abs_value=None,
        top_percent=None,
        fiber_atlas_path=None,
    ):
        """Return selected sampled trajectories and normalized mean weights.

        Unlike :meth:`from_path`, this preserves every selected trajectory for
        an exact mean of pairwise target/connectome-fiber similarities. It does
        not align target fibers to a shared anchor because each target is
        independently registered to each connectome fiber during matching.
        """
        normalized_weighting = str(weighting).strip().lower()
        if (
            normalized_weighting == "binary"
            and top_percent is None
            and self.is_fiber_values_path(target_path)
        ):
            top_percent = self.DEFAULT_BINARY_TOP_PERCENT

        fibers, values = self.load_target(
            target_path,
            sign=sign,
            min_abs_value=min_abs_value,
            top_percent=top_percent,
            fiber_atlas_path=fiber_atlas_path,
        )
        raw_weights = self.averaging_weights(values, normalized_weighting)
        trajectories = []
        weights = []
        for fiber, weight in zip(fibers, raw_weights):
            sampled = self.registration.sample(fiber, reverse=False)
            if sampled.shape[0] == 0 or weight == 0:
                continue
            trajectories.append(sampled)
            weights.append(float(weight))

        if not trajectories:
            raise ValueError("No non-empty, nonzero-weight target fibers remain.")
        weights = np.asarray(weights, dtype=np.float32)
        weight_sum = float(weights.sum())
        if not np.isfinite(weight_sum) or np.isclose(weight_sum, 0):
            raise ValueError("Target-fiber weights must have a finite nonzero sum.")
        return trajectories, weights / weight_sum


__all__ = ["TargetTrajectoryAverager"]
