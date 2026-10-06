import numpy as np

from calvin_utils.neuroimaging_utils.tract_utils.fiber_geometry_registration import (
    GeodesicFiberRegistration,
)
from calvin_utils.neuroimaging_utils.tract_utils.target_trajectory_averaging import (
    TargetTrajectoryAverager,
)


def test_average_orients_and_slides_shorter_target_fiber():
    registration = GeodesicFiberRegistration(sample_interval_mm=1.0)
    averager = TargetTrajectoryAverager(registration, raw_fiber_loader=None)
    longest = np.asarray([[1, 1, 1], [5, 1, 1]], dtype=np.float32)
    shorter_reversed = np.asarray([[4, 1, 1], [2, 1, 1]], dtype=np.float32)

    average = averager.average([longest, shorter_reversed])

    np.testing.assert_allclose(
        average,
        np.asarray(
            [[1, 1, 1], [2, 1, 1], [3, 1, 1], [4, 1, 1], [5, 1, 1]],
            dtype=np.float32,
        ),
    )


def test_averaging_weights_are_binary_or_raw_signed_values():
    values = np.asarray([-2.0, 4.0], dtype=np.float32)

    np.testing.assert_array_equal(
        TargetTrajectoryAverager.averaging_weights(values, "binary"),
        [1.0, 1.0],
    )
    np.testing.assert_array_equal(
        TargetTrajectoryAverager.averaging_weights(values, "weighted"),
        [-2.0, 4.0],
    )


def test_upper_tail_uses_raw_values_not_absolute_magnitude():
    values = np.asarray([-100.0, 1.0, 3.0], dtype=np.float32)

    selected = TargetTrajectoryAverager.upper_tail_mask(values, 5)

    np.testing.assert_array_equal(selected, [False, False, True])
