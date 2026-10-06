import numpy as np

from calvin_utils.neuroimaging_utils.tract_utils.fiber_geometry_registration import (
    GeodesicFiberRegistration,
)


def test_sliding_registration_finds_reversed_subtrajectory():
    registration = GeodesicFiberRegistration(sample_interval_mm=1.0)
    candidate = np.asarray([[1, 1, 1], [5, 1, 1]], dtype=np.float32)
    target = np.asarray(
        [[4, 1, 1], [3.5, 1, 1], [2, 1, 1]], dtype=np.float32
    )

    alignment = registration.align(candidate, target)

    assert alignment.orientation == "reverse"
    assert alignment.candidate_offset_samples == 0
    assert alignment.target_offset_samples == 1
    assert alignment.overlap_samples == 3
    np.testing.assert_allclose(alignment.offset_mm, -1.0)
    np.testing.assert_allclose(alignment.rms_distance_mm, 0.0, atol=1e-6)
    np.testing.assert_allclose(alignment.overlap_cosine, 1.0, atol=1e-6)
    np.testing.assert_allclose(alignment.coverage, 3 / 5)
    np.testing.assert_allclose(
        alignment.coverage_weighted_cosine,
        alignment.overlap_cosine * alignment.coverage,
    )


def test_registration_does_not_spatially_translate_coordinates():
    registration = GeodesicFiberRegistration(sample_interval_mm=1.0)
    first = np.asarray([[0, 0, 0], [2, 0, 0]], dtype=np.float32)
    displaced = np.asarray([[0, 10, 0], [2, 10, 0]], dtype=np.float32)

    alignment = registration.align(first, displaced)

    assert alignment.rms_distance_mm == 10.0
    assert alignment.overlap_cosine < 1.0
