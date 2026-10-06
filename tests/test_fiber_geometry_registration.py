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
    np.testing.assert_allclose(alignment.cosine_similarity, 1.0, atol=1e-6)


def test_registration_does_not_spatially_translate_coordinates():
    registration = GeodesicFiberRegistration(sample_interval_mm=1.0)
    first = np.asarray([[0, 0, 0], [2, 0, 0]], dtype=np.float32)
    displaced = np.asarray([[0, 10, 0], [2, 10, 0]], dtype=np.float32)

    alignment = registration.align(first, displaced)

    assert alignment.rms_distance_mm == 10.0
    assert alignment.cosine_similarity < 1.0


def test_sample_both_is_a_zero_cost_reverse_view_of_one_sampling():
    registration = GeodesicFiberRegistration(sample_interval_mm=0.75)
    fiber = np.asarray(
        [[1, 2, 3], [2, 4, 3], [5, 5, 4]],
        dtype=np.float32,
    )

    forward, reverse = registration.sample_both(fiber)

    np.testing.assert_allclose(forward, registration.sample(fiber, reverse=False))
    np.testing.assert_array_equal(reverse, forward[::-1])
    assert forward.dtype == np.float32


def test_broadcast_scores_equal_individual_alignment_scores():
    registration = GeodesicFiberRegistration(sample_interval_mm=1.0)
    target = registration.sample(
        np.asarray([[1, 1, 1], [4, 1, 1]], dtype=np.float32)
    )
    fibers = [
        np.asarray([[1, 1, 1], [3, 1, 1]], dtype=np.float32),
        np.asarray([[4, 1, 1], [2, 1, 1]], dtype=np.float32),
        np.asarray([[1, 2, 1], [3, 2, 1]], dtype=np.float32),
    ]
    sampled = [registration.sample(fiber) for fiber in fibers]
    candidates = np.stack(sampled)

    broadcast = registration.score_sampled_batch(
        candidates, target, similarity="valid_sample_cosine"
    )
    individual = np.asarray(
        [
            registration.align_sampled(candidate, target).cosine_similarity
            for candidate in sampled
        ]
    )

    np.testing.assert_allclose(broadcast, individual, rtol=1e-6, atol=1e-6)

    long_fibers = [
        np.asarray([[0, 1, 1], [5, 1, 1]], dtype=np.float32),
        np.asarray([[5, 1, 1], [0, 1, 1]], dtype=np.float32),
    ]
    long_sampled = [registration.sample(fiber) for fiber in long_fibers]
    long_candidates = np.stack(long_sampled)
    short_target = registration.sample(
        np.asarray([[2, 1, 1], [4, 1, 1]], dtype=np.float32)
    )
    broadcast = registration.score_sampled_batch(
        long_candidates, short_target, similarity="valid_sample_cosine"
    )
    individual = np.asarray(
        [
            registration.align_sampled(candidate, short_target).cosine_similarity
            for candidate in long_sampled
        ]
    )
    np.testing.assert_allclose(broadcast, individual, rtol=1e-6, atol=1e-6)
    assert broadcast.dtype == np.float32


def test_cosine_uses_only_valid_shorter_fiber_samples():
    registration = GeodesicFiberRegistration(sample_interval_mm=1.0)
    longer = registration.sample(
        np.asarray([[1, 1, 1], [5, 1, 1]], dtype=np.float32)
    )
    shorter_reversed = registration.sample(
        np.asarray([[4, 1, 1], [2, 1, 1]], dtype=np.float32)
    )

    alignment = registration.align_sampled(shorter_reversed, longer)
    broadcast = registration.score_sampled_batch(
        shorter_reversed[None, ...], longer
    )

    assert alignment.orientation == "reverse"
    assert alignment.candidate_offset_samples == 1
    np.testing.assert_allclose(alignment.cosine_similarity, 1.0, atol=1e-6)
    np.testing.assert_allclose(broadcast, [1.0], atol=1e-6)
