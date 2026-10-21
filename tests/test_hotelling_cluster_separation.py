"""Tests for pairwise Hotelling cluster separation."""

import unittest

import numpy as np

from calvin_utils.plotting_utils.hotelling_cluster_separation import (
    pairwise_hotelling_tests,
)


class PairwiseHotellingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.points = np.vstack(
            [
                rng.normal([0, 0, 0], 0.4, (12, 3)),
                rng.normal([3, 2, 1], 0.5, (13, 3)),
                rng.normal([-2, 4, 3], 0.6, (14, 3)),
            ]
        )
        self.labels = np.repeat(["a", "b", "c"], [12, 13, 14])

    def test_one_multivariate_test_is_returned_per_pair(self):
        results = pairwise_hotelling_tests(self.points, self.labels)
        self.assertEqual(len(results), 3)
        self.assertTrue((results["hotelling_t_squared"] >= 0).all())
        self.assertTrue((results["centroid_distance_l2"] > 0).all())
        self.assertTrue((results["df1"] == 3).all())

    def test_test_is_invariant_to_rigid_transformation(self):
        angle = np.deg2rad(29.0)
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        transformed = self.points @ rotation.T + [100.0, -20.0, 8.0]
        original = pairwise_hotelling_tests(self.points, self.labels)
        moved = pairwise_hotelling_tests(transformed, self.labels)
        np.testing.assert_allclose(
            original["centroid_distance_l2"],
            moved["centroid_distance_l2"],
        )
        np.testing.assert_allclose(
            original["hotelling_t_squared"],
            moved["hotelling_t_squared"],
        )
        np.testing.assert_allclose(original["p_value"], moved["p_value"])


if __name__ == "__main__":
    unittest.main()
