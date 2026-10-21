"""Tests for generic 2D/3D SD ellipsoids and separation tests."""

import unittest

import numpy as np

from calvin_utils.plotting_utils.scatter_ellipsoids import (
    add_cluster_ellipsoids_matplotlib,
    add_cluster_ellipsoids_plotly,
    calculate_cluster_ellipsoids,
    pairwise_cluster_separation_tests,
)


class ScatterEllipsoidTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1234)
        self.cluster_a = rng.normal(
            loc=[-2.0, 0.5, 1.0], scale=[0.5, 1.0, 1.5], size=(40, 3)
        )
        self.cluster_b = rng.normal(
            loc=[2.5, 1.0, -0.5], scale=[0.7, 0.8, 0.6], size=(45, 3)
        )
        self.cluster_c = rng.normal(
            loc=[0.0, 4.0, 2.0], scale=[0.6, 0.5, 0.9], size=(35, 3)
        )
        self.points = np.vstack([self.cluster_a, self.cluster_b, self.cluster_c])
        self.labels = np.repeat([0, 1, 2], [40, 45, 35])
        self.points_2d = self.points[:, :2]

    def test_surface_satisfies_requested_ellipsoid_equation(self):
        ellipsoids = calculate_cluster_ellipsoids(
            self.points, self.labels, n_std=1.0, resolution=18
        )
        ellipsoid = ellipsoids[0]

        normalized_radius_squared = (
            ((ellipsoid.x - ellipsoid.center[0]) / ellipsoid.radii[0]) ** 2
            + ((ellipsoid.y - ellipsoid.center[1]) / ellipsoid.radii[1]) ** 2
            + ((ellipsoid.z - ellipsoid.center[2]) / ellipsoid.radii[2]) ** 2
        )
        np.testing.assert_allclose(normalized_radius_squared, 1.0, atol=1e-12)
        np.testing.assert_allclose(
            ellipsoid.standard_deviation, self.cluster_a.std(axis=0, ddof=1)
        )

    def test_noise_is_excluded_by_default(self):
        points = np.vstack([self.points, [[20.0, 20.0, 20.0], [21.0, 21.0, 21.0]]])
        labels = np.concatenate([self.labels, [-1, -1]])
        ellipsoids = calculate_cluster_ellipsoids(points, labels)
        self.assertEqual(set(ellipsoids), {0, 1, 2})

    def test_two_dimensional_input_produces_ellipse(self):
        ellipsoid = calculate_cluster_ellipsoids(
            self.points_2d, self.labels, resolution=25
        )[0]
        self.assertEqual(ellipsoid.n_dimensions, 2)
        self.assertIsNone(ellipsoid.z)
        self.assertEqual(ellipsoid.x.shape, (25,))
        normalized_radius_squared = (
            ((ellipsoid.x - ellipsoid.center[0]) / ellipsoid.radii[0]) ** 2
            + ((ellipsoid.y - ellipsoid.center[1]) / ellipsoid.radii[1]) ** 2
        )
        np.testing.assert_allclose(normalized_radius_squared, 1.0, atol=1e-12)

    def test_labels_are_optional_for_a_single_point_cloud(self):
        ellipsoids = calculate_cluster_ellipsoids(self.cluster_a)
        self.assertEqual(set(ellipsoids), {0})
        self.assertEqual(ellipsoids[0].n_points, len(self.cluster_a))

    def test_input_layout_requires_observations_by_two_or_three_dimensions(self):
        with self.assertRaisesRegex(ValueError, "dimensions in columns"):
            calculate_cluster_ellipsoids(self.points[:, :1], self.labels)
        with self.assertRaisesRegex(ValueError, "dimensions in columns"):
            calculate_cluster_ellipsoids(self.points.T, self.labels)

    def test_every_cluster_pair_is_tested_and_holm_corrected(self):
        results = pairwise_cluster_separation_tests(self.points, self.labels)
        self.assertEqual(len(results), 3)
        self.assertTrue((results["p_value_adjusted"] >= results["p_value"]).all())
        self.assertTrue((results["centroid_distance_l2"] > 0).all())
        self.assertEqual(set(results["score"]), {"centroid_axis"})

    def test_centroid_axis_test_is_rigid_transform_invariant(self):
        angle = np.deg2rad(37.0)
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        transformed = self.points @ rotation.T + np.array([100.0, -30.0, 8.0])
        original = pairwise_cluster_separation_tests(self.points, self.labels)
        moved = pairwise_cluster_separation_tests(transformed, self.labels)

        np.testing.assert_allclose(original["t_statistic"], moved["t_statistic"])
        np.testing.assert_allclose(original["p_value"], moved["p_value"])
        np.testing.assert_allclose(
            original["centroid_distance_l2"], moved["centroid_distance_l2"]
        )

    def test_literal_l2_from_origin_option_is_available(self):
        results = pairwise_cluster_separation_tests(
            self.points, self.labels, score="l2_origin", p_adjust=None
        )
        expected = np.linalg.norm(self.cluster_a, axis=1).mean()
        self.assertAlmostEqual(results.iloc[0]["mean_a"], expected)
        self.assertEqual(results.iloc[0]["p_value"], results.iloc[0]["p_value_adjusted"])

    def test_two_dimensional_separation_test_has_two_axis_columns(self):
        results = pairwise_cluster_separation_tests(self.points_2d, self.labels)
        axis_columns = [column for column in results if column.startswith("axis_dimension_")]
        self.assertEqual(axis_columns, ["axis_dimension_1", "axis_dimension_2"])

    def test_plotly_helper_adds_one_surface_per_cluster(self):
        import plotly.graph_objects as go

        figure = go.Figure()
        ellipsoids = add_cluster_ellipsoids_plotly(
            figure, self.points, self.labels, resolution=8
        )
        self.assertEqual(len(ellipsoids), 3)
        self.assertEqual(len(figure.data), 3)
        self.assertTrue(all(trace.type == "surface" for trace in figure.data))

    def test_plotly_helper_adds_one_filled_ellipse_per_2d_cluster(self):
        import plotly.graph_objects as go

        figure = go.Figure()
        add_cluster_ellipsoids_plotly(
            figure, self.points_2d, self.labels, resolution=8
        )
        self.assertEqual(len(figure.data), 3)
        self.assertTrue(all(trace.type == "scatter" for trace in figure.data))
        self.assertTrue(all(trace.fill == "toself" for trace in figure.data))

    def test_matplotlib_helper_adds_one_filled_ellipse_per_2d_cluster(self):
        import matplotlib.pyplot as plt

        figure, axis = plt.subplots()
        add_cluster_ellipsoids_matplotlib(axis, self.points_2d, self.labels)
        self.assertEqual(len(axis.patches), 3)
        plt.close(figure)

    def test_matplotlib_helper_adds_soft_lit_3d_surfaces(self):
        import matplotlib.pyplot as plt

        figure = plt.figure()
        axis = figure.add_subplot(111, projection="3d")
        ellipsoids = add_cluster_ellipsoids_matplotlib(
            axis,
            self.points,
            self.labels,
            resolution=8,
            lighting=True,
            light_azimuth=315,
            light_altitude=45,
            light_strength=0.22,
        )
        self.assertEqual(len(ellipsoids), 3)
        self.assertEqual(len(axis.collections), 3)
        plt.close(figure)

    def test_matplotlib_helper_rejects_invalid_light_strength(self):
        import matplotlib.pyplot as plt

        figure = plt.figure()
        axis = figure.add_subplot(111, projection="3d")
        with self.assertRaisesRegex(ValueError, "light_strength"):
            add_cluster_ellipsoids_matplotlib(
                axis, self.points, self.labels, light_strength=1.1
            )
        plt.close(figure)

    def test_matplotlib_helper_adds_optional_ground_shadows_and_stems(self):
        import matplotlib.pyplot as plt

        figure = plt.figure()
        axis = figure.add_subplot(111, projection="3d")
        add_cluster_ellipsoids_matplotlib(
            axis,
            self.points,
            self.labels,
            resolution=8,
            ground_shadows=True,
            shadow_stems=True,
        )
        # Three shaded ellipsoids plus three filled ground footprints.
        self.assertEqual(len(axis.collections), 6)
        self.assertEqual(len(axis.lines), 3)
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
