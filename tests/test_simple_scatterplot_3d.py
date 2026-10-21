"""Tests for the generic publication-style 3D scatter plotter."""

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from calvin_utils.plotting_utils.simple_scatterplot_3d import SimpleScatterplot3D


class SimpleScatterplot3DTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.points = np.vstack(
            [
                rng.normal([0, 0, 2], 0.2, (8, 3)),
                rng.normal([3, 2, 4], 0.3, (9, 3)),
                rng.normal([6, -1, 7], 0.25, (10, 3)),
            ]
        )
        self.labels = np.repeat(["a", "b", "c"], [8, 9, 10])
        self.probabilities = np.linspace(0.5, 1.0, len(self.points))

    def test_requires_observations_by_three_dimensions(self):
        with self.assertRaisesRegex(ValueError, "dimensions in columns"):
            SimpleScatterplot3D(self.points[:, :2])

    def test_plot_accepts_general_labels_colors_and_ground_shadows(self):
        plotter = SimpleScatterplot3D(
            self.points, self.labels, self.probabilities
        )
        figure = plotter.plot(
            colors={"a": "#c15656", "b": "#5071a0", "c": "#9a8ed1"},
            names={"a": "First", "b": "Second", "c": "Third"},
            ellipsoids=True,
            ground_shadows=True,
            shadow_stems=True,
            show_ground_plane=True,
        )
        self.assertIs(figure, plotter.figure)
        self.assertEqual(len(plotter.cluster_separation_results), 3)
        self.assertIn(
            "hotelling_t_squared", plotter.cluster_separation_results
        )
        self.assertEqual(len(plotter.axis.lines), 3)
        self.assertIsNotNone(plotter.axis.get_legend())
        plt.close(figure)

    def test_export_preserves_requested_canvas(self):
        plotter = SimpleScatterplot3D(self.points, self.labels)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scatter.svg"
            result = plotter.export(
                path,
                figsize=(10, 4),
                ellipsoids=True,
                ground_shadows=True,
            )
            text = path.read_text()
            self.assertEqual(result, str(path))
            self.assertIn('width="720pt"', text)
            self.assertIn('height="288pt"', text)
            separation_path = (
                Path(directory)
                / "scatter_cluster_separation_hotelling.csv"
            )
            self.assertTrue(separation_path.exists())
            self.assertEqual(len(separation_path.read_text().splitlines()), 4)

    def test_plot_prints_hotelling_results(self):
        plotter = SimpleScatterplot3D(self.points, self.labels)
        output = StringIO()
        with redirect_stdout(output):
            figure = plotter.plot(ellipsoids=False)
        self.assertIn("Pairwise Hotelling tests", output.getvalue())
        self.assertIn("centroid_distance_l2", output.getvalue())
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()

