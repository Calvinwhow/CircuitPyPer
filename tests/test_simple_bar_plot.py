import os
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from calvin_utils.plotting_utils.simple_bar_plot import SimpleBarPlotWrapper


class TestSimpleBarPlot(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_draws_one_bar_per_row_with_duplicate_labels(self):
        df = pd.DataFrame({"label": ["A", "A", "B", "C"], "value": [1.0, -0.4, 0.7, 0.2]})

        ax = SimpleBarPlotWrapper(df).plot("value", "label", palette="viridis", annotate_values=True)

        self.assertEqual(len(ax.patches), len(df))
        self.assertEqual([tick.get_text() for tick in ax.get_xticklabels()], ["A", "A", "B", "C"])

    def test_flip_draws_horizontal_bars(self):
        df = pd.DataFrame({"label": ["A", "B"], "value": [1.0, 0.5]})

        ax = SimpleBarPlotWrapper(df).plot("value", "label", flip=True)

        self.assertEqual(len(ax.patches), len(df))
        self.assertEqual(ax.get_xlabel(), "value")
        self.assertEqual(ax.get_ylabel(), "label")

    def test_sorts_and_limits_rows(self):
        df = pd.DataFrame({"label": ["A", "B", "C"], "value": [0.1, 0.9, 0.4]})

        ax = SimpleBarPlotWrapper(df).plot("value", "label", sort_by_value=True, top_n=2)

        self.assertEqual(len(ax.patches), 2)
        self.assertEqual([tick.get_text() for tick in ax.get_xticklabels()], ["B", "C"])

    def test_saves_file_path_passed_as_out_dir(self):
        df = pd.DataFrame({"label": ["A", "B"], "value": [1.0, 0.5]})
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "plot.svg")

            SimpleBarPlotWrapper(df).plot("value", "label", out_dir=output_path)

            self.assertTrue(os.path.exists(output_path))
            self.assertGreater(os.path.getsize(output_path), 0)

    def test_rejects_empty_after_dropna(self):
        df = pd.DataFrame({"label": [None], "value": [1.0]})

        with self.assertRaisesRegex(ValueError, "No rows remain"):
            SimpleBarPlotWrapper(df).plot("value", "label")


if __name__ == "__main__":
    unittest.main()
