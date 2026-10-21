"""Tests for outcome-specific complete-case imaging regressions."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from calvin_utils.ml_utils.outcome_wise_imaging_regression import (
    OutcomeWiseImagingRegression,
)


class OutcomeWiseImagingRegressionTest(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(
            {
                "image": [f"image_{index}" for index in range(6)],
                "a": [1, 2, 3, 4, 5, 6],
                "b": [1, np.nan, 2, 4, np.nan, 8],
                "c": [6, 5, 4, 3, 2, 1],
            }
        )
        self.imaging = np.asarray(
            [
                [1, 2, 0],
                [2, 1, 0],
                [3, 4, 1],
                [4, 3, 1],
                [5, 6, 2],
                [6, 5, 2],
            ],
            dtype=float,
        )

    def test_preserves_outcome_specific_complete_case_cohorts(self):
        regression = OutcomeWiseImagingRegression(
            dataframe=self.frame,
            imaging_data=self.imaging,
            path_column="image",
            outcome_names=["a", "b", "c"],
            transform="rank",
        )

        np.testing.assert_array_equal(regression.observation_counts, [6, 4, 6])
        self.assertEqual(len(regression._groups), 2)
        _, observed, _ = regression.run_single_multiout_regression()
        self.assertEqual(observed.shape, (1, 3, 3))
        self.assertTrue(np.isfinite(observed).all())

    def test_seeded_padded_permutations_are_valid_and_reproducible(self):
        regression = OutcomeWiseImagingRegression(
            dataframe=self.frame,
            imaging_data=self.imaging,
            path_column="image",
            outcome_names=["a", "b", "c"],
        )
        first = regression.draw_permutation_indices(3, np.random.default_rng(9))
        second = regression.draw_permutation_indices(3, np.random.default_rng(9))

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.shape, (3, 3, 6))
        np.testing.assert_array_equal(np.sort(first[0, 1, :4]), np.arange(4))
        np.testing.assert_array_equal(first[:, 1, 4:], -1)

    def test_ordinal_ranking_reproduces_stable_historical_tie_order(self):
        values = np.asarray([[0], [0], [1], [0]], dtype=float)
        transformed = OutcomeWiseImagingRegression._transform(
            values, "rank_ordinal"
        )
        np.testing.assert_array_equal(
            transformed.ravel(), np.asarray([-1.5, -0.5, 1.5, 0.5])
        )

    def test_zero_missing_policy_retains_rows_and_applies_outcome_sign(self):
        regression = OutcomeWiseImagingRegression(
            dataframe=self.frame,
            imaging_data=self.imaging,
            path_column="image",
            outcome_names=["b"],
            transform="rank_ordinal",
            missing_policy="zero",
            outcome_signs=-1,
        )
        self.assertEqual(regression.observation_counts.tolist(), [6])
        expected = OutcomeWiseImagingRegression._transform(
            -self.frame[["b"]].fillna(0).to_numpy(float), "rank_ordinal"
        )
        np.testing.assert_array_equal(regression._groups[0]["y"], expected)

    def test_observed_maps_are_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observed.npy"
            regression = OutcomeWiseImagingRegression(
                dataframe=self.frame,
                imaging_data=self.imaging,
                path_column="image",
                outcome_names=["a", "b", "c"],
                observed_maps_path=path,
            )
            regression.run_single_multiout_regression()
            self.assertEqual(np.load(path).shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
