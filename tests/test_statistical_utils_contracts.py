"""Numerical contract tests for commonly reused statistical utilities."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from calvin_utils.permutation_analysis_utils.statsmodels_palm import (
    CalvinStatsmodelsPalm,
)
from calvin_utils.permutation_analysis_utils.voxelwise_regression_prep import (
    RegressionPrep,
)
from calvin_utils.statistical_utils.data_transformation import (
    min_max_normalize,
    min_max_normalize_minus_one_to_one,
    rank_transform,
    z_score_normalize,
)
from calvin_utils.statistical_utils.regression_utils import RegressOutCovariates


class StatisticalUtilsContractsTest(unittest.TestCase):
    def test_core_transforms_have_expected_ranges_and_order(self):
        series = pd.Series([1.0, 2.0, 4.0, 8.0])
        np.testing.assert_allclose(min_max_normalize(series), [0, 1 / 7, 3 / 7, 1])
        np.testing.assert_allclose(
            min_max_normalize_minus_one_to_one(series),
            [-1, -5 / 7, -1 / 7, 1],
        )
        self.assertAlmostEqual(float(z_score_normalize(series).mean()), 0.0)
        self.assertEqual(rank_transform(series).tolist(), [1.0, 2.0, 3.0, 4.0])

    def test_regressing_covariates_returns_orthogonal_residuals(self):
        covariate = np.arange(10, dtype=float)
        outcome = 3 * covariate + np.array([0, 1, 0, -1, 0, 1, 0, -1, 0, 0])
        residuals = RegressOutCovariates.regress_out_covariates_using_endog_exog(
            outcome, covariate
        )
        self.assertAlmostEqual(float(np.dot(residuals, covariate)), 0.0, places=9)
        self.assertAlmostEqual(float(np.mean(residuals)), 0.0, places=9)

    def test_simple_design_matrix_honors_intercept_setting(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.csv"
            frame = pd.DataFrame({"y": [1, 2, 3], "x": [4, 5, 6]})
            frame.to_csv(path, index=False)
            palm = CalvinStatsmodelsPalm(path, directory)
            loaded = palm.read_and_display_data()
            _, with_intercept = palm.define_design_matrix(
                "y ~ x", loaded, add_intercept=True
            )
            _, without_intercept = palm.define_design_matrix(
                "y ~ x", loaded, add_intercept=False
            )
            self.assertEqual(list(with_intercept.columns), ["Intercept", "x"])
            self.assertEqual(list(without_intercept.columns), ["x"])

    def test_regression_prep_transforms_subject_axis_and_keeps_indicators(self):
        arr = np.array(
            [
                [[0.0], [1.0]],
                [[0.0], [2.0]],
                [[1.0], [3.0]],
                [[1.0], [4.0]],
            ]
        )
        transformed = RegressionPrep.transform_array(
            arr, "standardize", keep_categorical=True
        )
        np.testing.assert_array_equal(transformed[:, 0, 0], arr[:, 0, 0])
        self.assertAlmostEqual(float(transformed[:, 1, 0].mean()), 0.0)
        self.assertAlmostEqual(float(transformed[:, 1, 0].std()), 1.0)


if __name__ == "__main__":
    unittest.main()
