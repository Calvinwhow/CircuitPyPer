"""Test datasets must be scored once, by the frozen map, and never fit on."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from calvin_utils.neuroimaging_utils.ccm_utils.optimization.test_evaluation import (
    evaluate_test_datasets,
    fitting_ids_from_datasets,
    save_test_evaluation,
)


def cohort(rng, n, n_features, prefix):
    X = rng.normal(size=(n, n_features))
    return {
        "X": X,
        "y": rng.normal(size=n),
        "ids": np.array([f"{prefix}_{index}" for index in range(n)]),
    }


class TestDatasetEvaluationTest(unittest.TestCase):
    def test_damage_is_the_cosine_with_the_frozen_map(self):
        rng = np.random.default_rng(3)
        data = cohort(rng, 9, 11, "test")
        optimized_map = rng.normal(size=11)
        predictions, summary = evaluate_test_datasets(
            optimized_map, {"held_out": data}
        )
        expected = (data["X"] @ optimized_map) / (
            np.linalg.norm(data["X"], axis=1) * np.linalg.norm(optimized_map)
        )
        np.testing.assert_allclose(predictions["damage"].to_numpy(), expected)
        self.assertAlmostEqual(
            float(summary.loc[0, "test_spearman_rho"]),
            float(spearmanr(expected, data["y"]).statistic),
        )

    def test_rescaling_the_map_does_not_change_the_reported_rho(self):
        rng = np.random.default_rng(5)
        data = cohort(rng, 12, 8, "test")
        optimized_map = rng.normal(size=8)
        _, summary = evaluate_test_datasets(optimized_map, {"held_out": data})
        _, rescaled = evaluate_test_datasets(
            optimized_map * 7.5, {"held_out": data}
        )
        self.assertAlmostEqual(
            float(summary.loc[0, "test_spearman_rho"]),
            float(rescaled.loc[0, "test_spearman_rho"]),
        )

    def test_patients_used_in_fitting_are_refused(self):
        rng = np.random.default_rng(7)
        fitting = {"fit": cohort(rng, 10, 6, "shared")}
        test = {"held_out": cohort(rng, 10, 6, "shared")}
        with self.assertRaises(ValueError) as caught:
            evaluate_test_datasets(
                rng.normal(size=6), test,
                fitting_ids=fitting_ids_from_datasets(fitting),
            )
        self.assertIn("shares", str(caught.exception))

    def test_map_building_ids_are_collected_alongside_fitting_cohorts(self):
        rng = np.random.default_rng(11)
        fitting = {"fit": cohort(rng, 6, 4, "score")}
        map_ids = np.array(["map_0", "map_1"])
        collected = fitting_ids_from_datasets(fitting, map_ids)
        self.assertEqual(collected, {"score_0", "score_1", "score_2", "score_3",
                                     "score_4", "score_5", "map_0", "map_1"})

    def test_missing_outcomes_are_dropped_and_counted(self):
        rng = np.random.default_rng(13)
        data = cohort(rng, 10, 5, "test")
        data["y"][[2, 6]] = np.nan
        predictions, summary = evaluate_test_datasets(
            rng.normal(size=5), {"held_out": data}
        )
        self.assertEqual(int(summary.loc[0, "n_patients"]), 8)
        self.assertEqual(int(summary.loc[0, "n_missing_outcomes"]), 2)
        self.assertNotIn("test_2", set(predictions["patient_id"]))

    def test_a_feature_space_mismatch_is_refused(self):
        rng = np.random.default_rng(17)
        with self.assertRaises(ValueError) as caught:
            evaluate_test_datasets(
                rng.normal(size=5), {"held_out": cohort(rng, 8, 9, "test")}
            )
        self.assertIn("features", str(caught.exception))

    def test_results_are_written_beside_the_other_stages(self):
        rng = np.random.default_rng(19)
        predictions, summary = evaluate_test_datasets(
            rng.normal(size=7), {"held_out": cohort(rng, 8, 7, "test")}
        )
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory) / "test_evaluation"
            with contextlib.redirect_stdout(io.StringIO()):
                save_test_evaluation(out_dir, predictions, summary)
            self.assertTrue((out_dir / "test_predictions.csv").is_file())
            self.assertTrue((out_dir / "test_summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
