"""A fold that flattens one outcome excludes it from fitting, not from scoring."""

import contextlib
import io
import unittest

import numpy as np

from calvin_utils.neuroimaging_utils.ccm_utils.optimization.fixed_cv import (
    evaluate_fixed_maps_outer_cv,
)
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.nested_cv import (
    ArrayLoader,
    evaluate_regression_pipeline_outer_cv,
)


def quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return function(*args, **kwargs)


class OuterFoldExclusionTest(unittest.TestCase):
    """One patient differs from the mode, exactly like Sec4AConcerns."""

    def setUp(self):
        rng = np.random.default_rng(4)
        self.X = rng.normal(size=(20, 30)).astype(np.float32)
        self.maps = {name: rng.normal(size=30) for name in ("a", "b", "c")}
        self.healthy = self.X @ rng.normal(size=30)
        self.degenerate = np.zeros(20)
        self.degenerate[7] = 1.0
        self.ids = np.array([f"p{i}" for i in range(20)])

    def run_fixed(self):
        loader = ArrayLoader({
            "healthy": {"niftis": self.X, "indep_var": self.healthy},
            "flat": {"niftis": self.X, "indep_var": self.degenerate},
        })
        return quiet(
            evaluate_fixed_maps_outer_cv, self.maps, loader,
            outer_folds=4, seed=1, data_mode="ram", max_iters=5,
            patient_ids={"healthy": self.ids, "flat": self.ids},
        )

    def test_a_flattened_outcome_no_longer_raises(self):
        predictions, summary, _ = self.run_fixed()
        self.assertEqual(len(summary), 2)
        flat = summary.set_index("dataset").loc["flat"]
        self.assertGreaterEqual(int(flat["n_folds_excluded_from_weight_fitting"]), 1)

    def test_every_patient_is_still_scored_exactly_once(self):
        predictions, _, _ = self.run_fixed()
        for name in ("healthy", "flat"):
            rows = predictions[predictions["dataset"] == name]
            self.assertEqual(len(rows), 20)
            self.assertEqual(rows["patient_id"].nunique(), 20)
            self.assertTrue(np.isfinite(rows["damage"]).all())

    def test_a_healthy_outcome_records_no_exclusions(self):
        _, summary, _ = self.run_fixed()
        healthy = summary.set_index("dataset").loc["healthy"]
        self.assertEqual(int(healthy["n_folds_excluded_from_weight_fitting"]), 0)

    def test_weights_are_still_fitted_in_every_fold(self):
        _, _, weights = self.run_fixed()
        self.assertEqual(sorted(weights["outer_fold"].unique()), [1, 2, 3, 4])

    def test_all_outcomes_flat_in_a_fold_is_still_refused(self):
        loader = ArrayLoader({
            "flat_one": {"niftis": self.X, "indep_var": self.degenerate},
            "flat_two": {"niftis": self.X, "indep_var": self.degenerate},
        })
        with self.assertRaises(ValueError) as caught:
            quiet(evaluate_fixed_maps_outer_cv, self.maps, loader,
                  outer_folds=4, seed=1, data_mode="ram", max_iters=5,
                  patient_ids={"flat_one": self.ids, "flat_two": self.ids})
        self.assertIn("no scoring dataset", str(caught.exception))

    def test_the_regression_pipeline_outer_loop_excludes_the_same_way(self):
        rng = np.random.default_rng(6)
        map_X = rng.normal(size=(20, 30)).astype(np.float32)
        map_ids = self.ids
        map_outcomes = {name: rng.normal(size=20) for name in ("m1", "m2", "m3")}
        scoring = {
            "healthy": {"X": self.X, "y": self.healthy, "ids": self.ids},
            "flat": {"X": self.X, "y": self.degenerate, "ids": self.ids},
        }
        predictions, summary, _, _, _ = quiet(
            evaluate_regression_pipeline_outer_cv,
            map_X, map_ids, map_outcomes, scoring,
            inner_folds=3, outer_folds=4, seed=1, max_iters=5,
        )
        flat = summary.set_index("dataset").loc["flat"]
        self.assertGreaterEqual(int(flat["n_folds_excluded_from_weight_fitting"]), 1)
        self.assertEqual(len(predictions[predictions["dataset"] == "flat"]), 20)


if __name__ == "__main__":
    unittest.main()
