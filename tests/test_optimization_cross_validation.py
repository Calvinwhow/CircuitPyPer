"""Inner-fold weight fitting scores each patient against maps excluding them."""

import contextlib
import io
import unittest

import numpy as np
from scipy.stats import spearmanr

from calvin_utils.neuroimaging_utils.ccm_utils.optimization.convergent_map_optimizer import LocalizationOptimizer
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.fold_maps import build_fold_maps
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.inner_cv import prepare_inner_folds
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.nested_cv import (
    FoldArrayLoader, evaluate_regression_pipeline_outer_cv,
    fit_weights_with_inner_cv,
)


class OptimizationCrossValidationTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(38)
        self.n, self.features = 12, 6
        self.ids = np.array([f"subject_{i}" for i in range(self.n)])
        self.map_X = rng.normal(size=(self.n, self.features)).astype(np.float32)
        self.score_X = self.map_X + rng.normal(scale=.2, size=self.map_X.shape)
        self.map_outcomes = {
            "basis_a": self.map_X @ rng.normal(size=self.features),
            "basis_b": self.map_X @ rng.normal(size=self.features),
            "basis_c": self.map_X @ rng.normal(size=self.features),
        }
        self.scoring = {
            "cognition": {
                "X": self.score_X, "ids": self.ids,
                "y": self.score_X @ rng.normal(size=self.features),
            },
            "motor": {
                "X": self.score_X, "ids": self.ids,
                "y": self.score_X @ rng.normal(size=self.features),
            },
        }

    def test_fold_maps_exclude_each_scored_patient_and_own_score_is_unchanged(self):
        full, fold_iter, cohorts, _ = prepare_inner_folds(
            self.map_X, self.ids, self.map_outcomes, self.scoring,
            inner_folds=3, seed=6,
        )
        folds = list(fold_iter)
        self.assertEqual(set(full), set(self.map_outcomes))
        for fold in folds:
            held_out = self.ids[fold["rows"]["cognition"]]
            train = ~np.isin(self.ids, held_out)
            expected, _ = build_fold_maps(self.map_X, self.map_outcomes, train)
            for name in full:
                np.testing.assert_allclose(fold["maps"][name], expected[name])

        row = int(folds[0]["rows"]["cognition"][0])
        changed = {name: values.copy() for name, values in self.map_outcomes.items()}
        for values in changed.values():
            values[row] += 1000
        _, altered_fold_iter, _, _ = prepare_inner_folds(
            self.map_X, self.ids, changed, self.scoring,
            inner_folds=3, seed=6,
        )
        altered_folds = list(altered_fold_iter)
        for name in full:
            np.testing.assert_allclose(folds[0]["maps"][name], altered_folds[0]["maps"][name])

        changed_X = self.map_X.copy()
        changed_X[row] += 1000
        _, altered_image_folds, _, _ = prepare_inner_folds(
            changed_X, self.ids, self.map_outcomes, self.scoring,
            inner_folds=3, seed=6,
        )
        altered_image_folds = list(altered_image_folds)
        for name in full:
            np.testing.assert_allclose(folds[0]["maps"][name], altered_image_folds[0]["maps"][name])

    def test_weight_objective_correlates_complete_inner_cv_cosines(self):
        full, folds, cohorts, _ = prepare_inner_folds(
            self.map_X, self.ids, self.map_outcomes, self.scoring,
            inner_folds=3, seed=6,
        )
        folds = list(folds)
        self.assertIs(cohorts["cognition"]["X"], cohorts["motor"]["X"])
        loader = FoldArrayLoader({
            name: {"niftis": data["X"], "indep_var": data["y"]}
            for name, data in cohorts.items()
        })
        with contextlib.redirect_stdout(io.StringIO()):
            optimizer = LocalizationOptimizer(
                full, loader, data_mode="ram", inner_folds=folds
            )
        for cognition, motor in zip(
            optimizer.engine._inner_projections["cognition"],
            optimizer.engine._inner_projections["motor"],
        ):
            self.assertIs(cognition[1], motor[1])
        candidates = np.array([[.3, -.1, .6], [.5, .2, -.4]])
        observed = optimizer.engine._rho_for_weights(candidates)
        for cohort_index, (name, data) in enumerate(cohorts.items()):
            for candidate_index, weights in enumerate(candidates):
                scores = np.empty(len(data["y"]))
                for fold in folds:
                    rows = fold["rows"][name]
                    maps = np.stack([
                        fold["maps"][map_name] / np.linalg.norm(fold["maps"][map_name])
                        for map_name in full
                    ])
                    combined = weights @ maps
                    scores[rows] = (data["X"][rows] @ combined) / (
                        np.linalg.norm(data["X"][rows], axis=1) * np.linalg.norm(combined)
                    )
                self.assertAlmostEqual(
                    observed[cohort_index, candidate_index],
                    spearmanr(scores, data["y"]).correlation,
                )

    def test_one_fit_exports_complete_inner_fold_scores(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            optimized, predictions, summary, weights = fit_weights_with_inner_cv(
                self.map_X, self.ids, self.map_outcomes, self.scoring,
                inner_folds=3, seed=6, max_iters=30,
            )
        self.assertEqual(optimized.shape, (1, self.features))
        self.assertEqual(len(predictions), self.n * len(self.scoring))
        self.assertTrue(np.isfinite(predictions["damage"]).all())
        self.assertTrue((predictions["damage"].abs() <= 1 + 1e-12).all())
        for name in self.scoring:
            self.assertEqual(
                set(predictions.loc[predictions.dataset == name, "patient_id"]),
                set(self.ids),
            )
        self.assertEqual(len(summary), len(self.scoring))
        self.assertEqual(len(weights), len(self.map_outcomes))
        self.assertIn("inner_cv_weight_fit_spearman_rho", summary)
        self.assertIn("final_map_apparent_spearman_rho", summary)

    def test_leave_one_out_calibration_uses_training_patient_scores(self):
        full, folds, cohorts, _ = prepare_inner_folds(
            self.map_X, self.ids, self.map_outcomes, self.scoring,
            inner_folds="loocv", seed=6,
        )
        folds = list(folds)
        loader = FoldArrayLoader({
            name: {"niftis": data["X"], "indep_var": data["y"]}
            for name, data in cohorts.items()
        })
        with contextlib.redirect_stdout(io.StringIO()):
            optimizer = LocalizationOptimizer(
                full, loader, data_mode="ram", inner_folds=folds,
                inner_score_mode="reference_z",
            )
        weights = np.array([.3, -.1, .6])
        observed = optimizer.engine.predict_inner(
            weights, adjusted=True
        )["cognition"]
        X = cohorts["cognition"]["X"]
        for fold in folds:
            rows = fold["rows"]["cognition"]
            self.assertEqual(len(rows), 1)
            maps = np.stack([
                fold["maps"][name] / np.linalg.norm(fold["maps"][name])
                for name in full
            ])
            weighted_map = weights @ maps
            raw = (X @ weighted_map) / (
                np.linalg.norm(X, axis=1) * np.linalg.norm(weighted_map)
            )
            reference = np.delete(raw, rows)
            expected = (raw[rows] - reference.mean()) / reference.std()
            np.testing.assert_allclose(observed[rows], expected)
        actual_rho = optimizer.engine._rho_for_weights(weights)[0]
        self.assertAlmostEqual(
            actual_rho, spearmanr(observed, cohorts["cognition"]["y"]).correlation
        )

    def test_batched_gradient_handles_more_than_fifty_fold_maps(self):
        rng = np.random.default_rng(71)
        X = rng.normal(size=(18, 64)).astype(np.float32)
        ids = np.arange(len(X)).astype(str)
        map_outcomes = {
            f"map_{index}": X @ rng.normal(size=X.shape[1])
            for index in range(53)
        }
        scoring = {"score": {"X": X, "ids": ids, "y": X @ rng.normal(size=X.shape[1])}}
        full, folds, cohorts, _ = prepare_inner_folds(
            X, ids, map_outcomes, scoring, inner_folds=3, seed=7,
        )
        loader = FoldArrayLoader({
            "score": {"niftis": cohorts["score"]["X"], "indep_var": cohorts["score"]["y"]}
        })
        with contextlib.redirect_stdout(io.StringIO()):
            optimizer = LocalizationOptimizer(
                full, loader, data_mode="ram", inner_folds=folds,
            )
        loss = optimizer.engine._loss_for_weights(optimizer.engine.W)
        gradient = optimizer.engine._forward_diff_grad(loss, batch_size=50)
        self.assertEqual(gradient.shape, (1, 53))
        self.assertTrue(np.isfinite(gradient).all())

    def test_outer_validation_is_excluded_from_maps_and_weight_selection(self):
        def run(map_outcomes, scoring):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return evaluate_regression_pipeline_outer_cv(
                    self.map_X, self.ids, map_outcomes, scoring,
                    inner_folds=2, outer_folds=3, seed=6, max_iters=2,
                )[0]

        before = run(self.map_outcomes, self.scoring)
        changed_maps = {
            name: values.copy() for name, values in self.map_outcomes.items()
        }
        changed_scoring = {
            name: {key: value.copy() for key, value in data.items()}
            for name, data in self.scoring.items()
        }
        for values in changed_maps.values():
            values[0] += 1000
        for data in changed_scoring.values():
            data["y"][0] += 1000
        after = run(changed_maps, changed_scoring)
        for name in self.scoring:
            first = before[
                (before.dataset == name) & (before.patient_id == self.ids[0])
            ].damage.iloc[0]
            second = after[
                (after.dataset == name) & (after.patient_id == self.ids[0])
            ].damage.iloc[0]
            self.assertAlmostEqual(first, second)


if __name__ == "__main__":
    unittest.main()
