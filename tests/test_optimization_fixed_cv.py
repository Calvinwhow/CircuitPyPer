"""Fixed prior maps must score patients excluded from weight fitting."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from calvin_utils.neuroimaging_utils.ccm_utils.optimization.fixed_cv import (
    _load_cohorts,
    _project_cohorts,
    cross_validate_fixed_maps,
)
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.convergent_map_optimizer import (
    LocalizationOptimizer,
)
from calvin_utils.neuroimaging_utils.ccm_utils.npy_utils import DataLoader
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.nested_cv import (
    FoldArrayLoader,
)


class FixedMapCVTest(unittest.TestCase):
    def test_same_image_file_shares_row_ids_without_an_ids_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            X = root / "X.npy"
            y_a = root / "a.npy"
            y_b = root / "b.npy"
            np.save(X, np.arange(30, dtype=float).reshape(6, 5))
            np.save(y_a, np.arange(6))
            np.save(y_b, np.arange(6)[::-1])
            manifest = root / "dataset_dict.json"
            manifest.write_text(json.dumps({
                "a": {"niftis": str(X), "indep_var": str(y_a)},
                "b": {"niftis": str(X), "indep_var": str(y_b)},
            }))
            loader = DataLoader(manifest)
            cohorts = _load_cohorts(loader, data_mode="memmap")
            np.testing.assert_array_equal(cohorts["a"]["ids"], cohorts["b"]["ids"])

    def test_cached_fixed_map_projection_matches_direct_scoring(self):
        rng = np.random.default_rng(12)
        X = rng.normal(size=(8, 17))
        y = rng.normal(size=8)
        maps = {"a": rng.normal(size=17), "b": rng.normal(size=17)}
        loader = FoldArrayLoader({"score": {"niftis": X, "indep_var": y}})
        cohorts = _load_cohorts(loader, data_mode="ram")
        _project_cohorts(maps, cohorts, loader)
        prepared = {"score": (cohorts["score"]["projected"], y)}
        with contextlib.redirect_stdout(io.StringIO()):
            direct = LocalizationOptimizer(maps, loader, data_mode="ram")
            cached = LocalizationOptimizer(
                maps, loader, data_mode="ram", precomputed_projections=prepared
            )
        weights = np.array([[.3, -.7], [.8, .2]])
        np.testing.assert_allclose(
            direct.engine._rho_for_weights(weights),
            cached.engine._rho_for_weights(weights),
        )

    def test_shared_patient_ids_are_held_out_in_every_dataset(self):
        rng = np.random.default_rng(91)
        X = rng.normal(size=(12, 20)).astype(np.float32)
        ids = np.array([f"patient_{i}" for i in range(len(X))])
        maps = {name: rng.normal(size=20) for name in ("map_a", "map_b")}
        y_a = X @ rng.normal(size=20)
        y_b = X @ rng.normal(size=20)

        def run(first, second):
            loader = FoldArrayLoader({
                "a": {"niftis": X, "indep_var": first},
                "b": {"niftis": X, "indep_var": second},
            })
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return cross_validate_fixed_maps(
                    maps, loader, folds=3, seed=4, data_mode="ram",
                    max_iters=2, patient_ids={"a": ids, "b": ids},
                )

        predictions, summary, weights = run(y_a, y_b)
        self.assertEqual(len(predictions), 24)
        self.assertEqual(len(summary), 2)
        self.assertEqual(len(weights), 6)
        a = predictions[predictions.dataset == "a"].set_index("patient_id")
        b = predictions[predictions.dataset == "b"].set_index("patient_id")
        np.testing.assert_array_equal(
            a.loc[ids, "outer_fold"], b.loc[ids, "outer_fold"]
        )

        # Alter this patient's outcomes in both datasets. Their own held-out
        # fold cannot use either altered value while fitting its weights.
        changed_a, changed_b = y_a.copy(), y_b.copy()
        changed_a[0] += 1000
        changed_b[0] -= 1000
        altered, _, _ = run(changed_a, changed_b)
        for name in ("a", "b"):
            before = predictions[
                (predictions.dataset == name) & (predictions.patient_id == ids[0])
            ].damage.iloc[0]
            after = altered[
                (altered.dataset == name) & (altered.patient_id == ids[0])
            ].damage.iloc[0]
            self.assertAlmostEqual(before, after)

    def test_multiple_datasets_require_patient_ids(self):
        X = np.arange(30, dtype=float).reshape(6, 5)
        loader = FoldArrayLoader({
            "a": {"niftis": X, "indep_var": np.arange(6)},
            "b": {"niftis": X, "indep_var": np.arange(6)},
        })
        with self.assertRaisesRegex(ValueError, "need patient IDs"):
            cross_validate_fixed_maps(
                {"a": np.ones(5), "b": np.arange(5)}, loader,
                folds=2, max_iters=1, data_mode="ram",
            )


if __name__ == "__main__":
    unittest.main()
