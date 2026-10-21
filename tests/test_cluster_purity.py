"""Tests for label-invariant Hungarian macro-F1 cluster purity."""

import unittest

import numpy as np

from calvin_utils.ml_utils.cluster_purity import hungarian_macro_f1


class HungarianMacroF1Test(unittest.TestCase):
    def test_perfect_recovery_is_invariant_to_cluster_names(self):
        true = ["motor", "motor", "cognitive", "cognitive", "emotional", "emotional"]
        predicted = [7, 7, 2, 2, 99, 99]
        result = hungarian_macro_f1(true, predicted)

        self.assertEqual(result.macro_f1, 1.0)
        self.assertTrue(result.perfect_parcellation)
        np.testing.assert_allclose(result.per_class["f1"], 1.0)
        np.testing.assert_allclose(result.per_cluster["f1"], 1.0)

    def test_one_to_one_matching_penalizes_fragmentation(self):
        true = np.repeat(["a", "b", "c"], 4)
        predicted = [10, 10, 11, 11, 20, 20, 20, 20, 30, 30, 30, 30]
        result = hungarian_macro_f1(true, predicted)

        self.assertAlmostEqual(result.macro_f1, (2 / 3 + 1 + 1) / 3)
        self.assertFalse(result.perfect_parcellation)
        unmatched = result.per_cluster[result.per_cluster["matched_true_class"].isna()]
        self.assertEqual(len(unmatched), 1)
        self.assertEqual(unmatched.iloc[0]["f1"], 0.0)

    def test_noise_reduces_recall_and_macro_f1(self):
        true = ["a", "a", "b", "b"]
        predicted = [0, -1, 1, 1]
        result = hungarian_macro_f1(true, predicted)

        self.assertEqual(result.n_noise, 1)
        self.assertAlmostEqual(
            result.per_class.set_index("true_class").loc["a", "f1"],
            2 / 3,
        )
        self.assertAlmostEqual(result.macro_f1, (2 / 3 + 1) / 2)

    def test_all_noise_returns_zero(self):
        result = hungarian_macro_f1(["a", "b"], [-1, -1])
        self.assertEqual(result.macro_f1, 0.0)
        self.assertEqual(result.n_clusters, 0)
        self.assertEqual(result.n_noise, 2)


if __name__ == "__main__":
    unittest.main()
