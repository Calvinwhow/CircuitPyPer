"""Behavioral contracts for small, reusable permutation utilities."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from calvin_utils.permutation_analysis_utils.get_palm_results import (
    aggregate_permutation_p_values,
)
from calvin_utils.permutation_analysis_utils.permute_dice_coefficient import (
    dice_coefficient,
)
from calvin_utils.permutation_analysis_utils.run_fsl_palm_script import (
    preprocess_colnames_for_regression,
    process_nifti_paths,
)


class PermutationUtilityContractTest(unittest.TestCase):
    def test_result_aggregation_sums_counts_and_normalizes_by_total_permutations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / "inputs"
            output_dir = root / "nested" / "outputs"
            input_dir.mkdir()
            pd.DataFrame([[1, 2], [3, 4]]).to_csv(
                input_dir / "batch_a_target.csv", header=False, index=False
            )
            pd.DataFrame([[3, 2], [1, 0]]).to_csv(
                input_dir / "batch_b_target.csv", header=False, index=False
            )

            result = aggregate_permutation_p_values(
                "target", num_perms=10, in_dir=str(input_dir), out_dir=str(output_dir)
            )

            np.testing.assert_allclose(result.to_numpy(), [[0.2, 0.2], [0.2, 0.2]])
            self.assertTrue((output_dir / "20_summed_targetp_values.csv").is_file())

    def test_result_aggregation_rejects_missing_inputs_and_invalid_counts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                aggregate_permutation_p_values("missing", 10, temp_dir, temp_dir)
            with self.assertRaises(ValueError):
                aggregate_permutation_p_values("missing", 0, temp_dir, temp_dir)

    def test_dice_coefficient_handles_overlap_empty_masks_and_shape_errors(self):
        first = np.array([1, 1, 0, 0])
        second = np.array([1, 0, 1, 0])
        self.assertEqual(dice_coefficient(first, second), 0.5)
        self.assertEqual(dice_coefficient(np.zeros(4), np.zeros(4)), 1.0)
        with self.assertRaises(ValueError):
            dice_coefficient(np.zeros(3), np.zeros(4))

    def test_palm_input_helpers_clean_columns_and_extract_subject_ids(self):
        dirty = pd.DataFrame({"1 value": [1], "group-name": [2]})
        cleaned = preprocess_colnames_for_regression(dirty)
        self.assertEqual(cleaned.columns.tolist(), ["var_1_value", "group_name"])

        with tempfile.TemporaryDirectory() as temp_dir:
            path_csv = Path(temp_dir) / "images.csv"
            paths = [
                "/data/sub-004/anatomical.nii.gz",
                "/data/control_27_mask.nii.gz",
                "/data/no_identifier.nii.gz",
            ]
            pd.Series(paths).to_csv(path_csv, header=False, index=False)
            extracted_paths, frame = process_nifti_paths(path_csv)

        self.assertEqual(extracted_paths, paths)
        self.assertEqual(frame.loc[0, "subject_id"], "004")
        self.assertEqual(frame.loc[1, "subject_id"], "27")
        self.assertTrue(pd.isna(frame.loc[2, "subject_id"]))


if __name__ == "__main__":
    unittest.main()
