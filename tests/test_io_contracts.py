"""Contract tests for the shared filesystem, spreadsheet, and NIfTI utilities."""

import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from calvin_utils.file_utils.csv_prep import CSVComposer
from calvin_utils.file_utils.dataframe_utilities import (
    preprocess_colnames_for_regression,
)
from calvin_utils.file_utils.file_path_collector import (
    glob_file_paths,
    glob_multiple_file_paths,
)
from calvin_utils.file_utils.import_functions import GiiNiiFileImport
from calvin_utils.neuroimaging_utils.nifti_utils.volume_io import NiftiIO


class IOContractsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mask_path = self.root / "mask.nii.gz"
        self.mask = np.array([1, 0, 1, 1, 0, 1, 0, 1], dtype=np.float32)
        nib.save(
            nib.Nifti1Image(self.mask.reshape(2, 2, 2), np.eye(4)),
            self.mask_path,
        )

    def _image(self, name, offset=0):
        path = self.root / name
        values = np.arange(8, dtype=np.float32) + offset
        nib.save(nib.Nifti1Image(values.reshape(2, 2, 2), np.eye(4)), path)
        return path, values

    def test_nifti_importer_preserves_file_order_and_applies_mask(self):
        first, first_values = self._image("sub-01_map.nii.gz")
        second, second_values = self._image("sub-02_map.nii.gz", offset=10)
        imported = GiiNiiFileImport(
            import_path=pd.Series([str(second), str(first)]),
            mask_path=str(self.mask_path),
            process_special_values=False,
        ).run()

        self.assertEqual(imported.shape, (int(self.mask.sum()), 2))
        np.testing.assert_array_equal(
            imported.iloc[:, 0], second_values[self.mask.astype(bool)]
        )
        np.testing.assert_array_equal(
            imported.iloc[:, 1], first_values[self.mask.astype(bool)]
        )

    def test_nifti_mask_and_unmask_round_trip(self):
        full = np.arange(8, dtype=np.float32)
        _, indices, masked = NiftiIO.mask_array(full, str(self.mask_path))
        restored = NiftiIO.unmask_array(
            masked, str(self.mask_path), fill_value=-1
        )
        np.testing.assert_array_equal(restored[indices], full[indices])
        np.testing.assert_array_equal(restored[~indices], -1)

    def test_path_collectors_find_and_optionally_save_files(self):
        self._image("a.nii.gz")
        self._image("b.nii.gz")
        single = glob_file_paths(str(self.root), "*.nii.gz", save=True)
        multiple = glob_multiple_file_paths(
            {str(self.root): "*.nii.gz"},
            save=True,
            save_path=str(self.root / "combined.csv"),
        )
        self.assertEqual(set(single["paths"]), set(multiple["paths"]))
        self.assertTrue((self.root / "path_df.csv").is_file())
        self.assertTrue((self.root / "combined.csv").is_file())

    def test_csv_composer_matches_bids_subjects_without_reusing_files(self):
        image_4, _ = self._image("sub-004_T1w.nii.gz")
        image_40, _ = self._image("sub-040_T1w.nii.gz")
        table = self.root / "clinical.csv"
        pd.DataFrame({"id": [4, 40], "age": [50, 60]}).to_csv(table, index=False)
        composer = CSVComposer(
            {
                "study": {
                    "nifti_path": str(self.root / "sub-*_T1w.nii.gz"),
                    "csv_path": str(table),
                    "subj_col": "id",
                    "covariate_col": {"Age": "age"},
                }
            },
            allow_loose_match=False,
        )
        composer.compose_df()
        rows = composer.composed_df.set_index("Subject")
        self.assertEqual(rows.loc["4", "Nifti_File_Path"], str(image_4))
        self.assertEqual(rows.loc["40", "Nifti_File_Path"], str(image_40))
        self.assertEqual(rows.loc["4", "Age"], 50)

    def test_regression_column_cleanup_is_deterministic(self):
        frame = pd.DataFrame(columns=["1 score", "group-name", "a/b.c"])
        cleaned = preprocess_colnames_for_regression(frame)
        self.assertEqual(
            list(cleaned.columns), ["var_1_score", "group_name", "a_b_c"]
        )


if __name__ == "__main__":
    unittest.main()
