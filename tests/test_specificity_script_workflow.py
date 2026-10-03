"""Tests for spreadsheet-backed script workflows and specificity orchestration."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pandas as pd

from calvin_utils.neuroimaging_utils.ccm_utils import symptom_specificity
from calvin_utils.neuroimaging_utils.ccm_utils.symptom_specificity import (
    NetworkSpecificityAnalysis,
)
from calvin_utils.permutation_analysis_utils.statsmodels_palm import (
    CalvinStatsmodelsPalm,
    SpreadsheetDataLoader,
)


class SpreadsheetWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.affine = np.eye(4)
        self.mask_path = self.root / "mask.nii.gz"
        nib.save(
            nib.Nifti1Image(
                np.array([[[1.0], [0.0]], [[1.0], [1.0]]]), self.affine
            ),
            self.mask_path,
        )

    def _write_image(self, name, values):
        path = self.root / name
        nib.save(
            nib.Nifti1Image(np.asarray(values, dtype=float).reshape(2, 2, 1), self.affine),
            path,
        )
        return str(path)

    def test_palm_prepares_rows_and_imports_the_prepared_image_column(self):
        paths = [
            self._write_image("subject_1.nii.gz", [1, 2, 3, 4]),
            self._write_image("subject_2.nii.gz", [4, 3, 2, 1]),
            self._write_image("subject_3.nii.gz", [2, 4, 6, 8]),
        ]
        csv_path = self.root / "input.csv"
        pd.DataFrame(
            {
                "image": paths,
                "outcome": [1.0, 2.0, np.nan],
                "group": ["keep", "drop", "keep"],
            }
        ).to_csv(csv_path, index=False)

        palm = CalvinStatsmodelsPalm(csv_path, self.root / "output")
        prepared = palm.prepare_spreadsheet_data(
            required_columns=["image", "outcome"],
            drop_rows=[("group", "equal", "drop")],
        )
        imported = palm.import_neuroimaging_data(
            file_column="image", mask_path=self.mask_path
        )

        self.assertEqual(prepared["outcome"].tolist(), [1.0])
        self.assertEqual(imported.shape, (3, 1))
        np.testing.assert_allclose(imported.iloc[:, 0], [1.0, 3.0, 4.0])

    def test_shared_loader_preserves_the_script_loader_contract(self):
        paths = [
            self._write_image("patient_1.nii.gz", [1, 2, 3, 4]),
            self._write_image("patient_2.nii.gz", [4, 3, 2, 1]),
        ]
        frame = pd.DataFrame(
            {"image": paths, "outcome": [10.0, 20.0], "age": [50.0, 60.0]}
        )
        loader = SpreadsheetDataLoader(
            frame,
            outcome_col="outcome",
            nifti_col="image",
            mask_path=self.mask_path,
            covariates_list=["age"],
            data_transform_method=None,
            dataset_name="cohort",
        )

        data = loader.load_dataset("cohort")
        self.assertEqual(data["niftis"].shape, (2, 3))
        np.testing.assert_allclose(data["niftis"][0], [1.0, 3.0, 4.0])
        np.testing.assert_allclose(data["indep_var"].ravel(), [10.0, 20.0])
        np.testing.assert_allclose(data["covariates"].ravel(), [50.0, 60.0])
        self.assertEqual(
            loader.load_dataset("cohort", nifti_type="niftis_ranked")["niftis"].shape,
            (2, 3),
        )

    def test_specificity_orchestrator_reproduces_notebook_data_flow(self):
        subject_paths = [
            self._write_image("case_1.nii.gz", [1, 0, 2, 3]),
            self._write_image("case_2.nii.gz", [2, 0, 3, 4]),
            self._write_image("case_3.nii.gz", [4, 0, 2, 1]),
            self._write_image("case_4.nii.gz", [3, 0, 1, 2]),
        ]
        csv_path = self.root / "specificity.csv"
        pd.DataFrame(
            {
                "image": subject_paths,
                "motor": [1.0, 2.0, 4.0, 3.0],
                "cognitive": [4.0, 3.0, 1.0, 2.0],
            }
        ).to_csv(csv_path, index=False)

        maps_dir = self.root / "maps"
        maps_dir.mkdir()
        target_path = maps_dir / "contrast_tval_0.nii.gz"
        nib.save(
            nib.Nifti1Image(
                np.array([1.0, 0.0, 0.5, 0.25]).reshape(2, 2, 1), self.affine
            ),
            target_path,
        )
        out_dir = self.root / "specificity_output"
        analysis = NetworkSpecificityAnalysis(
            input_path=csv_path,
            out_dir=out_dir,
            file_column="image",
            target_maps_directory=maps_dir,
            target_map_pattern="contrast_tval_0.nii.gz",
            mask_path=self.mask_path,
            label_dict={"motor": "Motor", "cognitive": "Cognitive"},
            required_columns=["image"],
            n_resamples=2,
            target_labels="Cognitive",
            target_name="Cognitive",
            other_name="Other",
            draw_comparison_plot=True,
        )

        with (
            patch.object(symptom_specificity.SpecificityAnalyzer, "run"),
            patch.object(
                symptom_specificity.SimpleBoxPlotWrapper,
                "plot",
                return_value=None,
            ) as plot,
        ):
            result = analysis.run()

        self.assertEqual(result.damage_scores.shape, (4, 1))
        self.assertEqual(len(result.observed_correlations), 2)
        self.assertEqual(list(result.group_comparison.columns), ["Cognitive", "Other"])
        plot.assert_called_once()
        analysis_dir = out_dir / "permutation"
        self.assertTrue((analysis_dir / "prepared_spreadsheet.csv").is_file())
        self.assertTrue((analysis_dir / "damage_scores.csv").is_file())
        self.assertTrue((analysis_dir / "observed_correlations.csv").is_file())
        self.assertTrue((analysis_dir / "group_comparison_r_values.csv").is_file())


if __name__ == "__main__":
    unittest.main()
