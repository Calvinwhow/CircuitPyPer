"""End-to-end contracts for voxelwise regression utilities and launchers."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import statsmodels.api as sm

from calvin_utils.permutation_analysis_utils.voxelwise_regression import (
    VoxelwiseRegression,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_regression_cli():
    path = PROJECT_ROOT / "scripts" / "05b_full_voxelwise_regression_cli.py"
    spec = importlib.util.spec_from_file_location("voxelwise_regression_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RegressionPipelineContractsTest(unittest.TestCase):
    def test_in_memory_tmap_matches_statsmodels(self):
        x = np.linspace(-2, 2, 12)
        X = np.column_stack([np.ones(len(x)), x])
        Y = np.column_stack(
            [
                1.5 + 2.0 * x + np.sin(np.arange(len(x))) * 0.1,
                -0.5 - 1.0 * x + np.cos(np.arange(len(x))) * 0.15,
            ]
        )
        observed = VoxelwiseRegression.fit_linear_tmap(
            X, Y, contrast=np.array([[0.0, 1.0]])
        )[0]
        expected = np.array([sm.OLS(Y[:, i], X).fit().tvalues[1] for i in range(2)])
        np.testing.assert_allclose(observed, expected, rtol=1e-4, atol=1e-4)

    def test_table_to_fit_and_cross_validation_pipeline(self):
        cli = load_regression_cli()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask_path = root / "mask.nii.gz"
            nib.save(
                nib.Nifti1Image(np.ones((2, 2, 2), dtype=np.float32), np.eye(4)),
                mask_path,
            )

            rows = []
            slopes = np.arange(1, 9, dtype=np.float32)
            for index, score in enumerate(np.linspace(-2, 2, 8)):
                image_path = root / f"sub-{index:02d}.nii.gz"
                noise = np.linspace(0, 0.1, 8) * ((index % 3) - 1)
                values = (1 + score * slopes + noise).reshape(2, 2, 2)
                nib.save(nib.Nifti1Image(values, np.eye(4)), image_path)
                rows.append((str(image_path), score))

            table_path = root / "subjects.csv"
            pd.DataFrame(rows, columns=["paths", "score"]).to_csv(
                table_path, index=False
            )
            out_dir = root / "regression"
            json_path, _, outcome, design, contrast = cli.prepare_from_table(
                input_path=str(table_path),
                out_dir=str(out_dir),
                sheet=None,
                formula="paths ~ score",
                voxelwise_vars=["paths"],
                voxelwise_interactions=[],
                add_intercept=True,
                drop_nans=["paths", "score"],
                drop_rows=None,
                one_hot=None,
                mask_path=str(mask_path),
                exchangeability_col=None,
                weights_col=None,
                data_transform_method=None,
                contrast_file=None,
                verbose=False,
            )

            self.assertEqual(list(outcome.columns), ["paths"])
            self.assertEqual(list(design.columns), ["Intercept", "score"])
            self.assertEqual(contrast.shape, (2, 2))
            payload = json.loads(Path(json_path).read_text())["neuroimaging_regression"]
            self.assertEqual(payload["neuroimaging_variables"], ["paths"])
            self.assertEqual(np.load(payload["design_matrix"]).shape, (8, 2, 1))
            self.assertEqual(np.load(payload["outcome_data"]).shape, (8, 1, 8))

            y_true, subject_files, regression_idx = cli._load_cv_inputs(
                json_path,
                dependent_source="auto",
                dependent_cols=None,
            )
            np.testing.assert_allclose(y_true, np.linspace(-2, 2, 8))
            self.assertEqual(subject_files.tolist(), [row[0] for row in rows])
            self.assertEqual(regression_idx, 0)

            cli.run_regression_from_json(
                json_path=json_path,
                out_dir=str(out_dir),
                mask_path=str(mask_path),
                regression_type="linear",
                n_permutations=0,
                all_outputs=False,
                cv="2",
                cv_dependent_source="auto",
                cv_dependent_cols=None,
            )

            self.assertTrue((out_dir / "beta_predictor_0.nii.gz").is_file())
            self.assertTrue((out_dir / "contrast_tval_1.nii.gz").is_file())
            self.assertTrue(
                (
                    out_dir
                    / "cross_validations"
                    / "scatterplots"
                    / "2_voxelwisemodel_aggregate-prediction_scatterplot.svg"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
