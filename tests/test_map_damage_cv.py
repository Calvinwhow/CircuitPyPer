"""Direct damage CV compares held-out native vectors with training t maps."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
from scipy.stats import rankdata

from calvin_utils.permutation_analysis_utils.map_damage_cv import cross_validated_map_damage
from calvin_utils.permutation_analysis_utils.voxelwise_regression import VoxelwiseRegression
from scripts.regression_pipeline import _load_native_cv_vectors


class MapDamageCVTest(unittest.TestCase):
    def test_inverse_regression_map_matches_existing_regression_math(self):
        rng = np.random.default_rng(91)
        patient_values = rng.normal(size=(9, 7)).astype(np.float32)
        patient_values[:4, 0] = 0  # tied fiber values
        for outcomes in (
            rng.normal(size=9).astype(np.float32),
            (np.arange(9) % 3 == 0).astype(np.float32),
        ):
            scores, folds = cross_validated_map_damage(patient_values, outcomes)
            self.assertEqual(folds.tolist(), list(range(1, 10)))

            regression = VoxelwiseRegression.__new__(VoxelwiseRegression)
            regression.contrast_matrix = np.ones((1, 1))
            regression.n_preds = 1
            expected = []
            for held_out in range(len(outcomes)):
                train = np.arange(len(outcomes)) != held_out
                image = rankdata(patient_values[train], axis=0).astype(np.float32)
                image -= image.mean(axis=0)
                design = outcomes[train]
                if np.unique(design).size > 2:
                    design = rankdata(design).astype(np.float32)
                    design -= design.mean()
                regression.n_obs = int(train.sum())
                _, t_map, _, _ = regression._linear_regression(
                    design[:, None], image, np.ones(train.sum())
                )
                patient = patient_values[held_out]
                t_map = t_map[0]
                expected.append(patient @ t_map / (
                    np.linalg.norm(patient) * np.linalg.norm(t_map)
                ))
            np.testing.assert_allclose(scores, expected, atol=1e-6)

    def test_held_out_outcome_cannot_change_own_damage_score(self):
        rng = np.random.default_rng(52)
        X = rng.normal(size=(12, 6)).astype(np.float32)
        y = X[:, 0] + rng.normal(scale=.2, size=12)
        first, _ = cross_validated_map_damage(X, y)
        changed = y.copy()
        changed[0] += 100
        second, _ = cross_validated_map_damage(X, changed)
        self.assertAlmostEqual(first[0], second[0], places=12)

    def test_native_comparators_keep_fiber_and_nifti_feature_spaces(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            fibers = np.empty(3, dtype=object)
            for index in range(3):
                fibers[index] = np.array([[index, 0, 0], [index, 1, 0]], dtype=np.float32)
            atlas = root / "atlas.npz"
            np.savez(atlas, fibers=fibers)
            fiber_paths = [root / f"patient_{index}.fib.npy" for index in range(2)]
            fiber_values = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
            for path, values in zip(fiber_paths, fiber_values):
                np.save(path, values)

            loaded_fibers = _load_native_cv_vectors(fiber_paths, atlas, "fiber")
            np.testing.assert_array_equal(loaded_fibers, fiber_values)
            with self.assertRaisesRegex(ValueError, "CV comparator is fiber"):
                _load_native_cv_vectors(fiber_paths, atlas, "nii")

            mask = root / "mask.nii.gz"
            selected = np.array([1, 0, 1, 0, 1, 1, 0, 1], dtype=np.float32)
            nib.save(nib.Nifti1Image(selected.reshape(2, 2, 2), np.eye(4)), mask)
            nifti_paths = [root / f"patient_{index}.nii.gz" for index in range(2)]
            image_values = np.array([
                [1, 10, 2, 10, 3, 4, 10, 5],
                [6, 10, 7, 10, 8, 9, 10, 11],
            ], dtype=np.float32)
            for path, values in zip(nifti_paths, image_values):
                nib.save(nib.Nifti1Image(values.reshape(2, 2, 2), np.eye(4)), path)

            loaded_niftis = _load_native_cv_vectors(nifti_paths, mask, "nii")
            np.testing.assert_array_equal(loaded_niftis, image_values[:, selected.astype(bool)])


if __name__ == "__main__":
    unittest.main()
