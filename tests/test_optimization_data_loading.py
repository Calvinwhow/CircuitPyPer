"""The optimizer opens scoring data once and keeps its chosen storage mode."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from calvin_utils.neuroimaging_utils.ccm_utils.npy_utils import DataLoader
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.convergent_map_optimizer import (
    LocalizationOptimizer,
    NiftiOptimizer,
)


class CountingLoader(DataLoader):
    def __init__(self, path, mmap_mode=None):
        super().__init__(path, mmap_mode=mmap_mode)
        self.calls = []

    def load_dataset(self, dataset_name, nifti_type='niftis', mmap_mode=None):
        self.calls.append((dataset_name, mmap_mode))
        return super().load_dataset(dataset_name, nifti_type, mmap_mode=mmap_mode)


class OptimizationDataLoadingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        manifest = {}
        for index, name in enumerate(('a', 'b')):
            images = np.array(
                [[1.0 + index, 2.0, 3.0], [2.0, 1.0 + index, 4.0],
                 [3.0, 4.0, 1.0 + index], [4.0, 3.0, 2.0 + index]]
            )
            if name == 'a':
                outcome = np.array([[1.0], [3.0], [2.0], [4.0]])
            else:
                images = np.vstack((images, [[5.0, 2.0, 1.0]]))
                outcome = np.array([[5.0], [1.0], [4.0], [2.0], [3.0]])
            covariates = np.ones((len(images), 1))
            paths = {}
            for key, values in (
                ('niftis', images),
                ('indep_var', outcome),
                ('covariates', covariates),
            ):
                path = root / f'{name}_{key}.npy'
                np.save(path, values)
                paths[key] = str(path)
            manifest[name] = paths
        self.manifest_path = root / 'dataset_dict.json'
        self.manifest_path.write_text(json.dumps(manifest))
        self.maps = {'a': np.array([0.1, 0.2, 0.3]),
                     'b': np.array([0.3, 0.2, 0.1])}

    def test_loader_supports_default_and_per_call_memory_mapping(self):
        loader = DataLoader(self.manifest_path, mmap_mode='r')
        self.assertIsInstance(loader.load_dataset('a')['niftis'], np.memmap)
        self.assertNotIsInstance(
            loader.load_dataset('a', mmap_mode=None)['niftis'], np.memmap
        )

    def test_optimizer_reuses_memory_maps_for_each_score(self):
        loader = CountingLoader(self.manifest_path)
        optimizer = LocalizationOptimizer(
            self.maps, loader, data_mode='memmap', mode='unweighted'
        )
        self.assertIs(NiftiOptimizer, LocalizationOptimizer)
        self.assertIsInstance(optimizer.get_dataset('a')[0], np.memmap)
        self.assertEqual(loader.calls, [('a', 'r'), ('b', 'r')])

        avg_map = optimizer._converge_maps()
        X, _ = optimizer.get_dataset('a')
        expected_similarity = (
            (X @ avg_map.T).ravel()
            / (np.linalg.norm(X, axis=1) * np.linalg.norm(avg_map))
        )
        np.testing.assert_allclose(
            optimizer._calculate_similarity(X, avg_map), expected_similarity
        )
        optimizer.engine._rho_array(avg_map)
        optimizer.engine._rho_array(avg_map)
        self.assertEqual(loader.calls, [('a', 'r'), ('b', 'r')])

    def test_ram_mode_overrides_loader_default(self):
        loader = CountingLoader(self.manifest_path, mmap_mode='r')
        optimizer = LocalizationOptimizer(
            self.maps, loader, data_mode='ram', mode='unweighted'
        )
        self.assertNotIsInstance(optimizer.get_dataset('a')[0], np.memmap)
        self.assertEqual(loader.calls, [('a', None), ('b', None)])

    def test_rho_pairs_each_cohort_with_its_own_outcomes(self):
        optimizer = LocalizationOptimizer(
            self.maps, DataLoader(self.manifest_path),
            data_mode='memmap', mode='unweighted'
        )
        candidates = np.vstack((optimizer._converge_maps(), optimizer.MAPS[0]))
        observed = optimizer.engine._rho_array(candidates)
        self.assertEqual(observed.shape, (2, 2))
        for row, name in enumerate(optimizer.dataset_names):
            X, y = optimizer.get_dataset(name)
            expected = []
            for candidate in candidates:
                result = spearmanr(
                    optimizer._calculate_similarity(X, candidate[None, :]),
                    y.ravel(),
                )
                expected.append(getattr(result, 'statistic', result.correlation))
            np.testing.assert_allclose(observed[row], expected)

    def test_many_component_maps_are_independent_of_scoring_datasets(self):
        rng = np.random.default_rng(12)
        maps = {f'basis_{i}': rng.normal(size=3) for i in range(125)}
        sizes = {name: i + 1 for i, name in enumerate(maps)}
        loader = CountingLoader(self.manifest_path)
        optimizer = LocalizationOptimizer(
            maps, loader, data_mode='memmap', mode='weighted',
            map_sample_sizes=sizes,
        )
        candidates = rng.normal(size=(8, len(maps)))
        expected = optimizer.engine._rho_array(candidates @ optimizer.MAPS)
        observed = optimizer.engine._rho_for_weights(candidates)
        np.testing.assert_allclose(observed, expected, atol=1e-12)
        base_loss = optimizer.engine._loss_for_weights(optimizer.engine.W)
        gradient = optimizer.engine._forward_diff_grad(base_loss, batch_size=17)
        self.assertEqual(gradient.shape, optimizer.W.shape)
        self.assertEqual(loader.calls, [('a', 'r'), ('b', 'r')])

    def test_memory_mapping_rejects_nonfinite_data_without_copying(self):
        paths = json.loads(self.manifest_path.read_text())
        images = np.load(paths['a']['niftis'])
        images[0, 0] = np.nan
        np.save(paths['a']['niftis'], images)
        with self.assertRaisesRegex(ValueError, 'Memory-mapped data must be cleaned'):
            LocalizationOptimizer(
                self.maps, DataLoader(self.manifest_path),
                data_mode='memmap', mode='unweighted'
            )

if __name__ == '__main__':
    unittest.main()
