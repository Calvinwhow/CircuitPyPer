"""Tests for the nested parameter search used by the permuted pipeline."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "cluster_pipeline_permuted.py"
SPEC = importlib.util.spec_from_file_location("cluster_pipeline_permuted", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
NestedParameterSearchClusterer = MODULE.NestedParameterSearchClusterer


class FakeBrainUmap:
    init_calls = []
    hdbscan_calls = []

    def __init__(self, data, **kwargs):
        self.kwargs = kwargs
        self.__class__.init_calls.append(dict(kwargs))
        if kwargs["n_neighbors"] == 5:
            self.cluster_labels = np.array([0, 0, 0, 1, 1, 1])
        else:
            self.cluster_labels = np.array([-1, -1, -1, -1, -1, -1])

    def _run_hdbscan(self, min_cluster_size, cluster_voxels):
        self.__class__.hdbscan_calls.append((min_cluster_size, cluster_voxels))
        if self.kwargs["n_neighbors"] == 5 and min_cluster_size == 3:
            labels = np.array([10, 10, 20, 20, 30, 30])
        else:
            labels = self.cluster_labels
        return labels, np.ones(len(labels)), np.ones(3)


class NestedParameterSearchClustererTest(unittest.TestCase):
    def setUp(self):
        FakeBrainUmap.init_calls = []
        FakeBrainUmap.hdbscan_calls = []

    def test_selects_maximum_and_repeats_search_for_each_call(self):
        candidates = [
            {
                "metric": "cosine",
                "n_components": 3,
                "n_neighbors": 5,
                "min_dist": 0.0,
                "min_cluster_size": 2,
            },
            {
                "metric": "cosine",
                "n_components": 3,
                "n_neighbors": 5,
                "min_dist": 0.0,
                "min_cluster_size": 3,
            },
            {
                "metric": "cosine",
                "n_components": 3,
                "n_neighbors": 10,
                "min_dist": 0.0,
                "min_cluster_size": 2,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            clusterer = NestedParameterSearchClusterer(
                true_classes=["a", "a", "b", "b", "c", "c"],
                candidates=candidates,
                out_dir=directory,
                brain_umap_factory=FakeBrainUmap,
            )
            maps = np.zeros((4, 6))
            observed_labels = clusterer(maps, {})
            permuted_labels = clusterer(maps, {})

            np.testing.assert_array_equal(
                observed_labels, [10, 10, 20, 20, 30, 30]
            )
            np.testing.assert_array_equal(permuted_labels, observed_labels)
            self.assertEqual(len(FakeBrainUmap.init_calls), 4)
            self.assertEqual(clusterer.n_calls, 2)

            sweep = pd.read_csv(Path(directory) / "observed_parameter_sweep.csv")
            self.assertEqual(len(sweep), 3)
            best = json.loads(
                (Path(directory) / "observed_best_parameters.json").read_text()
            )
            self.assertEqual(best["min_cluster_size"], 3)
            self.assertEqual(best["hungarian_macro_f1"], 1.0)

            metadata = clusterer.analysis_metadata()
            self.assertTrue(metadata["selection_repeated_for_every_permutation"])
            self.assertEqual(metadata["n_candidates"], 3)

    def test_rejects_non_3d_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "three UMAP components"):
                NestedParameterSearchClusterer(
                    true_classes=["a", "b"],
                    candidates=[
                        {
                            "metric": "cosine",
                            "n_components": 2,
                            "n_neighbors": 5,
                            "min_dist": 0.0,
                            "min_cluster_size": 2,
                        }
                    ],
                    out_dir=directory,
                    brain_umap_factory=FakeBrainUmap,
                )


if __name__ == "__main__":
    unittest.main()
