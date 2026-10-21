"""Tests for the checkpointed permutation cluster-purity runner."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from calvin_utils.ml_utils.permuted_cluster_purity import PermutedClusterPurity


class FakeRegression:
    def __init__(self):
        self.n_obs = 8
        self.n_outputs = 6
        self.exchangeability_blocks = None
        self.permutation_indices = []

    def run_single_multiout_regression(
        self, *, permutation=False, permutation_index=None
    ):
        if permutation:
            self.permutation_indices.append(np.asarray(permutation_index).copy())
        marker = len(self.permutation_indices)
        t_values = np.full((1, 4, self.n_outputs), marker, dtype=float)
        return np.zeros_like(t_values), t_values, np.zeros_like(t_values)


class SequencedClusterer:
    def __init__(self):
        self.calls = 0

    def __call__(self, maps, params):
        outputs = (
            [10, 10, 20, 20, 30, 30],
            [0, 0, 0, 1, 1, 1],
            [0, -1, 1, 1, 2, 2],
        )
        labels = outputs[self.calls]
        self.calls += 1
        return np.asarray(labels)


class PermutedClusterPurityTest(unittest.TestCase):
    def test_runner_calls_metric_checkpoints_and_calculates_empirical_p(self):
        regression = FakeRegression()
        clusterer = SequencedClusterer()
        with tempfile.TemporaryDirectory() as directory:
            runner = PermutedClusterPurity(
                true_classes=["a", "a", "b", "b", "c", "c"],
                outcome_names=[f"outcome_{index}" for index in range(6)],
                out_dir=directory,
                regression=regression,
                clusterer=clusterer,
                verbose=False,
            )
            results = runner.run(2, checkpoint_every=1)

            self.assertEqual(results["observed"].macro_f1, 1.0)
            self.assertEqual(len(regression.permutation_indices), 2)
            self.assertTrue(
                all(
                    np.array_equal(
                        np.sort(index, axis=1),
                        np.tile(np.arange(8), (6, 1)),
                    )
                    for index in regression.permutation_indices
                )
            )
            inference = results["overall_inference"].iloc[0]
            self.assertEqual(inference["n_null_at_least_observed"], 0)
            self.assertAlmostEqual(inference["empirical_p_value"], 1 / 3)

            root = Path(directory)
            expected = {
                "observed_purity.csv",
                "observed_per_class_f1.csv",
                "observed_per_cluster_f1.csv",
                "permutation_purity.csv",
                "permutation_per_class_f1.csv",
                "permutation_per_cluster_f1.csv",
                "purity_inference.csv",
                "matched_cluster_f1_inference.csv",
                "permutation_indices.npy",
                "permutation_cluster_labels.npy",
            }
            self.assertTrue(all((root / name).exists() for name in expected))
            self.assertEqual(len(pd.read_csv(root / "permutation_purity.csv")), 2)

    def test_resume_can_extend_an_existing_null_distribution(self):
        with tempfile.TemporaryDirectory() as directory:
            first = PermutedClusterPurity(
                true_classes=["a", "a", "b", "b", "c", "c"],
                outcome_names=[f"outcome_{index}" for index in range(6)],
                out_dir=directory,
                regression=FakeRegression(),
                clusterer=SequencedClusterer(),
                verbose=False,
            )
            first.run(1)

            regression = FakeRegression()
            resumed = PermutedClusterPurity(
                true_classes=["a", "a", "b", "b", "c", "c"],
                outcome_names=[f"outcome_{index}" for index in range(6)],
                out_dir=directory,
                regression=regression,
                clusterer=SequencedClusterer(),
                verbose=False,
            )
            results = resumed.run(3, resume=True)

            self.assertEqual(len(results["permutation_summary"]), 3)
            self.assertEqual(len(regression.permutation_indices), 2)
            self.assertEqual(
                np.load(Path(directory) / "permutation_cluster_labels.npy").shape,
                (3, 6),
            )

    def test_reads_outcome_names_from_regression_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = Path(directory) / "dataset_dict.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "neuroimaging_regression": {
                            "outcome_names": ["motor", "cognitive", "emotional"]
                        }
                    }
                )
            )
            regression = FakeRegression()
            regression.n_outputs = 3
            runner = PermutedClusterPurity(
                true_classes={
                    "motor": "motor",
                    "cognitive": "cognitive",
                    "emotional": "emotional",
                },
                json_path=metadata_path,
                out_dir=directory,
                regression=regression,
                clusterer=lambda maps, params: np.arange(3),
                verbose=False,
            )

            self.assertEqual(
                runner.outcome_names, ["motor", "cognitive", "emotional"]
            )

    def test_records_custom_clusterer_metadata(self):
        class MetadataClusterer:
            def __call__(self, maps, params):
                return np.array([0, 0, 1, 1, 2, 2])

            def analysis_metadata(self):
                return {"selection_repeated_for_every_permutation": True}

        with tempfile.TemporaryDirectory() as directory:
            runner = PermutedClusterPurity(
                true_classes=["a", "a", "b", "b", "c", "c"],
                outcome_names=[f"outcome_{index}" for index in range(6)],
                out_dir=directory,
                regression=FakeRegression(),
                clusterer=MetadataClusterer(),
                verbose=False,
            )
            runner.run(1)
            parameters = json.loads(
                (Path(directory) / "purity_analysis_parameters.json").read_text()
            )
            self.assertTrue(
                parameters["clusterer"][
                    "selection_repeated_for_every_permutation"
                ]
            )


if __name__ == "__main__":
    unittest.main()
