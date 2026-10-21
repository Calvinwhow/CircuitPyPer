"""Empirical cluster-purity testing by regression-outcome permutation.

Each null iteration independently shuffles the subject rows of every outcome,
refits all voxelwise regressions, reruns UMAP and HDBSCAN, and evaluates the
result with Hungarian-aligned macro-F1. This reproduces the permutation null
used by the original UMAP-regression workflow while making every shuffle seeded
and reproducible.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from calvin_utils.ml_utils.cluster_purity import hungarian_macro_f1


class PermutedClusterPurity:
    """Test N-class cluster recovery against a refitted permutation null.

    Parameters
    ----------
    true_classes : sequence or mapping
        Prespecified class for each regression output. A sequence must follow
        serialized output-channel order. A mapping maps every outcome name to
        its true class; names may be passed directly or read from new
        ``RegressionPrep`` metadata.
    out_dir : path-like
        Directory for observed scores, checkpoints, null results, and empirical
        p-values.
    json_path, mask_path : path-like, optional
        Serialized regression input and its mask. Required unless a prepared
        ``regression`` object is supplied.
    outcome_names : sequence, optional
        Ordered names corresponding to the regression output channels.
    umap_params : mapping, optional
        Parameters passed to :class:`BrainUmap`. Defaults reproduce the recent
        3D map-clustering setup (3 components, 12 neighbors, zero minimum
        distance, cosine metric, minimum cluster size 5, no projection).
    outcome_name_patterns_to_flip : sequence of str, optional
        Substrings identifying outcome maps whose sign should be reversed
        before every observed and permuted clustering.
    random_seed : int, default 42
        Seed for the reproducible sequence of subject permutations.
    contrast_index : int, default 0
        Regression contrast used to construct the map-by-voxel clustering
        input.
    regression : object, optional
        Prepared regression object. Primarily useful for dependency injection;
        it must implement ``run_single_multiout_regression`` and expose
        ``n_obs``, ``n_outputs``, and optionally ``exchangeability_blocks``.
    clusterer : callable, optional
        Override accepting ``(voxel_by_output_maps, umap_params)`` and returning
        one predicted label per output. By default, BrainUmap is used.
    """

    DEFAULT_UMAP_PARAMS = {
        "n_components": 3,
        "n_neighbors": 12,
        "min_dist": 0.0,
        "metric": "cosine",
        "projection": None,
        "min_cluster_size": 5,
        "mask": None,
        "cluster_voxels": False,
        "visualize_failed_clusters": True,
        "verbose": False,
        "random_state": 42,
    }

    def __init__(
        self,
        *,
        true_classes,
        out_dir,
        json_path=None,
        mask_path=None,
        outcome_names: Sequence[str] | None = None,
        umap_params: Mapping | None = None,
        outcome_name_patterns_to_flip: Sequence[str] | None = None,
        random_seed: int = 42,
        contrast_index: int = 0,
        regression_type: str = "linear",
        regression=None,
        clusterer=None,
        verbose: bool = True,
    ):
        self.out_dir = Path(out_dir)
        self.json_path = Path(json_path) if json_path is not None else None
        self.random_seed = int(random_seed)
        self.contrast_index = int(contrast_index)
        self.verbose = bool(verbose)
        if regression is None:
            if json_path is None:
                raise ValueError("json_path is required when regression is not supplied.")
            from calvin_utils.permutation_analysis_utils.voxelwise_regression import (
                VoxelwiseRegression,
            )

            regression = VoxelwiseRegression(
                json_path,
                mask_path=mask_path,
                out_dir=str(self.out_dir / "regression_maps"),
                regression_type=regression_type,
            )
        self.regression = regression
        self.clusterer = clusterer
        self.n_outputs = int(self.regression.n_outputs)
        self.n_observations = int(self.regression.n_obs)

        if outcome_names is None and self.json_path is not None:
            outcome_names = self._read_serialized_outcome_names(self.json_path)
        if outcome_names is None:
            outcome_names = [f"output_{index}" for index in range(self.n_outputs)]
        self.outcome_names = [str(name) for name in outcome_names]
        if len(self.outcome_names) != self.n_outputs:
            raise ValueError("outcome_names must contain one name per regression output.")

        if isinstance(true_classes, Mapping):
            missing = [name for name in self.outcome_names if name not in true_classes]
            if missing:
                raise ValueError(f"true_classes is missing outcomes: {missing}")
            resolved_classes = [true_classes[name] for name in self.outcome_names]
        else:
            resolved_classes = list(true_classes)
        if len(resolved_classes) != self.n_outputs:
            raise ValueError("true_classes must contain one class per regression output.")
        if pd.isna(np.asarray(resolved_classes, dtype=object)).any():
            raise ValueError("true_classes cannot contain missing values.")
        self.true_classes = np.asarray(
            [str(label) for label in resolved_classes], dtype=object
        )

        self.umap_params = dict(self.DEFAULT_UMAP_PARAMS)
        self.umap_params.update(dict(umap_params or {}))
        if self.umap_params.get("n_components") != 3:
            raise ValueError("permuted 3D cluster purity requires n_components=3.")
        if self.umap_params.get("cluster_voxels", False):
            raise ValueError("cluster_voxels must be False for outcome-map clustering.")
        if self.umap_params.get("mask") is not None:
            raise ValueError(
                "BrainUmap mask must be None because regression maps are already masked."
            )
        if self.umap_params.get("cols_to_flip"):
            raise ValueError(
                "Use outcome_name_patterns_to_flip instead of BrainUmap cols_to_flip."
            )

        patterns = list(outcome_name_patterns_to_flip or [])
        self.outcome_name_patterns_to_flip = [str(value) for value in patterns]
        self.flip_indices = [
            index
            for index, name in enumerate(self.outcome_names)
            if any(pattern in name for pattern in self.outcome_name_patterns_to_flip)
        ]
        self.observed_result = None
        self.observed_cluster_labels = None

    @staticmethod
    def _read_serialized_outcome_names(json_path):
        """Read optional output-channel names written by RegressionPrep."""
        with open(json_path, "r") as stream:
            specification = json.load(stream)["neuroimaging_regression"]
        names = specification.get("outcome_names")
        return None if names is None else [str(name) for name in names]

    @staticmethod
    def _atomic_csv(frame, path):
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)

    @staticmethod
    def _atomic_npy(array, path):
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with open(temporary, "wb") as stream:
            np.save(stream, array)
        os.replace(temporary, path)

    @staticmethod
    def _holm_adjust(p_values):
        values = np.asarray(p_values, dtype=float)
        order = np.argsort(values)
        ordered = values[order]
        corrected = np.maximum.accumulate(
            (len(values) - np.arange(len(values))) * ordered
        )
        restored = np.empty_like(corrected)
        restored[order] = np.minimum(corrected, 1.0)
        return restored

    def _prepare_clustering_maps(self, t_statistics):
        values = np.asarray(t_statistics, dtype=float)
        if values.ndim != 3:
            raise ValueError(
                "regression T statistics must have shape "
                "(contrasts, voxels, outputs)."
            )
        if not 0 <= self.contrast_index < values.shape[0]:
            raise ValueError("contrast_index is outside the regression output.")
        maps = values[self.contrast_index, :, :].copy()
        if maps.shape[1] != self.n_outputs:
            raise ValueError("regression output count changed during analysis.")
        if self.flip_indices:
            maps[:, self.flip_indices] *= -1.0
        return maps

    def _cluster_maps(self, maps):
        if self.clusterer is not None:
            labels = self.clusterer(maps, dict(self.umap_params))
        else:
            from calvin_utils.ml_utils.brain_umap import BrainUmap

            labels = BrainUmap(maps, **self.umap_params).cluster_labels
        labels = np.asarray(labels)
        if labels.ndim != 1 or len(labels) != self.n_outputs:
            raise ValueError("clusterer must return one label per regression output.")
        return labels

    def _fit_cluster_and_score(self, *, permutation_index=None):
        permutation = permutation_index is not None
        _, t_statistics, _ = self.regression.run_single_multiout_regression(
            permutation=permutation,
            permutation_index=permutation_index,
        )
        labels = self._cluster_maps(self._prepare_clustering_maps(t_statistics))
        result = hungarian_macro_f1(self.true_classes, labels, noise_label=-1)
        return labels, result

    def _draw_permutation_indices(self, n_permutations):
        rng = np.random.default_rng(self.random_seed)
        custom_draw = getattr(self.regression, "draw_permutation_indices", None)
        if custom_draw is not None:
            return custom_draw(n_permutations, rng)
        blocks = getattr(self.regression, "exchangeability_blocks", None)
        indices = np.empty(
            (n_permutations, self.n_outputs, self.n_observations), dtype=int
        )
        for iteration in range(n_permutations):
            for output_index in range(self.n_outputs):
                if blocks is None:
                    indices[iteration, output_index] = rng.permutation(
                        self.n_observations
                    )
                    continue
                block_values = np.asarray(blocks).ravel()
                if len(block_values) != self.n_observations:
                    raise ValueError(
                        "exchangeability_blocks must contain one value per observation."
                    )
                current = np.arange(self.n_observations)
                for block in pd.unique(block_values):
                    members = np.flatnonzero(block_values == block)
                    current[members] = rng.permutation(members)
                indices[iteration, output_index] = current
        return indices

    def _save_observed(self, labels, result):
        summary = pd.DataFrame([{"analysis": "observed", **result.summary()}])
        per_class = result.per_class.copy()
        per_class.insert(0, "analysis", "observed")
        per_cluster = result.per_cluster.copy()
        per_cluster.insert(0, "analysis", "observed")
        contingency = result.contingency.copy()
        contingency.to_csv(self.out_dir / "observed_contingency.csv")
        self._atomic_csv(summary, self.out_dir / "observed_purity.csv")
        self._atomic_csv(per_class, self.out_dir / "observed_per_class_f1.csv")
        self._atomic_csv(
            per_cluster, self.out_dir / "observed_per_cluster_f1.csv"
        )
        membership = pd.DataFrame(
            {
                "outcome": self.outcome_names,
                "true_class": self.true_classes,
                "predicted_cluster": labels,
            }
        )
        self._atomic_csv(
            membership, self.out_dir / "observed_cluster_membership.csv"
        )

    def _analysis_parameters(self, n_permutations):
        parameters = {
            "metric": "hungarian_macro_f1",
            "n_permutations": int(n_permutations),
            "random_seed": self.random_seed,
            "contrast_index": self.contrast_index,
            "outcome_names": self.outcome_names,
            "true_classes": self.true_classes.tolist(),
            "outcome_name_patterns_to_flip": self.outcome_name_patterns_to_flip,
            "flipped_outcomes": [
                self.outcome_names[index] for index in self.flip_indices
            ],
            "umap_params": self.umap_params,
            "permutation": (
                "an independent subject-row permutation for each outcome channel"
            ),
        }
        regression_metadata = getattr(self.regression, "analysis_metadata", None)
        if regression_metadata is not None:
            parameters["regression"] = regression_metadata()
        clusterer_metadata = getattr(self.clusterer, "analysis_metadata", None)
        if clusterer_metadata is not None:
            parameters["clusterer"] = clusterer_metadata()
        return json.loads(json.dumps(parameters, default=str))

    def _save_parameters(self, n_permutations, *, resume):
        parameters = self._analysis_parameters(n_permutations)
        path = self.out_dir / "purity_analysis_parameters.json"
        if resume and path.exists():
            previous = json.loads(path.read_text())
            comparable_previous = {
                key: value
                for key, value in previous.items()
                if key != "n_permutations"
            }
            comparable_current = {
                key: value
                for key, value in parameters.items()
                if key != "n_permutations"
            }
            if comparable_previous != comparable_current:
                raise ValueError(
                    "Existing checkpoint parameters do not match this analysis. "
                    "Use a new out_dir or run with resume=False."
                )
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(parameters, indent=2))
        os.replace(temporary, path)

    def _load_checkpoint(self, resume, n_permutations):
        paths = {
            "summary": self.out_dir / "permutation_purity.csv",
            "per_class": self.out_dir / "permutation_per_class_f1.csv",
            "per_cluster": self.out_dir / "permutation_per_cluster_f1.csv",
        }
        frames = {}
        for key, path in paths.items():
            frames[key] = (
                pd.read_csv(path) if resume and path.exists() else pd.DataFrame()
            )
            if not frames[key].empty:
                if "permutation" not in frames[key]:
                    raise ValueError(f"Invalid permutation checkpoint: {path}")
                frames[key] = frames[key].loc[
                    frames[key]["permutation"].astype(int) < n_permutations
                ].copy()
        if not frames["summary"].empty:
            frames["summary"] = frames["summary"].drop_duplicates(
                subset="permutation", keep="last"
            )
        for key in ("per_class", "per_cluster"):
            if not frames[key].empty:
                frames[key] = frames[key].drop_duplicates(keep="last")
        return paths, frames

    def _save_checkpoint(self, paths, frames, cluster_labels):
        self._atomic_csv(frames["summary"], paths["summary"])
        self._atomic_csv(frames["per_class"], paths["per_class"])
        self._atomic_csv(frames["per_cluster"], paths["per_cluster"])
        self._atomic_npy(
            cluster_labels,
            self.out_dir / "permutation_cluster_labels.npy",
        )

    def _calculate_inference(self, frames, n_permutations):
        summary = frames["summary"].sort_values("permutation")
        null_scores = summary["hungarian_macro_f1"].to_numpy(float)
        observed_score = self.observed_result.macro_f1
        n_exceeding = int(np.sum(null_scores >= observed_score - 1e-12))
        overall = pd.DataFrame(
            [
                {
                    "metric": "hungarian_macro_f1",
                    "observed": observed_score,
                    "n_permutations": n_permutations,
                    "n_null_at_least_observed": n_exceeding,
                    "empirical_p_value": (1 + n_exceeding)
                    / (n_permutations + 1),
                    "n_perfect_null": int(
                        np.sum(np.isclose(null_scores, 1.0, atol=1e-12, rtol=0.0))
                    ),
                    "perfect_null_rate": float(
                        np.mean(np.isclose(null_scores, 1.0, atol=1e-12, rtol=0.0))
                    ),
                }
            ]
        )
        self._atomic_csv(overall, self.out_dir / "purity_inference.csv")

        class_rows = []
        permutation_class = frames["per_class"]
        for observed_row in self.observed_result.per_class.to_dict("records"):
            true_class = str(observed_row["true_class"])
            null = permutation_class.loc[
                permutation_class["true_class"].astype(str) == true_class,
                "f1",
            ].to_numpy(float)
            observed_f1 = float(observed_row["f1"])
            exceed = int(np.sum(null >= observed_f1 - 1e-12))
            class_rows.append(
                {
                    "true_class": true_class,
                    "observed_cluster": observed_row["matched_cluster"],
                    "observed_f1": observed_f1,
                    "n_permutations": len(null),
                    "n_null_at_least_observed": exceed,
                    "empirical_p_value": (1 + exceed) / (len(null) + 1),
                }
            )
        class_inference = pd.DataFrame(class_rows)
        class_inference["empirical_p_value_holm"] = self._holm_adjust(
            class_inference["empirical_p_value"]
        )
        self._atomic_csv(
            class_inference,
            self.out_dir / "matched_cluster_f1_inference.csv",
        )
        return overall, class_inference

    def run(
        self,
        n_permutations: int = 1000,
        *,
        resume: bool = True,
        checkpoint_every: int = 1,
    ):
        """Run the observed analysis and the requested permutation null."""
        if not isinstance(n_permutations, (int, np.integer)) or n_permutations < 1:
            raise ValueError("n_permutations must be a positive integer.")
        if not isinstance(checkpoint_every, (int, np.integer)) or checkpoint_every < 1:
            raise ValueError("checkpoint_every must be a positive integer.")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._save_parameters(n_permutations, resume=resume)

        observed_labels, observed_result = self._fit_cluster_and_score()
        self.observed_cluster_labels = observed_labels
        self.observed_result = observed_result
        self._save_observed(observed_labels, observed_result)
        if self.verbose:
            print(
                "[PermutedClusterPurity] Observed Hungarian macro-F1: "
                f"{observed_result.macro_f1:.6f}"
            )
            print(observed_result.per_class.to_string(index=False))

        permutation_indices = self._draw_permutation_indices(n_permutations)
        self._atomic_npy(
            permutation_indices,
            self.out_dir / "permutation_indices.npy",
        )
        paths, frames = self._load_checkpoint(resume, n_permutations)
        completed = (
            set(frames["summary"]["permutation"].astype(int))
            if not frames["summary"].empty
            else set()
        )
        labels_path = self.out_dir / "permutation_cluster_labels.npy"
        if resume and labels_path.exists():
            saved_labels = np.load(labels_path)
            if saved_labels.ndim != 2 or saved_labels.shape[1] != self.n_outputs:
                raise ValueError(
                    "existing permutation_cluster_labels.npy has incompatible shape."
                )
            permutation_labels = np.full(
                (n_permutations, self.n_outputs), -2, dtype=int
            )
            rows_to_copy = min(n_permutations, saved_labels.shape[0])
            permutation_labels[:rows_to_copy] = saved_labels[:rows_to_copy]
        else:
            if resume and not frames["summary"].empty:
                raise ValueError(
                    "Permutation CSV checkpoint exists but its cluster-label "
                    "checkpoint is missing. Use resume=False to restart."
                )
            permutation_labels = np.full(
                (n_permutations, self.n_outputs), -2, dtype=int
            )

        since_checkpoint = 0
        for iteration in range(n_permutations):
            if iteration in completed:
                continue
            labels, result = self._fit_cluster_and_score(
                permutation_index=permutation_indices[iteration]
            )
            permutation_labels[iteration] = labels.astype(int)
            summary_row = {"permutation": iteration, **result.summary()}
            frames["summary"] = pd.concat(
                [frames["summary"], pd.DataFrame([summary_row])],
                ignore_index=True,
            )
            per_class = result.per_class.copy()
            per_class.insert(0, "permutation", iteration)
            frames["per_class"] = pd.concat(
                [frames["per_class"], per_class], ignore_index=True
            )
            per_cluster = result.per_cluster.copy()
            per_cluster.insert(0, "permutation", iteration)
            frames["per_cluster"] = pd.concat(
                [frames["per_cluster"], per_cluster], ignore_index=True
            )
            since_checkpoint += 1
            if self.verbose:
                print(
                    f"[PermutedClusterPurity] {iteration + 1}/{n_permutations}: "
                    f"macro-F1={result.macro_f1:.6f}, "
                    f"clusters={result.n_clusters}, noise={result.n_noise}"
                )
            if since_checkpoint >= checkpoint_every:
                self._save_checkpoint(paths, frames, permutation_labels)
                since_checkpoint = 0

        self._save_checkpoint(paths, frames, permutation_labels)
        frames["summary"] = frames["summary"].sort_values(
            "permutation"
        ).reset_index(drop=True)
        if len(frames["summary"]) != n_permutations:
            raise RuntimeError(
                "permutation checkpoint does not contain the requested number of runs."
            )
        overall, class_inference = self._calculate_inference(
            frames, n_permutations
        )
        if self.verbose:
            print("[PermutedClusterPurity] Overall inference:")
            print(overall.to_string(index=False))
            print("[PermutedClusterPurity] Matched-cluster inference:")
            print(class_inference.to_string(index=False))
            print(f"[PermutedClusterPurity] Results saved to: {self.out_dir}")
        return {
            "observed": observed_result,
            "permutation_summary": frames["summary"],
            "permutation_per_class": frames["per_class"],
            "permutation_per_cluster": frames["per_cluster"],
            "overall_inference": overall,
            "matched_cluster_inference": class_inference,
        }


__all__ = ["PermutedClusterPurity"]
