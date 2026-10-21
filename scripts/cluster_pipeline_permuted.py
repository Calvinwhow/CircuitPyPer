#!/usr/bin/env python3
"""Permutation inference for the symptom-map clustering pipeline.

This is the subject-level, inferential counterpart to ``cluster_pipeline.py``.
It does not permute already-created maps.  For the observed data and every
null realization it:

    1. fits one ranked imaging regression per clinical outcome;
    2. represents every outcome by its voxel/fiber t-statistic map;
    3. runs 3-D UMAP and HDBSCAN; and
    4. scores recovery of the prespecified outcome classes with
       Hungarian-aligned macro-F1.

Two clustering modes are available:

``fixed``
    Use one prespecified UMAP/HDBSCAN configuration.  This is appropriate when
    the parameters were chosen independently of the present inference.

``nested_search``
    Search a prespecified parameter grid on the observed data *and repeat the
    same search inside every permutation*.  The statistic is the maximum
    Hungarian macro-F1 over the grid.  This is the valid mode when several
    clustering solutions must be tried before choosing the observed one.

This is intentionally a CONFIG-style script, analogous to
``cluster_pipeline.py``.  Change values below or override the documented
environment variables, then run::

    python scripts/cluster_pipeline_permuted.py

Outputs are written under ``OUT_ROOT/<modality>/``.  The permutation runner is
checkpointed, so interrupted analyses can be resumed with the same settings.
"""

from __future__ import annotations

import json
import os
import sys
from itertools import product
from pathlib import Path

TMPDIR = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(TMPDIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMPDIR / "xdg_cache"))

import numpy as np
import pandas as pd

CIRCUIT_PYPER_DIR = Path(__file__).resolve().parents[1]
if str(CIRCUIT_PYPER_DIR) not in sys.path:
    sys.path.insert(0, str(CIRCUIT_PYPER_DIR))

from calvin_utils.ml_utils.brain_umap import BrainUmap
from calvin_utils.ml_utils.cluster_purity import hungarian_macro_f1
from calvin_utils.ml_utils.outcome_wise_imaging_regression import (
    OutcomeWiseImagingRegression,
)
from calvin_utils.ml_utils.permuted_cluster_purity import PermutedClusterPurity


# =============================================================================
# CONFIG
# =============================================================================

INPUT_PATH = os.environ.get(
    "SCA_INPUT_PATH",
    "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/"
    "optimized_master_list_filtered.csv",
)
RESULTS_ROOT = os.environ.get(
    "SCA_ROOT",
    "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimization/"
    "symptom_on_lhs",
)
OUT_ROOT = Path(
    os.environ.get(
        "SCA_PERMUTED_CLUSTER_OUT_ROOT",
        str(Path(RESULTS_ROOT) / "permuted_cluster_purity"),
    )
)

VOLUME_MASK = os.environ.get(
    "VOLUME_MASK",
    str(CIRCUIT_PYPER_DIR / "resources" / "MNI152_T1_2mm_brain_mask.nii"),
)
FIBER_ATLAS = os.environ.get(
    "FIBER_ATLAS",
    "/Volumes/OneTouch/resources/Atlas_tck_MNI/Atlas_all30_MNI.npz",
)

MODALITIES = [
    value
    for value in os.environ.get("ONLY_MODALITIES", "network,vlsm,fiber").split(",")
    if value
]
MODALITY_CONFIG = {
    "network": {
        "path_column": os.environ.get("NETWORK_PATH_COLUMN", "connectivity_t_path"),
        "mask_path": os.environ.get("MASK_NETWORK", VOLUME_MASK),
        "missing_policy": "complete_case",
        "outcome_sign": 1,
        "min_cluster_size": 5,
    },
    "vlsm": {
        "path_column": os.environ.get("VLSM_PATH_COLUMN", "Nifti_File_Path"),
        "mask_path": os.environ.get("MASK_VLSM", VOLUME_MASK),
        "missing_policy": "complete_case",
        "outcome_sign": 1,
        "min_cluster_size": 5,
    },
    "fiber": {
        "path_column": os.environ.get("FIBER_PATH_COLUMN", "fiber_path_guerrera"),
        "mask_path": os.environ.get("MASK_FIBER", FIBER_ATLAS),
        # These reproduce the approved historical fiber-map pipeline.
        "missing_policy": "zero",
        "outcome_sign": -1,
        "min_cluster_size": 7,
    },
}

# Prespecified reference classes.  The discovered HDBSCAN labels are never
# treated as ground truth; they are label-invariantly matched to these classes.
MOTOR_VARS = [
    "Gait",
    "HeelToShinTestLeft",
    "HeelToShinTestRight",
    "FingerToNoseTestLeft",
    "FingerToNoseTestRight",
    "LimbAtaxia",
    "Speech",
    "Oculomotor",
]
COGNITIVE_VARS = [
    "SematicFluencyRawS",
    "PhonemicFluencyRawS",
    "CategorySwitchRawS",
    "VerbalRegSum",
    "DigitSpanForwardRawS",
    "DigitSpanBackwardRawS",
    "CubeDrawRawS",
    "VerbalRecallRawS",
    "SimiliarityRawS",
    "GoNoGoRawS",
]
EMOTIONAL_VARS = [
    "Sec1ADifficultFocus",
    "Sec1AEasilyDistracted",
    "Sec1AOntheGo",
    "Sec1AFeelsCompelled",
    "Sec1AFeelsDriven",
    "Sec1BWorries",
    "Sec1BRepeats",
    "Sec1BMentallyStuck",
    "Sec1BCauseDistress",
    "Sec2AActHastily",
    "Sec2ARapidChanges",
    "Sec2ACryingLaughing",
    "Sec2AOverAnxious",
    "Sec2BLackOfPleasure",
    "Sec2BNegativeAttitude",
    "Sec2BUneasyWithLife",
    "Sec2BSadDepressed",
    "Sec3ARepetitiveMovements",
    "Sec3ASensoryExp",
    "Sec3BSensitive",
    "Sec3BOverwhelmed",
    "Sec4ACommunicates",
    "Sec4AConcerns",
    "Sec4ASeesHearsThings",
    "Sec4BTroubleUnderstand",
    "Sec4BDistant",
    "Sec4BIndifferent",
    "Sec5AAngry",
    "Sec5AUpset",
    "Sec5AIntolerant",
    "Sec5AArgumentative",
    "Sec5Bimmature",
    "Sec5BUnaware",
    "Sec5BManner",
    "Sec5BTrusting",
]

# Optional CSV with columns ``symptom`` and ``domain``.  Set this to use any
# N-class problem without editing the three lists above.
OUTCOME_DOMAIN_CSV = os.environ.get("OUTCOME_DOMAIN_CSV", "")

# Clinical and imaging ranks reproduce the historical symptom-map generation.
DATA_TRANSFORM_METHOD = os.environ.get("DATA_TRANSFORM_METHOD", "rank_ordinal")
OUTCOME_NAME_PATTERNS_TO_FLIP = [
    value
    for value in os.environ.get("COLS_TO_FLIP", "Raw,VerbalRegSum").split(",")
    if value
]

N_PERMUTATIONS = int(os.environ.get("N_PERMUTATIONS", "1000"))
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))
CHECKPOINT_EVERY = int(os.environ.get("CHECKPOINT_EVERY", "1"))
RESUME = os.environ.get("RESUME", "1").lower() not in {"0", "false", "no"}

# ``fixed`` tests a prespecified solution.  ``nested_search`` selects the best
# observed solution and repeats the identical selection process under every
# permutation, thereby accounting for the parameter search in the p-value.
CLUSTER_MODE = os.environ.get("PERMUTED_CLUSTER_MODE", "fixed").lower()

FIXED_PARAMS = {
    "n_components": 3,
    "n_neighbors": int(os.environ.get("FINAL_N_NEIGHBORS", "12")),
    "min_dist": float(os.environ.get("FINAL_MIN_DIST", "0.0")),
    "metric": os.environ.get("FINAL_METRIC", "cosine"),
    "projection": None,
    "mask": None,
    "cluster_voxels": False,
    "visualize_failed_clusters": True,
    "verbose": False,
    "random_state": int(os.environ.get("UMAP_RANDOM_STATE", "42")),
}

# Search dimensions.  UMAP is fitted once per combination of the first four
# lists, then HDBSCAN is cheaply rerun for every minimum cluster size.  A broad
# grid can be very expensive because it is repeated N_PERMUTATIONS + 1 times.
SEARCH_METRIC = [
    value
    for value in os.environ.get("SWEEP_METRIC", "cosine").split(",")
    if value
]
SEARCH_N_COMPONENTS = [
    int(value)
    for value in os.environ.get("SWEEP_N_COMPONENTS", "3").split(",")
    if value
]
SEARCH_N_NEIGHBORS = [
    int(value)
    for value in os.environ.get("SWEEP_N_NEIGHBORS", "8,12,20").split(",")
    if value
]
SEARCH_MIN_DIST = [
    float(value)
    for value in os.environ.get("SWEEP_MIN_DIST", "0.0,0.1").split(",")
    if value
]
SEARCH_MIN_CLUSTER_SIZE = [
    int(value)
    for value in os.environ.get("SWEEP_MIN_CLUSTER_SIZE", "3,5,7").split(",")
    if value
]


# =============================================================================
# PARAMETER-SELECTION CLUSTERER
# =============================================================================


class NestedParameterSearchClusterer:
    """Return the highest-purity clustering from a prespecified search grid.

    ``PermutedClusterPurity`` calls this object once for the observed maps and
    once for every permuted set of maps.  Selection is therefore nested inside
    the permutation loop, and the resulting null is a distribution of maxima.
    """

    def __init__(
        self,
        *,
        true_classes,
        candidates,
        out_dir,
        random_state=42,
        brain_umap_factory=BrainUmap,
    ):
        self.true_classes = np.asarray(true_classes, dtype=object)
        self.candidates = [dict(candidate) for candidate in candidates]
        if not self.candidates:
            raise ValueError("nested parameter search requires at least one candidate")
        if any(candidate.get("n_components") != 3 for candidate in self.candidates):
            raise ValueError("all candidates must use three UMAP components")
        self.out_dir = Path(out_dir)
        self.random_state = int(random_state)
        self.brain_umap_factory = brain_umap_factory
        self.n_calls = 0
        self.last_best_parameters = None

    @staticmethod
    def _embedding_key(candidate):
        return (
            candidate["metric"],
            candidate["n_components"],
            candidate["n_neighbors"],
            candidate["min_dist"],
        )

    def __call__(self, maps, _umap_params):
        rows = []
        best = None
        grouped = {}
        for candidate in self.candidates:
            grouped.setdefault(self._embedding_key(candidate), []).append(candidate)

        for (metric, n_components, n_neighbors, min_dist), candidates in grouped.items():
            first_size = candidates[0]["min_cluster_size"]
            try:
                umapper = self.brain_umap_factory(
                    data=maps,
                    n_components=n_components,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    metric=metric,
                    min_cluster_size=first_size,
                    projection=None,
                    mask=None,
                    cluster_voxels=False,
                    visualize_failed_clusters=False,
                    verbose=False,
                    random_state=self.random_state,
                )
            except Exception as exc:  # keep the rest of a prespecified grid usable
                for candidate in candidates:
                    rows.append({**candidate, "status": "failed", "error": str(exc)})
                continue

            for candidate in candidates:
                if candidate["min_cluster_size"] == first_size:
                    labels = np.asarray(umapper.cluster_labels)
                else:
                    labels, _, _ = umapper._run_hdbscan(  # reuse this UMAP embedding
                        candidate["min_cluster_size"], False
                    )
                    labels = np.asarray(labels)
                score = hungarian_macro_f1(
                    self.true_classes, labels, noise_label=-1
                )
                row = {
                    **candidate,
                    "status": "ok",
                    **score.summary(),
                }
                rows.append(row)

                # Candidate order is the deterministic final tie-breaker.
                rank = (
                    score.macro_f1,
                    -score.n_noise,
                    -abs(score.n_clusters - score.n_classes),
                )
                if best is None or rank > best[0]:
                    best = (rank, labels.copy(), dict(candidate), score)

        if best is None:
            failures = [row.get("error", "unknown failure") for row in rows]
            raise RuntimeError(
                "every nested clustering candidate failed; first failure: "
                + (failures[0] if failures else "none recorded")
            )

        _, best_labels, best_parameters, best_score = best
        self.last_best_parameters = best_parameters
        if self.n_calls == 0:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(
                self.out_dir / "observed_parameter_sweep.csv", index=False
            )
            payload = {
                **best_parameters,
                "hungarian_macro_f1": best_score.macro_f1,
                "n_clusters": best_score.n_clusters,
                "n_noise": best_score.n_noise,
            }
            (self.out_dir / "observed_best_parameters.json").write_text(
                json.dumps(payload, indent=2)
            )
        self.n_calls += 1
        return best_labels

    def analysis_metadata(self):
        return {
            "selection": "maximum Hungarian macro-F1 over a prespecified grid",
            "selection_repeated_for_every_permutation": True,
            "random_state": self.random_state,
            "n_candidates": len(self.candidates),
            "candidates": self.candidates,
        }


# =============================================================================
# PIPELINE
# =============================================================================


def load_outcome_classes(input_path=INPUT_PATH):
    """Return an ordered outcome-to-class mapping and validate source columns."""
    if OUTCOME_DOMAIN_CSV:
        table = pd.read_csv(OUTCOME_DOMAIN_CSV)
        required = {"symptom", "domain"}
        if not required.issubset(table.columns):
            raise ValueError(
                f"OUTCOME_DOMAIN_CSV must contain {sorted(required)}: "
                f"{OUTCOME_DOMAIN_CSV}"
            )
        if table["symptom"].duplicated().any():
            raise ValueError("OUTCOME_DOMAIN_CSV contains duplicate symptoms")
        outcome_classes = dict(
            zip(table["symptom"].astype(str), table["domain"].astype(str))
        )
    else:
        outcome_classes = {
            **{name: "motor" for name in MOTOR_VARS},
            **{name: "cognitive" for name in COGNITIVE_VARS},
            **{name: "emotional" for name in EMOTIONAL_VARS},
        }

    # Stable ordering matters for exactly reproducible UMAP embeddings.
    outcome_classes = dict(sorted(outcome_classes.items()))
    header = pd.read_csv(input_path, nrows=0).columns
    missing = [name for name in outcome_classes if name not in header]
    if missing:
        raise ValueError(f"input CSV is missing configured outcomes: {missing}")
    return outcome_classes


def search_candidates():
    """Build the Cartesian parameter grid in deterministic order."""
    return [
        {
            "metric": metric,
            "n_components": n_components,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "min_cluster_size": min_cluster_size,
        }
        for metric, n_components, n_neighbors, min_dist, min_cluster_size in product(
            SEARCH_METRIC,
            SEARCH_N_COMPONENTS,
            SEARCH_N_NEIGHBORS,
            SEARCH_MIN_DIST,
            SEARCH_MIN_CLUSTER_SIZE,
        )
    ]


def run_modality(modality, outcome_classes):
    """Run observed clustering and its complete regression-permutation null."""
    if modality not in MODALITY_CONFIG:
        raise ValueError(f"unknown modality: {modality}")
    configuration = MODALITY_CONFIG[modality]
    out_dir = OUT_ROOT / modality
    out_dir.mkdir(parents=True, exist_ok=True)
    outcomes = list(outcome_classes)

    print(f"[{modality}] loading imaging data and preparing outcome cohorts", flush=True)
    regression = OutcomeWiseImagingRegression(
        csv_path=INPUT_PATH,
        path_column=configuration["path_column"],
        outcome_names=outcomes,
        mask_path=configuration["mask_path"],
        transform=DATA_TRANSFORM_METHOD,
        missing_policy=configuration["missing_policy"],
        outcome_signs=configuration["outcome_sign"],
        observed_maps_path=out_dir / "observed_t_statistics.npy",
    )
    pd.DataFrame(
        {
            "outcome": outcomes,
            "true_class": [outcome_classes[name] for name in outcomes],
            "n_observations": regression.observation_counts,
        }
    ).to_csv(out_dir / "outcome_observation_counts.csv", index=False)

    params = {
        **FIXED_PARAMS,
        "min_cluster_size": configuration["min_cluster_size"],
    }
    clusterer = None
    if CLUSTER_MODE == "nested_search":
        candidates = search_candidates()
        n_embeddings = len(
            {
                (
                    item["metric"],
                    item["n_components"],
                    item["n_neighbors"],
                    item["min_dist"],
                )
                for item in candidates
            }
        )
        print(
            f"[{modality}] nested search: {len(candidates)} candidates, "
            f"{n_embeddings} UMAP fits per realization, "
            f"{n_embeddings * (N_PERMUTATIONS + 1):,} total UMAP fits",
            flush=True,
        )
        clusterer = NestedParameterSearchClusterer(
            true_classes=[outcome_classes[name] for name in outcomes],
            candidates=candidates,
            out_dir=out_dir,
            random_state=params["random_state"],
        )
    elif CLUSTER_MODE != "fixed":
        raise ValueError("PERMUTED_CLUSTER_MODE must be 'fixed' or 'nested_search'")

    analysis = PermutedClusterPurity(
        true_classes=outcome_classes,
        outcome_names=outcomes,
        out_dir=out_dir,
        regression=regression,
        clusterer=clusterer,
        umap_params=params,
        outcome_name_patterns_to_flip=OUTCOME_NAME_PATTERNS_TO_FLIP,
        random_seed=RANDOM_SEED,
        verbose=True,
    )
    result = analysis.run(
        n_permutations=N_PERMUTATIONS,
        resume=RESUME,
        checkpoint_every=CHECKPOINT_EVERY,
    )
    observed = result["observed"].macro_f1
    p_value = result["overall_inference"]["empirical_p_value"].iloc[0]
    print(
        f"[{modality}] complete: observed macro-F1={observed:.6f}; "
        f"empirical p={p_value:.6g}; results={out_dir}",
        flush=True,
    )
    return result


def main():
    if not MODALITIES:
        raise ValueError("ONLY_MODALITIES selected no modalities")
    unknown = [name for name in MODALITIES if name not in MODALITY_CONFIG]
    if unknown:
        raise ValueError(f"ONLY_MODALITIES contains unknown values: {unknown}")
    if N_PERMUTATIONS < 1:
        raise ValueError("N_PERMUTATIONS must be positive")

    outcome_classes = load_outcome_classes()
    counts = pd.Series(list(outcome_classes.values())).value_counts().to_dict()
    print(f"Input: {INPUT_PATH}")
    print(f"Output root: {OUT_ROOT}")
    print(f"Clustering mode: {CLUSTER_MODE}")
    print(
        f"Outcomes: {len(outcome_classes)} | "
        + ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
    )
    for modality in MODALITIES:
        run_modality(modality, outcome_classes)


if __name__ == "__main__":
    main()
