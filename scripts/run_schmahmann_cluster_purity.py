#!/usr/bin/env python3
"""Run outcome-permuted cluster-purity analyses for the Schmahmann data."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import pandas as pd

from calvin_utils.ml_utils.outcome_wise_imaging_regression import (
    OutcomeWiseImagingRegression,
)
from calvin_utils.ml_utils.permuted_cluster_purity import PermutedClusterPurity


PROJECT_ROOT = Path(
    "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimization/"
    "symptom_on_lhs"
)
MASTER_CSV = Path(
    "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/"
    "optimized_master_list_filtered.csv"
)
VOLUME_MASK = Path(
    "/Users/cu135/Software_Local/calvin_utils_project/circuit_pyper/"
    "resources/MNI152_T1_2mm_brain_mask.nii"
)
FIBER_ATLAS = Path("/Volumes/OneTouch/resources/Atlas_tck_MNI/Atlas_all30_MNI.npz")

MODALITIES = {
    "vlsm": {
        "path_column": "Nifti_File_Path",
        "mask_path": VOLUME_MASK,
        "reference_csv": PROJECT_ROOT
        / "vlsm_regressions_clusters/cluster_regression_input.csv",
        "min_cluster_size": 5,
        "missing_policy": "complete_case",
        "outcome_sign": 1,
    },
    "network": {
        "path_column": "connectivity_t_path",
        "mask_path": VOLUME_MASK,
        "reference_csv": PROJECT_ROOT
        / "network_regressions-HigherIsBetter_Clusters_Rerun/"
        "cluster_regression_input.csv",
        "min_cluster_size": 5,
        "missing_policy": "complete_case",
        "outcome_sign": 1,
    },
    "fiber": {
        "path_column": "fiber_path_guerrera",
        "mask_path": FIBER_ATLAS,
        "reference_csv": PROJECT_ROOT
        / "fiber_regressions_clusters/cluster_regression_input.csv",
        "min_cluster_size": 7,
        # The approved three-cluster fiber solution used the historical
        # HigherIsWorse-Ranked pipeline: all outcomes inverted, missing values
        # retained as zero, then stable ordinal ranking.
        "missing_policy": "zero",
        "outcome_sign": -1,
    },
}


def _load_reference(path):
    reference = pd.read_csv(path)
    required = {"symptom", "domain"}
    if not required.issubset(reference.columns):
        raise ValueError(f"Reference CSV lacks {sorted(required)}: {path}")
    if reference["symptom"].duplicated().any():
        raise ValueError(f"Reference CSV contains duplicate symptoms: {path}")
    return reference


def run_modality(
    modality,
    *,
    n_permutations,
    output_root,
    resume,
    checkpoint_every,
    random_seed,
):
    configuration = MODALITIES[modality]
    reference = _load_reference(configuration["reference_csv"])
    outcomes = reference["symptom"].astype(str).tolist()
    true_classes = dict(
        zip(reference["symptom"].astype(str), reference["domain"].astype(str))
    )
    output_directory = output_root / modality
    output_directory.mkdir(parents=True, exist_ok=True)

    print(f"[{modality}] Loading source imaging and preparing outcome cohorts...")
    regression = OutcomeWiseImagingRegression(
        csv_path=MASTER_CSV,
        path_column=configuration["path_column"],
        outcome_names=outcomes,
        mask_path=configuration["mask_path"],
        # Historical symptom-map generation used stable ordinal ranks rather
        # than average ranks for tied clinical scores and imaging values.
        transform="rank_ordinal",
        missing_policy=configuration["missing_policy"],
        outcome_signs=configuration["outcome_sign"],
        observed_maps_path=output_directory / "observed_t_statistics.npy",
    )
    cohort_counts = pd.DataFrame(
        {
            "outcome": outcomes,
            "true_class": [true_classes[name] for name in outcomes],
            "n_observations": regression.observation_counts,
        }
    )
    cohort_counts.to_csv(output_directory / "outcome_observation_counts.csv", index=False)

    analysis = PermutedClusterPurity(
        true_classes=true_classes,
        outcome_names=outcomes,
        out_dir=output_directory,
        regression=regression,
        random_seed=random_seed,
        outcome_name_patterns_to_flip=["Raw", "VerbalRegSum"],
        umap_params={
            "n_components": 3,
            "n_neighbors": 12,
            "min_dist": 0.0,
            "metric": "cosine",
            "projection": None,
            "min_cluster_size": configuration["min_cluster_size"],
            "mask": None,
            "cluster_voxels": False,
            "visualize_failed_clusters": True,
            "verbose": False,
            "random_state": 42,
        },
        verbose=True,
    )
    results = analysis.run(
        n_permutations=n_permutations,
        resume=resume,
        checkpoint_every=checkpoint_every,
    )
    print(
        f"[{modality}] complete: observed macro-F1="
        f"{results['observed'].macro_f1:.6f}; results={output_directory}"
    )
    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modality",
        action="append",
        choices=tuple(MODALITIES),
        help="Modality to run; repeat the flag for several. Default: all three.",
    )
    parser.add_argument("--n-permutations", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "permuted_cluster_purity_from_master",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore existing checkpoints and restart the requested analyses.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    modalities = args.modality or list(MODALITIES)
    for modality in modalities:
        run_modality(
            modality,
            n_permutations=args.n_permutations,
            output_root=args.output_root,
            resume=not args.restart,
            checkpoint_every=args.checkpoint_every,
            random_seed=args.random_seed,
        )
        gc.collect()


if __name__ == "__main__":
    main()
