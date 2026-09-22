#!/usr/bin/env python3
"""Batch fixed-map optimization on Schmahmann, then test maps in RCP.

Edit MODALITIES and TARGET_GROUPS, run ``--dry-run``, then run ``--run``.
CSV preparation is delegated to optimization_pipeline_fixed_maps; RCP lesion
scoring is delegated to the existing BIDS scoring script.
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy.stats import ConstantInputWarning, spearmanr

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import scripts.optimization_pipeline_fixed_maps as fixed


# =============================================================================
# CONFIG
# =============================================================================

SCA_ROOT = Path(
    "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimization/symptom_on_lhs"
)
SCA_CSV = Path(
    "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimzation/"
    "optimized_master_list_filtered_HigherIsWorse.csv"
)
RCP_METADATA = Path("/Volumes/OneTouch/01x_Dhand_RCPStrokes/metadata")
RCP_MASTER = RCP_METADATA / "master.csv"
RCP_SCORER_DIR = Path("/Volumes/OneTouch/01x_Dhand_RCPStrokes/BIDS/code")
OUTPUT_ROOT = SCA_ROOT / "fixed_map_transfer_domain_3way"
VOLUME_MASK = PROJECT / "resources/MNI152_T1_2mm_brain_mask.nii"
FIBER_ATLAS = Path("/Volumes/OneTouch/resources/Atlas_tck_MNI/Atlas_all30_MNI.npz")

MODALITIES = OrderedDict([
    ("network", ("connectivity_t_path", "nii", VOLUME_MASK)),
    ("vlsm", ("Nifti_File_Path", "nii", VOLUME_MASK)),
    ("fiber", ("fiber_path_guerrera", "fiber", FIBER_ATLAS)),
])

TARGET_GROUPS = OrderedDict([
    ("motor", [
        "Gait", "HeelToShinTestLeft", "HeelToShinTestRight",
        "FingerToNoseTestLeft", "FingerToNoseTestRight", "LimbAtaxia",
        "Speech", "Oculomotor",
    ]),
    ("cognitive", [
        "CategorySwitchRawS", "CubeDrawRawS", "DigitSpanBackwardRawS",
        "DigitSpanForwardRawS", "GoNoGoRawS", "PhonemicFluencyRawS",
        "SematicFluencyRawS", "SimiliarityRawS", "VerbalRecallRawS",
        "VerbalRegSum",
    ]),
    ("emotional", [
        "Sec1ADifficultFocus", "Sec1AEasilyDistracted", "Sec1AFeelsCompelled",
        "Sec1AFeelsDriven", "Sec1AOntheGo", "Sec1BCauseDistress",
        "Sec1BMentallyStuck", "Sec1BRepeats", "Sec1BWorries",
        "Sec2AActHastily", "Sec2ACryingLaughing", "Sec2AOverAnxious",
        "Sec2ARapidChanges", "Sec2BLackOfPleasure", "Sec2BNegativeAttitude",
        "Sec2BSadDepressed", "Sec2BUneasyWithLife", "Sec3ARepetitiveMovements",
        "Sec3ASensoryExp", "Sec3BOverwhelmed", "Sec3BSensitive",
        "Sec4ACommunicates", "Sec4AConcerns", "Sec4ASeesHearsThings",
        "Sec4BDistant", "Sec4BIndifferent", "Sec4BTroubleUnderstand",
        "Sec5AAngry", "Sec5AArgumentative", "Sec5AIntolerant", "Sec5AUpset",
        "Sec5BManner", "Sec5BTrusting", "Sec5BUnaware", "Sec5Bimmature",
    ]),
])

OUTER_FOLDS = 5
MAX_ITERS = 500
RENDER_GIF = False
GIF_OPTIONS = {"fps": 5, "max_frames": 60, "view": "auto"}


def all_targets():
    return list(OrderedDict.fromkeys(
        target for group in TARGET_GROUPS.values() for target in group
    ))


def component_maps(modality):
    image_col, output_type, _ = MODALITIES[modality]
    extension = "fib.npy" if output_type == "fiber" else "nii.gz"
    root = SCA_ROOT / f"{modality}_regressions_HigherIsWorse-Ranked"
    return OrderedDict(
        (target, str(root / f"{target}-on-{image_col}" / "regression" /
                     f"contrast_tval_0.{extension}"))
        for target in all_targets()
    )


def scoring_specs(group, image_col):
    return OrderedDict(
        (target, {
            "csv_path": str(SCA_CSV), "nifti_column": image_col,
            "symptom_column": target, "subject_column": "subid",
        })
        for target in TARGET_GROUPS[group]
    )


def optimized_map_path(modality, group):
    _, output_type, _ = MODALITIES[modality]
    extension = "fib.npy" if output_type == "fiber" else "nii.gz"
    return (OUTPUT_ROOT / "optimization" / f"{modality}_{group}"
            / f"final_model/optimized_map.{extension}")


def validate(modalities, groups):
    missing = []
    for modality in modalities:
        missing.extend(
            f"{modality}/{target}: {path}" for target, path in component_maps(modality).items()
            if not Path(path).is_file()
        )
    for path in (SCA_CSV, RCP_MASTER, VOLUME_MASK):
        if not path.is_file():
            missing.append(str(path))
    if "fiber" in modalities and not FIBER_ATLAS.is_file():
        missing.append(str(FIBER_ATLAS))
    print(f"Planned fits: {len(modalities) * len(groups)}")
    for modality in modalities:
        for group in groups:
            print(f"  {modality}/{group}: {len(component_maps(modality))} maps, "
                  f"{len(TARGET_GROUPS[group])} scoring outcomes")
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))


def run_optimization(modalities, groups, force, render_gif):
    for modality in modalities:
        image_col, output_type, mask = MODALITIES[modality]
        for group in groups:
            out_dir = OUTPUT_ROOT / "optimization" / f"{modality}_{group}"
            if optimized_map_path(modality, group).is_file() and not force:
                print(f"Reusing {modality}/{group}")
                continue
            fixed.MAP_FILES = component_maps(modality)
            fixed.SCORING_DATASETS = scoring_specs(group, image_col)
            fixed.SCORING_MANIFEST_PATH = None
            fixed.SUBJECT_COL = "subid"
            fixed.OUT_DIR = str(out_dir)
            fixed.MASK_PATH = str(mask)
            fixed.MAP_OUTPUT_TYPE = output_type
            fixed.OUTER_FOLDS = OUTER_FOLDS
            fixed.MAX_ITERS = MAX_ITERS
            fixed.RENDER_GIF = render_gif
            fixed.GIF_OPTIONS = dict(GIF_OPTIONS)
            fixed.main()


def heldout_outcomes():
    motor = {
        "Gait": "bars__bars_gait",
        "HeelToShinTestLeft": "bars__bars_left_heel_shin",
        "HeelToShinTestRight": "bars__bars_right_heel_shin",
        "FingerToNoseTestLeft": "bars__bars_left_finger_to_nose",
        "FingerToNoseTestRight": "bars__bars_right_finger_to_nose",
        "Speech": "bars__bars_speech", "Oculomotor": "bars__bars_oculomotor",
        "LimbAtaxia": [
            "bars__bars_left_heel_shin", "bars__bars_right_heel_shin",
            "bars__bars_left_finger_to_nose", "bars__bars_right_finger_to_nose",
        ],
    }
    cognitive = {
        target: f"ccas__{target}" for target in TARGET_GROUPS["cognitive"]
        if target != "VerbalRegSum"
    }
    emotional = {
        target: f"cnrs__{target}" for target in TARGET_GROUPS["emotional"]
        if target != "Sec2AActHastily"
    }
    return {"motor": motor, "cognitive": cognitive, "emotional": emotional}


def outcome_value(row, source, finite_number):
    if isinstance(source, list):
        values = [finite_number(row.get(column)) for column in source]
        return sum(values) if all(math.isfinite(value) for value in values) else math.nan
    return finite_number(row.get(source))


def fieldnames(rows):
    return list(OrderedDict.fromkeys(key for row in rows for key in row))


def run_heldout(modalities, groups):
    if str(RCP_SCORER_DIR) not in sys.path:
        sys.path.insert(0, str(RCP_SCORER_DIR))
    from score_lesions_against_fwe_maps import (
        fdr_bh, finite_number, load_csv, prepare_maps, score_rows, write_csv,
    )
    rows, fields = load_csv(RCP_MASTER)
    specs = []
    for modality in modalities:
        for group in groups:
            path = optimized_map_path(modality, group)
            if not path.is_file():
                raise FileNotFoundError(path)
            specs.append({
                "map_id": f"{modality}_{group}", "modality": modality,
                "map_domain": group, "contrast_index": "optimized",
                "map_path": str(path),
                "score_column": f"damage_cosine_fixedopt_{modality}_{group}",
            })
    qc = score_rows(rows, specs, prepare_maps(specs))
    reports = []
    outcomes = heldout_outcomes()
    for spec in specs:
        damage = np.asarray([finite_number(row.get(spec["score_column"])) for row in rows])
        current = []
        for outcome, source in outcomes[spec["map_domain"]].items():
            values = np.asarray([outcome_value(row, source, finite_number) for row in rows])
            valid = np.isfinite(damage) & np.isfinite(values)
            rho = pvalue = math.nan
            if valid.sum() >= 3 and np.unique(damage[valid]).size > 1 and np.unique(values[valid]).size > 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConstantInputWarning)
                    result = spearmanr(damage[valid], values[valid])
                rho, pvalue = float(result.correlation), float(result.pvalue)
            current.append({
                "map_id": spec["map_id"], "modality": spec["modality"],
                "target_group": spec["map_domain"], "map_path": spec["map_path"],
                "outcome": outcome,
                "outcome_source": " + ".join(source) if isinstance(source, list) else source,
                "n": int(valid.sum()), "spearman_rho": rho, "spearman_p": pvalue,
            })
        for row, adjusted in zip(current, fdr_bh([row["spearman_p"] for row in current])):
            row["spearman_p_fdr_bh_within_map"] = adjusted
        reports.extend(current)
    out = OUTPUT_ROOT / "heldout_rcp"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "heldout_predictions.csv", rows,
              fields + [spec["score_column"] for spec in specs])
    write_csv(out / "heldout_scoring_qc.csv", qc, fieldnames(qc))
    write_csv(out / "heldout_correlations.csv", reports, fieldnames(reports))
    write_csv(
        out / "map_manifest.csv", specs,
        ["map_id", "modality", "map_domain", "contrast_index", "map_path",
         "score_column", "shape", "voxel_size_mm", "nonzero_voxels_in_mask"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--render-gif", action=argparse.BooleanOptionalAction,
        default=RENDER_GIF,
    )
    parser.add_argument("--modalities", nargs="+", choices=MODALITIES, default=list(MODALITIES))
    parser.add_argument("--groups", nargs="+", choices=TARGET_GROUPS, default=list(TARGET_GROUPS))
    args = parser.parse_args()
    validate(args.modalities, args.groups)
    if args.dry_run or not args.run:
        return
    run_optimization(args.modalities, args.groups, args.force, args.render_gif)
    run_heldout(args.modalities, args.groups)
    print(f"Results: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
