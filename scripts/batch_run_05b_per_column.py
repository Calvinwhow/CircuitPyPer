#!/usr/bin/env python3
"""
Minimal batch wrapper around `05b_full_voxelwise_regression.py`.

For each selected column:
1) write a two-column CSV: `paths`, `<predictor>`
2) fit `paths ~ <predictor>`
3) run `reg.run_cross_validation(y_true=<predictor series>, subject_files=<paths series>)`
4) write one row to a summary CSV
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd


_TMPDIR = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_TMPDIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_TMPDIR / "xdg_cache"))


_SCRIPTS_DIR = Path(__file__).resolve().parent
_CIRCUIT_PYPER_DIR = _SCRIPTS_DIR.parent
if str(_CIRCUIT_PYPER_DIR) not in sys.path:
    sys.path.insert(0, str(_CIRCUIT_PYPER_DIR))

from calvin_utils.permutation_analysis_utils.voxelwise_regression import VoxelwiseRegression


CV_LABELS = ["loocv", "2", "5", "10", "leave_all_in"]
AVERAGE_CV_LABELS = ["loocv", "2", "5", "10"]


SCRIPT_ARGS: dict[str, Any] = {
    "input_csv": "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/master_list_redo.csv",
    "mask_path": "/Users/cu135/Software_Local/calvin_utils_project/circuit_pyper/resources/MNI152_T1_2mm_brain_mask.nii",
    "paths_col": "onetouch_path",
    "out_root": "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/cross_validations/batch_loocv_0perms_n34",
    "summary_csv": "batch_05b_summary.csv",
    "columns": [
        "Gait",
        "HeelToShinTestLeft",
        "HeelToShinTestRight",
        "FingerToNoseTestLeft",
        "FingerToNoseTestRight",
        "LimbAtaxia",
        "Speech",
        "Oculomotor",
        "TotalBarsScore",
        "delta_days_bars",
        "delta_days_ccas_x",
        "SematicFluencyRawS",
        "SematicFluencyFailS",
        "PhonemicFluencyRawS",
        "PhonemicFluencyFailS",
        "CategorySwitchRawS",
        "CategorySwitchFailS",
        "VerbalRegSum",
        "DigitSpanForwardRawS",
        "DigitSpanForwardFailS",
        "DigitSpanBackwardRawS",
        "DigitSpanBackwardFailS",
        "CubeDrawRawS",
        "CubeDrawFailS",
        "VerbalRecallRawS",
        "VerbalRecallFailS",
        "SimiliarityPair1RawS",
        "SimiliarityPair2RawS",
        "SimiliarityPair3RawS",
        "SimiliarityPair4RawS",
        "SimiliarityRawS",
        "SimiliarityFailS",
        "GoNoGoRawS",
        "GoNoGoFailS",
        "AFFECTAssessments_Angryoraggress",
        "AFFECTAssessments_Difficultywith",
        "AFFECTAssessments_Emotionallylab",
        "AFFECTAssessments_Expressesillog",
        "AFFECTAssessments_Lacksempathyis",
        "AFFECTAssessments_Showseasysenso",
        "AffectRawS",
        "AffectFailS",
        "TotalCCASRawScore",
        "TotalCCASFailScore",
        "MemoryTotal",
        "ExecutiveTotal",
        "LanguageTotal",
        "VisualTotal",
        "AffectTotal",
        "delta_days_ccas_y",
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
        "ScoreCol1A",
        "ScoreCol1B",
        "ScoreCol2A",
        "ScoreCol2B",
        "ScoreCol3A",
        "ScoreCol3B",
        "ScoreCol4A",
        "ScoreCol4B",
        "ScoreCol5A",
        "ScoreCol5B",
        "TotalSection1Score",
        "TotalSection2Score",
        "TotalSection3Score",
        "TotalSection4Score",
        "TotalSection5Score",
        "CNRSTotColAScore",
        "CNRSTotColBScore",
        "CNRSTotScore",
    ],
    "all_numeric": False,
    "exclude_cols": [
        "selected",
        "subid",
        "session",
        "id",
        "local_id",
        "paths",
        "onetouch_path",
        "_resolved_path",
    ],
    "filter_col": "selected",
    "filter_value": 1,
    "drop_nans": ["cosine_Total_BARS", "TotalCCASFailScore", "AffectFailS"],
    "cv": 'loocv',
    "n_permutations": 0,
    "data_transform": None,
    "verbose": True,
}


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")


def _normalize_cv(raw: Any) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text in {"", "0", "none", "off", "false", "no"}:
        return None
    return str(raw)


def _extract_metrics(svg_path: Path) -> dict[str, float]:
    text = " ".join(svg_path.read_text(errors="ignore").split())

    def grab(pattern: str) -> float:
        match = re.search(pattern, text)
        return float(match.group(1)) if match else float("nan")

    return {
        "spearman_rho": grab(r"Rho\s*=\s*([+-]?\d+(?:\.\d+)?)"),
        "spearman_p": grab(r"Rho\s*=\s*[+-]?\d+(?:\.\d+)?\s*,\s*p\s*=\s*([0-9.eE+-]+)"),
        "pearson_r": grab(r"R\s*=\s*([+-]?\d+(?:\.\d+)?)"),
        "pearson_p": grab(r"R\s*=\s*[+-]?\d+(?:\.\d+)?\s*,\s*p\s*=\s*([0-9.eE+-]+)"),
        "rmse": grab(r"RMSE\s*=\s*([0-9.eE+-]+)"),
        "mae": grab(r"MAE\s*=\s*([0-9.eE+-]+)"),
    }


def _strip_nii_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def _find_fwe_maps(out_dir: Path) -> list[Path]:
    return sorted(out_dir.glob("contrast_tval_FWE_*.nii*"))


def _dot_product_with_mask(map_path: Path, mask_data: np.ndarray) -> float:
    map_data = np.asanyarray(nib.load(str(map_path)).dataobj)
    return float(np.nansum(map_data * mask_data))


def _append_summary_row(summary_csv: Path, row: dict[str, Any]) -> None:
    if summary_csv.exists():
        summary_df = pd.read_csv(summary_csv)
        if "predictor" in summary_df.columns and "predictor" in row:
            summary_df = summary_df[summary_df["predictor"] != row["predictor"]]
        summary_df = pd.concat([summary_df, pd.DataFrame([row])], ignore_index=True)
    else:
        summary_df = pd.DataFrame([row])
    summary_df.to_csv(summary_csv, index=False)


def _find_cv_plot(out_dir: Path, cv: str) -> Path:
    scatter_dir = out_dir / "cross_validations" / "scatterplots"
    matches = sorted(scatter_dir.glob(f"{cv}_contrast_0_correlation*scatterplot.svg"))
    if not matches:
        raise FileNotFoundError(f"No CV scatterplot found in {scatter_dir}")
    return matches[0]


def _valid_cv_labels(n_obs: int) -> list[str]:
    labels = ["loocv", "2", "leave_all_in"]
    if n_obs >= 5:
        labels.insert(2, "5")
    if n_obs >= 10:
        labels.insert(3, "10")
    return labels


def _cv_arg_from_label(cv_label: str) -> str | int:
    if cv_label in {"2", "5", "10"}:
        return int(cv_label)
    return cv_label


def _all_cv_plots_exist(out_dir: Path, cv_labels: list[str]) -> bool:
    scatter_dir = out_dir / "cross_validations" / "scatterplots"
    return scatter_dir.exists() and all(any(scatter_dir.glob(f"{cv}_contrast_0_correlation*scatterplot.svg")) for cv in cv_labels)


def _collect_all_cv_metrics(out_dir: Path, cv_labels: list[str]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for cv in CV_LABELS:
        if cv in cv_labels:
            plot_path = _find_cv_plot(out_dir, cv)
            metrics = _extract_metrics(plot_path)
            row[f"{cv}_cv_plot"] = str(plot_path)
            for key, value in metrics.items():
                row[f"{cv}_{key}"] = value
        else:
            row[f"{cv}_cv_plot"] = ""
            for key in ("spearman_rho", "spearman_p", "pearson_r", "pearson_p", "rmse", "mae"):
                row[f"{cv}_{key}"] = float("nan")
    return row


def _empty_cv_metrics() -> dict[str, Any]:
    row: dict[str, Any] = {}
    for cv in CV_LABELS:
        row[f"{cv}_cv_plot"] = ""
        for key in ("spearman_rho", "spearman_p", "pearson_r", "pearson_p", "rmse", "mae"):
            row[f"{cv}_{key}"] = float("nan")
    return row


def _average_cv_metrics(cv_metrics: dict[str, Any]) -> dict[str, float]:
    averaged: dict[str, float] = {}
    metric_names = ("spearman_rho", "spearman_p", "pearson_r", "pearson_p", "rmse", "mae")
    for metric_name in metric_names:
        values = [cv_metrics.get(f"{cv}_{metric_name}", float("nan")) for cv in AVERAGE_CV_LABELS]
        averaged[f"avg_{metric_name}"] = float(np.nanmean(values)) if not np.all(np.isnan(values)) else float("nan")
    return averaged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run 05b once per CSV column.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--mask-path", required=True)
    parser.add_argument("--paths-col", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--summary-csv", default="batch_05b_summary.csv")
    parser.add_argument("--columns", nargs="*", default=None)
    parser.add_argument("--all-numeric", action="store_true")
    parser.add_argument("--exclude-col", action="append", default=[])
    parser.add_argument("--filter-col", default=None)
    parser.add_argument("--filter-value", default=None)
    parser.add_argument("--drop-nans", nargs="*", default=None)
    parser.add_argument("--cv", default="2")
    parser.add_argument("--n-permutations", type=int, default=0)
    parser.add_argument("--data-transform", default=None, choices=["none", "standardize", "rank"])
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cv_value = _normalize_cv(args.cv)

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_csv = out_root / Path(args.summary_csv).name
    mask_data = np.asanyarray(nib.load(str(args.mask_path)).dataobj)

    df = pd.read_csv(args.input_csv)
    if args.filter_col is not None:
        df = df[df[args.filter_col].astype(str) == str(args.filter_value)].copy()
    if args.drop_nans:
        df = df.dropna(subset=list(args.drop_nans)).copy()

    if args.all_numeric:
        columns = [c for c in df.select_dtypes(include=["number"]).columns if c not in set(args.exclude_col)]
    else:
        columns = list(args.columns or [])

    for predictor in columns:
        out_dir = out_root / _safe_name(predictor)
        out_dir.mkdir(parents=True, exist_ok=True)

        tmp = pd.DataFrame(
            {
                "paths": df[args.paths_col].astype(str),
                predictor: pd.to_numeric(df[predictor], errors="coerce"),
            }
        ).dropna()

        input_csv = out_dir / "input.csv"
        input_df = tmp[["paths", predictor]]
        input_df.to_csv(input_csv, index=False)

        contrast_path = out_dir / "contrast.json"
        contrast_path.write_text(json.dumps([[0, 1]], indent=2))

        cmd = [
            sys.executable,
            str(_SCRIPTS_DIR / "05b_full_voxelwise_regression.py"),
            "fit",
            "--input-path",
            str(input_csv),
            "--out-dir",
            str(out_dir),
            "--mask-path",
            str(args.mask_path),
            "--formula",
            f"paths ~ {predictor}",
            "--voxelwise-var",
            "paths",
            "--add-intercept",
            "true",
            "--drop-nans",
            "paths",
            predictor,
            "--contrast-file",
            str(contrast_path),
            "--n-permutations",
            str(args.n_permutations),
            "--all-outputs",
            "false",
        ]
        if args.data_transform and args.data_transform != "none":
            cmd += ["--data-transform", args.data_transform]
        if args.verbose:
            cmd += ["--verbose"]

        subprocess.run(cmd, check=True)

        if cv_value is not None and len(input_df) >= 2:
            cv_labels = _valid_cv_labels(len(input_df))
            if not _all_cv_plots_exist(out_dir, cv_labels):
                reg = VoxelwiseRegression(
                    str(out_dir / "dataset_dict.json"),
                    mask_path=str(args.mask_path),
                    out_dir=str(out_dir),
                    regression_type="linear",
                    n_permutations=0,
                )
                for cv in cv_labels:
                    reg._evaluate_map(
                        cv=_cv_arg_from_label(cv),
                        y_true=input_df[predictor],
                        subject_files=input_df["paths"],
                    )
            cv_metrics = _collect_all_cv_metrics(out_dir, cv_labels)
        else:
            cv_metrics = _empty_cv_metrics()
        avg_cv_metrics = _average_cv_metrics(cv_metrics)

        fwe_dots = {
            f"dot_{_strip_nii_suffix(map_path.name)}": _dot_product_with_mask(map_path, mask_data)
            for map_path in _find_fwe_maps(out_dir)
        }

        row = {
            "predictor": predictor,
            "n": len(input_df),
            "out_dir": str(out_dir),
            "cv": "" if cv_value is None else "all",
            **cv_metrics,
            **avg_cv_metrics,
            **fwe_dots,
        }
        _append_summary_row(summary_csv, row)

    return 0


def _script_argv(cfg: dict[str, Any]) -> list[str]:
    argv = [
        "--input-csv",
        str(cfg["input_csv"]),
        "--mask-path",
        str(cfg["mask_path"]),
        "--paths-col",
        str(cfg["paths_col"]),
        "--out-root",
        str(cfg["out_root"]),
        "--summary-csv",
        str(cfg.get("summary_csv", "batch_05b_summary.csv")),
        "--drop-nans",
        *[str(col) for col in cfg.get("drop_nans", [])],
        "--cv",
        str(cfg.get("cv", "2")),
        "--n-permutations",
        str(cfg.get("n_permutations", 0)),
    ]
    if cfg.get("data_transform") is not None:
        argv += ["--data-transform", str(cfg["data_transform"])]
    if cfg.get("all_numeric", False):
        argv.append("--all-numeric")
        for col in cfg.get("exclude_cols", []):
            argv += ["--exclude-col", str(col)]
    else:
        columns = cfg.get("columns", [])
        if columns:
            argv += ["--columns", *[str(col) for col in columns]]
    if cfg.get("filter_col") is not None:
        argv += ["--filter-col", str(cfg["filter_col"]), "--filter-value", str(cfg.get("filter_value", ""))]
    if cfg.get("verbose", True):
        argv.append("--verbose")
    return argv


if __name__ == "__main__":
    if len(sys.argv) == 1:
        raise SystemExit(main(_script_argv(SCRIPT_ARGS)))
    raise SystemExit(main())
