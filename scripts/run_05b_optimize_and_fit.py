#!/usr/bin/env python3
"""
End-to-end driver for the Schmahmann optimization workflow:

1) Optimize which session per subject is `selected=1` to maximize cv=2 Spearman rho
   for the contrast-only branch (contrast t-map cosine), logging every eval.
2) Run the voxelwise regression (05b script) exactly once on the optimized selection
   (filters to selected==1), producing the usual regression outputs + cv=2 plot.

This script is intentionally a *caller* of:
- `optimize_selected_cv2_contrast.py` (optimization)
- `05b_full_voxelwise_regression.py` (regression)

It does not modify either script’s logic.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import pandas as pd


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_05b_optimize_and_fit.py",
        description="Optimize selected sessions (cv=2 contrast rho) then run 05b regression once.",
    )
    p.add_argument("--input-csv", required=True, help="Master CSV with multiple sessions per subject.")
    p.add_argument("--opt-dir", required=True, help="Directory to write optimization log + scatterplots + regression outputs.")
    p.add_argument("--mask-path", required=True, help="Mask NIfTI used for vectorizing and regression.")

    p.add_argument("--subject-col", default="id", help="Subject ID column. Default: id")
    p.add_argument("--selected-col", default="selected", help="Selection flag column. Default: selected")
    p.add_argument("--path-col", default="onetouch_path", help="NIfTI path column to use. Default: onetouch_path")
    p.add_argument("--score-col", default="TotalBarsScore", help="Predictor column. Default: TotalBarsScore")

    p.add_argument("--passes", type=int, default=10, help="Optimization passes. Default: 10")
    p.add_argument("--seed", type=int, default=0, help="Optimization seed. Default: 0")
    p.add_argument("--plot-mode", choices=["none", "best-only", "all"], default="all", help="Optimization scatterplot mode.")

    p.add_argument("--regression-out-dir", default=None, help="Regression output dir (default: <opt-dir>/regression).")
    p.add_argument("--cv", default="2", help="Regression CV to run (default: 2). Set empty to skip.")
    p.add_argument("--n-permutations", type=int, default=0, help="Permutations for regression (default: 0).")
    p.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    opt_dir = Path(args.opt_dir)
    opt_dir.mkdir(parents=True, exist_ok=True)
    regression_out_dir = Path(args.regression_out_dir) if args.regression_out_dir else (opt_dir / "regression")
    regression_out_dir.mkdir(parents=True, exist_ok=True)

    scripts_dir = Path(__file__).resolve().parent
    optimizer_path = scripts_dir / "optimize_selected_cv2_contrast.py"
    regression_path = scripts_dir / "05b_full_voxelwise_regression.py"

    optimizer_mod = _load_module_from_path("_optimize_selected_cv2_contrast", optimizer_path)
    regression_mod = _load_module_from_path("_full_voxelwise_regression_05b", regression_path)

    # 1) Run optimization -> writes optimized CSV + log/plots in opt-dir
    optimized_csv = opt_dir / "optimized_master_list.csv"
    if args.verbose:
        print(f"[opt] writing to {optimized_csv}")

    cfg = optimizer_mod.OptimizeConfig(
        input_path=args.input_csv,
        opt_dir=str(opt_dir),
        out_csv=str(optimized_csv),
        mask_path=args.mask_path,
        subject_col=args.subject_col,
        selected_col=args.selected_col,
        paths_col=("onetouch_path" if args.path_col == "onetouch_file_path" else args.path_col),
        score_col=args.score_col,
        add_intercept=True,
        contrast=[0.0, 1.0],
        passes=int(args.passes),
        seed=int(args.seed),
        plot_mode=args.plot_mode,
        resample_interpolation="nearest",
        candidate_condition=None,
        candidate_value=None,
        verbose=bool(args.verbose),
    )
    optimizer_mod.optimize_selected_cv2(cfg)

    # 2) Prepare regression input CSV so formula stays exactly: paths ~ TotalBarsScore
    reg_input_csv = opt_dir / "optimized_for_regression.csv"
    df = pd.read_csv(optimized_csv)
    if args.path_col == "onetouch_file_path" and "onetouch_path" in df.columns:
        path_col = "onetouch_path"
    else:
        path_col = args.path_col
    if path_col not in df.columns:
        raise SystemExit(f"Missing path column: {path_col}")
    df["paths"] = df[path_col].astype(str)
    df.to_csv(reg_input_csv, index=False)

    # 3) Create contrast file [[0,1]] and run 05b regression filtered to selected==1
    contrast_path = opt_dir / "contrast.json"
    contrast_path.write_text(json.dumps([[0, 1]], indent=2))

    cmd = [
        "fit",
        "--input-path",
        str(reg_input_csv),
        "--out-dir",
        str(regression_out_dir),
        "--mask-path",
        str(args.mask_path),
        "--formula",
        "paths ~ TotalBarsScore",
        "--voxelwise-var",
        "paths",
        "--add-intercept",
        "true",
        "--drop-nans",
        "paths",
        "TotalBarsScore",
        "--drop-row",
        args.selected_col,
        "not",
        "1",
        "--contrast-file",
        str(contrast_path),
        "--n-permutations",
        str(int(args.n_permutations)),
        "--all-outputs",
        "false",
    ]
    if args.cv:
        cmd += ["--cv", str(args.cv)]
    if args.verbose:
        cmd += ["--verbose"]

    if args.verbose:
        print(f"[reg] 05b args: {' '.join(cmd)}")
    regression_mod.main(cmd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

