#!/usr/bin/env python3
"""
Batch run voxelwise regression + permutation testing for many predictor columns.

This script:
- Prepares tensors via RegressionPrep
- Runs VoxelwiseRegression.run() (NO cross-validation)
- Runs permutation testing via `n_permutations` (set >0)
- Saves NIfTI outputs into one subfolder per predictor

Designed for large batches:
- Iterates over many predictors from a CSV
- Logs per-predictor status to a summary CSV

Typical usage (Schmahmann):
`python circuit_pyper/scripts/batch_voxelwise_permutation_screen.py \\
  --input-csv /Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/best_selection.csv \\
  --out-root /Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/permutation_screen \\
  --mask-path /Users/cu135/Software_Local/calvin_utils_project/circuit_pyper/resources/MNI152_T1_2mm_brain_mask.nii \\
  --paths-col _resolved_path \\
  --n-permutations 1000 \\
  --columns TotalBarsScore Gait Speech`

Notes
-----
- Formula per predictor is: `paths ~ <predictor>` with `add_intercept=True`.
- Contrast defaults to `[0, 1]` (tests the predictor coefficient).
- Predictors must be numeric. Non-numeric or missing columns are skipped and logged.
- permutations=0 is supported but disables permutation testing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


_TMPDIR = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(_TMPDIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_TMPDIR / "xdg_cache"))

# Ensure `import calvin_utils...` works when running from repo root without installation.
_CIRCUIT_PYPER_DIR = Path(__file__).resolve().parents[1]
if str(_CIRCUIT_PYPER_DIR) not in sys.path:
    sys.path.insert(0, str(_CIRCUIT_PYPER_DIR))


def _parse_bool(v: str) -> bool:
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {v}")


def _safe_name(name: str, max_len: int = 80) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")
    return s[:max_len] if len(s) > max_len else s


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _resolve_paths(df: pd.DataFrame, paths_col: str) -> pd.Series:
    if paths_col in df.columns:
        return df[paths_col].astype(str)
    if "_resolved_path" in df.columns:
        return df["_resolved_path"].astype(str)
    raise ValueError(f"paths_col not found: {paths_col} (and no _resolved_path present)")


def iter_columns_from_file(path: str) -> list[str]:
    p = Path(path)
    cols = []
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        cols.append(s)
    return cols


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="batch_voxelwise_permutation_screen.py",
        description="Batch voxelwise regression + permutation testing across many predictor columns (no CV).",
    )
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--mask-path", required=True)

    ap.add_argument("--filter-col", default=None, help="Optional: keep rows where filter-col == filter-value.")
    ap.add_argument("--filter-value", default=None, help="Value used with --filter-col.")
    # Backward-compatible aliases (deprecated): --selected-col/--selected-value
    ap.add_argument("--selected-col", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--selected-value", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--paths-col", default="_resolved_path", help="NIfTI path column (default _resolved_path).")

    ap.add_argument("--n-permutations", type=int, default=1000, help="Number of permutations (0 disables).")
    ap.add_argument("--data-transform", default=None, choices=["none", "standardize", "rank"], help="Tensor transform (default none).")

    ap.add_argument("--columns", nargs="*", default=None, help="Predictor columns to run.")
    ap.add_argument("--columns-file", default=None, help="Text file with one predictor column per line.")
    ap.add_argument("--all-numeric", action="store_true", help="Run all numeric columns (after filtering).")
    ap.add_argument("--exclude-col", action="append", default=[], help="Exclude columns when using --all-numeric. Repeatable.")

    ap.add_argument("--overwrite", type=_parse_bool, default=False, help="Overwrite existing predictor output dirs.")
    ap.add_argument("--verbose", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    out_root = Path(args.out_root)
    _safe_mkdir(out_root)
    summary_csv = out_root / "batch_summary.csv"

    df = pd.read_csv(args.input_csv)

    # normalize deprecated selected-col args into filter-col/value
    if args.filter_col is None and args.selected_col is not None:
        args.filter_col = args.selected_col
        args.filter_value = args.selected_value

    if args.filter_col is not None:
        if args.filter_col not in df.columns:
            raise SystemExit(f"Missing filter-col: {args.filter_col}")
        if args.filter_value is None:
            raise SystemExit("--filter-value is required when --filter-col is set.")
        m = df[args.filter_col].astype(str) == str(args.filter_value)
        df = df.loc[m].copy()
        if df.empty:
            raise SystemExit(f"No rows remain after filtering {args.filter_col} == {args.filter_value}.")

    # Ensure a consistent dependent var column name
    df = df.copy()
    df["paths"] = _resolve_paths(df, args.paths_col)

    # Determine predictors to run
    cols: list[str] = []
    if args.columns_file:
        cols.extend(iter_columns_from_file(args.columns_file))
    if args.columns:
        cols.extend(args.columns)
    if args.all_numeric:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cols.extend([c for c in numeric_cols if c not in set(args.exclude_col or [])])
    cols = [c for c in dict.fromkeys(cols) if c]  # unique preserve order
    if not cols:
        raise SystemExit("No predictors provided. Use --columns/--columns-file or --all-numeric.")

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing predictor columns in CSV: {missing[:20]}" + (" ..." if len(missing) > 20 else ""))

    # Lazy imports: keep help fast
    from calvin_utils.permutation_analysis_utils.statsmodels_palm import CalvinStatsmodelsPalm
    from calvin_utils.permutation_analysis_utils.voxelwise_regression_prep import RegressionPrep
    from calvin_utils.permutation_analysis_utils.voxelwise_regression import VoxelwiseRegression

    # CalvinStatsmodelsPalm is mostly a helper; we can point it at input-csv but we will override its df.
    cal = CalvinStatsmodelsPalm(input_csv_path=args.input_csv, output_dir=str(out_root), sheet=None)
    cal.df = df  # bypass read_data

    data_transform = None if args.data_transform in {None, "none"} else args.data_transform

    results: list[dict] = []
    for pred_col in cols:
        t0 = time.time()
        safe = _safe_name(pred_col)
        out_dir = out_root / safe
        status = "ok"
        note = ""
        n_used = 0
        nunique = 0
        err = ""

        try:
            if out_dir.exists() and any(out_dir.iterdir()) and (not args.overwrite):
                status = "skip_exists"
                note = "output_dir_exists"
                results.append(
                    {
                        "predictor": pred_col,
                        "out_dir": str(out_dir),
                        "status": status,
                        "note": note,
                        "n": "",
                        "n_unique": "",
                        "seconds": round(time.time() - t0, 3),
                        "error": "",
                    }
                )
                continue

            _safe_mkdir(out_dir)

            # Build per-predictor dataframe (drop rows missing predictor or paths)
            tmp = df[["paths", pred_col]].copy()
            tmp[pred_col] = pd.to_numeric(tmp[pred_col], errors="coerce")
            tmp = tmp.dropna(subset=["paths", pred_col]).copy()
            n_used = int(len(tmp))
            nunique = int(tmp[pred_col].nunique(dropna=True))

            if n_used < 6:
                status = "skip_too_few_rows"
                note = "n<6_after_dropna"
                raise RuntimeError(note)

            # Formula + design/outcome
            formula = f"paths ~ {pred_col}"
            outcome_df, design = cal.define_design_matrix(
                formula,
                tmp,
                add_intercept=True,
                voxelwise_variable_list=["paths"],
                voxelwise_interaction_terms=[],
            )

            # Ensure predictor is a simple scalar column (no dummy expansion)
            if design.shape[1] != 2:
                status = "skip_non_scalar_design"
                note = f"design_cols={design.columns.tolist()}"
                raise RuntimeError(note)

            # Contrast for predictor coefficient
            contrast_df = cal.finalize_contrast_matrix(design_matrix=design, contrast_matrix=[[0, 1]])

            preparer = RegressionPrep(
                design_matrix=design,
                contrast_matrix=contrast_df,
                outcome_df=outcome_df,
                out_dir=str(out_dir),
                voxelwise_variables=["paths"],
                voxelwise_interactions=[],
                mask_path=args.mask_path,
                exchangeability_block=None,
                data_transform_method=data_transform,
                weights=None,
                formula=formula,
            )
            _, json_path = preparer.run()

            reg = VoxelwiseRegression(
                json_path,
                mask_path=args.mask_path,
                out_dir=str(out_dir),
                regression_type="linear",
                n_permutations=int(args.n_permutations),
            )
            reg.run()  # no CV

        except Exception as e:
            if status == "ok":
                status = "error"
            err = f"{type(e).__name__}: {e}"
            if args.verbose:
                print(f"[{pred_col}] {status}: {err}")
        finally:
            results.append(
                {
                    "predictor": pred_col,
                    "out_dir": str(out_dir),
                    "status": status,
                    "note": note,
                    "n": n_used,
                    "n_unique": nunique,
                    "seconds": round(time.time() - t0, 3),
                    "error": err,
                }
            )

            # stream-write progress so long runs are resumable
            pd.DataFrame(results[-1:]).to_csv(
                summary_csv,
                mode="a",
                header=not summary_csv.exists(),
                index=False,
            )

    if args.verbose:
        print(f"[write] {summary_csv}")
    return 0


if __name__ == "__main__":
    # Run-without-CLI pattern:
    # - Edit `selected = {...}` in `batch_voxelwise_regression_cross_validation.py`
    # - Then run this file with no CLI args.
    def _default_argv() -> list[str]:
        try:
            from batch_voxelwise_regression_cross_validation import selected as cfg  # type: ignore
        except Exception:
            cfg = {}
        input_csv = cfg.get("input_csv", "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/best_selection.csv")
        mask_path = cfg.get(
            "mask_path",
            "/Users/cu135/Software_Local/calvin_utils_project/circuit_pyper/resources/MNI152_T1_2mm_brain_mask.nii",
        )
        paths_col = cfg.get("paths_col", "_resolved_path")
        out_root = cfg.get(
            "perm_out_root",
            "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/permutation_screen",
        )
        exclude = cfg.get(
            "exclude_cols",
            ["subid", "session", "id", "local_id", "paths", "onetouch_path", "_resolved_path", "selected"],
        )
        argv = [
            "--input-csv",
            str(input_csv),
            "--out-root",
            str(out_root),
            "--mask-path",
            str(mask_path),
            "--paths-col",
            str(paths_col),
            "--n-permutations",
            str(cfg.get("n_permutations", 0)),
        ]
        if cfg.get("all_numeric", False):
            argv.append("--all-numeric")
            for c in exclude:
                argv += ["--exclude-col", str(c)]
        else:
            for c in cfg.get("columns", []):
                argv.append(str(c))
        # Optional generic filter
        if cfg.get("filter_col") is not None:
            argv += ["--filter-col", str(cfg["filter_col"]), "--filter-value", str(cfg.get("filter_value", ""))]
        if cfg.get("verbose", True):
            argv.append("--verbose")
        return argv

    argv = None if len(sys.argv) > 1 else _default_argv()
    raise SystemExit(main(argv))
