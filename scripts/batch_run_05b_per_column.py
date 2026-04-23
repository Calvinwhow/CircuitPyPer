#!/usr/bin/env python3
"""
Very simple batch wrapper around `05b_full_voxelwise_regression.py`.

For each predictor column:
1) Build a per-predictor input CSV with:
   - `paths` (voxelwise outcome image path)
   - `<predictor>` (scalar predictor)
2) Call `05b_full_voxelwise_regression.py fit` with:
   - `formula = "paths ~ <predictor>"`
   - `add_intercept=True`
   - `contrast_matrix = [[0, 1]]`
   - `n_permutations = 0` (default; configurable)
   - `cv = 2` (default; configurable)
3) Extract rho (and p) from the saved SVG scatterplot and write a summary CSV.

This intentionally does NOT implement any "fast" math itself — it just drives 05b.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import nibabel as nib
import pandas as pd


# Ensure we can import sibling scripts + calvin_utils when running from repo root.
_SCRIPTS_DIR = Path(__file__).resolve().parent
_CIRCUIT_PYPER_DIR = _SCRIPTS_DIR.parent
if str(_CIRCUIT_PYPER_DIR) not in sys.path:
    sys.path.insert(0, str(_CIRCUIT_PYPER_DIR))


### Script-mode config (edit these and run this file directly) ###
SCRIPT_ARGS: dict[str, Any] = {
    "input_csv": "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/master_list_redo.csv",
    "mask_path": "/Users/cu135/Software_Local/calvin_utils_project/circuit_pyper/resources/MNI152_T1_2mm_brain_mask.nii",
    "paths_col": "onetouch_path",
    "out_root": "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimzation/batch_0cv_100perms_n34",
    # Summary CSV is always written into out_root (basename only).
    "summary_csv": "batch_05b_summary.csv",
    # Cross-validation: loocv | 2 | 5 | 10 | 0 (or none) for permutation-only.
    "cv": 0,
    "n_permutations": 1000,
    # Columns: either explicit list, or set all_numeric=True.
    "columns": [
    "Gait",
    "HeelToShinTestLeft",
    "HeelToShinTestRight",
    "FingerToNoseTestLeft",
    "FingerToNoseTestRight",
    "LimbAtaxia",
    "Speech",
    "Oculomotor",
    "TotalBarsScore"
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
    # Optional row filter
    "filter_col": "selected",
    "filter_value": 1,
    # Behavior flags
    "skip_missing_paths": True,
    "overwrite": False,
    "continue_on_error": False,
    "echo_05b": False,
    "verbose": True,
    # Optional: none | standardize | rank
    "data_transform": None,
    # Rows with NaN in these columns are dropped before 05b prep.
    "drop_nans": ["AffectFailS"],
}

### Script ###

def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _safe_name(name: str, max_len: int = 80) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")
    return s[:max_len] if len(s) > max_len else s


def _normalize_cv(raw: Any) -> str | None:
    """
    Normalize CV arg.
    Returns None to mean "disable CV".
    """
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in {"", "0", "none", "off", "false", "no"}:
        return None
    return str(raw)


def _find_fwe_tval_maps(out_dir: Path) -> list[Path]:
    """
    Match voxelwise_regression.py naming convention exactly:
      contrast_tval_FWE_{c}.nii(.gz)
    """
    return sorted(out_dir.glob("contrast_tval_FWE_*.nii*"))


def _is_complete(out_dir: Path, *, cv: str | None, require_fwe: bool = False) -> bool:
    """
    Best-effort completion check so we only skip truly finished runs.
    If `cv` is set, we expect the cv contrast-0 scatterplot to exist.
    """
    if not out_dir.exists():
        return False
    if not (out_dir / "dataset_dict.json").exists():
        return False
    if cv:
        scatter_dir = out_dir / "cross_validations" / "scatterplots"
        fwe_ok = (not require_fwe) or bool(_find_fwe_tval_maps(out_dir))
        if scatter_dir.exists():
            # Accept either legacy naming or the newer dependent-variable-qualified naming.
            if any(scatter_dir.glob(f"{cv}_contrast_0_correlation*_scatterplot.svg")):
                return fwe_ok
            if any(scatter_dir.glob("2_contrast_0_correlation*_scatterplot.svg")):
                return fwe_ok
        return False
    if require_fwe:
        return bool(_find_fwe_tval_maps(out_dir))
    return any(out_dir.rglob("*.nii")) or any(out_dir.rglob("*.nii.gz"))


def _run_and_tee(cmd: list[str], *, log_path: Path, echo: bool) -> None:
    """
    Run a subprocess, writing all output to `log_path`.
    If `echo=True`, also stream lines to stdout so the wrapper doesn't look hung.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log_f:
        log_f.write(" ".join(cmd) + "\n\n")
        log_f.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log_f.write(line)
            if echo:
                print(line.rstrip("\n"), flush=True)
        rc = proc.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)


def _extract_metrics_from_svg(svg_path: Path) -> dict[str, float]:
    """
    Extracts Rho/Pearson/RMSE/MAE lines written by calvin_utils.statistical_utils.scatterplot.simple_scatter.
    Best-effort: returns NaNs if parsing fails.
    """
    if not svg_path.exists():
        return {
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "rmse": float("nan"),
            "mae": float("nan"),
        }
    txt = svg_path.read_text(errors="ignore")
    flat = " ".join(txt.split())

    def _m(pat: str) -> float:
        m = re.search(pat, flat)
        return float(m.group(1)) if m else float("nan")

    # Matches: "Rho = 0.88, p = 2.46e-16"
    rho = _m(r"Rho\s*=\s*([+-]?\d+(?:\.\d+)?)")
    sp = _m(r"Rho\s*=\s*[+-]?\d+(?:\.\d+)?\s*,\s*p\s*=\s*([0-9.eE+-]+)")
    r = _m(r"R\s*=\s*([+-]?\d+(?:\.\d+)?)")
    rp = _m(r"R\s*=\s*[+-]?\d+(?:\.\d+)?\s*,\s*p\s*=\s*([0-9.eE+-]+)")
    rmse = _m(r"RMSE\s*=\s*([0-9.eE+-]+)")
    mae = _m(r"MAE\s*=\s*([0-9.eE+-]+)")

    return {
        "spearman_rho": rho,
        "spearman_p": sp,
        "pearson_r": r,
        "pearson_p": rp,
        "rmse": rmse,
        "mae": mae,
    }


def _strip_nii_suffix(name: str) -> str:
    s = str(name)
    if s.endswith(".nii.gz"):
        return s[: -len(".nii.gz")]
    if s.endswith(".nii"):
        return s[: -len(".nii")]
    return s


def _dot_product_with_mask(map_path: Path, *, mask_data: np.ndarray) -> float:
    """
    Returns sum(map * mask) in image space.
    This is the dot product of the full-volume arrays (flattened), ignoring NaNs.
    """
    img = nib.load(str(map_path))
    data = np.asanyarray(img.dataobj)
    if data.shape != mask_data.shape:
        raise ValueError(f"Shape mismatch: map {data.shape} vs mask {mask_data.shape} for {map_path.name}")
    return float(np.nansum(data * mask_data))


def _append_row(summary_csv: Path, row: dict[str, Any]) -> None:
    """
    Append a row while allowing dynamic columns (e.g., permutation/FWE metrics).
    We rewrite the CSV each time so new columns are preserved.
    """
    if summary_csv.exists():
        prev = pd.read_csv(summary_csv)
        out = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
    else:
        out = pd.DataFrame([row])
    out.to_csv(summary_csv, index=False)


def _run_posthoc_permutations(*, dataset_json: Path, out_dir: Path, mask_path: str, n_permutations: int) -> None:
    """
    Workaround runner for permutation/FWE maps.

    Why this exists:
    - Some regression code paths can make the built-in permutation runner crash
      when n_permutations>0.
    - We avoid that by running the main 05b fit with n_permutations=0 (so CV works),
      then re-running a fit here and running a wrapper permutation loop that correctly
      handles 1D R2 vectors.
    """
    if n_permutations < 1:
        return

    from calvin_utils.permutation_analysis_utils.voxelwise_regression import VoxelwiseRegression

    reg = VoxelwiseRegression(
        str(dataset_json),
        mask_path=mask_path,
        out_dir=str(out_dir),
        regression_type="linear",
        n_permutations=0,
    )
    reg.run()  # fit once to populate T/R2, etc. (no permutations)

    # Custom permutation loop to avoid numpy AxisError in core code when R2 is 1D.
    Tp = np.zeros_like(reg.T)
    R2p = np.zeros_like(reg.R2)
    for _ in range(int(n_permutations)):
        _, permT, permR2 = reg.voxelwise_regression(permutation=True)
        max_statsT = np.nanpercentile(np.abs(permT), 99.99, axis=1)  # (n_contrasts,)
        max_statsR2 = float(np.nanpercentile(np.abs(permR2), 99.99))  # scalar
        Tp += (max_statsT[:, None] > np.abs(reg.T)).astype(int)
        R2p += (max_statsR2 > reg.R2).astype(int)
    reg.Tp = Tp / float(n_permutations)
    reg.R2p = R2p / float(n_permutations)
    reg._save_result_maps()  # writes contrast_tval_FWE_* and contrast_pval_FWE_* etc.


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="batch_run_05b_per_column.py", description="Run 05b per predictor column.")
    ap.add_argument("--input-csv", required=True)
    ap.add_argument("--mask-path", required=True)
    ap.add_argument("--paths-col", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument(
        "--summary-csv",
        default=None,
        help=(
            "Summary CSV filename. Always written into --out-root "
            "(default: batch_05b_summary.csv). If you pass a path, only the basename is used."
        ),
    )

    ap.add_argument("--columns", nargs="*", default=None)
    ap.add_argument("--all-numeric", action="store_true")
    ap.add_argument("--exclude-col", action="append", default=[])

    ap.add_argument("--filter-col", default=None)
    ap.add_argument("--filter-value", default=None)
    ap.add_argument(
        "--skip-missing-paths",
        action="store_true",
        help="Drop rows where the NIfTI path does not exist (recommended).",
    )
    ap.add_argument(
        "--echo-05b",
        action="store_true",
        help="Stream 05b output to console (it is always saved to <out_dir>/05b.log).",
    )

    ap.add_argument("--cv", default="2")
    ap.add_argument("--n-permutations", type=int, default=0)
    ap.add_argument("--data-transform", default=None, choices=["none", "standardize", "rank"])
    ap.add_argument(
        "--drop-nans",
        nargs="*",
        default=["AffectFails"],
        help="Drop rows with NaNs in these columns during 05b prep (default: AffectFails).",
    )

    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to the next column if 05b fails for a predictor (default: stop immediately).",
    )
    ap.add_argument("--verbose", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cv_value = _normalize_cv(args.cv)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_name = Path(args.summary_csv).name if args.summary_csv else "batch_05b_summary.csv"
    summary_csv = out_root / summary_name
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and summary_csv.exists():
        summary_csv.unlink()
    if args.verbose:
        print(f"[config] out_root={out_root}")
        print(f"[config] summary_csv={summary_csv}")

    # Load mask once (used for dot products of permutation/FWE maps).
    mask_img = nib.load(str(args.mask_path))
    mask_data = np.asanyarray(mask_img.dataobj)

    def _tail_file(path: Path, *, n_lines: int = 80) -> str:
        if not path.exists():
            return ""
        try:
            with path.open("r", errors="ignore") as f:
                lines = f.readlines()
            return "".join(lines[-n_lines:])
        except Exception:
            return ""

    df = pd.read_csv(args.input_csv)
    if args.filter_col is not None:
        if args.filter_col not in df.columns:
            raise SystemExit(f"Missing filter_col: {args.filter_col}")
        if args.filter_value is None:
            raise SystemExit("--filter-value is required when --filter-col is set.")
        df = df[df[args.filter_col].astype(str) == str(args.filter_value)].copy()
        if df.empty:
            raise SystemExit("No rows after filtering.")

    if args.paths_col not in df.columns and "_resolved_path" in df.columns:
        paths_col = "_resolved_path"
    else:
        paths_col = args.paths_col
    if paths_col not in df.columns:
        raise SystemExit(f"Missing paths_col: {paths_col}")

    # Determine predictors
    cols: list[str] = []
    if args.columns:
        cols.extend(args.columns)
    if args.all_numeric:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cols.extend([c for c in numeric_cols if c not in set(args.exclude_col or [])])
    cols = [c for c in dict.fromkeys(cols) if c]
    if not cols:
        raise SystemExit("No columns to run. Provide --columns or --all-numeric.")

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing predictor columns: {missing[:20]}" + (" ..." if len(missing) > 20 else ""))

    requested_drop_nans = [str(c) for c in (args.drop_nans or []) if str(c).strip()]

    for i, pred in enumerate(cols, start=1):
        t0 = time.time()
        out_dir = out_root / _safe_name(pred)
        status = "ok"
        err = ""

        try:
            require_fwe = int(args.n_permutations) > 0
            if (not args.overwrite) and _is_complete(
                out_dir,
                cv=cv_value,
                require_fwe=require_fwe,
            ):
                status = "skip_exists"
                raise RuntimeError("output_complete_exists")

            out_dir.mkdir(parents=True, exist_ok=True)

            # Build per-predictor input csv (minimal)
            tmp_cols: dict[str, Any] = {
                "paths": df[paths_col].astype(str),
                pred: pd.to_numeric(df[pred], errors="coerce"),
            }
            for col in requested_drop_nans:
                if col in {"paths", pred}:
                    continue
                if col not in df.columns:
                    raise SystemExit(
                        f"Missing drop-nans column '{col}' in input data. "
                        f"Pass --drop-nans with valid columns or empty list to disable."
                    )
                tmp_cols[col] = df[col]
            tmp = pd.DataFrame(tmp_cols)
            drop_subset = ["paths", pred]
            for col in requested_drop_nans:
                if col == "paths":
                    continue
                if col not in drop_subset and col in tmp.columns:
                    drop_subset.append(col)
            tmp = tmp.dropna(subset=drop_subset)
            if args.skip_missing_paths:
                tmp = tmp[tmp["paths"].map(lambda p: Path(p).exists())].copy()
            if len(tmp) < 6:
                status = "skip_too_few_rows"
                raise RuntimeError("n<6_after_dropna")

            in_csv = out_dir / "input.csv"
            tmp.to_csv(in_csv, index=False)

            contrast_path = out_dir / "contrast.json"
            contrast_path.write_text(json.dumps([[0, 1]], indent=2))

            drop_nans_05b = ["paths", pred]
            for col in requested_drop_nans:
                if col not in drop_nans_05b:
                    drop_nans_05b.append(col)

            argv_05b = [
                sys.executable,
                str(_SCRIPTS_DIR / "05b_full_voxelwise_regression.py"),
                "fit",
                "--input-path",
                str(in_csv),
                "--out-dir",
                str(out_dir),
                "--mask-path",
                str(args.mask_path),
                "--formula",
                f"paths ~ {pred}",
                "--voxelwise-var",
                "paths",
                "--add-intercept",
                "true",
                "--drop-nans",
                *drop_nans_05b,
                "--contrast-file",
                str(contrast_path),
                "--n-permutations",
                # Run permutations post-hoc in this wrapper to avoid permutation crashes in some code paths.
                "0",
                "--all-outputs",
                "false",
            ]
            if args.data_transform and args.data_transform != "none":
                argv_05b += ["--data-transform", args.data_transform]
            if cv_value is not None:
                argv_05b += [
                    "--cv",
                    str(cv_value),
                    # Evaluate CV scatterplots against the scalar predictor column (design-matrix),
                    # i.e. correlate contrast-cosine damage scores vs the predictor vector.
                    "--cv-dependent-source",
                    "design",
                    "--cv-dependent-col",
                    pred,
                ]
            if args.verbose:
                argv_05b += ["--verbose"]

            log_path = out_dir / "05b.log"
            print(f"[{i}/{len(cols)}] {pred}: running (n={len(tmp)}) -> {out_dir}", flush=True)
            print(f"[{i}/{len(cols)}] {pred}: log -> {log_path}", flush=True)
            _run_and_tee(argv_05b, log_path=log_path, echo=bool(args.echo_05b))
            print(f"[{i}/{len(cols)}] {pred}: done ({round(time.time() - t0, 1)}s)", flush=True)

            # If requested, run permutation/FWE maps post-hoc into the same out_dir.
            perm_status = "skipped"
            if int(args.n_permutations) > 0:
                dataset_json = out_dir / "dataset_dict.json"
                if not dataset_json.exists():
                    raise FileNotFoundError(f"Missing {dataset_json}")
                print(f"[{i}/{len(cols)}] {pred}: permutations={int(args.n_permutations)} (post-hoc)", flush=True)
                _run_posthoc_permutations(
                    dataset_json=dataset_json,
                    out_dir=out_dir,
                    mask_path=str(args.mask_path),
                    n_permutations=int(args.n_permutations),
                )
                produced_fwe = _find_fwe_tval_maps(out_dir)
                if not produced_fwe:
                    raise RuntimeError(
                        f"Permutation step finished but no FWE maps found in {out_dir} "
                        f"(expected files like contrast_tval_FWE_0.nii.gz)."
                    )
                perm_status = "ok"

            metrics = {
                "spearman_rho": float("nan"),
                "spearman_p": float("nan"),
                "pearson_r": float("nan"),
                "pearson_p": float("nan"),
                "rmse": float("nan"),
                "mae": float("nan"),
            }
            plot_path = Path("")
            if cv_value is not None:
                # Parse cv contrast plot (prefers design-dependent naming).
                scatter_dir = out_dir / "cross_validations" / "scatterplots"
                plot_path = scatter_dir / f"{cv_value}_contrast_0_correlation__actual_design_{_safe_name(pred)}_scatterplot.svg"
                if not plot_path.exists():
                    # fallback to any matching newer naming (handles truncation/sanitization)
                    matches = sorted(scatter_dir.glob(f"{cv_value}_contrast_0_correlation__actual_design_*_scatterplot.svg"))
                    plot_path = matches[0] if matches else plot_path
                if not plot_path.exists():
                    # legacy names
                    plot_path = scatter_dir / f"{cv_value}_contrast_0_correlation_scatterplot.svg"
                if not plot_path.exists():
                    plot_path = scatter_dir / "2_contrast_0_correlation_scatterplot.svg"
                metrics = _extract_metrics_from_svg(plot_path)

            # Permutation outputs: dot-product any FWE t-maps with the mask and store in summary.
            # Maps are saved by VoxelwiseRegression as: contrast_tval_FWE_{c}.nii.gz
            fwe_maps = _find_fwe_tval_maps(out_dir)
            fwe_dots: dict[str, float] = {}
            for mp in fwe_maps:
                key = f"dot_{_strip_nii_suffix(mp.name)}"
                try:
                    fwe_dots[key] = _dot_product_with_mask(mp, mask_data=mask_data)
                except Exception as e:
                    fwe_dots[key] = float("nan")
                    if args.verbose:
                        print(f"[{pred}] warn: {type(e).__name__}: {e}", flush=True)

            row = {
                "predictor": pred,
                "n": int(len(tmp)),
                "out_dir": str(out_dir),
                "status": status,
                "seconds": round(time.time() - t0, 3),
                "cv": "" if cv_value is None else str(cv_value),
                "cv_plot": str(plot_path) if plot_path and plot_path.exists() else "",
                "log_path": str(log_path),
                "n_permutations_requested": int(args.n_permutations),
                "permutations_status": perm_status,
                **metrics,
                **fwe_dots,
                "error": "",
            }

        except Exception as e:
            if status == "ok":
                status = "error"
            err = f"{type(e).__name__}: {e}"
            row = {
                "predictor": pred,
                "n": "",
                "out_dir": str(out_dir),
                "status": status,
                "seconds": round(time.time() - t0, 3),
                "cv_plot": "",
                "log_path": str(out_dir / "05b.log"),
                "spearman_rho": float("nan"),
                "spearman_p": float("nan"),
                "pearson_r": float("nan"),
                "pearson_p": float("nan"),
                "rmse": float("nan"),
                "mae": float("nan"),
                "error": err,
            }
            if args.verbose:
                print(f"[{pred}] {status}: {err}")

        _append_row(summary_csv, row)
        if status == "error" and (not args.continue_on_error):
            # Fail fast, but leave breadcrumbs in the summary CSV.
            log_tail = _tail_file(Path(row.get("log_path", "")), n_lines=120)
            if log_tail:
                print(f"[{pred}] tail({row.get('log_path')}):\n{log_tail}", flush=True)
            raise SystemExit(f"Stopping on error for predictor '{pred}'. Re-run with --continue-on-error to batch past failures.")

    if args.verbose:
        print(f"[write] {summary_csv}")
    return 0


if __name__ == "__main__":
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
            "--cv",
            str(cfg.get("cv", "2")),
            "--n-permutations",
            str(int(cfg.get("n_permutations", 0))),
        ]

        if cfg.get("data_transform") is not None:
            argv += ["--data-transform", str(cfg["data_transform"])]
        if cfg.get("drop_nans") is not None:
            argv += ["--drop-nans", *[str(c) for c in cfg.get("drop_nans", [])]]

        if cfg.get("all_numeric", False):
            argv.append("--all-numeric")
            for c in cfg.get("exclude_cols", []):
                argv += ["--exclude-col", str(c)]
        else:
            cols = cfg.get("columns", [])
            if cols:
                argv += ["--columns", *[str(c) for c in cols]]

        if cfg.get("filter_col") is not None:
            argv += ["--filter-col", str(cfg["filter_col"]), "--filter-value", str(cfg.get("filter_value", ""))]
        if cfg.get("skip_missing_paths", False):
            argv.append("--skip-missing-paths")
        if cfg.get("echo_05b", False):
            argv.append("--echo-05b")
        if cfg.get("overwrite", False):
            argv.append("--overwrite")
        if cfg.get("continue_on_error", False):
            argv.append("--continue-on-error")
        if cfg.get("verbose", True):
            argv.append("--verbose")
        return argv

    # "MATLAB script" behavior: no CLI args => read SCRIPT_ARGS and run.
    if len(sys.argv) == 1:
        raise SystemExit(main(_script_argv(SCRIPT_ARGS)))

    # CLI behavior remains available if explicit args are provided.
    raise SystemExit(main())
