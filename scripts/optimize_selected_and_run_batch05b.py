#!/usr/bin/env python3
"""
Optimize per-subject session selection (`selected`) to maximize cv=2 Spearman rho
for the contrast-only scoring, then run 05b (via `batch_run_05b_per_column.py`)
for `TotalBarsScore` on the optimized selection.

This is a "just run it" script: edit the CONFIG dict below and execute.
"""

from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


# ----------------------------
# Simple, editable defaults
# ----------------------------
CONFIG: dict[str, Any] = {
    # Master list (WILL BE OVERWRITTEN: selected column).
    "master_csv": "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/master_list_redo.csv",
    # Optimization outputs + batch 05b outputs.
    "opt_dir": "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimzation",
    # Mask used by optimizer and 05b.
    "mask_path": "/Users/cu135/Software_Local/calvin_utils_project/circuit_pyper/resources/MNI152_T1_2mm_brain_mask.nii",
    # Required columns in master_csv
    "subject_col": "subid",
    "session_col": "ses",  # only for sanity checks / debugging
    "paths_col": "onetouch_path",
    "selected_col": "selected",
    "predictor_col": "TotalBarsScore",
    # Optimization knobs
    "passes": 10,
    "seed": 0,
    "plot_mode": "all",  # none|best-only|all
    # Batch 05b knobs
    "cv": 2,
    "n_permutations": 0,
    "overwrite_outputs": True,
    "skip_missing_paths": True,
    # Verbosity
    "verbose": True,
}


def _assert_exactly_one_selected(df: pd.DataFrame, *, subject_col: str, selected_col: str) -> None:
    c = pd.to_numeric(df[selected_col], errors="coerce").fillna(0).astype(int)
    by = c.groupby(df[subject_col]).sum()
    bad = by[(by != 1)]
    if not bad.empty:
        examples = bad.head(20)
        raise SystemExit(
            f"Constraint violation: expected exactly one `{selected_col}`==1 per `{subject_col}`.\n"
            f"Examples (first 20):\n{examples.to_string()}"
        )


def main() -> int:
    scripts_dir = Path(__file__).resolve().parent
    opt_mod_path = scripts_dir / "optimize_selected_cv2_contrast.py"
    batch_mod_path = scripts_dir / "batch_run_05b_per_column.py"
    if not opt_mod_path.exists():
        raise SystemExit(f"Missing script: {opt_mod_path}")
    if not batch_mod_path.exists():
        raise SystemExit(f"Missing script: {batch_mod_path}")

    master_csv = Path(CONFIG["master_csv"])
    opt_dir = Path(CONFIG["opt_dir"])
    opt_dir.mkdir(parents=True, exist_ok=True)

    if CONFIG.get("verbose", True):
        print(f"[config] master_csv={master_csv}", flush=True)
        print(f"[config] opt_dir={opt_dir}", flush=True)

    # Backup master CSV (since we're overwriting selected)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = opt_dir / f"{master_csv.stem}.backup_{ts}{master_csv.suffix}"
    shutil.copy2(master_csv, backup_path)
    if CONFIG.get("verbose", True):
        print(f"[backup] {backup_path}", flush=True)

    # Lazy import optimizer
    import importlib.util

    spec = importlib.util.spec_from_file_location("_optimize_selected_cv2_contrast", str(opt_mod_path))
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to import optimizer from {opt_mod_path}")
    opt_mod = importlib.util.module_from_spec(spec)
    # Required for `@dataclass` evaluation on some Python builds:
    # dataclasses looks up `sys.modules[cls.__module__]` while processing the class.
    sys.modules[spec.name] = opt_mod
    spec.loader.exec_module(opt_mod)

    optimized_csv = opt_dir / "optimized_master_list.csv"
    cfg = opt_mod.OptimizeConfig(
        input_path=str(master_csv),
        opt_dir=str(opt_dir),
        out_csv=str(optimized_csv),
        mask_path=str(CONFIG["mask_path"]),
        subject_col=str(CONFIG["subject_col"]),
        selected_col=str(CONFIG["selected_col"]),
        paths_col=str(CONFIG["paths_col"]),
        score_col=str(CONFIG["predictor_col"]),
        add_intercept=True,
        contrast=[0.0, 1.0],
        passes=int(CONFIG["passes"]),
        seed=int(CONFIG["seed"]),
        plot_mode=str(CONFIG["plot_mode"]),
        resample_interpolation="nearest",
        candidate_condition=None,
        candidate_value=None,
        verbose=bool(CONFIG.get("verbose", True)),
    )

    if CONFIG.get("verbose", True):
        print("[opt] starting", flush=True)
    opt_mod.optimize_selected_cv2(cfg)
    if CONFIG.get("verbose", True):
        print(f"[opt] wrote {optimized_csv}", flush=True)

    # Sanity check constraint (exactly one session per subject)
    opt_df = pd.read_csv(optimized_csv)
    _assert_exactly_one_selected(
        opt_df,
        subject_col=str(CONFIG["subject_col"]),
        selected_col=str(CONFIG["selected_col"]),
    )
    if CONFIG.get("verbose", True):
        n_subj = opt_df[str(CONFIG["subject_col"])].nunique(dropna=True)
        print(f"[check] exactly-one-selected OK (subjects={n_subj})", flush=True)

    # Overwrite master CSV with optimized version
    tmp_path = master_csv.with_suffix(master_csv.suffix + ".tmp")
    opt_df.to_csv(tmp_path, index=False)
    tmp_path.replace(master_csv)
    if CONFIG.get("verbose", True):
        print(f"[write] overwrote {master_csv}", flush=True)

    # Run batch 05b (TotalBarsScore only) filtered to selected==1
    if CONFIG.get("verbose", True):
        print("[05b] starting batch_run_05b_per_column", flush=True)

    cmd = [
        sys.executable,
        str(batch_mod_path),
        "--input-csv",
        str(master_csv),
        "--mask-path",
        str(CONFIG["mask_path"]),
        "--paths-col",
        str(CONFIG["paths_col"]),
        "--out-root",
        str(opt_dir / "05b_TotalBarsScore"),
        "--summary-csv",
        str(opt_dir / "batch_05b_summary.csv"),
        "--columns",
        str(CONFIG["predictor_col"]),
        "--cv",
        str(int(CONFIG["cv"])),
        "--n-permutations",
        str(int(CONFIG["n_permutations"])),
        "--filter-col",
        str(CONFIG["selected_col"]),
        "--filter-value",
        "1",
        "--verbose",
    ]
    if CONFIG.get("skip_missing_paths", True):
        cmd.append("--skip-missing-paths")
    if CONFIG.get("overwrite_outputs", True):
        cmd.append("--overwrite")

    import subprocess

    subprocess.run(cmd, check=True)

    if CONFIG.get("verbose", True):
        print(f"[done] outputs in {opt_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
