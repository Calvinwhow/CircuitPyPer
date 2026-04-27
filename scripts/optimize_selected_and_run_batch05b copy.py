#!/usr/bin/env python3
"""
Simple coordinate-ascent optimizer for per-subject `selected` rows.

Goal:
- exactly one selected row per subject
- run `batch_run_05b_per_column.py` on the currently selected rows
- score the result by `abs(avg_spearman_rho)` for the symptom of interest
- keep session swaps that improve that objective
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CONFIG: dict[str, Any] = {
    "master_csv": "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/master_list_redo.csv",
    "opt_dir": "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimzation/AffectFailS",
    "mask_path": "/Users/cu135/Software_Local/calvin_utils_project/circuit_pyper/resources/MNI152_T1_2mm_brain_mask.nii",
    "subject_col": "subid",
    "session_col": "session",
    "paths_col": "onetouch_path",
    "selected_col": "selected",
    "predictor_col": "AffectFailS",  #AffectFailS # TotalCCASRawScore
    "passes": 15,
    "seed": 0,
    "cv": "loocv",
    "n_permutations": 0,
    "drop_nans": ["AffectFailS"],
    "verbose": True,
}


def _assert_exactly_one_selected(df: pd.DataFrame, *, subject_col: str, selected_col: str) -> None:
    selected = pd.to_numeric(df[selected_col], errors="coerce").fillna(0).astype(int)
    counts = selected.groupby(df[subject_col]).sum()
    bad = counts[counts != 1]
    if not bad.empty:
        raise SystemExit(f"Expected exactly one `{selected_col}`==1 per `{subject_col}`.\n{bad.head(20).to_string()}")


def _initialize_selection(df: pd.DataFrame, *, subject_col: str, selected_col: str) -> dict[Any, int]:
    selection: dict[Any, int] = {}
    for subject, index in df.groupby(subject_col).groups.items():
        rows = list(index)
        chosen = None
        for row in rows:
            if int(pd.to_numeric(pd.Series([df.at[row, selected_col]]), errors="coerce").fillna(0).iloc[0]) == 1:
                chosen = row
                break
        selection[subject] = rows[0] if chosen is None else chosen
    return selection


def _apply_selection(df: pd.DataFrame, selection: dict[Any, int], *, selected_col: str) -> pd.DataFrame:
    out = df.copy()
    out[selected_col] = 0
    out.loc[list(selection.values()), selected_col] = 1
    return out


def _append_log(log_csv: Path, row: dict[str, Any]) -> None:
    if log_csv.exists():
        prev = pd.read_csv(log_csv)
        prev = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
    else:
        prev = pd.DataFrame([row])
    prev.to_csv(log_csv, index=False)


def _score_selection(
    *,
    df: pd.DataFrame,
    eval_id: int,
    scripts_dir: Path,
    opt_dir: Path,
    mask_path: str,
    paths_col: str,
    selected_col: str,
    predictor_col: str,
    cv: str,
    n_permutations: int,
    drop_nans: list[str],
) -> tuple[float, Path, Path]:
    batch_script = scripts_dir / "batch_run_05b_per_column.py"
    eval_dir = opt_dir / "evals" / f"eval_{eval_id:06d}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_root = eval_dir / "05b"

    input_csv = eval_dir / "master.csv"
    summary_csv = out_root / "batch_05b_summary.csv"
    df.to_csv(input_csv, index=False)

    cmd = [
        sys.executable,
        str(batch_script),
        "--input-csv",
        str(input_csv),
        "--mask-path",
        str(mask_path),
        "--paths-col",
        str(paths_col),
        "--out-root",
        str(out_root),
        "--summary-csv",
        summary_csv.name,
        "--columns",
        str(predictor_col),
        "--drop-nans",
        *[str(col) for col in drop_nans],
        "--cv",
        str(cv),
        "--n-permutations",
        str(int(n_permutations)),
        "--filter-col",
        str(selected_col),
        "--filter-value",
        "1",
        "--verbose",
    ]
    subprocess.run(cmd, check=True)

    summary_df = pd.read_csv(summary_csv)
    row = summary_df.loc[summary_df["predictor"] == predictor_col]
    if row.empty:
        raise RuntimeError(f"Missing predictor row `{predictor_col}` in {summary_csv}")
    avg_rho = float(row.iloc[0]["avg_spearman_rho"])
    return float(abs(avg_rho)), summary_csv, eval_dir


def main() -> int:
    scripts_dir = Path(__file__).resolve().parent
    master_csv = Path(CONFIG["master_csv"])
    opt_dir = Path(CONFIG["opt_dir"])
    opt_dir.mkdir(parents=True, exist_ok=True)
    evals_dir = opt_dir / "evals"
    evals_dir.mkdir(parents=True, exist_ok=True)
    log_csv = opt_dir / "optimization_log.csv"

    if CONFIG.get("verbose", True):
        print(f"[config] master_csv={master_csv}", flush=True)
        print(f"[config] opt_dir={opt_dir}", flush=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_csv = opt_dir / f"{master_csv.stem}.backup_{timestamp}{master_csv.suffix}"
    shutil.copy2(master_csv, backup_csv)
    if CONFIG.get("verbose", True):
        print(f"[backup] {backup_csv}", flush=True)

    df = pd.read_csv(master_csv)
    required = [
        CONFIG["subject_col"],
        CONFIG["session_col"],
        CONFIG["paths_col"],
        CONFIG["selected_col"],
        CONFIG["predictor_col"],
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")

    subject_col = str(CONFIG["subject_col"])
    session_col = str(CONFIG["session_col"])
    paths_col = str(CONFIG["paths_col"])
    selected_col = str(CONFIG["selected_col"])
    predictor_col = str(CONFIG["predictor_col"])
    drop_nans = [str(col) for col in CONFIG.get("drop_nans", [])]

    selection = _initialize_selection(df, subject_col=subject_col, selected_col=selected_col)
    rng = np.random.default_rng(int(CONFIG.get("seed", 0)))
    eval_id = 0

    current_df = _apply_selection(df, selection, selected_col=selected_col)
    _assert_exactly_one_selected(current_df, subject_col=subject_col, selected_col=selected_col)
    best_score, best_summary_csv, best_eval_dir = _score_selection(
        df=current_df,
        eval_id=eval_id,
        scripts_dir=scripts_dir,
        opt_dir=opt_dir,
        mask_path=str(CONFIG["mask_path"]),
        paths_col=paths_col,
        selected_col=selected_col,
        predictor_col=predictor_col,
        cv=str(CONFIG["cv"]),
        n_permutations=int(CONFIG["n_permutations"]),
        drop_nans=drop_nans,
    )
    _append_log(
        log_csv,
        {
            "eval_id": eval_id,
            "pass": 0,
            "subject": "",
            "trial_row": "",
            "trial_session": "",
            "score_abs_avg_rho": best_score,
            "summary_csv": str(best_summary_csv),
            "eval_dir": str(best_eval_dir),
            "note": "init",
        },
    )
    eval_id += 1

    if CONFIG.get("verbose", True):
        print(f"[init] abs(avg_spearman_rho)={best_score:.6f}", flush=True)

    subject_order = list(selection.keys())
    for pass_idx in range(int(CONFIG.get("passes", 1))):
        improved = False
        rng.shuffle(subject_order)

        for subject in subject_order:
            subject_rows = list(df.index[df[subject_col] == subject])
            if len(subject_rows) <= 1:
                continue

            current_row = int(selection[subject])
            best_subject_row = current_row
            best_subject_score = best_score

            for trial_row in subject_rows:
                trial_row = int(trial_row)
                if trial_row == current_row:
                    continue

                trial_selection = dict(selection)
                trial_selection[subject] = trial_row
                trial_df = _apply_selection(df, trial_selection, selected_col=selected_col)

                score, summary_csv, eval_dir = _score_selection(
                    df=trial_df,
                    eval_id=eval_id,
                    scripts_dir=scripts_dir,
                    opt_dir=opt_dir,
                    mask_path=str(CONFIG["mask_path"]),
                    paths_col=paths_col,
                    selected_col=selected_col,
                    predictor_col=predictor_col,
                    cv=str(CONFIG["cv"]),
                    n_permutations=int(CONFIG["n_permutations"]),
                    drop_nans=drop_nans,
                )
                _append_log(
                    log_csv,
                    {
                        "eval_id": eval_id,
                        "pass": int(pass_idx + 1),
                        "subject": subject,
                        "trial_row": trial_row,
                        "trial_session": df.at[trial_row, session_col],
                        "score_abs_avg_rho": score,
                        "summary_csv": str(summary_csv),
                        "eval_dir": str(eval_dir),
                        "note": "trial",
                    },
                )
                eval_id += 1

                if CONFIG.get("verbose", True):
                    print(
                        f"[pass {pass_idx + 1}] subject={subject} session={df.at[trial_row, session_col]} "
                        f"abs(avg_rho)={score:.6f}",
                        flush=True,
                    )

                if np.isfinite(score) and score > best_subject_score:
                    best_subject_score = score
                    best_subject_row = trial_row

            if best_subject_row != current_row:
                selection[subject] = best_subject_row
                best_score = best_subject_score
                improved = True
                if CONFIG.get("verbose", True):
                    print(
                        f"[best] subject={subject} session={df.at[best_subject_row, session_col]} "
                        f"abs(avg_rho)={best_score:.6f}",
                        flush=True,
                    )

        if not improved:
            if CONFIG.get("verbose", True):
                print(f"[stop] no improvement on pass {pass_idx + 1}", flush=True)
            break

    optimized_df = _apply_selection(df, selection, selected_col=selected_col)
    _assert_exactly_one_selected(optimized_df, subject_col=subject_col, selected_col=selected_col)

    optimized_csv = opt_dir / "optimized_master_list.csv"
    optimized_df.to_csv(optimized_csv, index=False)

    master_tmp = master_csv.with_suffix(master_csv.suffix + ".tmp")
    optimized_df.to_csv(master_tmp, index=False)
    master_tmp.replace(master_csv)

    final_score, final_summary_csv, final_eval_dir = _score_selection(
        df=optimized_df,
        eval_id=eval_id,
        scripts_dir=scripts_dir,
        opt_dir=opt_dir,
        mask_path=str(CONFIG["mask_path"]),
        paths_col=paths_col,
        selected_col=selected_col,
        predictor_col=predictor_col,
        cv=str(CONFIG["cv"]),
        n_permutations=int(CONFIG["n_permutations"]),
        drop_nans=drop_nans,
    )
    _append_log(
        log_csv,
        {
            "eval_id": eval_id,
            "pass": int(CONFIG.get("passes", 0)),
            "subject": "",
            "trial_row": "",
            "trial_session": "",
            "score_abs_avg_rho": final_score,
            "summary_csv": str(final_summary_csv),
            "eval_dir": str(final_eval_dir),
            "note": "final",
        },
    )

    if CONFIG.get("verbose", True):
        print(f"[write] {optimized_csv}", flush=True)
        print(f"[write] overwrote {master_csv}", flush=True)
        print(f"[write] {log_csv}", flush=True)
        print(f"[final] abs(avg_spearman_rho)={final_score:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
