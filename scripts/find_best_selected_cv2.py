#!/usr/bin/env python3
"""
Find the best per-subject session selection (`selected=1`) that maximizes
cv=2 Spearman rho for the *contrast-only* branch used in
`VoxelwiseRegression._evaluate_map(cv=2)`.

This script is built for your Schmahmann dataset layout:
- Many rows per subject (sessions)
- `selected` marks the chosen session per subject
- NIfTI paths live in `onetouch_path` (alias: onetouch_file_path)
- Predictor is `TotalBarsScore`
- Regression formula is effectively: paths ~ TotalBarsScore (add_intercept=True)
- Contrast is [0, 1] (i.e., test TotalBarsScore coefficient)

Objective (contrast-only cv=2)
-----------------------------
For each 2-fold CV split over subjects (fixed split by subject order):
- Fit voxelwise OLS on the training subjects (one session per subject).
- Compute the *contrast t-map* for the specified contrast row.
- For each held-out subject, compute scalar prediction:
    cosine(subject_image_vector, contrast_tmap_vector)
- Compute Spearman rho between those predictions and:
    actual_y = mean(subject_image_vector)

Outputs (written into --out-dir)
-------------------------------
- `best_selection.csv` (full input CSV with updated selected column)
- `cv2_optimization_log.csv` (every trial rho + plot_path)
- `scatterplots/eval_XXXXXX_scatterplot.svg` (when plot-mode=all)
- `best_selection.summary.json`
- `regression/` (optional) outputs from 05b regression script run once on best selection
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

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


def _coerce_value(raw: Any) -> Any:
    if raw is None:
        return None
    s = str(raw)
    try:
        return int(s)
    except Exception:
        pass
    try:
        return float(s)
    except Exception:
        return s


def _cosine_similarity_rows(A: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    num = A @ b
    den = (np.linalg.norm(A, axis=1) * (np.linalg.norm(b) + eps)) + eps
    return num / den


@dataclass
class FoldStats:
    XtX: np.ndarray  # (p,p)
    XTY: np.ndarray  # (p,v)
    yTy: np.ndarray  # (v,)
    dof: int
    XtX_inv: np.ndarray  # (p,p)
    tmap: np.ndarray  # (v,)


def _build_fold_stats(X_train: np.ndarray, Y_train: np.ndarray, C: np.ndarray, eps: float = 1e-12) -> FoldStats:
    """
    Build sufficient stats and t-map for a single fold.
    X_train: (n,p)
    Y_train: (n,v)
    C: (1,p)
    """
    n, p = X_train.shape
    XtX = X_train.T @ X_train  # (p,p)
    XTY = X_train.T @ Y_train  # (p,v)
    yTy = (Y_train * Y_train).sum(axis=0)  # (v,)
    dof = max(n - p, 1)
    XtX_inv = np.linalg.pinv(XtX)
    B = XtX_inv @ XTY  # (p,v)
    sse = yTy - (B * XTY).sum(axis=0)  # (v,)
    mse = sse / float(dof)
    denom_scalar = float((C @ XtX_inv @ C.T).squeeze())
    denom = np.sqrt(np.maximum(denom_scalar, eps) * np.maximum(mse, eps))
    tmap = (C @ B).squeeze(axis=0) / (denom + eps)
    return FoldStats(XtX=XtX, XTY=XTY, yTy=yTy, dof=dof, XtX_inv=XtX_inv, tmap=tmap.astype(np.float32, copy=False))


def _update_fold_stats_replace_one(
    fs: FoldStats,
    *,
    x_old: np.ndarray,  # (p,)
    y_old: np.ndarray,  # (v,)
    x_new: np.ndarray,  # (p,)
    y_new: np.ndarray,  # (v,)
    C: np.ndarray,  # (1,p)
    eps: float = 1e-12,
) -> None:
    """
    In-place update fold sufficient stats for replacing one training row.
    """
    # XtX update: rank-1
    fs.XtX += np.outer(x_new, x_new) - np.outer(x_old, x_old)
    # XTY update: for each predictor, add scaled y vector
    fs.XTY += (x_new[:, None] * y_new[None, :]) - (x_old[:, None] * y_old[None, :])
    # yTy update
    fs.yTy += (y_new * y_new) - (y_old * y_old)

    fs.XtX_inv = np.linalg.pinv(fs.XtX)
    B = fs.XtX_inv @ fs.XTY  # (p,v)
    sse = fs.yTy - (B * fs.XTY).sum(axis=0)
    mse = sse / float(fs.dof)
    denom_scalar = float((C @ fs.XtX_inv @ C.T).squeeze())
    denom = np.sqrt(np.maximum(denom_scalar, eps) * np.maximum(mse, eps))
    fs.tmap = ((C @ B).squeeze(axis=0) / (denom + eps)).astype(np.float32, copy=False)


def _cv2_predict_from_tmaps(
    *,
    Y: np.ndarray,  # (n,v)
    tmap_fold_for_subject: np.ndarray,  # (n,v) but each row points to fold tmap; we'll pass tmap0,tmap1 and fold assignment
    fold_assignment: np.ndarray,  # (n,) values 0/1 indicating which fold subject is in (as TEST fold)
    tmap0: np.ndarray,
    tmap1: np.ndarray,
) -> np.ndarray:
    """
    Pred[i] = cosine(Y[i], tmap_of_fold_where_i_is_test)
    fold_assignment[i]=0 => subject in fold0 test => use tmap0
    fold_assignment[i]=1 => use tmap1
    """
    m0 = fold_assignment == 0
    m1 = ~m0
    preds = np.zeros((Y.shape[0],), dtype=np.float32)
    if m0.any():
        preds[m0] = _cosine_similarity_rows(Y[m0], tmap0)
    if m1.any():
        preds[m1] = _cosine_similarity_rows(Y[m1], tmap1)
    return preds


def _spearman(pred: np.ndarray, actual: np.ndarray) -> float:
    from scipy.stats import spearmanr

    rho, _ = spearmanr(pred, actual, nan_policy="omit")
    return float(rho) if rho is not None else float("nan")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="find_best_selected_cv2.py",
        description="Multi-start optimize selected sessions to maximize cv=2 Spearman rho (contrast-only), saving all trial plots and rhos.",
    )
    p.add_argument("--input-csv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--mask-path", required=True)

    p.add_argument("--selected-col", default="selected")
    p.add_argument("--paths-col", default="onetouch_path")
    p.add_argument("--score-col", default="TotalBarsScore")
    p.add_argument("--subject-col", default="id")

    p.add_argument("--add-intercept", type=_parse_bool, default=True)
    p.add_argument("--contrast", nargs="+", type=float, default=[0.0, 1.0])
    p.add_argument("--passes", type=int, default=10)

    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--seed-stop", type=int, default=20)
    p.add_argument("--plot-mode", choices=["none", "best-only", "all"], default="all")
    p.add_argument("--resample-interpolation", default="nearest")
    p.add_argument(
        "--path-rewrite",
        nargs=2,
        action="append",
        default=None,
        metavar=("FROM", "TO"),
        help="Optional path rewrite rule applied when a NIfTI path does not exist. Repeatable.",
    )

    p.add_argument("--run-regression", type=_parse_bool, default=True, help="Run 05b regression once on best selection.")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.paths_col == "onetouch_file_path":
        args.paths_col = "onetouch_path"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scatterplots").mkdir(parents=True, exist_ok=True)
    log_csv_path = out_dir / "cv2_optimization_log.csv"

    df = pd.read_csv(args.input_csv)
    for col in (args.selected_col, args.paths_col, args.score_col, args.subject_col):
        if col not in df.columns:
            raise SystemExit(f"Missing required column: {col}")

    # candidates: all rows with required fields
    cand_df = df.dropna(subset=[args.paths_col, args.score_col, args.subject_col]).reset_index(drop=False).rename(
        columns={"index": "_orig_index"}
    )
    cand_df = cand_df.sort_values("_orig_index").reset_index(drop=True)

    # mask + cache
    import nibabel as nib
    from nilearn import image

    mask_img = nib.load(args.mask_path)
    mask_flat = (mask_img.get_fdata().reshape(-1) > 0)
    v = int(mask_flat.sum())
    if v < 10:
        raise SystemExit("Mask appears too small; check --mask-path.")

    # Resolve/repair NIfTI paths (the CSV may contain stale mount prefixes).
    rewrite_rules = list(args.path_rewrite or [])
    # Default repair for this dataset if user doesn't supply one.
    if not rewrite_rules:
        rewrite_rules = [
            ("/Volumes/OneTouch/chmahmann_SCA_Atrophy/", "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/"),
        ]

    def resolve_path(p: str) -> str | None:
        p = str(p)
        if os.path.exists(p):
            return p
        for src, dst in rewrite_rules:
            if src in p:
                cand = p.replace(src, dst)
                if os.path.exists(cand):
                    return cand
        return None

    resolved = []
    missing = 0
    for p in cand_df[args.paths_col].astype(str).tolist():
        rp = resolve_path(p)
        if rp is None:
            missing += 1
        resolved.append(rp)
    cand_df["_resolved_path"] = resolved
    if missing:
        cand_df = cand_df.dropna(subset=["_resolved_path"]).reset_index(drop=True)
        if args.verbose:
            print(f"[paths] dropped rows with missing nifti: {missing}")

    unique_paths = sorted(set(cand_df["_resolved_path"].astype(str).tolist()))
    if args.verbose:
        print(f"[cache] unique_paths={len(unique_paths)} voxels={v}")

    vec_cache: dict[str, np.ndarray] = {}
    mean_cache: dict[str, float] = {}
    for pth in unique_paths:
        img = nib.load(pth)
        if img.shape != mask_img.shape or (not np.allclose(img.affine, mask_img.affine)):
            img = image.resample_to_img(
                img,
                mask_img,
                interpolation=args.resample_interpolation,
                force_resample=True,
                copy_header=True,
            )
        data = img.get_fdata(dtype=np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        vec = data.reshape(-1)[mask_flat]
        vec_cache[pth] = vec.astype(np.float32, copy=False)
        mean_cache[pth] = float(np.nanmean(vec))

    # group rows by subject
    subj_to_rows: dict[Any, list[int]] = {}
    for i, subj in enumerate(cand_df[args.subject_col].tolist()):
        subj_to_rows.setdefault(subj, []).append(i)
    subjects = list(subj_to_rows.keys())
    n_subj = len(subjects)
    if args.verbose:
        print(f"[subjects] n={n_subj}")

    # fixed 2-fold split by subject order (same as numpy array_split)
    subj_idx = np.arange(n_subj)
    folds = np.array_split(subj_idx, 2)
    fold_assignment = np.zeros((n_subj,), dtype=np.int8)
    fold_assignment[folds[1]] = 1  # fold 1 test

    def build_x(score: float) -> np.ndarray:
        if args.add_intercept:
            return np.asarray([1.0, float(score)], dtype=np.float32)
        return np.asarray([float(score)], dtype=np.float32)

    C = np.asarray([args.contrast], dtype=np.float32)

    # Initialize selection: take pre-existing selected==1 if present per subject, else first row.
    initial_sel_row = np.zeros((n_subj,), dtype=int)
    for si, subj in enumerate(subjects):
        rows = subj_to_rows[subj]
        chosen = None
        for r in rows:
            if _coerce_value(cand_df.loc[r, args.selected_col]) == 1:
                chosen = r
                break
        initial_sel_row[si] = chosen if chosen is not None else rows[0]

    row_paths = cand_df["_resolved_path"].astype(str).tolist()
    row_scores = pd.to_numeric(cand_df[args.score_col], errors="raise").to_numpy(dtype=np.float32)
    row_orig_idx = cand_df["_orig_index"].to_numpy(dtype=int)
    row_vecs = [vec_cache[p] for p in row_paths]
    row_means = np.asarray([mean_cache[p] for p in row_paths], dtype=np.float32)

    def selection_to_arrays(sel_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        Y = np.stack([row_vecs[r] for r in sel_rows], axis=0).astype(np.float32, copy=False)
        actual = row_means[sel_rows].astype(np.float32, copy=False)
        scores = row_scores[sel_rows].astype(np.float32, copy=False)
        return Y, actual, scores

    def build_design(scores: np.ndarray) -> np.ndarray:
        if args.add_intercept:
            return np.column_stack([np.ones_like(scores), scores]).astype(np.float32, copy=False)
        return scores[:, None].astype(np.float32, copy=False)

    # --- fast optimizer core (no plotting/logging) ---
    def run_one_seed(seed: int) -> tuple[float, np.ndarray]:
        rng = np.random.default_rng(int(seed))
        sel_rows = initial_sel_row.copy()
        Y, actual, scores = selection_to_arrays(sel_rows)
        X = build_design(scores)
        if C.shape[1] != X.shape[1]:
            raise SystemExit(f"Contrast length {C.shape[1]} != n_preds {X.shape[1]}")

        # Precompute which subjects are training for each fold (train = not test fold)
        test0 = fold_assignment == 0
        test1 = ~test0
        train_for_fold0 = test1  # fold0 test => train is fold1
        train_for_fold1 = test0  # fold1 test => train is fold0

        fs0 = _build_fold_stats(X[train_for_fold0], Y[train_for_fold0], C)
        fs1 = _build_fold_stats(X[train_for_fold1], Y[train_for_fold1], C)
        preds = _cv2_predict_from_tmaps(Y=Y, tmap_fold_for_subject=None, fold_assignment=fold_assignment, tmap0=fs0.tmap, tmap1=fs1.tmap)
        best_rho = _spearman(preds, actual)

        order = np.arange(n_subj)
        for _pass in range(int(args.passes)):
            rng.shuffle(order)
            improved = False
            for si in order:
                subj = subjects[int(si)]
                cand_rows = subj_to_rows[subj]
                if len(cand_rows) <= 1:
                    continue
                cur_row = int(sel_rows[si])
                cur_best_row = cur_row
                cur_best_rho = best_rho

                # determine which fold stats to update: subject contributes to training of the *other* fold
                test_fold = int(fold_assignment[si])
                affected_fold = 1 - test_fold

                for r in cand_rows:
                    r = int(r)
                    if r == cur_row:
                        continue

                    # swap in-place
                    y_old = Y[si].copy()
                    x_old = X[si].copy()
                    actual_old = float(actual[si])
                    score_old = float(scores[si])

                    sel_rows[si] = r
                    Y[si] = row_vecs[r]
                    actual[si] = row_means[r]
                    scores[si] = row_scores[r]
                    X[si] = build_x(scores[si])

                    if affected_fold == 0:
                        _update_fold_stats_replace_one(fs0, x_old=x_old, y_old=y_old, x_new=X[si], y_new=Y[si], C=C)
                    else:
                        _update_fold_stats_replace_one(fs1, x_old=x_old, y_old=y_old, x_new=X[si], y_new=Y[si], C=C)

                    # update predictions:
                    # - subjects in test set of affected_fold (their tmap changed)
                    # - this subject itself (its Y changed) in its own test fold
                    if affected_fold == 0:
                        m = fold_assignment == 0
                        preds[m] = _cosine_similarity_rows(Y[m], fs0.tmap)
                    else:
                        m = fold_assignment == 1
                        preds[m] = _cosine_similarity_rows(Y[m], fs1.tmap)
                    # this subject prediction (might already be in m, but safe to reassign)
                    if test_fold == 0:
                        preds[si] = _cosine_similarity_rows(Y[si : si + 1], fs0.tmap)[0]
                    else:
                        preds[si] = _cosine_similarity_rows(Y[si : si + 1], fs1.tmap)[0]

                    rho = _spearman(preds, actual)
                    if np.isfinite(rho) and rho > cur_best_rho:
                        cur_best_rho = rho
                        cur_best_row = r

                    # revert (restore fold stats too by swapping back)
                    y_new = Y[si].copy()
                    x_new = X[si].copy()

                    sel_rows[si] = cur_row
                    Y[si] = y_old
                    actual[si] = actual_old
                    scores[si] = score_old
                    X[si] = x_old

                    if affected_fold == 0:
                        _update_fold_stats_replace_one(fs0, x_old=x_new, y_old=y_new, x_new=x_old, y_new=y_old, C=C)
                        m = fold_assignment == 0
                        preds[m] = _cosine_similarity_rows(Y[m], fs0.tmap)
                    else:
                        _update_fold_stats_replace_one(fs1, x_old=x_new, y_old=y_new, x_new=x_old, y_new=y_old, C=C)
                        m = fold_assignment == 1
                        preds[m] = _cosine_similarity_rows(Y[m], fs1.tmap)
                    if test_fold == 0:
                        preds[si] = _cosine_similarity_rows(Y[si : si + 1], fs0.tmap)[0]
                    else:
                        preds[si] = _cosine_similarity_rows(Y[si : si + 1], fs1.tmap)[0]

                # commit best row for this subject (update stats/preds accordingly if changed)
                if cur_best_row != cur_row:
                    # apply swap once (no revert)
                    y_old = Y[si].copy()
                    x_old = X[si].copy()
                    sel_rows[si] = cur_best_row
                    Y[si] = row_vecs[cur_best_row]
                    actual[si] = row_means[cur_best_row]
                    scores[si] = row_scores[cur_best_row]
                    X[si] = build_x(scores[si])

                    if affected_fold == 0:
                        _update_fold_stats_replace_one(fs0, x_old=x_old, y_old=y_old, x_new=X[si], y_new=Y[si], C=C)
                        m = fold_assignment == 0
                        preds[m] = _cosine_similarity_rows(Y[m], fs0.tmap)
                    else:
                        _update_fold_stats_replace_one(fs1, x_old=x_old, y_old=y_old, x_new=X[si], y_new=Y[si], C=C)
                        m = fold_assignment == 1
                        preds[m] = _cosine_similarity_rows(Y[m], fs1.tmap)
                    if test_fold == 0:
                        preds[si] = _cosine_similarity_rows(Y[si : si + 1], fs0.tmap)[0]
                    else:
                        preds[si] = _cosine_similarity_rows(Y[si : si + 1], fs1.tmap)[0]

                    best_rho = _spearman(preds, actual)
                    improved = True

            if not improved:
                break
        return float(best_rho), sel_rows

    # --- sweep seeds quickly ---
    best_seed = None
    best_rho = float("-inf")
    best_sel = None

    for seed in range(int(args.seed_start), int(args.seed_stop)):
        rho, sel = run_one_seed(seed)
        if args.verbose:
            print(f"[seed {seed}] rho={rho:.4f}")
        if np.isfinite(rho) and rho > best_rho:
            best_rho = rho
            best_seed = seed
            best_sel = sel

    if best_seed is None or best_sel is None:
        raise SystemExit("No seed produced a finite rho.")

    if args.verbose:
        print(f"[best] seed={best_seed} rho={best_rho:.4f}")

    # --- final run with logging/plots for the best seed ---
    from calvin_utils.statistical_utils.scatterplot import simple_scatter

    if log_csv_path.exists():
        log_csv_path.unlink()

    def log_row(row: dict[str, Any]) -> None:
        exists = log_csv_path.exists()
        pd.DataFrame([row]).to_csv(log_csv_path, mode="a", header=not exists, index=False)

    def maybe_plot(eval_id: int, preds: np.ndarray, actual: np.ndarray) -> str:
        if args.plot_mode == "none":
            return ""
        dataset_name = f"eval_{eval_id:06d}"
        plot_df = pd.DataFrame({"pred": preds, "actual": actual})
        simple_scatter(
            df=plot_df,
            x_col="pred",
            y_col="actual",
            dataset_name=dataset_name,
            out_dir=str(out_dir),
            x_label="Predicted (contrast cosine)",
            y_label="Actual (mean image intensity)",
            show=False,
        )
        return str(out_dir / "scatterplots" / f"{dataset_name}_scatterplot.svg")

    def run_one_seed_logged(seed: int) -> tuple[float, np.ndarray]:
        rng = np.random.default_rng(int(seed))
        sel_rows = initial_sel_row.copy()
        Y, actual, scores = selection_to_arrays(sel_rows)
        X = build_design(scores)
        if C.shape[1] != X.shape[1]:
            raise SystemExit(f"Contrast length {C.shape[1]} != n_preds {X.shape[1]}")

        test0 = fold_assignment == 0
        test1 = ~test0
        train_for_fold0 = test1
        train_for_fold1 = test0

        fs0 = _build_fold_stats(X[train_for_fold0], Y[train_for_fold0], C)
        fs1 = _build_fold_stats(X[train_for_fold1], Y[train_for_fold1], C)
        preds = _cv2_predict_from_tmaps(
            Y=Y,
            tmap_fold_for_subject=None,
            fold_assignment=fold_assignment,
            tmap0=fs0.tmap,
            tmap1=fs1.tmap,
        )
        rho = _spearman(preds, actual)

        eval_id = 0
        plot_path = maybe_plot(eval_id, preds, actual) if args.plot_mode in {"all", "best-only"} else ""
        log_row(
            {
                "eval_id": eval_id,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "seed": int(seed),
                "pass": 0,
                "subject": "",
                "trial_row": "",
                "kept_row": "",
                "trial_orig_index": "",
                "kept_orig_index": "",
                "trial_path": "",
                "kept_path": "",
                "rho": float(rho),
                "plot_path": plot_path,
                "note": "init",
            }
        )
        eval_id += 1

        order = np.arange(n_subj)
        best_rho_local = float(rho)

        for pass_idx in range(int(args.passes)):
            rng.shuffle(order)
            improved = False
            for si in order:
                subj = subjects[int(si)]
                cand_rows = subj_to_rows[subj]
                if len(cand_rows) <= 1:
                    continue
                cur_row = int(sel_rows[si])
                test_fold = int(fold_assignment[si])
                affected_fold = 1 - test_fold

                cur_best_row = cur_row
                cur_best_rho = best_rho_local

                for r in cand_rows:
                    r = int(r)
                    if r == cur_row:
                        continue

                    # snapshot
                    y_old = Y[si].copy()
                    x_old = X[si].copy()
                    actual_old = float(actual[si])
                    score_old = float(scores[si])

                    # swap
                    sel_rows[si] = r
                    Y[si] = row_vecs[r]
                    actual[si] = row_means[r]
                    scores[si] = row_scores[r]
                    X[si] = build_x(scores[si])

                    if affected_fold == 0:
                        _update_fold_stats_replace_one(fs0, x_old=x_old, y_old=y_old, x_new=X[si], y_new=Y[si], C=C)
                        m = fold_assignment == 0
                        preds[m] = _cosine_similarity_rows(Y[m], fs0.tmap)
                    else:
                        _update_fold_stats_replace_one(fs1, x_old=x_old, y_old=y_old, x_new=X[si], y_new=Y[si], C=C)
                        m = fold_assignment == 1
                        preds[m] = _cosine_similarity_rows(Y[m], fs1.tmap)

                    # subject itself (Y changed)
                    if test_fold == 0:
                        preds[si] = _cosine_similarity_rows(Y[si : si + 1], fs0.tmap)[0]
                    else:
                        preds[si] = _cosine_similarity_rows(Y[si : si + 1], fs1.tmap)[0]

                    rho_trial = _spearman(preds, actual)
                    plot_path = maybe_plot(eval_id, preds, actual) if args.plot_mode == "all" else ""
                    log_row(
                        {
                            "eval_id": eval_id,
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "seed": int(seed),
                            "pass": int(pass_idx + 1),
                            "subject": str(subj),
                            "trial_row": int(r),
                            "kept_row": int(cur_row),
                            "trial_orig_index": int(row_orig_idx[int(r)]),
                            "kept_orig_index": int(row_orig_idx[int(cur_row)]),
                            "trial_path": row_paths[int(r)],
                            "kept_path": row_paths[int(cur_row)],
                            "rho": float(rho_trial),
                            "plot_path": plot_path,
                            "note": "trial",
                        }
                    )
                    eval_id += 1

                    if np.isfinite(rho_trial) and rho_trial > cur_best_rho:
                        cur_best_rho = float(rho_trial)
                        cur_best_row = r

                    # revert swap (and revert affected fold stats by replacing back)
                    y_new = Y[si].copy()
                    x_new = X[si].copy()

                    sel_rows[si] = cur_row
                    Y[si] = y_old
                    actual[si] = actual_old
                    scores[si] = score_old
                    X[si] = x_old

                    if affected_fold == 0:
                        _update_fold_stats_replace_one(fs0, x_old=x_new, y_old=y_new, x_new=x_old, y_new=y_old, C=C)
                        m = fold_assignment == 0
                        preds[m] = _cosine_similarity_rows(Y[m], fs0.tmap)
                    else:
                        _update_fold_stats_replace_one(fs1, x_old=x_new, y_old=y_new, x_new=x_old, y_new=y_old, C=C)
                        m = fold_assignment == 1
                        preds[m] = _cosine_similarity_rows(Y[m], fs1.tmap)
                    if test_fold == 0:
                        preds[si] = _cosine_similarity_rows(Y[si : si + 1], fs0.tmap)[0]
                    else:
                        preds[si] = _cosine_similarity_rows(Y[si : si + 1], fs1.tmap)[0]

                # commit best for subject
                if cur_best_row != cur_row:
                    y_old = Y[si].copy()
                    x_old = X[si].copy()

                    sel_rows[si] = cur_best_row
                    Y[si] = row_vecs[cur_best_row]
                    actual[si] = row_means[cur_best_row]
                    scores[si] = row_scores[cur_best_row]
                    X[si] = build_x(scores[si])

                    if affected_fold == 0:
                        _update_fold_stats_replace_one(fs0, x_old=x_old, y_old=y_old, x_new=X[si], y_new=Y[si], C=C)
                        m = fold_assignment == 0
                        preds[m] = _cosine_similarity_rows(Y[m], fs0.tmap)
                    else:
                        _update_fold_stats_replace_one(fs1, x_old=x_old, y_old=y_old, x_new=X[si], y_new=Y[si], C=C)
                        m = fold_assignment == 1
                        preds[m] = _cosine_similarity_rows(Y[m], fs1.tmap)
                    if test_fold == 0:
                        preds[si] = _cosine_similarity_rows(Y[si : si + 1], fs0.tmap)[0]
                    else:
                        preds[si] = _cosine_similarity_rows(Y[si : si + 1], fs1.tmap)[0]

                    best_rho_local = _spearman(preds, actual)
                    improved = True

                    if args.plot_mode == "best-only":
                        plot_path = maybe_plot(eval_id, preds, actual)
                        log_row(
                            {
                                "eval_id": eval_id,
                                "timestamp": datetime.now().isoformat(timespec="seconds"),
                                "seed": int(seed),
                                "pass": int(pass_idx + 1),
                                "subject": str(subj),
                                "trial_row": int(cur_best_row),
                                "kept_row": int(cur_row),
                                "trial_orig_index": int(row_orig_idx[int(cur_best_row)]),
                                "kept_orig_index": int(row_orig_idx[int(cur_row)]),
                                "trial_path": row_paths[int(cur_best_row)],
                                "kept_path": row_paths[int(cur_row)],
                                "rho": float(best_rho_local),
                                "plot_path": plot_path,
                                "note": "global_best",
                            }
                        )
                        eval_id += 1

            if not improved:
                break

        return float(best_rho_local), sel_rows

    rho_logged, sel_logged = run_one_seed_logged(int(best_seed))

    # Write best selection CSV
    out_csv = out_dir / "best_selection.csv"
    out_df = df.copy()
    out_df[args.selected_col] = 0
    chosen_orig = cand_df.iloc[sel_logged]["_orig_index"].to_list()
    out_df.loc[chosen_orig, args.selected_col] = 1
    # Also include resolved paths for reproducibility.
    out_df["_resolved_path"] = None
    out_df.loc[cand_df["_orig_index"].to_list(), "_resolved_path"] = cand_df["_resolved_path"].to_list()
    out_df.to_csv(out_csv, index=False)

    summary = {
        "best_seed": int(best_seed),
        "best_rho": float(rho_logged),
        "n_subjects": int(n_subj),
        "n_candidates": int(len(cand_df)),
        "input_csv": args.input_csv,
        "out_dir": str(out_dir),
        "out_csv": str(out_csv),
        "log_csv": str(log_csv_path),
        "mask_path": args.mask_path,
        "subject_col": args.subject_col,
        "selected_col": args.selected_col,
        "paths_col": args.paths_col,
        "score_col": args.score_col,
        "add_intercept": bool(args.add_intercept),
        "contrast": list(map(float, args.contrast)),
        "passes": int(args.passes),
        "seed_start": int(args.seed_start),
        "seed_stop": int(args.seed_stop),
        "plot_mode": args.plot_mode,
    }
    (out_dir / "best_selection.summary.json").write_text(json.dumps(summary, indent=2))

    # Optional: run regression once using 05b script on selected==1
    if args.run_regression:
        from importlib.util import spec_from_file_location, module_from_spec

        regression_script = Path(__file__).resolve().parent / "05b_full_voxelwise_regression.py"
        spec = spec_from_file_location("_reg05b", str(regression_script))
        mod = module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(mod)

        regression_out = out_dir / "regression"
        regression_out.mkdir(parents=True, exist_ok=True)

        # Create a regression input with a `paths` column pointing at onetouch_path so the formula stays identical.
        reg_input = out_dir / "best_selection_for_regression.csv"
        reg_df = out_df.copy()
        # Use resolved paths (guaranteed to exist for candidate rows).
        reg_df["paths"] = reg_df["_resolved_path"].fillna(reg_df.get(args.paths_col, "")).astype(str)
        reg_df.to_csv(reg_input, index=False)

        contrast_path = out_dir / "contrast.json"
        contrast_path.write_text(json.dumps([[0, 1]], indent=2))

        cmd = [
            "fit",
            "--input-path",
            str(reg_input),
            "--out-dir",
            str(regression_out),
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
            "0",
            "--all-outputs",
            "false",
            "--cv",
            "2",
        ]
        if args.verbose:
            cmd += ["--verbose"]
        mod.main(cmd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
