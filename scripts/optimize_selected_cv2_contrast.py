#!/usr/bin/env python3
"""
Optimize which per-subject session is marked `selected=1` to maximize the
cv=2 Spearman rho for the *contrast-only* branch used by:
`VoxelwiseRegression._evaluate_map(cv=2)`.

This is intentionally separate from `05b_full_voxelwise_regression.py`.

What it optimizes
-----------------
- Assumes your dependent variable is voxelwise (`paths` -> NIfTI per row).
- Fits a voxelwise OLS model on each cv fold, forms the contrast t-map for the
  requested contrast row, then scores held-out subjects as:
    cosine(subject_image_vector, contrast_tmap_vector)
- Computes Spearman rho between those scalar predictions and:
    actual_y = the scalar score column (e.g. TotalBarsScore) for the held-out rows.

Outputs
-------
Writes into `--opt-dir`:
- `cv2_optimization_log.csv` (eval-by-eval rho + scatterplot path)
- `scatterplots/eval_XXXXXX_scatterplot.svg` (one per eval when plot-mode=all)
- `optimized_master_list.csv` (same as input with updated selected column)
- `optimized_master_list.summary.json`

Notes
-----
- Requires `--mask-path` that matches (or can resample) your input NIfTIs.
- Uses greedy coordinate ascent: for each subject, try each session candidate,
  keep the session that improves global rho.
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


# Reduce noisy matplotlib/fontconfig cache warnings in headless/script contexts.
_TMPDIR = Path(os.environ.get("TMPDIR", "/tmp"))
os.environ.setdefault("MPLCONFIGDIR", str(_TMPDIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_TMPDIR / "xdg_cache"))

# Ensure `import calvin_utils...` works when running from repo root without installation.
_CIRCUIT_PYPER_DIR = Path(__file__).resolve().parents[1]
if str(_CIRCUIT_PYPER_DIR) not in sys.path:
    sys.path.insert(0, str(_CIRCUIT_PYPER_DIR))


def _coerce_value(raw: str) -> Any:
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


def _parse_bool(v: str) -> bool:
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {v}")


def _cosine_similarity_rows(A: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    num = A @ b
    den = (np.linalg.norm(A, axis=1) * (np.linalg.norm(b) + eps)) + eps
    return num / den


def _fit_tmap_for_contrast(
    *,
    X: np.ndarray,
    Y: np.ndarray,
    C: np.ndarray,
    XtX_inv: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    OLS per-voxel:
      B = (X'X)^-1 X'Y
      t = (C B) / sqrt( (C (X'X)^-1 C') * MSE )
    Returns tmap shape (v,).
    """
    n, p = X.shape
    B = XtX_inv @ (X.T @ Y)  # (p,v)
    resid = Y - (X @ B)  # (n,v)
    dof = max(n - p, 1)
    mse = (resid * resid).sum(axis=0, keepdims=True) / float(dof)  # (1,v)
    num = (C @ B)  # (1,v)
    denom_scalar = float((C @ XtX_inv @ C.T).squeeze())
    denom = np.sqrt(np.maximum(denom_scalar, eps) * np.maximum(mse, eps))
    return (num / (denom + eps)).squeeze(axis=0).astype(np.float32, copy=False)


def _cv2_predictions(*, X: np.ndarray, Y: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Match the current behavior of `VoxelwiseRegression.run_prediction_cv` when
    `XTX_inv` has already been computed by a prior full-dataset regression:
    the fold regressions reuse the full-dataset (X'X)^-1.
    """
    n = X.shape[0]
    XtX_inv_full = np.linalg.pinv(X.T @ X)
    idx = np.arange(n)
    fold_idx = np.array_split(idx, 2)
    splits = []
    for k in range(2):
        test = fold_idx[k]
        train = np.concatenate([fold_idx[j] for j in range(2) if j != k])
        splits.append((train, test))

    preds = np.zeros((n,), dtype=np.float32)
    for train, test in splits:
        tmap = _fit_tmap_for_contrast(X=X[train], Y=Y[train], C=C, XtX_inv=XtX_inv_full)
        preds[test] = _cosine_similarity_rows(Y[test], tmap)
    return preds


def _mask_and_vectorize(
    img_path: str,
    *,
    mask_img,
    mask_flat: np.ndarray,
    resample_interpolation: str,
    verbose: bool,
) -> np.ndarray:
    import nibabel as nib
    from nilearn import image

    img = nib.load(img_path)
    if img.shape != mask_img.shape or (not np.allclose(img.affine, mask_img.affine)):
        if verbose:
            print(f"[resample] {img_path} -> mask space")
        img = image.resample_to_img(
            img,
            mask_img,
            interpolation=resample_interpolation,
            force_resample=True,
            copy_header=True,
        )
    data = img.get_fdata(dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data.reshape(-1)[mask_flat]


@dataclass(frozen=True)
class OptimizeConfig:
    input_path: str
    opt_dir: str
    out_csv: str
    mask_path: str
    subject_col: str
    selected_col: str
    paths_col: str
    score_col: str
    add_intercept: bool
    contrast: list[float]
    passes: int
    seed: int
    plot_mode: str
    resample_interpolation: str
    candidate_condition: str | None
    candidate_value: str | None
    verbose: bool


def optimize_selected_cv2(cfg: OptimizeConfig) -> None:
    from scipy.stats import spearmanr

    base_dir = Path(cfg.opt_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "scatterplots").mkdir(parents=True, exist_ok=True)
    log_csv_path = base_dir / "cv2_optimization_log.csv"

    df = pd.read_csv(cfg.input_path)
    for col in (cfg.subject_col, cfg.selected_col, cfg.paths_col, cfg.score_col):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    cand_df = df.copy()
    if cfg.candidate_condition is not None and cfg.candidate_value is not None:
        v = _coerce_value(cfg.candidate_value)
        if cfg.candidate_condition == "equal":
            cand_df = cand_df[cand_df[cfg.selected_col] == v]
        elif cfg.candidate_condition == "not":
            cand_df = cand_df[cand_df[cfg.selected_col] != v]
        elif cfg.candidate_condition == "above":
            cand_df = cand_df[pd.to_numeric(cand_df[cfg.selected_col], errors="coerce") > float(v)]
        elif cfg.candidate_condition == "below":
            cand_df = cand_df[pd.to_numeric(cand_df[cfg.selected_col], errors="coerce") < float(v)]
        else:
            raise ValueError("candidate-condition must be one of: equal, above, below, not")

    cand_df = cand_df.dropna(subset=[cfg.paths_col, cfg.score_col, cfg.subject_col]).reset_index(drop=False).rename(
        columns={"index": "_orig_index"}
    )
    cand_df = cand_df.sort_values("_orig_index").reset_index(drop=True)

    unique_paths = sorted(set(cand_df[cfg.paths_col].astype(str).tolist()))
    if cfg.verbose:
        print(f"[cache] unique images={len(unique_paths)}")

    # Load/Vectorize images using the same importer used by RegressionPrep so the
    # voxel ordering and masking matches 05b.
    from calvin_utils.file_utils.import_functions import GiiNiiFileImport

    importer = GiiNiiFileImport(
        import_path=pd.Series(unique_paths),
        mask_path=cfg.mask_path if cfg.mask_path is not None else "default",
        transpose=False,
    )
    loaded = importer.run()
    if not isinstance(loaded, pd.DataFrame):
        raise TypeError(f"Expected GiiNiiFileImport.run() to return a DataFrame, got {type(loaded)}")
    arr = loaded.to_numpy(dtype=np.float32).T  # (n_paths, n_vox)
    # Match RegressionPrep._handle_nans semantics (global finite extrema for inf replacement).
    finite = arr[np.isfinite(arr)]
    max_val = float(finite.max()) if finite.size else 0.0
    min_val = float(finite.min()) if finite.size else 0.0
    arr = np.nan_to_num(arr, nan=0.0, posinf=max_val, neginf=min_val)
    if arr.shape[0] != len(unique_paths):
        raise ValueError(f"Importer returned {arr.shape[0]} observations for {len(unique_paths)} paths.")
    vec_cache: dict[str, np.ndarray] = {pth: arr[i, :] for i, pth in enumerate(unique_paths)}

    # Group candidates by subject
    subj_to_rows: dict[Any, list[int]] = {}
    for i, subj in enumerate(cand_df[cfg.subject_col].tolist()):
        subj_to_rows.setdefault(subj, []).append(i)
    subjects = list(subj_to_rows.keys())

    # Initialize selection: prefer rows already selected==1 (if any), else first row per subject.
    selected: dict[Any, int] = {}
    for subj, rows in subj_to_rows.items():
        chosen = None
        for r in rows:
            if _coerce_value(cand_df.loc[r, cfg.selected_col]) == 1:
                chosen = r
                break
        selected[subj] = chosen if chosen is not None else rows[0]

    # Materialize per-row arrays
    row_paths = cand_df[cfg.paths_col].astype(str).tolist()
    row_vecs = [vec_cache[p] for p in row_paths]
    row_scores = pd.to_numeric(cand_df[cfg.score_col], errors="raise").to_numpy(dtype=np.float32)
    row_orig_idx = cand_df["_orig_index"].to_numpy(dtype=int)

    rng = np.random.default_rng(int(cfg.seed))

    def build_X(xvals: np.ndarray) -> np.ndarray:
        if cfg.add_intercept:
            return np.column_stack([np.ones_like(xvals), xvals]).astype(np.float32, copy=False)
        return xvals[:, None].astype(np.float32, copy=False)

    def materialize_selection(sel: dict[Any, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build (Y_sel, x_score, actual_y) in the same observation order 05b will use:
        rows sorted by their original row index in the master table.
        """
        chosen = [int(sel[s]) for s in subjects]
        chosen_sorted = sorted(chosen, key=lambda r: int(row_orig_idx[int(r)]))
        Y_sel = np.stack([row_vecs[int(r)] for r in chosen_sorted], axis=0).astype(np.float32, copy=False)
        x_score = row_scores[np.asarray(chosen_sorted, dtype=int)].astype(np.float32, copy=False)
        # Objective: correlate contrast-cosine damage score with the scalar dependent variable (score_col)
        # e.g. TotalBarsScore. This is a true (n_obs,) vector.
        actual_y = x_score
        return Y_sel, x_score, actual_y

    C = np.asarray([cfg.contrast], dtype=np.float32)
    Y_sel, x_score, actual_y = materialize_selection(selected)
    X = build_X(x_score)
    if C.shape[1] != X.shape[1]:
        raise ValueError(f"Contrast length {C.shape[1]} != n_preds {X.shape[1]}")

    def log_row(row: dict[str, Any]) -> None:
        exists = log_csv_path.exists()
        pd.DataFrame([row]).to_csv(log_csv_path, mode="a", header=not exists, index=False)

    def maybe_plot(eval_id: int, preds: np.ndarray, actual: np.ndarray) -> str:
        from calvin_utils.statistical_utils.scatterplot import simple_scatter
        import matplotlib.pyplot as plt

        dataset_name = f"eval_{eval_id:06d}"
        plot_df = pd.DataFrame({"pred": preds, "actual": actual})
        simple_scatter(
            df=plot_df,
            x_col="pred",
            y_col="actual",
            dataset_name=dataset_name,
            out_dir=str(base_dir),
            x_label="Predicted (contrast cosine)",
            y_label=f"Actual ({cfg.score_col})",
            show=False,
        )
        # Avoid figure leak when plot_mode=all.
        plt.close("all")
        return str(base_dir / "scatterplots" / f"{dataset_name}_scatterplot.svg")

    eval_id = 0
    preds = _cv2_predictions(X=X, Y=Y_sel, C=C)
    best_rho, _ = spearmanr(preds, actual_y, nan_policy="omit")
    best_rho = float(best_rho) if best_rho is not None else float("nan")
    best_selected = dict(selected)

    plot_path = ""
    if cfg.plot_mode in {"all", "best-only"}:
        plot_path = maybe_plot(eval_id, preds, actual_y)
    log_row(
        {
            "eval_id": eval_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "pass": 0,
            "subject": "",
            "trial_row": "",
            "kept_row": "",
            "trial_orig_index": "",
            "kept_orig_index": "",
            "trial_path": "",
            "kept_path": "",
            "rho": best_rho,
            "plot_path": plot_path,
            "note": "init",
        }
    )
    eval_id += 1

    if cfg.verbose:
        print(f"[init] rho={best_rho:.4f} subjects={len(subjects)}")

    for pass_idx in range(int(cfg.passes)):
        improved = False
        order = subjects.copy()
        rng.shuffle(order)
        for subj in order:
            rows = subj_to_rows[subj]
            if len(rows) <= 1:
                continue
            cur_row = int(selected[subj])

            cur_best_row = cur_row
            cur_best_rho = best_rho

            for r in rows:
                if r == cur_row:
                    continue

                # trial swap (selection change affects observation ordering -> rematerialize)
                prev_row = int(selected[subj])
                selected[subj] = int(r)
                Y_sel, x_score, actual_y = materialize_selection(selected)
                X = build_X(x_score)
                preds = _cv2_predictions(X=X, Y=Y_sel, C=C)
                rho, _ = spearmanr(preds, actual_y, nan_policy="omit")
                rho = float(rho) if rho is not None else float("nan")

                plot_path = ""
                if cfg.plot_mode == "all":
                    plot_path = maybe_plot(eval_id, preds, actual_y)

                log_row(
                    {
                        "eval_id": eval_id,
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "pass": int(pass_idx + 1),
                        "subject": str(subj),
                        "trial_row": int(r),
                        "kept_row": int(cur_row),
                        "trial_orig_index": int(row_orig_idx[int(r)]),
                        "kept_orig_index": int(row_orig_idx[int(cur_row)]),
                        "trial_path": row_paths[int(r)],
                        "kept_path": row_paths[int(cur_row)],
                        "rho": rho,
                        "plot_path": plot_path,
                        "note": "trial",
                    }
                )
                eval_id += 1

                if cfg.verbose:
                    print(f"[pass {pass_idx+1}] subj={subj} try={r} rho={rho:.4f}")

                if np.isfinite(rho) and rho > cur_best_rho:
                    cur_best_rho = rho
                    cur_best_row = int(r)

                # restore selection for next trial
                selected[subj] = prev_row

            # commit best for this subject
            selected[subj] = int(cur_best_row)
            Y_sel, x_score, actual_y = materialize_selection(selected)

            if cur_best_rho > best_rho:
                best_rho = float(cur_best_rho)
                best_selected = dict(selected)
                improved = True
                if cfg.verbose:
                    print(f"[best] rho={best_rho:.4f}")
                if cfg.plot_mode == "best-only":
                    X = build_X(x_score)
                    preds = _cv2_predictions(X=X, Y=Y_sel, C=C)
                    plot_path = maybe_plot(eval_id, preds, actual_y)
                    log_row(
                        {
                            "eval_id": eval_id,
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "pass": int(pass_idx + 1),
                            "subject": str(subj),
                            "trial_row": int(cur_best_row),
                            "kept_row": int(cur_row),
                            "trial_orig_index": int(row_orig_idx[int(cur_best_row)]),
                            "kept_orig_index": int(row_orig_idx[int(cur_row)]),
                            "trial_path": row_paths[int(cur_best_row)],
                            "kept_path": row_paths[int(cur_row)],
                            "rho": best_rho,
                            "plot_path": plot_path,
                            "note": "global_best",
                        }
                    )
                    eval_id += 1

        if not improved:
            if cfg.verbose:
                print(f"[stop] no improvement on pass {pass_idx+1}")
            break

    # Write optimized selected column back to full df
    out_df = df.copy()
    out_df[cfg.selected_col] = 0
    chosen_rows = [int(best_selected[s]) for s in subjects]
    chosen_orig_idx = cand_df.iloc[chosen_rows]["_orig_index"].to_list()
    out_df.loc[chosen_orig_idx, cfg.selected_col] = 1

    # Enforce exactly one selected per subject in the *full* table:
    # - If a subject had no valid candidates (e.g., missing paths/score), it may have 0 selected.
    #   In that case, pick the first row for that subject so downstream filtering stays well-defined.
    # - If a subject somehow has >1 selected, collapse to the first.
    for subj, sub_idx in out_df.groupby(cfg.subject_col).groups.items():
        idx_list = list(sub_idx)
        sel_sum = int(pd.to_numeric(out_df.loc[idx_list, cfg.selected_col], errors="coerce").fillna(0).astype(int).sum())
        if sel_sum == 1:
            continue
        out_df.loc[idx_list, cfg.selected_col] = 0
        out_df.loc[idx_list[0], cfg.selected_col] = 1

    out_csv_path = Path(cfg.out_csv)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv_path, index=False)

    summary = {
        "best_rho": float(best_rho),
        "n_subjects": int(len(subjects)),
        "n_candidates": int(len(cand_df)),
        "input_path": cfg.input_path,
        "out_csv": str(out_csv_path),
        "opt_dir": str(base_dir),
        "log_csv": str(log_csv_path),
        "mask_path": cfg.mask_path,
        "subject_col": cfg.subject_col,
        "selected_col": cfg.selected_col,
        "paths_col": cfg.paths_col,
        "score_col": cfg.score_col,
        "add_intercept": bool(cfg.add_intercept),
        "contrast": cfg.contrast,
        "passes": int(cfg.passes),
        "seed": int(cfg.seed),
        "plot_mode": cfg.plot_mode,
        "resample_interpolation": cfg.resample_interpolation,
    }
    (out_csv_path.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2))

    if cfg.verbose:
        print(f"[write] {out_csv_path}")
        print(f"[write] {log_csv_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="optimize_selected_cv2_contrast.py",
        description="Optimize per-subject selected session to maximize cv=2 Spearman rho (contrast-only).",
    )
    p.add_argument("--input-path", required=True, help="CSV with multiple sessions per subject.")
    p.add_argument("--opt-dir", required=True, help="Directory to write scatterplots and optimization CSV log.")
    p.add_argument("--out-csv", required=True, help="Output CSV with updated selected column.")
    p.add_argument("--mask-path", required=True, help="Mask NIfTI for vectorizing and alignment.")

    p.add_argument(
        "--subject-col",
        default="id",
        help="Subject identifier column (groups sessions). Default: id",
    )
    p.add_argument("--selected-col", default="selected", help="Selection flag column to optimize.")
    p.add_argument(
        "--paths-col",
        default="onetouch_path",
        help="Column containing NIfTI paths. Default: onetouch_path (OneTouch-mounted paths).",
    )
    p.add_argument("--score-col", default="TotalBarsScore", help="Scalar predictor column in formula RHS.")

    p.add_argument("--add-intercept", type=_parse_bool, default=True, help="Include intercept in design (true/false).")
    p.add_argument("--contrast", nargs="+", type=float, default=[0.0, 1.0], help="Contrast row (length must equal n_preds).")

    p.add_argument("--passes", type=int, default=10, help="Greedy coordinate-ascent passes.")
    p.add_argument("--seed", type=int, default=0, help="Random seed (subject order per pass).")
    p.add_argument("--plot-mode", choices=["none", "best-only", "all"], default="all", help="Which evals get scatterplots.")
    p.add_argument("--resample-interpolation", default="nearest", help="nilearn resample interpolation if needed.")

    p.add_argument("--candidate-condition", default=None, help="Optional: equal/above/below/not applied to selected-col to restrict candidate rows.")
    p.add_argument("--candidate-value", default=None, help="Value used with --candidate-condition.")

    p.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Backward/alias handling: user may call it onetouch_file_path.
    # Normalize to actual column name if needed.
    if args.paths_col == "onetouch_file_path":
        args.paths_col = "onetouch_path"

    cfg = OptimizeConfig(
        input_path=args.input_path,
        opt_dir=args.opt_dir,
        out_csv=args.out_csv,
        mask_path=args.mask_path,
        subject_col=args.subject_col,
        selected_col=args.selected_col,
        paths_col=args.paths_col,
        score_col=args.score_col,
        add_intercept=args.add_intercept,
        contrast=list(args.contrast),
        passes=args.passes,
        seed=args.seed,
        plot_mode=args.plot_mode,
        resample_interpolation=args.resample_interpolation,
        candidate_condition=args.candidate_condition,
        candidate_value=args.candidate_value,
        verbose=args.verbose,
    )
    optimize_selected_cv2(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
