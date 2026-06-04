#!/usr/bin/env python3
"""
Script equivalent of:
`circuit_pyper/notebooks/neuroimaging_notebooks/convergent_causal_mapping/05b_full_voxelwise_regression.ipynb`

Goal: encapsulate the same voxelwise-regression workflow behind a fast CLI.

Core workflow
-------------
1) Load a CSV/XLS(X) with subject covariates + neuroimaging path columns
2) Optional preprocessing (drop NaNs, drop rows by condition, one-hot encode)
3) Build design/outcome matrices from a patsy formula (supports voxelwise vars)
4) Build contrasts (basic by default, or load a custom contrast matrix)
5) Prepare tensors/JSON via RegressionPrep
6) Fit voxelwise regression (and optionally cross-validation)
7) Optionally predict maps for a new dataset using saved beta maps
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
    """Try int -> float -> str."""
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


def _parse_data_transform(v: str | None) -> str | None:
    if v is None:
        return None
    s = v.strip().lower()
    if s in {"none", "null", "no", ""}:
        return None
    if s in {"standardize", "rank"}:
        return s
    raise argparse.ArgumentTypeError("data-transform must be one of: none, standardize, rank")


def _load_contrast_matrix(path: str) -> list[list[float]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.suffix.lower() == ".json":
        obj = json.loads(p.read_text())
        if not isinstance(obj, list):
            raise ValueError("Contrast JSON must be a list of rows.")
        return obj
    df = pd.read_csv(p, header=None)
    return df.to_numpy(dtype=float).tolist()


def _maybe_save_tables(out_dir: str, *, df: pd.DataFrame, design: pd.DataFrame, outcome: pd.DataFrame, contrast: pd.DataFrame) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "preprocessed_data.csv"), index=False)
    design.to_csv(os.path.join(out_dir, "design_matrix.csv"), index=False)
    outcome.to_csv(os.path.join(out_dir, "outcome_df.csv"), index=False)
    contrast.to_csv(os.path.join(out_dir, "contrast_matrix.csv"), index=False)


def prepare_from_table(
    *,
    input_path: str,
    out_dir: str,
    sheet: str | None,
    formula: str,
    voxelwise_vars: list[str],
    voxelwise_interactions: list[str],
    add_intercept: bool,
    drop_nans: list[str] | None,
    drop_rows: list[tuple[str, str, str]] | None,
    one_hot: list[str] | None,
    mask_path: str | None,
    exchangeability_col: str | None,
    weights_col: str | None,
    data_transform_method: str | None,
    contrast_file: str | None,
    verbose: bool,
) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Lazy imports to keep `--help` fast and avoid heavy matplotlib/seaborn imports unless needed.
    from calvin_utils.permutation_analysis_utils.statsmodels_palm import CalvinStatsmodelsPalm
    from calvin_utils.permutation_analysis_utils.voxelwise_regression_prep import RegressionPrep

    cal_palm = CalvinStatsmodelsPalm(input_csv_path=input_path, output_dir=out_dir, sheet=sheet)
    df = cal_palm.read_and_display_data()

    if verbose:
        print(f"[load] rows={len(df)} cols={len(df.columns)}")

    if drop_nans:
        df = cal_palm.drop_nans_from_columns(columns_to_drop_from=list(drop_nans))
        if verbose:
            print(f"[drop_nans] remaining rows={len(df)}")

    if drop_rows:
        for col, cond, val in drop_rows:
            df, _ = cal_palm.drop_rows_based_on_value(col, cond, _coerce_value(val))
        if verbose:
            print(f"[drop_rows] remaining rows={len(df)}")

    if one_hot:
        for col in one_hot:
            if col not in df.columns:
                raise ValueError(f"one-hot column not found: {col}")
            dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
            df = df.join(dummies)
        if verbose:
            print(f"[one_hot] cols now={len(df.columns)}")

    outcome_df, design_matrix = cal_palm.define_design_matrix(
        formula,
        df,
        add_intercept=add_intercept,
        voxelwise_variable_list=list(voxelwise_vars or []),
        voxelwise_interaction_terms=list(voxelwise_interactions or []),
    )

    if contrast_file:
        contrast_matrix = _load_contrast_matrix(contrast_file)
    else:
        contrast_matrix = cal_palm.generate_basic_contrast_matrix(design_matrix)
    contrast_matrix_df = cal_palm.finalize_contrast_matrix(design_matrix=design_matrix, contrast_matrix=contrast_matrix)

    exchangeability_block = None
    if exchangeability_col:
        if exchangeability_col not in df.columns:
            raise ValueError(f"exchangeability-col not found: {exchangeability_col}")
        exchangeability_block = pd.to_numeric(df[exchangeability_col], errors="raise").astype(int).to_numpy()

    weights = None
    if weights_col:
        if weights_col not in df.columns:
            raise ValueError(f"weights-col not found: {weights_col}")
        weights = pd.to_numeric(df[weights_col], errors="raise").astype(float).to_numpy()

    preparer = RegressionPrep(
        design_matrix=design_matrix,
        contrast_matrix=contrast_matrix_df,
        outcome_df=outcome_df,
        out_dir=out_dir,
        voxelwise_variables=list(voxelwise_vars or []),
        voxelwise_interactions=list(voxelwise_interactions or []),
        mask_path=mask_path,
        exchangeability_block=exchangeability_block,
        data_transform_method=data_transform_method,
        weights=weights,
        formula=formula,
    )
    _, json_path = preparer.run()
    _maybe_save_tables(out_dir, df=df, design=design_matrix, outcome=outcome_df, contrast=contrast_matrix_df)
    if verbose:
        print(f"[prep] wrote {json_path}")
    return json_path, df, outcome_df, design_matrix, contrast_matrix_df


def run_regression_from_json(
    *,
    json_path: str,
    out_dir: str | None,
    mask_path: str | None,
    regression_type: str,
    n_permutations: int,
    all_outputs: bool,
    cv: str | None,
    cv_dependent_source: str,
    cv_dependent_cols: list[str] | None,
) -> None:
    from calvin_utils.permutation_analysis_utils.voxelwise_regression import VoxelwiseRegression

    jp = Path(json_path)
    if not jp.exists():
        raise FileNotFoundError(str(jp))
    resolved_out_dir = out_dir or str(jp.parent)
    os.makedirs(resolved_out_dir, exist_ok=True)

    reg = VoxelwiseRegression(
        json_path,
        mask_path=mask_path,
        out_dir=resolved_out_dir,
        regression_type=regression_type,
        n_permutations=int(n_permutations),
    )
    if all_outputs:
        reg.run_all_outputs()
    else:
        reg.run()

    if cv:
        if str(cv).lower() in {"all"}:
            reg.run_cross_validation(
                dependent_variable_source=cv_dependent_source,
                dependent_cols=cv_dependent_cols,
            )
        else:
            try:
                cv_arg: str | int = int(cv)
            except Exception:
                cv_arg = str(cv).lower()
            reg._evaluate_map(
                cv=cv_arg,
                dependent_variable_source=cv_dependent_source,
                dependent_cols=cv_dependent_cols,
            )


def run_prediction(
    *,
    json_path: str,
    out_dir: str,
    mask_path: str | None,
    params_dir: str,
    regression_type: str,
) -> None:
    from calvin_utils.permutation_analysis_utils.voxelwise_regression import VoxelwiseRegression

    os.makedirs(out_dir, exist_ok=True)
    reg = VoxelwiseRegression(
        json_path,
        mask_path=mask_path,
        out_dir=out_dir,
        regression_type=regression_type,
        n_permutations=0,
    )
    pred = reg._run_prediction_switch(temp_dir=params_dir)
    reg.PREDICTIONS = np.asarray(pred, dtype=np.float32)
    reg._save_result_maps()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="05b_full_voxelwise_regression.py",
        description="CLI wrapper for full voxelwise regression workflow (05b notebook).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common_prep_args(sp: argparse.ArgumentParser, *, default_add_intercept: bool) -> None:
        sp.add_argument("--input-path", required=True, help="CSV (or Excel) with covariates + neuroimaging paths.")
        sp.add_argument("--sheet", default=None, help="Excel sheet name (if input-path is Excel).")
        sp.add_argument("--out-dir", required=True, help="Output directory (prep artifacts + regression outputs).")
        sp.add_argument("--mask-path", default=None, help="Brain mask path (recommended for NIfTI workflows).")
        sp.add_argument("--formula", required=True, help="Patsy-like formula, e.g. 'paths ~ TotalBarsScore'.")
        sp.add_argument("--voxelwise-var", action="append", default=[], help="Column name for a voxelwise (image-path) variable. Repeatable.")
        sp.add_argument("--voxelwise-interaction", action="append", default=[], help="Interaction term involving voxelwise vars. Repeatable.")
        sp.add_argument("--add-intercept", type=_parse_bool, default=default_add_intercept, help="Include intercept in design matrix (true/false).")
        sp.add_argument("--drop-nans", nargs="*", default=None, help="Drop rows with NaNs in these columns.")
        sp.add_argument("--drop-row", nargs=3, action="append", default=None, metavar=("COLUMN", "COND", "VALUE"), help="Drop rows based on condition: equal/above/below/not.")
        sp.add_argument("--one-hot", action="append", default=None, help="One-hot encode this categorical column. Repeatable.")
        sp.add_argument("--exchangeability-col", default=None, help="Column to use as exchangeability block labels (int-like).")
        sp.add_argument("--weights-col", default=None, help="Column to use as positive weights (float-like).")
        sp.add_argument("--data-transform", type=_parse_data_transform, default=None, help="none|standardize|rank (applied to tensors).")
        sp.add_argument("--contrast-file", default=None, help="Optional custom contrast matrix (CSV/TSV or JSON list-of-lists).")
        sp.add_argument("--verbose", action="store_true", help="Print progress info.")

    sp_prep = sub.add_parser("prep", help="Prepare tensors + dataset_dict.json (no regression).")
    add_common_prep_args(sp_prep, default_add_intercept=True)

    sp_fit = sub.add_parser("fit", help="Prepare + run voxelwise regression (like the notebook).")
    add_common_prep_args(sp_fit, default_add_intercept=True)
    sp_fit.add_argument("--regression-type", default="linear", choices=["linear", "naive_bayes"], help="Regression type.")
    sp_fit.add_argument("--n-permutations", type=int, default=0, help="Number of permutations (0 disables).")
    sp_fit.add_argument("--all-outputs", type=_parse_bool, default=True, help="Run all outcome channels (true/false).")
    sp_fit.add_argument("--cv", default=None, help="Cross-validation: 'loocv', an int (e.g. 2), or 'all'.")
    sp_fit.add_argument(
        "--cv-dependent-source",
        default="outcome",
        choices=["outcome", "design"],
        help="Which vector to correlate against in CV scatterplots (default: outcome).",
    )
    sp_fit.add_argument(
        "--cv-dependent-col",
        action="append",
        default=None,
        help="When --cv-dependent-source=design, evaluate against this design_matrix column (repeatable). "
        "If omitted, evaluates against all design columns.",
    )

    sp_run = sub.add_parser("run", help="Run regression from an existing dataset_dict.json.")
    sp_run.add_argument("--json-path", required=True, help="Path to dataset_dict.json produced by prep.")
    sp_run.add_argument("--out-dir", default=None, help="Output directory (default: json-path parent).")
    sp_run.add_argument("--mask-path", default=None, help="Mask path (optional; json may already include mask_path).")
    sp_run.add_argument("--regression-type", default="linear", choices=["linear", "naive_bayes"], help="Regression type.")
    sp_run.add_argument("--n-permutations", type=int, default=0, help="Number of permutations (0 disables).")
    sp_run.add_argument("--all-outputs", type=_parse_bool, default=True, help="Run all outcome channels (true/false).")
    sp_run.add_argument("--cv", default=None, help="Cross-validation: 'loocv', an int (e.g. 2), or 'all'.")
    sp_run.add_argument(
        "--cv-dependent-source",
        default="outcome",
        choices=["outcome", "design"],
        help="Which vector to correlate against in CV scatterplots (default: outcome).",
    )
    sp_run.add_argument(
        "--cv-dependent-col",
        action="append",
        default=None,
        help="When --cv-dependent-source=design, evaluate against this design_matrix column (repeatable). "
        "If omitted, evaluates against all design columns.",
    )

    sp_predict = sub.add_parser("predict", help="Predict maps for a dataset using saved beta maps from a fitted model.")
    group = sp_predict.add_mutually_exclusive_group(required=True)
    group.add_argument("--json-path", help="Prepared dataset_dict.json to predict from.")
    group.add_argument("--input-path", help="CSV/XLS(X) to prepare then predict from (uses the same prep args).")
    sp_predict.add_argument("--sheet", default=None, help="Excel sheet name (if input-path is Excel).")
    sp_predict.add_argument("--out-dir", required=True, help="Output directory for prediction_*.nii(.gz) maps.")
    sp_predict.add_argument("--mask-path", default=None, help="Mask path (optional; recommended).")
    sp_predict.add_argument("--params-dir", required=True, help="Directory containing fitted params (beta_predictor_*.nii*).")
    sp_predict.add_argument("--regression-type", default="linear", choices=["linear", "naive_bayes"], help="Regression type.")
    sp_predict.add_argument("--formula", default=None, help="Formula (required if using --input-path).")
    sp_predict.add_argument("--voxelwise-var", action="append", default=[], help="Voxelwise variable column. Repeatable.")
    sp_predict.add_argument("--voxelwise-interaction", action="append", default=[], help="Voxelwise interaction term. Repeatable.")
    sp_predict.add_argument("--add-intercept", type=_parse_bool, default=False, help="Include intercept in design matrix (true/false).")
    sp_predict.add_argument("--drop-nans", nargs="*", default=None, help="Drop rows with NaNs in these columns.")
    sp_predict.add_argument("--drop-row", nargs=3, action="append", default=None, metavar=("COLUMN", "COND", "VALUE"), help="Drop rows based on condition.")
    sp_predict.add_argument("--one-hot", action="append", default=None, help="One-hot encode this categorical column. Repeatable.")
    sp_predict.add_argument("--exchangeability-col", default=None, help="Column to use as exchangeability block labels (int-like).")
    sp_predict.add_argument("--weights-col", default=None, help="Column to use as positive weights (float-like).")
    sp_predict.add_argument("--data-transform", type=_parse_data_transform, default=None, help="none|standardize|rank (applied to tensors).")
    sp_predict.add_argument("--contrast-file", default=None, help="Optional custom contrast matrix (CSV/TSV or JSON).")
    sp_predict.add_argument("--verbose", action="store_true", help="Print progress info.")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "prep":
        prepare_from_table(
            input_path=args.input_path,
            out_dir=args.out_dir,
            sheet=args.sheet,
            formula=args.formula,
            voxelwise_vars=args.voxelwise_var,
            voxelwise_interactions=args.voxelwise_interaction,
            add_intercept=args.add_intercept,
            drop_nans=args.drop_nans,
            drop_rows=args.drop_row,
            one_hot=args.one_hot,
            mask_path=args.mask_path,
            exchangeability_col=args.exchangeability_col,
            weights_col=args.weights_col,
            data_transform_method=args.data_transform,
            contrast_file=args.contrast_file,
            verbose=args.verbose,
        )
        return 0

    if args.cmd == "fit":
        json_path, *_ = prepare_from_table(
            input_path=args.input_path,
            out_dir=args.out_dir,
            sheet=args.sheet,
            formula=args.formula,
            voxelwise_vars=args.voxelwise_var,
            voxelwise_interactions=args.voxelwise_interaction,
            add_intercept=args.add_intercept,
            drop_nans=args.drop_nans,
            drop_rows=args.drop_row,
            one_hot=args.one_hot,
            mask_path=args.mask_path,
            exchangeability_col=args.exchangeability_col,
            weights_col=args.weights_col,
            data_transform_method=args.data_transform,
            contrast_file=args.contrast_file,
            verbose=args.verbose,
        )
        run_regression_from_json(
            json_path=json_path,
            out_dir=args.out_dir,
            mask_path=args.mask_path,
            regression_type=args.regression_type,
            n_permutations=args.n_permutations,
            all_outputs=args.all_outputs,
            cv=args.cv,
            cv_dependent_source=args.cv_dependent_source,
            cv_dependent_cols=args.cv_dependent_col,
        )
        return 0

    if args.cmd == "run":
        run_regression_from_json(
            json_path=args.json_path,
            out_dir=args.out_dir,
            mask_path=args.mask_path,
            regression_type=args.regression_type,
            n_permutations=args.n_permutations,
            all_outputs=args.all_outputs,
            cv=args.cv,
            cv_dependent_source=args.cv_dependent_source,
            cv_dependent_cols=args.cv_dependent_col,
        )
        return 0

    if args.cmd == "predict":
        if args.json_path:
            json_path = args.json_path
        else:
            if not args.formula:
                raise SystemExit("--formula is required when using --input-path for predict.")
            json_path, *_ = prepare_from_table(
                input_path=args.input_path,
                out_dir=args.out_dir,
                sheet=args.sheet,
                formula=args.formula,
                voxelwise_vars=args.voxelwise_var,
                voxelwise_interactions=args.voxelwise_interaction,
                add_intercept=args.add_intercept,
                drop_nans=args.drop_nans,
                drop_rows=args.drop_row,
                one_hot=args.one_hot,
                mask_path=args.mask_path,
                exchangeability_col=args.exchangeability_col,
                weights_col=args.weights_col,
                data_transform_method=args.data_transform,
                contrast_file=args.contrast_file,
                verbose=args.verbose,
            )

        run_prediction(
            json_path=json_path,
            out_dir=args.out_dir,
            mask_path=args.mask_path,
            params_dir=args.params_dir,
            regression_type=args.regression_type,
        )
        return 0

    raise SystemExit(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
