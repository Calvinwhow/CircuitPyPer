# symbolic_regression.py
# Install:  pip install pysr==0.15.1 statsmodels scikit-learn matplotlib

from __future__ import annotations
import tempfile
import shutil
"""Symbolic-regression model helpers."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import KFold
from calvin_utils.statistical_utils.response_surface_utils.regression_response_surface import GLMPlot


def _pysr_regressor_class():
    try:
        from pysr import PySRRegressor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Symbolic regression requires the optional 'ml' dependencies. "
            "Install calvin_utils[ml]."
        ) from exc
    return PySRRegressor


class LeanSymbolicRegressor:
    """
    Noise-robust symbolic regression for tiny clinical datasets (≈30–100 rows).

    Core workflow
    -------------
    >>> sr = LeanSymbolicRegressor(max_nodes=12, k_folds=5, random_state=42)
    >>> sr.fit(X, y)         # numpy arrays or pandas objects
    >>> sr.summary(X, y)     # prints CV R², OLS refit metrics & coefficients
    >>> sr.plot_actual_vs_pred(X, y); plt.show()
    >>> sr.plot_residuals(X, y);      plt.show()
    NOTE: cross validation doesnt make sense as expression changes slightly with each fold
    """

    # ---------------------------- construction ------------------------- #
    def __init__(self, *, k_folds: int = 5, max_nodes: int = 12, parsimony: float = 1.0, niterations: int = 400, population_size: int = 30, random_state: int = 42) -> None:
        self.k_folds = k_folds
        self.max_nodes = max_nodes
        self.parsimony = parsimony
        self.niterations = niterations
        self.population_size = population_size
        self.random_state = random_state
        self._scratch_dir = tempfile.mkdtemp(prefix="pysr_scratch_")
        self.expr_: str | None = None
        self.cv_stats_: dict[str, float] | None = None
        self._model = None
        self._ols_res: sm.regression.linear_model.RegressionResults | None = None

    # ---------------------------- main API ----------------------------- #
    def fit_one_shot(self, X, y, **pysr_kwargs):
        """
        Fit PySRRegressor once on the full dataset (no cross-validation).

        Parameters
        ----------
        X, y : array-like
            Full training data.
        **pysr_kwargs
            Extra arguments passed straight to PySRRegressor(...).
        """
        self._model = _pysr_regressor_class()(**pysr_kwargs)
        self._model.fit(X, y)
        self.expr_     = self._model.sympy()
        preds          = self.predict(X)
        self.train_r2_ = 1 - ((y - preds) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        return self
    
    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> None:
        """Cross-validate to pick the simplest, highest-CV-R² expression, then
        refit that expression on the full dataset for interpretability stats."""
        X_arr, y_arr = self._to_arrays(X, y)
        scores: dict[str, list[float]] = {}

        for mdl in self._cv_models(X_arr, y_arr):
            expr = str(mdl.get_best())
            scores.setdefault(expr, []).append(self._score(mdl, X_arr, y_arr))

        # choose expression with highest median CV R²
        self.expr_, self.cv_stats_ = self._select_expr(scores)

        # final fit on all data (locked random_state keeps determinism)
        self._model = self._new_pysr()
        self._model.set_params()                 # reset
        self._model.fit(X_arr, y_arr.ravel())                    # may retrain
        # ensure expression is identical (rare stochastic drift)
        if str(self._model.get_best()) != self.expr_:
            self._model = self._model  # still fine – expression re-selected

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        self._check_fitted()
        return self._model.predict(self._to_arrays(X))

    def ols(self, X, y):
        self._check_fitted()
        X_arr, y_arr = self._to_arrays(X, y)
        preds = self.predict(X_arr)
        X_design = sm.add_constant(preds)
        self._ols_res = sm.OLS(y_arr, X_design).fit()
        return self._ols_res.params, self._ols_res.bse
    
    ### Methods to choose a specific equation for prediction ###
    def set_expression(self, index: int) -> None:
        """
        Select a specific equation for subsequent predictions.

        Parameters
        ----------
        index : int
            Index of the chosen equation.
        """
        self._check_fitted()

        if not hasattr(self._model, 'equations_'):
            raise RuntimeError("PySR model is not fitted yet.")

        if index < 0 or index >= len(self._model.equations_):
            raise IndexError("Equation index out of range.")

        self.selected_index_ = index
        self.expr_ = self._model.equations_.iloc[index]["sympy_format"]
        print(f"Selected equation (index {index}): {self.expr_}")

    def predict_chosen(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Predict using the explicitly chosen PySR equation.

        Parameters
        ----------
        X : array-like
            Input features.

        Returns
        -------
        np.ndarray
            Predictions based on the chosen equation.
        """
        self._check_fitted()

        if not hasattr(self, 'selected_index_'):
            raise RuntimeError("No equation selected yet. Call `.set_expression(index)` first.")

        X_arr = self._to_arrays(X)
        return self._model.predict(X_arr, index=self.selected_index_)
    
    # ---------------------------- summary api -------------------------- #
    def summary(self, X, y) -> None:
        """Key statistics: expression, CV R², OLS coefficients, R², AIC."""
        self._check_fitted()
        print(f"\nBest expression:\n  {self.expr_}")
        if self._model is not None and hasattr(self._model, "score"):
            try:
                score = self._model.score(*self._to_arrays(X, y))
                print(f"PySR model R² on provided data: {score:.3f}")
            except Exception as e:
                print(f"Could not compute PySR model R²: {e}")

        if self.cv_stats_ is not None:
            print(
            f"\nCV R²  (median ± IQR): "
            f"{self.cv_stats_['median']:.3f} ± {self.cv_stats_['iqr']:.3f}"
            )
        else:
            print("\nCV R²  (median ± IQR): Not available")

        try:
            beta, se = self.ols(X, y)
            names = ["Intercept"] + [f"θ{i}" for i in range(1, len(beta))]
            print("\nCoefficients (OLS refit):")
            for nm, b, s in zip(names, beta, se):
                print(f"{nm:<9}: {b:>9.4f} ± {s:.4f}")
            if self._ols_res is not None:
                print(
                    f"\nR² (OLS)   : {self._ols_res.rsquared:.3f}"
                    f"\nAdj-R²      : {self._ols_res.rsquared_adj:.3f}"
                    f"\nAIC         : {self._ols_res.aic:.1f}"
                )
            else:
                print("\nR² (OLS), Adj-R², AIC: Not available")
        except Exception as e:
            print(f"\nOLS refit failed: {e}")
        
    def plot_actual_vs_pred(self, X, y, ax: plt.Axes | None = None) -> plt.Axes:
        self._check_fitted()
        X_arr, y_arr = self._to_arrays(X, y)
        y_hat = self.predict_chosen(X_arr)

        ax = ax or plt.subplots()[1]
        ax.scatter(y_arr, y_hat, alpha=0.8)
        lims = [min(y_arr.min(), y_hat.min()), max(y_arr.max(), y_hat.max())]
        ax.plot(lims, lims, linewidth=1)
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        return ax

    def plot_residuals(self, X, y, ax: plt.Axes | None = None) -> plt.Axes:
        self._check_fitted()
        X_arr, y_arr = self._to_arrays(X, y)
        y_hat = self.predict_chosen(X_arr)
        res = y_arr.ravel() - y_hat

        ax = ax or plt.subplots()[1]
        ax.scatter(y_hat, res, alpha=0.8)
        ax.axhline(0, linewidth=1)
        ax.set_xlabel("Fitted")
        ax.set_ylabel("Residual")
        ax.set_title("Residuals vs Fitted")
        return ax
    
    def plot_with_glmplot(self, data_df: pd.DataFrame, formula: str, plot_residuals: bool = True):
        gp = GLMPlot(model_results=self, data_df=data_df, formula=formula)
        gp.run(plot_residuals)

    # --------------------------- internals ----------------------------- #
    def _cv_models(self, X, y):
        kf = KFold(self.k_folds, shuffle=True, random_state=self.random_state)
        for tr_idx, _ in kf.split(X):
            mdl = self._new_pysr()
            mdl.fit(X[tr_idx], y[tr_idx].ravel())
            yield mdl

    def _new_pysr(self):
        return _pysr_regressor_class()(
            model_selection="best",
            niterations=self.niterations,
            population_size=self.population_size,
            maxsize=self.max_nodes,
            parsimony=self.parsimony,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["exp", "log"],
            turbo=True,                   # use turbo for stability
            random_state=self.random_state,
            deterministic=True,
            progress=False,
            procs=0,                      # safer for Jupyter
            multithreading=False,
            timeout_in_seconds=120,        # avoid infinite hangs
            tempdir=self._scratch_dir,
            output_directory=self._scratch_dir,
            delete_tempfiles=False,   # so that we can delete it manually later
            verbosity=0,              # silence console chatter
        )

    def cleanup_scratch(self):
        """Call when you’re completely done and want to blow away all pysr files."""
        shutil.rmtree('outputs', ignore_errors=True)

    # -- helpers -------------------------------------------------------- #
    @staticmethod
    def _to_arrays(X, y: None | pd.Series | np.ndarray = None):
        """Convert pandas/NumPy input to NumPy arrays, preserving order."""
        X_arr = X.values if isinstance(X, (pd.DataFrame, pd.Series)) else np.asarray(X)
        if y is None:
            return X_arr
        y_arr = y.values if isinstance(y, (pd.DataFrame, pd.Series)) else np.asarray(y)
        return X_arr, y_arr

    @staticmethod
    def _score(model, X, y):
        y_hat = model.predict(X)
        return 1.0 - np.sum((y.ravel() - y_hat) ** 2) / np.var(y)

    @staticmethod
    def _select_expr(score_dict):
        expr = max(score_dict, key=lambda e: np.median(score_dict[e]))
        q1, med, q3 = np.percentile(score_dict[expr], [25, 50, 75])
        return expr, {"median": med, "iqr": q3 - q1, "all": score_dict[expr]}

    def _check_fitted(self):
        if self._model is None:
            raise RuntimeError("Call .fit() before this operation.")
