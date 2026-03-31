import pandas as pd
import numpy as np

from calvin_utils.statistical_utils.classification_statistics import ComprehensiveMulticlassROC
from calvin_utils.statistical_utils.statistical_measurements import model_diagnostics
from calvin_utils.statistical_utils.regression_evaluation import RegressionPerformanceEvaluator


class BrainModelEvaluation:
    """
    Evaluate brain-model predictions against observed outcomes.

    Modes
    -----
    classification
        Uses ComprehensiveMulticlassROC on prediction columns (scores/probabilities)
        and observation columns (one-hot or ordinal labels).

    regression
        If fitted_model is provided, runs model_diagnostics on that model.
        If fitted_model is None, computes scatter + RMSE + F-stat + R^2 using
        RegressionPerformanceEvaluator. This mode requires exactly one prediction
        column, one observation column, and exog_df (design matrix).

    Validation/Coercion
    -------------------
    - Validates requested columns exist.
    - Coerces predictions/observations to numeric.
    - Aligns indices across frames and drops rows with NaNs.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        task: str,
        prediction_cols: list[str] | None = None,
        observation_cols: list[str] | None = None,
        fitted_model=None,
        exog_df: pd.DataFrame | None = None,
        normalization: str | None = "true",
        thresholds: dict | None = None,
        out_dir: str | None = None,
        show: bool = True,
    ):
        """
        Parameters
        ----------
        df : pd.DataFrame
            Source dataframe containing predictions and observations.
        task : str
            'classification' or 'regression'.
        prediction_cols : list[str] | None
            Column names for prediction scores/probabilities.
        observation_cols : list[str] | None
            Column names for observed outcomes (one-hot or ordinal).
        fitted_model : statsmodels result | None
            If provided in regression mode, model_diagnostics is used.
        exog_df : pd.DataFrame | None
            Design matrix for regression metrics when fitted_model is None.
        normalization : str | None
            Confusion-matrix normalization for classification.
        thresholds : dict | None
            Optional thresholds for classification.
        out_dir : str | None
            Output directory for plots (if show=True).
        show : bool
            If False, suppresses plotting and only prepares/returns evaluators.
        """
        self.df = df
        self.task = task
        self.prediction_cols = prediction_cols
        self.observation_cols = observation_cols
        self.fitted_model = fitted_model
        self.exog_df = exog_df
        self.normalization = normalization
        self.thresholds = thresholds
        self.out_dir = out_dir
        self.show = show
        
        
    ### Validation and Coercion ###
    def _validate_columns(self, cols, name):
        missing = [c for c in cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing {name}: {missing}")

    def _coerce_numeric_df(self, df, name):
        out = df.copy()
        for col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        if out.isna().all().all():
            raise ValueError(f"{name} has no numeric data after coercion.")
        return out

    def _align_and_dropna(self, *dfs):
        idx = dfs[0].index
        for d in dfs[1:]:
            idx = idx.intersection(d.index)
        dfs = [d.loc[idx] for d in dfs]
        combined = pd.concat(dfs, axis=1)
        combined = combined.dropna()
        splits = []
        start = 0
        for d in dfs:
            end = start + d.shape[1]
            splits.append(combined.iloc[:, start:end])
            start = end
        return splits if len(splits) > 1 else splits[0]

    ### Business Logic ###
    def _run_classification(self):
        if not self.prediction_cols or not self.observation_cols:
            raise ValueError("prediction_cols and observation_cols are required for classification.")
        self._validate_columns(self.prediction_cols, "prediction_cols")
        self._validate_columns(self.observation_cols, "observation_cols")
        predictions_df = self._coerce_numeric_df(self.df[self.prediction_cols], "predictions_df")
        observation_df = self._coerce_numeric_df(self.df[self.observation_cols], "observation_df")
        predictions_df, observation_df = self._align_and_dropna(predictions_df, observation_df)
        if predictions_df.empty:
            raise ValueError("No valid rows after dropping NaNs for classification.")
        evaluator = ComprehensiveMulticlassROC(
            fitted_model=None,
            predictions_df=predictions_df,
            observation_df=observation_df,
            normalization=self.normalization,
            thresholds=self.thresholds,
            out_dir=self.out_dir,
        )
        if self.show:
            evaluator.run()
        else:
            evaluator.get_predictions()
            evaluator.get_observations()
        return evaluator

    def _run_regression(self):
        if self.fitted_model is not None:
            model_diagnostics(self.fitted_model)
            return self.fitted_model
        if not self.prediction_cols or not self.observation_cols:
            raise ValueError("prediction_cols and observation_cols are required for regression when fitted_model is None.")
        if len(self.prediction_cols) != 1 or len(self.observation_cols) != 1:
            raise ValueError("Regression mode requires exactly one prediction column and one observation column.")
        if self.exog_df is None:
            raise ValueError("Regression mode requires exog_df (design matrix) when fitted_model is None.")

        self._validate_columns(self.prediction_cols, "prediction_cols")
        self._validate_columns(self.observation_cols, "observation_cols")
        predictions_df = self._coerce_numeric_df(self.df[self.prediction_cols], "predictions_df")
        observation_df = self._coerce_numeric_df(self.df[self.observation_cols], "observation_df")
        predictions_df, observation_df, exog_df = self._align_and_dropna(
            predictions_df, observation_df, self.exog_df
        )
        if predictions_df.empty:
            raise ValueError("No valid rows after dropping NaNs for regression.")

        evaluator = RegressionPerformanceEvaluator(
            df=pd.concat([predictions_df, observation_df], axis=1),
            prediction_col=self.prediction_cols[0],
            outcome_col=self.observation_cols[0],
            design_matrix=exog_df.to_numpy(),
            out_dir=self.out_dir,
            dataset_name="Regression",
            show=self.show,
        )
        return evaluator.run()

    
    ### Public API ###    
    def run(self):
        if self.task == "classification":
            return self._run_classification()
        if self.task == "regression":
            return self._run_regression()
        raise ValueError("task must be 'classification' or 'regression'")

