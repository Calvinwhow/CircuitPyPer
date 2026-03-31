import numpy as np
from scipy.stats import f

from calvin_utils.statistical_utils.scatterplot import simple_scatter


def calculate_ssr(observations, predictions):
    y_hat = observations
    y_bar = np.mean(predictions)
    ssr = np.sum((y_hat - y_bar) ** 2)
    return ssr


def calculate_sse(observations, predictions):
    y = observations
    y_hat = predictions
    sse = np.sum((y - y_hat) ** 2)
    return sse


def calculate_ssto(observations):
    y = observations
    y_bar = np.mean(y)
    ssto = np.sum((y - y_bar) ** 2)
    return ssto


def calculate_msr(ssr, num_regressors):
    return ssr / (num_regressors - 1)


def calculate_mse(sse, num_regressors, num_observations):
    return sse / (num_observations - num_regressors)


def calculate_f_stat(msr, mse, num_regressors, num_observations):
    f_stat = msr / mse
    dfn = num_regressors - 1
    dfd = num_observations - num_regressors
    p_value = f.sf(f_stat, dfn, dfd)
    return f_stat, p_value


def run_goodness_of_fit(target_outcome_matrix, predictions, target_design_matrix):
    ssr = calculate_ssr(target_outcome_matrix, predictions)
    sse = calculate_sse(target_outcome_matrix, predictions)
    num_regressors = target_design_matrix.shape[1]
    num_observations = len(target_outcome_matrix)
    msr = calculate_msr(ssr, num_regressors)
    mse = calculate_mse(sse, num_regressors, num_observations)
    f_stat, p_value = calculate_f_stat(msr, mse, num_regressors, num_observations)
    return f_stat, p_value


def calculate_r_squared(observations, predictions):
    sse = np.sum((observations - predictions) ** 2)
    y_mean = np.mean(observations)
    ssto = np.sum((observations - y_mean) ** 2)
    r_squared = 1 - (sse / ssto)
    return r_squared


class RegressionPerformanceEvaluator:
    """
    Regression evaluation helper:
    - Scatter plot (simple_scatter)
    - RMSE
    - F-statistic goodness-of-fit
    - R-squared
    """

    def __init__(
        self,
        df,
        *,
        prediction_col: str,
        outcome_col: str,
        design_matrix,
        out_dir: str | None = None,
        dataset_name: str = "Regression",
        x_label: str | None = None,
        y_label: str | None = None,
        show: bool = True,
    ):
        self.df = df
        self.prediction_col = prediction_col
        self.outcome_col = outcome_col
        self.design_matrix = design_matrix
        self.out_dir = out_dir
        self.dataset_name = dataset_name
        self.x_label = x_label
        self.y_label = y_label
        self.show = show

    def run(self):
        predictions = self.df[self.prediction_col].to_numpy().astype(float)
        observations = self.df[self.outcome_col].to_numpy().astype(float)

        simple_scatter(
            self.df,
            x_col=self.prediction_col,
            y_col=self.outcome_col,
            dataset_name=self.dataset_name,
            out_dir=self.out_dir,
            x_label=self.x_label or self.prediction_col,
            y_label=self.y_label or self.outcome_col,
            show=self.show,
        )

        rmse = np.sqrt(np.mean((observations - predictions) ** 2))
        f_stat, p_value = run_goodness_of_fit(observations, predictions, self.design_matrix)
        r2 = calculate_r_squared(observations, predictions)

        print(f"RMSE: {rmse}")
        print(f"F-statistic: {f_stat}, p-value: {p_value}")
        print(f"R-squared: {r2}")

        return {
            "rmse": rmse,
            "f_stat": f_stat,
            "p_value": p_value,
            "r_squared": r2,
        }
