import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, kruskal
import patsy
import os 

class ModelFreeInteractionPlot:
    '''
    This class calculates the mean and standard error of the mean (SEM) for specified groupings in the data 
    and plots the mean +/- SEM across several categorical values. It generates an interaction plot without 
    using a statistical model.
    '''
    
    @staticmethod
    def calculate_means_and_sem(data_df, group1_column, group2_column, outcome_column):
        """
        Calculate means and standard errors of the mean (SEM) for specified groupings.

        Args:
        data_df (pd.DataFrame): DataFrame containing the data.
        group1_column (str): Name of the first grouping column (categorical).
        group2_column (str): Name of the second grouping column (categorical).
        outcome_column (str): Name of the outcome column (continuous).

        Returns:
        pd.DataFrame: Summary DataFrame with columns for the group levels, mean, standard deviation, count, and SEM.

        The function groups the data by `group1_column` and `group2_column`, calculates the mean, standard deviation, 
        and count of the `outcome_column` for each group, and then computes the SEM for each group.
        """
        # Group data and calculate mean, standard deviation, and count of samples
        summary_df = data_df.groupby([group2_column, group1_column])[outcome_column].agg(['mean', 'std', 'count']).reset_index()
        summary_df.columns = [group2_column, group1_column, 'Average', 'STD', 'Count']

        # Calculate SEM only for groups with more than one data point
        summary_df['SEM'] = summary_df.apply(lambda row: row['STD'] / np.sqrt(row['Count']) if row['Count'] > 1 else np.nan, axis=1)

        # Drop any rows with NaN values in 'Average'
        summary_df.dropna(subset=['Average'], inplace=True)

        return summary_df

    @staticmethod
    def plot_interaction_error_bar(summary_df, group1_column, group2_column, out_dir=None):
        """
        Plot an interaction error bar plot showing the relationship between two factors with SEM.

        Args:
        summary_df (pd.DataFrame): Summary DataFrame containing the calculated means and SEM.
        group1_column (str): Name of the first grouping column (categorical).
        group2_column (str): Name of the second grouping column (categorical).
        out_dir (str, optional): Directory to save the plot. If None, the plot is not saved.

        The function generates an interaction plot with error bars representing the SEM. 
        The x-axis represents the levels of `group1_column`, the y-axis represents the mean outcome, 
        and different lines represent the levels of `group2_column`.
        """
        sns.set(style="whitegrid")
        plt.figure(figsize=(8, 6))

        # Unique categories in the second grouping variable
        group2_categories = summary_df[group2_column].unique()

        # Colors for each line
        colors = sns.color_palette("tab10", len(group2_categories))

        for idx, category in enumerate(group2_categories):
            # Subset for each category of group 2
            subset = summary_df[summary_df[group2_column] == category]
            # Plot points. If SEM is NaN, error bars are not drawn.
            plt.errorbar(subset[group1_column], subset['Average'], capsize=6, yerr=subset['SEM'], fmt='-o', label=category, color=colors[idx])

        plt.xlabel(group1_column)
        plt.ylabel('Average Outcome')
        plt.title('Interaction Plot of Average Outcomes with SEM')
        plt.legend(title=group2_column, loc='upper right')
        plt.xticks(rotation=0)  # Rotate x-axis labels for better readability

        # Save the figure if a directory is provided
        if out_dir:
            plt.savefig(f"{out_dir}/interaction_means_error_bars.png", bbox_inches='tight')
            plt.savefig(f"{out_dir}/interaction_means_error_bars.svg", bbox_inches='tight')
            print(f'Saved to {out_dir}/interaction_means_error_bars.svg')
        
        # Show the plot
        plt.show()
        
    @staticmethod
    def diagnose_data(data_df, group1_column, group2_column, outcome_column):
        group2_categories = data_df[group2_column].unique()

        for category in group2_categories:
            subset = data_df[data_df[group2_column] == category]
            print(f"Category: {category}")
            groups = [subset[subset[group1_column] == group][outcome_column].values for group in subset[group1_column].unique()]
            
            for i, group in enumerate(subset[group1_column].unique()):
                print(f"  Group {group} size: {len(groups[i])}, values: {groups[i]}")
                
            if any(len(group) <= 1 for group in groups):
                print("  Issue: One or more groups have insufficient data points.")
            elif all(np.all(group == groups[0][0]) for group in groups):
                print("  Issue: All values in the groups are the same.")
                
    @staticmethod
    def perform_kruskal_wallis_test(data_df, group1_column, group2_column, outcome_column):
        """
        Performs the Kruskal-Wallis test to compare the distribution of outcome values across
        multiple groups defined by group1_column within each level of group2_column.

        Parameters:
        - data_df: DataFrame containing the data.
        - group1_column: String, the name of the first grouping column.
        - group2_column: String, the name of the second grouping column.
        - outcome_column: String, the name of the outcome column.

        Returns:
        - results: Dictionary containing the p-values of the Kruskal-Wallis test for each level of group2_column.
        """
        results = {}
        # Get unique categories in the second grouping variable
        group2_categories = data_df[group2_column].unique()

        for category in group2_categories:
            # Filter data for each category of group2
            subset = data_df[data_df[group2_column] == category]

            # Collect the data from each group in group1_column, removing NaN values
            groups = [subset[subset[group1_column] == group][outcome_column].dropna().values for group in subset[group1_column].unique()]

            # Check if all groups have data after removing NaNs
            if all(len(group) > 0 for group in groups):
                # Perform Kruskal-Wallis test
                statistic, p_value = kruskal(*groups)
                results[category] = p_value
            else:
                results[category] = 'Not enough data to perform test'

        return results

    @staticmethod
    def perform_contrast(data_df, contrast_column, outcome_column):
        """
        Compare data across all cohorts between two levels using the Mann-Whitney U test.

        Parameters:
        - data_df (pd.DataFrame): DataFrame containing the data.
        - contrast_column (str): The column that specifies the subgroup levels, which must contain exactly two unique values.
        - outcome_column (str): The column containing the data to compare (continuous).

        Returns:
        - result (dict): A dictionary containing the U statistic and p-value of the test, or an error message if there is insufficient data.
        """
        # Ensure there are exactly two unique values in the contrast column
        unique_values = data_df[contrast_column].unique()
        if len(unique_values) != 2:
            return {'Error': 'Contrast column must have exactly two unique values'}

        # Assign the unique values to low and high
        low_value, high_value = unique_values

        # Filter data into two groups based on the levels
        low_data = data_df[data_df[contrast_column] == low_value][outcome_column].dropna()
        high_data = data_df[data_df[contrast_column] == high_value][outcome_column].dropna()

        # Perform the Mann-Whitney U test
        if len(low_data) > 0 and len(high_data) > 0:
            statistic, p_value = mannwhitneyu(low_data, high_data, alternative='two-sided')
            return {'U statistic': statistic, 'p-value': p_value}
        else:
            return {'Error': 'Insufficient data'}

class QuickANOVAPlot:
    """
    QuickANOVAPlot is a utility class for generating interaction plots from a fitted model and design matrix.

    Parameters:
    -----------
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        The fitted model object from statsmodels.
    design_matrix : pandas.DataFrame or numpy.ndarray
        The design matrix used for the model. This should include all predictor variables used in the model.
    out_dir : str, optional
        Directory to save the plot. If None, the plot will not be saved.

    Methods:
    --------
    make_predictions():
        Generates predictions from the model using the design matrix.
    
    plot_predictions(x_var, hue_var, cohort_var, y_label='Predictions', title='Interaction Plot of Predictions'):
        Plots the interaction effects using the specified variables.

    Parameters for plot_predictions:
    --------------------------------
    x_var : str
        The variable to be plotted on the x-axis.
    hue_var : str
        The variable to differentiate lines within the plot.
    cohort_var : str, optional
        The variable to differentiate cohorts with different colors. If None, cohorts are not differentiated.
    y_label : str, optional
        Label for the y-axis. Default is 'Predictions'.
    title : str, optional
        Title for the plot. Default is 'Interaction Plot of Predictions'.

    Notes:
    ------
    - The class expects the design matrix to contain both categorical and continuous variables.
    - It can handle models with more than two or three variables, but the plot will only display the interaction between the specified variables.
    - Categorical variables should be properly encoded (e.g., using pandas' categorical dtype or similar).
    - The class is flexible and can generalize to different models, provided the necessary variables are specified for plotting.
    """

    def __init__(self, model, design_matrix, out_dir=None):
        self.model = model
        self.design_matrix = design_matrix
        self.predictions = None
        self.out_dir = out_dir

    def make_predictions(self):
        """Generates predictions from the model using the design matrix."""
        self.predictions = self.model.predict(self.design_matrix)

    def plot_predictions(self, x_var, hue_var, cohort_var=None, y='predictions', error_bar='ci'):
        """
        Plots the interaction effects using the specified variables.

        Parameters:
        -----------
        x_var : str
            The variable to be plotted on the x-axis.
        hue_var : str
            The variable to differentiate lines within the plot.
        cohort_var : str, optional
            The variable to differentiate cohorts with different colors. If None, cohorts are not differentiated.
        y : str, optional
            The variable to plot on the y-axis. If 'predictions' (default), will plot the model's predictions (Estimated Marginal Means).
        error_bar : str, optional
            The size of the confidence interval to draw when aggregating with an estimator. Default is 'ci'.
            options: 'ci' | 'sd' | 'se' | None
        """
        if self.predictions is None:
            raise ValueError("Predictions have not been made yet. Call make_predictions() first.")

        # Convert the design matrix to a DataFrame if it isn't already
        if not isinstance(self.design_matrix, pd.DataFrame):
            self.design_matrix = pd.DataFrame(self.design_matrix)

        # Add predictions to the design matrix DataFrame
        self.design_matrix['predictions'] = self.predictions

        # Ensure variables are in the design matrix
        if x_var not in self.design_matrix.columns or hue_var not in self.design_matrix.columns or (cohort_var and cohort_var not in self.design_matrix.columns):
            raise ValueError("Variables for x, hue, or cohort are not in the design matrix.")
        # Plot the interaction plot with traces for cohorts
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 8))
        y_label=y.capitalize()
        title=f'Plot of {y_label}'

        # Iterate through each cohort and plot
        if cohort_var is not None:
            cohort_colors = sns.color_palette("tab10", len(self.design_matrix[cohort_var].unique()))
            for i, cohort in enumerate(self.design_matrix[cohort_var].unique()):
                cohort_data = self.design_matrix[self.design_matrix[cohort_var] == cohort]

                # Plot with solid and dashed lines
                for j, hue in enumerate(cohort_data[hue_var].unique()):
                    hue_data = cohort_data[cohort_data[hue_var] == hue]
                    sns.lineplot(
                        x=x_var, 
                        y=y, 
                        data=hue_data,
                        color=cohort_colors[i],
                        linestyle='--' if j % 2 == 0 else '-',
                        marker='o',
                        label=f'{cohort} ({hue})',
                        errorbar=error_bar
                        )
        else:
            # Plot without cohort differentiation
            sns.lineplot(
                x=x_var, 
                y=y, 
                hue=hue_var, 
                data=self.design_matrix,
                palette='tab10',
                style=hue_var,
                markers=True,
                dashes=[(10, 10), ''],  # Dashed line for first level, solid line for second level
                errorbar=error_bar
            )

        plt.xlabel(x_var)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend(title=cohort_var, loc='upper right')
        sns.despine()
        plt.grid(False)
        if self.out_dir is not None:
            plt.savefig(os.path.join(self.out_dir, 'anova_plot.svg'))
        plt.show()
        
    def plot_bar(self, x_var, hue_var, cohort_var=None, y='predictions', y_label='Predictions', title='Bar Plot of Predictions', error_bar='ci'):
        """
        Creates a bar plot using the specified variables.

        Parameters:
        -----------
        x_var : str
            The variable to be plotted on the x-axis.
        hue_var : str
            The variable to differentiate bars within the plot.
        cohort_var : str, optional
            The variable to differentiate cohorts with different colors. If None, cohorts are not differentiated.
        y : str, optional
            The variable to plot on the y-axis. If 'predictions' (default), will plot the model's predictions.
        y_label : str, optional
            Label for the y-axis. Default is 'Predictions'.
        title : str, optional
            Title for the plot. Default is 'Bar Plot of Predictions'.
        error_bar : str, optional
            The size of the confidence interval to draw when aggregating with an estimator. Default is 'ci'.
            options: 'ci' | 'sd' | 'se' | None
        """
        if self.predictions is None:
            raise ValueError("Predictions have not been made yet. Call make_predictions() first.")

        # Convert the design matrix to a DataFrame if it isn't already
        if not isinstance(self.design_matrix, pd.DataFrame):
            self.design_matrix = pd.DataFrame(self.design_matrix)

        # Add predictions to the design matrix DataFrame
        self.design_matrix['predictions'] = self.predictions

        # Ensure variables are in the design matrix
        if x_var not in self.design_matrix.columns or hue_var not in self.design_matrix.columns or (cohort_var and cohort_var not in self.design_matrix.columns):
            raise ValueError("Variables for x, hue, or cohort are not in the design matrix.")

        # Plot the bar plot
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 8))
        y_label = y_label.capitalize()
        title = title

        # Bar plot with hue and cohort differentiation
        if cohort_var is not None:
            sns.barplot(
                x=x_var, 
                y=y, 
                hue=hue_var, 
                data=self.design_matrix,
                errorbar=error_bar,
                palette='tab10'
            )
            plt.legend(title=cohort_var, loc='upper right')
        else:
            sns.barplot(
                x=x_var, 
                y=y, 
                hue=hue_var, 
                data=self.design_matrix,
                errorbar=error_bar,
                palette='tab10'
            )

        plt.xlabel(x_var)
        plt.ylabel(y_label)
        plt.title(title)
        sns.despine()
        plt.grid(False)
        if self.out_dir is not None:
            plt.savefig(os.path.join(self.out_dir, 'bar_plot.svg'))
        plt.show()


class QuickRegressionMarginalPlot:
    """
    QuickRegressionMarginalPlot is a utility class for generating estimated marginal mean (EMM)
    plots for standard regressions (e.g., OLS) fitted with statsmodels.

    It varies one predictor (x_var) across its range and holds all other predictors constant
    at their means (or mean +/- N * std if stdevs are provided). Optionally, it stratifies
    predictions by hue/cohort variables and overlays the distribution of observed values.

    Parameters:
    -----------
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        The fitted model object from statsmodels.
    data_df : pandas.DataFrame
        Original dataframe used to fit the model.
    out_dir : str, optional
        Directory to save the plot. If None, the plot will not be saved.
    formula : str, optional
        The regression formula. If not provided, the class will attempt to infer it from the model.
    """

    def __init__(self, model, data_df, out_dir=None, formula=None):
        self.model = model
        self.data_df = data_df
        self.out_dir = out_dir
        self.formula = formula or getattr(getattr(model, 'model', None), 'formula', None)
        if self.formula is None:
            raise ValueError("Formula could not be inferred from the model. Please provide formula explicitly.")

        self.design_info = getattr(getattr(model, 'model', None), 'data', None)
        self.design_info = getattr(self.design_info, 'design_info', None)

    def _infer_y_var(self, y_var=None):
        if y_var is not None:
            return y_var
        if self.formula and '~' in self.formula:
            return self.formula.split('~')[0].strip()
        return getattr(getattr(self.model, 'model', None), 'endog_names', None)

    def _get_levels(self, series):
        series = series.dropna()
        if series.empty:
            return []
        if pd.api.types.is_categorical_dtype(series):
            return list(series.cat.categories)
        if pd.api.types.is_numeric_dtype(series):
            return sorted(series.unique())
        return list(series.unique())

    def _get_constant_value(self, var, stdev, fixed_values=None):
        if fixed_values and var in fixed_values:
            return fixed_values[var]
        series = self.data_df[var].dropna()
        if series.empty:
            return np.nan
        if pd.api.types.is_numeric_dtype(series):
            mean = series.mean()
            std = series.std()
            if np.isfinite(std):
                return mean + (stdev * std)
            return mean
        # Fallback for categorical/object variables
        mode = series.mode()
        if not mode.empty:
            return mode.iloc[0]
        return series.iloc[0]

    def _build_design_matrix(self, pred_df):
        if self.design_info is not None:
            return patsy.build_design_matrices([self.design_info], pred_df, return_type='dataframe')[0]
        rhs = self.formula.split('~', 1)[1].strip()
        return patsy.dmatrix(rhs, pred_df, return_type='dataframe')

    def build_predictions(self,
                          x_var,
                          y_var=None,
                          hue_var=None,
                          cohort_var=None,
                          stdevs=None,
                          n_points=100,
                          x_range=None,
                          x_values=None,
                          fixed_values=None,
                          include_ci=False):
        """
        Build a prediction DataFrame for EMM-style plotting.

        Parameters:
        -----------
        x_var : str
            The variable to vary across its range.
        y_var : str, optional
            The dependent variable name. If None, inferred from formula.
        hue_var : str, optional
            Variable to differentiate lines.
        cohort_var : str, optional
            Variable to differentiate cohorts (colors).
        stdevs : list of float, optional
            List of standard deviation offsets to apply to OTHER continuous variables.
            Default is [0] (mean).
        n_points : int, optional
            Number of points to use for x_var when continuous. Default 100.
        x_range : tuple, optional
            (min, max) to override x_var range.
        x_values : array-like, optional
            Explicit x values to use. If provided, overrides n_points/x_range.
        fixed_values : dict, optional
            Explicit values for any covariates to hold constant.
        include_ci : bool, optional
            If True, include 95% CI columns (ci_low/ci_high) if supported by the model.
        """
        y_var = self._infer_y_var(y_var)
        if y_var is None:
            raise ValueError("Could not infer y_var. Please specify y_var explicitly.")

        if x_var not in self.data_df.columns:
            raise ValueError(f"x_var '{x_var}' is not in data_df")
        if y_var not in self.data_df.columns:
            raise ValueError(f"y_var '{y_var}' is not in data_df")

        stdevs = stdevs or [0]

        x_series = self.data_df[x_var].dropna()
        if x_values is None:
            if pd.api.types.is_numeric_dtype(x_series) and x_series.nunique() > 12:
                if x_range is None:
                    x_range = (x_series.min(), x_series.max())
                x_values = np.linspace(x_range[0], x_range[1], n_points)
            else:
                x_values = self._get_levels(x_series)

        hue_levels = self._get_levels(self.data_df[hue_var]) if hue_var else [None]
        cohort_levels = self._get_levels(self.data_df[cohort_var]) if cohort_var else [None]

        other_vars = [
            col for col in self.data_df.columns
            if col not in {x_var, y_var, hue_var, cohort_var}
        ]

        rows = []
        for stdev in stdevs:
            for cohort in cohort_levels:
                for hue in hue_levels:
                    pred_df = pd.DataFrame({x_var: x_values})
                    if hue_var is not None:
                        pred_df[hue_var] = hue
                    if cohort_var is not None:
                        pred_df[cohort_var] = cohort

                    for var in other_vars:
                        pred_df[var] = self._get_constant_value(var, stdev, fixed_values=fixed_values)

                    pred_df['_stdev'] = stdev
                    if hue_var is not None:
                        pred_df['_hue'] = hue
                    if cohort_var is not None:
                        pred_df['_cohort'] = cohort
                    rows.append(pred_df)

        pred_df = pd.concat(rows, ignore_index=True)

        design_matrix = self._build_design_matrix(pred_df)
        if include_ci:
            try:
                prediction = self.model.get_prediction(design_matrix)
                pred_df['predictions'] = prediction.predicted_mean
                ci = prediction.conf_int()
                pred_df['ci_low'] = ci[:, 0]
                pred_df['ci_high'] = ci[:, 1]
            except Exception as e:
                print(f"get_prediction failed ({e}); falling back to predict().")
                pred_df['predictions'] = self.model.predict(design_matrix)
        else:
            pred_df['predictions'] = self.model.predict(design_matrix)

        return pred_df

    def plot_predictions(self,
                         x_var,
                         y_var=None,
                         hue_var=None,
                         cohort_var=None,
                         stdevs=None,
                         n_points=100,
                         x_range=None,
                         x_values=None,
                         fixed_values=None,
                         include_ci=False,
                         show_actual=True,
                         actual_alpha=0.35,
                         actual_size=25,
                         palette='tab10',
                         save_name='regression_marginal_plot'):
        """
        Plot EMM-style predictions and overlay the distribution of observed values.
        """
        pred_df = self.build_predictions(
            x_var=x_var,
            y_var=y_var,
            hue_var=hue_var,
            cohort_var=cohort_var,
            stdevs=stdevs,
            n_points=n_points,
            x_range=x_range,
            x_values=x_values,
            fixed_values=fixed_values,
            include_ci=include_ci
        )

        y_var = self._infer_y_var(y_var)

        sns.set(style='whitegrid')
        plt.figure(figsize=(10, 6))

        # Build palettes
        stdevs = stdevs or [0]
        hue_levels = self._get_levels(self.data_df[hue_var]) if hue_var else [None]
        cohort_levels = self._get_levels(self.data_df[cohort_var]) if cohort_var else [None]

        if cohort_var is not None:
            color_levels = cohort_levels
        elif hue_var is not None:
            color_levels = hue_levels
        elif len(stdevs) > 1:
            color_levels = stdevs
        else:
            color_levels = [None]

        palette_colors = sns.color_palette(palette, len(color_levels))
        color_map = {lvl: palette_colors[i] for i, lvl in enumerate(color_levels)}

        # Plot actual data distribution
        if show_actual:
            scatter_kwargs = dict(
                data=self.data_df,
                x=x_var,
                y=y_var,
                alpha=actual_alpha,
                s=actual_size,
                edgecolor='none'
            )
            if cohort_var is not None:
                sns.scatterplot(
                    hue=cohort_var,
                    style=hue_var if hue_var is not None else None,
                    palette=color_map,
                    legend=False,
                    **scatter_kwargs
                )
            elif hue_var is not None:
                sns.scatterplot(hue=hue_var, palette=color_map, legend=False, **scatter_kwargs)
            else:
                sns.scatterplot(legend=False, **scatter_kwargs)

        # Line style and marker maps
        stdev_styles = ['-', '--', ':', '-.']
        stdev_style_map = {s: stdev_styles[i % len(stdev_styles)] for i, s in enumerate(stdevs)}
        markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
        hue_marker_map = {lvl: markers[i % len(markers)] for i, lvl in enumerate(hue_levels) if lvl is not None}

        # Plot predictions
        for stdev in stdevs:
            for cohort in cohort_levels:
                for hue in hue_levels:
                    subset = pred_df[pred_df['_stdev'] == stdev]
                    if cohort_var is not None:
                        subset = subset[subset['_cohort'] == cohort]
                    if hue_var is not None:
                        subset = subset[subset['_hue'] == hue]

                    if subset.empty:
                        continue

                    label_parts = []
                    if cohort_var is not None:
                        label_parts.append(f"{cohort_var}={cohort}")
                    if hue_var is not None:
                        label_parts.append(f"{hue_var}={hue}")
                    if len(stdevs) > 1:
                        label_parts.append(f"stdev={stdev}")
                    label = ', '.join(label_parts) if label_parts else 'predictions'

                    if cohort_var is not None:
                        line_color = color_map[cohort]
                    elif hue_var is not None:
                        line_color = color_map[hue]
                    elif len(stdevs) > 1:
                        line_color = color_map[stdev]
                    else:
                        line_color = 'C0'

                    if hue_var is not None and cohort_var is not None and len(stdevs) == 1:
                        line_style = stdev_styles[hue_levels.index(hue) % len(stdev_styles)]
                    else:
                        line_style = stdev_style_map.get(stdev, '-')

                    marker = None
                    if hue_var is not None and cohort_var is not None:
                        marker = hue_marker_map.get(hue, 'o')

                    plt.plot(
                        subset[x_var],
                        subset['predictions'],
                        label=label,
                        color=line_color,
                        linestyle=line_style,
                        marker=marker
                    )

                    if include_ci and 'ci_low' in subset.columns and 'ci_high' in subset.columns:
                        plt.fill_between(
                            subset[x_var],
                            subset['ci_low'],
                            subset['ci_high'],
                            color=line_color,
                            alpha=0.15
                        )

        plt.xlabel(x_var)
        plt.ylabel(y_var)
        plt.title('Regression EMM Plot')
        plt.legend(loc='upper right')
        sns.despine()
        plt.grid(False)

        if self.out_dir is not None:
            plt.savefig(os.path.join(self.out_dir, f'{save_name}.svg'))
            plt.savefig(os.path.join(self.out_dir, f'{save_name}.png'))
        plt.show()
