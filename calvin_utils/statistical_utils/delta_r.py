import pandas as pd
import numpy as np

class DataPreparation:
    """
    A class used to segregate data by categorical variables and optional cohorts.

    Attributes:
    ----------
    df : pd.DataFrame
        The input dataframe containing the data.
    dependent_variable : str
        The name of the dependent variable column.
    independent_variable : str
        The name of the independent variable column.
    categorical_variable : str
        The name of the categorical variable column used for splitting.
    cohort_variable : str, optional
        The name of the cohort variable column used for additional splitting. Default is None.

    Methods:
    -------
    segregate_data():
        Splits the dataframe into nested dictionaries by cohort and category.

    Example Usage:
    -------------
    dependent_variable = 'Z_Scored_Percent_Cognitive_Improvement'
    independent_variable = 'Subiculum_Connectivity_T_Redone'
    categorical_variable = 'Age_Group'
    cohort_variable = 'City'  # Optional, can be None if not needed

    data_prep = DataPreparation(data_df, dependent_variable, independent_variable, categorical_variable, cohort_variable)
    category_dataframes = data_prep.category_dataframes

    for cohort, categories in category_dataframes.items():
        print(f"Cohort: {cohort}")
        for category, cat_df in categories.items():
            print(f"  Category: {category}")
            print(cat_df.head())
    """

    def __init__(self, df, dependent_variable, independent_variable, categorical_variable, cohort_variable=None):
        self.dependent_variable = dependent_variable
        self.independent_variable = independent_variable
        self.categorical_variable = categorical_variable
        self.cohort_variable = cohort_variable
        self.df = self.prepare_dataframe(df)
        self.category_dataframes = self.segregate_data()
        self.category_dataframes = self.reverse_dictionary()
        
    def prepare_dataframe(self, df):
        if self.cohort_variable is None:
            df = df.loc[:, [self.dependent_variable, self.independent_variable, self.categorical_variable]]
        else:
            df = df.loc[:, [self.dependent_variable, self.independent_variable, self.categorical_variable, self.cohort_variable]]
        return df
    
    def segregate_data(self):
        """
        Splits the dataframe into nested dictionaries by cohort and category.

        Returns:
        -------
        dict
            A nested dictionary where data is first split by cohort (if provided) and then by category.
            Each entry corresponds to a subset of the data for a specific cohort and category.
        """
        # Initialize dictionary to hold dataframes for each category
        category_dataframes = {}
        unique_categories = self.df[self.categorical_variable].unique()
        # We do not need to split by cohort, as we can iterate over each cohort during plotting. 
        for category in unique_categories:
            category_dataframes[category] = self.df[self.df[self.categorical_variable] == category]
        category_dataframes['all_categories'] = self.df
        return category_dataframes
    
    def reverse_dictionary(self):
        reversed_dict = {k: self.category_dataframes[k] for k in reversed(self.category_dataframes)}
        return reversed_dict
    

import seaborn as sns
import matplotlib.pyplot as plt

class ScatterPlot:
    """
    A class to create scatter plots with linear regression lines for each category and cohort.

    Attributes:
    ----------
    data_prep : DataPreparation
        An instance of the DataPreparation class containing the processed data.
    colors : list
        List of colors for each category.

    Methods:
    -------
    get_colors(colors_list):
        Generates a list of colors for plotting.
    get_scatterplots():
        Creates scatter plots with linear regression lines for each category and cohort.
    plot():
        Plots the scatter plots.
    """

    def __init__(self, data_prep, colors_list=None, confidence_intervals=False):
        self.data_prep = data_prep
        self.colors = self.get_colors(colors_list)
        self.scatterplots_dict = self.get_scatterplots(confidence_intervals)
        
    def get_colors(self, colors_list):
        if colors_list is not None:
            return sns.color_palette(colors_list, len(self.data_prep.df[self.data_prep.cohort_variable].unique()))
        else:
            try:
                return sns.color_palette("tab10", len(self.data_prep.df[self.data_prep.cohort_variable].unique()))
            except:
                return sns.color_palette("tab10")

    def get_scatterplots(self,confidence_intervals):
        scatterplots_dict = {}
        for name, dataframe in self.data_prep.category_dataframes.items():
            fig, ax = plt.subplots(figsize=(6, 6))
            
            if self.data_prep.cohort_variable is not None: 
                for i, cohort in enumerate(dataframe[self.data_prep.cohort_variable].unique()):
                    cohort_df = dataframe[dataframe[self.data_prep.cohort_variable] == cohort]
                    sns.regplot(x=self.data_prep.independent_variable, 
                                y=self.data_prep.dependent_variable, 
                                data=cohort_df, ax=ax, label=cohort, color=self.colors[i],
                                ci=confidence_intervals)
            else:
                sns.regplot(x=self.data_prep.independent_variable, 
                            y=self.data_prep.dependent_variable, 
                            data=dataframe, ax=ax, label=name, color=self.colors[0],
                            ci=confidence_intervals)
                
            ax.legend()
            plt.title(name)
            plt.tight_layout()
            scatterplots_dict[name] = fig
            plt.close()

        return scatterplots_dict

    def plot(self):
        import warnings
        for name, fig in self.scatterplots_dict.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", UserWarning)
                    fig.show()
            except UserWarning:
                from IPython.display import display
                display(fig)
                
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

class BarPlot:
    """
    A class to create bar plots summarizing the correlation strengths for each category and cohort.

    Attributes:
    ----------
    data_prep : DataPreparation
        An instance of the DataPreparation class containing the processed data.
    colors : list
        List of colors for each cohort.

    Methods:
    -------
    get_colors(colors_list):
        Generates a list of colors for plotting.
    get_barplots(method):
        Creates bar plots summarizing the correlation strengths for each category and cohort.
    plot():
        Plots the bar plots.
    """

    def __init__(self, data_prep, colors_list=None, method='pearson'):
        self.data_prep = data_prep
        self.colors = self.get_colors(colors_list)
        self.method = method
        self.barplots_dict = self.get_barplots()
        self.set_shared_y_limits()
        
    def get_colors(self, colors_list):
        if colors_list is not None:
            return sns.color_palette(colors_list, len(self.data_prep.df[self.data_prep.cohort_variable].unique()))
        else:
            try:
                return sns.color_palette("tab10", len(self.data_prep.df[self.data_prep.cohort_variable].unique()))
            except:
                return sns.color_palette("tab10")

    def get_barplots(self):
        barplots_dict = {}
        all_correlations = []

        for name, dataframe in self.data_prep.category_dataframes.items():
            correlations = {}
            p_values = {}

            if self.data_prep.cohort_variable is not None:
                for i, cohort in enumerate(dataframe[self.data_prep.cohort_variable].unique()):
                    cohort_df = dataframe[dataframe[self.data_prep.cohort_variable] == cohort]
                    if self.method == 'pearson':
                        correlation, p_value = pearsonr(cohort_df[self.data_prep.independent_variable], cohort_df[self.data_prep.dependent_variable])
                    else:  # spearman
                        correlation, p_value = spearmanr(cohort_df[self.data_prep.independent_variable], cohort_df[self.data_prep.dependent_variable])
                    correlations[cohort] = correlation
                    p_values[cohort] = p_value
                    all_correlations.append(correlation)
            else:
                if self.method == 'pearson':
                    correlation, p_value = pearsonr(dataframe[self.data_prep.independent_variable], dataframe[self.data_prep.dependent_variable])
                else:  # spearman
                    correlation, p_value = spearmanr(dataframe[self.data_prep.independent_variable], dataframe[self.data_prep.dependent_variable])
                correlations[name] = correlation
                p_values[name] = p_value
                all_correlations.append(correlation)

            fig, ax = plt.subplots(figsize=(6, 6))
            bars = sns.barplot(y=list(correlations.keys()), x=list(correlations.values()), ax=ax, palette=self.colors[:len(correlations)])
            ax.set_xlabel('Correlation Strength')
            ax.set_title(name)

            # Outline bars with p-value < 0.05
            for j, bar in enumerate(bars.patches):
                cohort_name = list(correlations.keys())[j]
                if p_values[cohort_name] < 0.05:
                    bar.set_edgecolor('black')
                    bar.set_linewidth(1.5)

            plt.tight_layout()
            barplots_dict[name] = fig
            plt.close()

        self.y_min = min(all_correlations)
        self.y_max = max(all_correlations)

        return barplots_dict

    def set_shared_y_limits(self):
        for name, fig in self.barplots_dict.items():
            ax = fig.get_axes()[0]
            ax.set_xlim(self.y_min, self.y_max)

    def plot(self):
        import warnings
        for name, fig in self.barplots_dict.items():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", UserWarning)
                    fig.show()
            except UserWarning:
                from IPython.display import display
                display(fig)

class CombinedPlotV1:
    """
    A class to create a combined plot of scatter plots and bar plots.

    Attributes:
    ----------
    scatter_plot : ScatterPlot
        An instance of the ScatterPlot class containing the scatter plots.
    bar_plot : BarPlot
        An instance of the BarPlot class containing the bar plots.
    out_dir : str
        Directory to save the output plot.

    Methods:
    -------
    create_figure():
        Creates a combined figure with scatter plots and bar plots.
    """

    def __init__(self, scatter_plot, bar_plot, out_dir=None):
        self.scatter_plot = scatter_plot
        self.bar_plot = bar_plot
        self.out_dir = out_dir
        self.fig, self.axes = self.create_figure()

    def create_figure(self):
        """
        Creates a combined figure with scatter plots and bar plots.
        """
        categories = list(self.scatter_plot.data_prep.category_dataframes.keys())
        num_categories = len(categories)

        fig, axes = plt.subplots(num_categories, 2, figsize=(15, 6 * num_categories))

        for i, category in enumerate(categories):
            # Scatter Plot
            scatter_ax = axes[i, 0]
            scatter_fig = self.scatter_plot.scatterplots_dict[category]
            scatter_canvas = scatter_fig.canvas
            scatter_canvas.draw()
            scatter_image = scatter_canvas.buffer_rgba()
            scatter_ax.imshow(scatter_image)
            scatter_ax.axis('off')

            # Bar Plot
            bar_ax = axes[i, 1]
            bar_fig = self.bar_plot.barplots_dict[category]
            bar_canvas = bar_fig.canvas
            bar_canvas.draw()
            bar_image = bar_canvas.buffer_rgba()
            bar_ax.imshow(bar_image)
            bar_ax.axis('off')

        plt.tight_layout()

        if self.out_dir:
            plt.savefig(f"{self.out_dir}/combined_plot.svg", format='svg')
        plt.close()
        return fig, axes
    
    def plot(self):
        import warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                self.fig.show()
        except UserWarning:
            from IPython.display import display
            display(self.fig)
            
import numpy as np

class PermutationAnalysis:
    def __init__(self, data_prep, onetail=True, correlation_method='pearson', abs_correl=False):
        """
        Params:
        onetail (bool): whether to perform 1-tail testing or not. 
            1-tail testing compares if the correlation is larger than the raw value of all other correlations. 
            2-tail testing compares the absolute value of the correlations.
        correlation_method (str): what correlation to use. pearson | spearman | kendall 
        abs_correl (bool): if to measure the difference in absolute correlations
        Notes:
        The similarity by chance is 1-p. The difference by chance is the p value outputted. 
        """
        self.data_prep = data_prep
        self.observed_correlations = {}
        self.permuted_differences = {}
        self.p_values = {}
        self.onetail = onetail
        self.correlation_method = correlation_method
        self.abs_correl = abs_correl

    def calculate_observed_differences(self, df):
        """Calculate the observed differences for each cohort within a dataframe."""
        observed_correlations = {}
        for cohort in df[self.data_prep.cohort_variable].unique():
            cohort_df = df[df[self.data_prep.cohort_variable] == cohort]
            correlation = cohort_df[self.data_prep.independent_variable].corr(cohort_df[self.data_prep.dependent_variable], method=self.correlation_method)
            observed_correlations[cohort] = correlation

        observed_differences = {}
        cohorts = list(observed_correlations.keys())
        for i in range(len(cohorts)):
            for j in range(i + 1, len(cohorts)):
                cohort1, cohort2 = cohorts[i], cohorts[j]
                if self.abs_correl:
                    diff = np.abs(observed_correlations[cohort1]) - np.abs(observed_correlations[cohort2])
                else:
                    diff = observed_correlations[cohort1] - observed_correlations[cohort2]
                
                observed_differences[(cohort1, cohort2)] = diff if self.onetail else np.abs(diff)

        return observed_differences

    def calculate_permuted_differences(self, df, n_permutations=1000):
        """Permute the data and calculate the permuted differences for each cohort within a dataframe."""
        permuted_differences = []

        for _ in range(n_permutations):
            permuted_df = df.copy()
            permuted_df[self.data_prep.dependent_variable] = np.random.permutation(permuted_df[self.data_prep.dependent_variable].values)
            correlations = {}
            for cohort in df[self.data_prep.cohort_variable].unique():
                cohort_df = permuted_df[permuted_df[self.data_prep.cohort_variable] == cohort]
                correlation = cohort_df[self.data_prep.independent_variable].corr(cohort_df[self.data_prep.dependent_variable], method=self.correlation_method)
                correlations[cohort] = correlation

            permuted_diffs = {}
            cohorts = list(correlations.keys())
            for i in range(len(cohorts)):
                for j in range(i + 1, len(cohorts)):
                    cohort1, cohort2 = cohorts[i], cohorts[j]
                    if self.abs_correl:
                        diff = np.abs(correlations[cohort1]) - np.abs(correlations[cohort2])
                    else:
                        diff = correlations[cohort1] - correlations[cohort2]
                    permuted_diffs[(cohort1, cohort2)] = diff if self.onetail else np.abs(diff)

            permuted_differences.append(permuted_diffs)

        return permuted_differences

    def calculate_p_values(self, observed_differences, permuted_differences):
        """Calculate p-values based on the observed and permuted differences."""
        p_values = {}
        for (cohort1, cohort2), observed_diff in observed_differences.items():
            permuted_diffs = [permuted[(cohort1, cohort2)] for permuted in permuted_differences]
            p_value = np.mean([diff >= observed_diff for diff in permuted_diffs])
            p_values[(cohort1, cohort2, f'delta = {observed_diff}')] = p_value

        return p_values

    def run_analysis(self, n_permutations=1000):
        """Run the entire permutation analysis for each dataframe in the data_prep object."""
        for category, df in self.data_prep.category_dataframes.items():
            observed_differences = self.calculate_observed_differences(df)
            permuted_differences = self.calculate_permuted_differences(df, n_permutations)
            p_values = self.calculate_p_values(observed_differences, permuted_differences)
            self.p_values[category] = p_values
            
import numpy as np
import pandas as pd
from pprint import pprint

class CrossCategoryPermutationAnalysis:
    def __init__(self, perm_analysis, onetail=True, correlation_method='pearson', abs_correl=False):
        """
        Params:
        onetail (bool): whether to perform 1-tail testing or not. 
            1-tail testing compares if the correlation is larger than the raw value of all other correlations. 
            2-tail testing compares the absolute value of the correlations. 
        abs_correl bool: whether to measure difference between absolute correlations.
        Notes:
        The similarity by chance is 1-p. The difference by chance is the p value outputted. 
        """
        self.perm_analysis = perm_analysis
        self.observed_correlations = {}
        self.permuted_correlations = []
        self.observed_differences = {}
        self.permuted_differences = []
        self.p_values = {}
        self.onetail = onetail
        self.correlation_method = correlation_method
        self.abs_correl = abs_correl

    def calculate_observed_correlations(self):
        """Calculate the observed correlations for each cohort within each category dataframe."""
        for category, df in self.perm_analysis.data_prep.category_dataframes.items():
            self.observed_correlations[category] = {}
            for cohort in df[self.perm_analysis.data_prep.cohort_variable].unique():
                cohort_df = df[df[self.perm_analysis.data_prep.cohort_variable] == cohort]
                correlation = cohort_df[self.perm_analysis.data_prep.independent_variable].corr(cohort_df[self.perm_analysis.data_prep.dependent_variable], method=self.correlation_method)
                self.observed_correlations[category][cohort] = correlation

    def calculate_observed_differences(self):
        """Calculate the differences between correlations of every cohort in one category dataframe against every cohort in other category dataframes."""
        categories = list(self.observed_correlations.keys())
        for i in range(len(categories)):
            for j in range(i + 1, len(categories)):
                category1, category2 = categories[i], categories[j]
                self.observed_differences[(category1, category2)] = {}
                for cohort1, corr1 in self.observed_correlations[category1].items():
                    for cohort2, corr2 in self.observed_correlations[category2].items():
                        if self.abs_correl:
                            diff = np.abs(corr1) - np.abs(corr2)
                        else:
                            diff = corr1 - corr2
                        self.observed_differences[(category1, category2)][(cohort1, cohort2)] = diff if self.onetail else np.abs(diff)


    def calculate_permuted_differences(self, n_permutations=1000):
        """Permute the data and calculate the permuted differences for each permutation."""
        categories = list(self.perm_analysis.data_prep.category_dataframes.keys())
        all_data = pd.concat(self.perm_analysis.data_prep.category_dataframes.values())

        for _ in range(n_permutations):
            permuted_data = all_data.copy()
            permuted_data[self.perm_analysis.data_prep.dependent_variable] = np.random.permutation(permuted_data[self.perm_analysis.data_prep.dependent_variable].values)

            permuted_correlations = {}
            for category in categories:
                permuted_correlations[category] = {}
                df = permuted_data[permuted_data[self.perm_analysis.data_prep.categorical_variable] == category]
                for cohort in df[self.perm_analysis.data_prep.cohort_variable].unique():
                    cohort_df = df[df[self.perm_analysis.data_prep.cohort_variable] == cohort]
                    correlation = cohort_df[self.perm_analysis.data_prep.independent_variable].corr(cohort_df[self.perm_analysis.data_prep.dependent_variable], method=self.correlation_method)
                    permuted_correlations[category][cohort] = correlation

            permuted_diffs = {}
            for i in range(len(categories)):
                for j in range(i + 1, len(categories)):
                    category1, category2 = categories[i], categories[j]
                    permuted_diffs[(category1, category2)] = {}
                    for cohort1, corr1 in permuted_correlations[category1].items():
                        for cohort2, corr2 in permuted_correlations[category2].items():
                            if self.abs_correl:
                                diff = np.abs(corr1) - np.abs(corr2)
                            else:
                                diff = corr1 - corr2
                            permuted_diffs[(category1, category2)][(cohort1, cohort2)] = diff if self.onetail else np.abs(diff)

            self.permuted_differences.append(permuted_diffs)

    def calculate_p_values(self):
        """Calculate p-values based on the observed and permuted differences."""
        categories = list(self.observed_correlations.keys())
        for i in range(len(categories)):
            for j in range(i + 1, len(categories)):
                category1, category2 = categories[i], categories[j]
                self.p_values[(category1, category2)] = {}
                for (cohort1, cohort2), observed_diff in self.observed_differences[(category1, category2)].items():
                    permuted_diffs = [permuted[(category1, category2)][(cohort1, cohort2)] for permuted in self.permuted_differences if (category1, category2) in permuted and (cohort1, cohort2) in permuted[(category1, category2)]]
                    p_value = np.mean([diff >= observed_diff for diff in permuted_diffs])
                    self.p_values[(category1, category2)][(cohort1, cohort2, f'delta = {observed_diff}')] = p_value

    def run_cross_category_permutation_analysis(self, n_permutations=1000):
        """Run the entire cross-category permutation analysis."""
        self.calculate_observed_correlations()
        self.calculate_observed_differences()
        self.calculate_permuted_differences(n_permutations)
        self.calculate_p_values()
        
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

class CombinedPlot:
    """
    A class to create a combined plot of scatter plots and bar plots.

    Attributes:
    ----------
    scatter_plot : ScatterPlot
        An instance of the ScatterPlot class containing the scatter plots.
    bar_plot : BarPlot
        An instance of the BarPlot class containing the bar plots.
    out_dir : str
        Directory to save the output plot.

    Methods:
    -------
    create_figure():
        Creates a combined figure with scatter plots and bar plots.
    plot():
        Displays the combined figure.
    save_figure():
        Saves the combined figure.
    """

    def __init__(self, scatter_plot, bar_plot, out_dir=None):
        self.scatter_plot = scatter_plot
        self.bar_plot = bar_plot
        self.out_dir = out_dir
        self.fig, self.axes = self.create_figure()
        self.set_shared_x_limits()

    def set_shared_x_limits(self):
        """
        Sets the x-limits for scatter plots and bar plots to be shared separately.
        """
        scatter_x_limits = [self.axes[i, 0].get_xlim() for i in range(self.axes.shape[0])]
        scatter_x_min = min([lim[0] for lim in scatter_x_limits])
        scatter_x_max = max([lim[1] for lim in scatter_x_limits])

        bar_x_limits = [self.axes[i, 1].get_xlim() for i in range(self.axes.shape[0])]
        bar_x_min = min([lim[0] for lim in bar_x_limits])
        bar_x_max = max([lim[1] for lim in bar_x_limits])

        for i in range(self.axes.shape[0]):
            self.axes[i, 0].set_xlim(scatter_x_min, scatter_x_max)
            self.axes[i, 1].set_xlim(bar_x_min, bar_x_max)

    def create_figure(self):
        """
        Creates a combined figure with scatter plots and bar plots.
        """
        categories = list(self.scatter_plot.data_prep.category_dataframes.keys())
        num_categories = len(categories)

        fig, axes = plt.subplots(num_categories, 2, figsize=(15, 6 * num_categories))

        for i, category in enumerate(categories):
            # Scatter Plot
            scatter_ax = axes[i, 0]
            scatter_data = self.scatter_plot.data_prep.category_dataframes[category]
            if self.scatter_plot.data_prep.cohort_variable is not None:
                for j, cohort in enumerate(scatter_data[self.scatter_plot.data_prep.cohort_variable].unique()):
                    cohort_df = scatter_data[scatter_data[self.scatter_plot.data_prep.cohort_variable] == cohort]
                    sns.regplot(
                        x=self.scatter_plot.data_prep.independent_variable,
                        y=self.scatter_plot.data_prep.dependent_variable,
                        data=cohort_df,
                        ax=scatter_ax,
                        label=cohort,
                        color=self.scatter_plot.colors[j],
                        ci=None
                    )
            else:
                sns.regplot(
                    x=self.scatter_plot.data_prep.independent_variable,
                    y=self.scatter_plot.data_prep.dependent_variable,
                    data=scatter_data,
                    ax=scatter_ax,
                    label=category,
                    color=self.scatter_plot.colors[0],
                    ci=None
                )
            scatter_ax.legend()
            scatter_ax.set_title(category)
            scatter_ax.set_xlabel(self.scatter_plot.data_prep.independent_variable)
            scatter_ax.set_ylabel(self.scatter_plot.data_prep.dependent_variable)

            # Bar Plot
            bar_ax = axes[i, 1]
            bar_data = self.bar_plot.data_prep.category_dataframes[category]
            correlations = []
            p_values = []
            cohorts = []

            if self.bar_plot.data_prep.cohort_variable is not None:
                for j, cohort in enumerate(bar_data[self.bar_plot.data_prep.cohort_variable].unique()):
                    cohort_df = bar_data[bar_data[self.bar_plot.data_prep.cohort_variable] == cohort]
                    if self.bar_plot.method == 'pearson':
                        correlation, p_value = pearsonr(
                            cohort_df[self.bar_plot.data_prep.independent_variable],
                            cohort_df[self.bar_plot.data_prep.dependent_variable]
                        )
                    else:
                        correlation, p_value = spearmanr(
                            cohort_df[self.bar_plot.data_prep.independent_variable],
                            cohort_df[self.bar_plot.data_prep.dependent_variable]
                        )
                    correlations.append(correlation)
                    p_values.append(p_value)
                    cohorts.append(cohort)
            else:
                if self.bar_plot.method == 'pearson':
                    correlation, p_value = pearsonr(
                        bar_data[self.bar_plot.data_prep.independent_variable],
                        bar_data[self.bar_plot.data_prep.dependent_variable]
                    )
                else:
                    correlation, p_value = spearmanr(
                        bar_data[self.bar_plot.data_prep.independent_variable],
                        bar_data[self.bar_plot.data_prep.dependent_variable]
                    )
                correlations.append(correlation)
                p_values.append(p_value)
                cohorts.append(category)
            
            sns.barplot(x=correlations, y=cohorts, ax=bar_ax, palette=self.bar_plot.colors[:len(cohorts)])
            bar_ax.set_xlabel('Correlation Strength')
            bar_ax.set_title(category)

            # Outline bars with p-value < 0.05
            for bar, p_value in zip(bar_ax.patches, p_values):
                if p_value < 0.05:
                    bar.set_edgecolor('black')
                    bar.set_linewidth(1.5)

        plt.tight_layout()
        self.bar_plot.set_shared_y_limits()
        return fig, axes

    def save_figure(self):
        """
        Saves the combined figure to the specified output directory.
        """
        if self.out_dir:
            self.fig.savefig(f"{self.out_dir}/combined_plot.svg", format='svg')

    def plot(self):
        """
        Displays the combined figure.
        """
        import warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)
                self.fig.show()
        except UserWarning:
            from IPython.display import display
            display(self.fig)

import numpy as np

class ContrastAnalysis:
    def __init__(self, perm_analysis, onetail=True, correlation_method='pearson', abs_correl=False):
        """
        Params:
        onetail (bool): whether to perform 1-tail testing or not. 
            1-tail testing compares if the correlation is larger than the raw value of all other correlations. 
            2-tail testing compares the absolute value of the correlations. 
        abs_correl (bool): whether to measure the difference between absolute correlations
        Notes:
        The similarity by chance is 1-p. The difference by chance is the p value outputted. 
        """
        self.perm_analysis = perm_analysis
        self.summed_correlations = {}
        self.cross_dataframe_differences = {}
        self.cross_dataframe_p_values = {}
        self.onetail = onetail
        self.correlation_method = correlation_method
        self.abs_correl = abs_correl

    def summate_correlations(self):
        """Summate the correlations within each dataframe."""
        for category, df in self.perm_analysis.data_prep.category_dataframes.items():
            summed_correlation = 0
            for cohort in df[self.perm_analysis.data_prep.cohort_variable].unique():
                cohort_df = df[df[self.perm_analysis.data_prep.cohort_variable] == cohort]
                correlation = cohort_df[self.perm_analysis.data_prep.independent_variable].corr(cohort_df[self.perm_analysis.data_prep.dependent_variable], method=self.correlation_method)
                summed_correlation += correlation
            self.summed_correlations[category] = summed_correlation

    def calculate_cross_dataframe_differences(self, n_permutations=1000):
        """Calculate the difference between summed correlations from different dataframes and calculate p-values."""
        categories = list(self.summed_correlations.keys())
        for i in range(len(categories)):
            for j in range(i + 1, len(categories)):
                category1, category2 = categories[i], categories[j]
                if self.abs_correl:
                    observed_diff = np.abs(self.summed_correlations[category1]) - np.abs(self.summed_correlations[category2])
                else:
                    observed_diff = self.summed_correlations[category1] - self.summed_correlations[category2]
                self.cross_dataframe_differences[(category1, category2)] = observed_diff if self.onetail else np.abs(observed_diff)

                permuted_diffs = []
                for _ in range(n_permutations):
                    permuted_df1 = self.perm_analysis.data_prep.category_dataframes[category1].copy()
                    permuted_df2 = self.perm_analysis.data_prep.category_dataframes[category2].copy()

                    permuted_df1[self.perm_analysis.data_prep.dependent_variable] = np.random.permutation(permuted_df1[self.perm_analysis.data_prep.dependent_variable].values)
                    permuted_df2[self.perm_analysis.data_prep.dependent_variable] = np.random.permutation(permuted_df2[self.perm_analysis.data_prep.dependent_variable].values)

                    permuted_sum1 = 0
                    for cohort in permuted_df1[self.perm_analysis.data_prep.cohort_variable].unique():
                        cohort_df = permuted_df1[permuted_df1[self.perm_analysis.data_prep.cohort_variable] == cohort]
                        correlation = cohort_df[self.perm_analysis.data_prep.independent_variable].corr(cohort_df[self.perm_analysis.data_prep.dependent_variable], method=self.correlation_method)
                        permuted_sum1 += correlation

                    permuted_sum2 = 0
                    for cohort in permuted_df2[self.perm_analysis.data_prep.cohort_variable].unique():
                        cohort_df = permuted_df2[permuted_df2[self.perm_analysis.data_prep.cohort_variable] == cohort]
                        correlation = cohort_df[self.perm_analysis.data_prep.independent_variable].corr(cohort_df[self.perm_analysis.data_prep.dependent_variable], method=self.correlation_method)
                        permuted_sum2 += correlation

                    if self.abs_correl:
                        permuted_diff = np.abs(permuted_sum1) - np.abs(permuted_sum2)
                    else:
                        permuted_diff = permuted_sum1 - permuted_sum2
                    permuted_diffs.append(permuted_diff if self.onetail else np.abs(permuted_diff))

                p_value = np.mean([diff >= observed_diff for diff in permuted_diffs])
                self.cross_dataframe_p_values[(category1, category2, f'delta = {observed_diff}')] = p_value

import os
from pprint import pformat

def save_dicts_as_py(out_dir, **dicts):
    """
    Save dictionaries as .py files in the specified directory with pretty-print formatting.

    Parameters:
    out_dir (str): Directory to save the Python files.
    **dicts: Arbitrary number of dictionaries to save, with keys as file names (without .py extension).
    """

    # Save each dictionary as a Python file
    for file_name, data in dicts.items():
        try:
            with open(os.path.join(out_dir, f"{file_name}.py"), 'w') as py_file:
                py_file.write(f"{file_name} = {pformat(data)}\n")
            print(f"Successfully saved {file_name}.py")
        except Exception as e:
            print(f"An unexpected error occurred while saving {file_name}.py: {e}")


from contextlib import contextmanager
from itertools import combinations
from scipy.stats import t


class DeltaCorrelationAnalysis:
    """Multigroup Pearson-r analysis with bootstraps, permutations, and plots."""

    # ---------- basic ----------
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy(deep=False)
        self.boot_corrs: dict[str, np.ndarray] = {}
        self.perm_results: dict[tuple[str, str], tuple[float, float]] = {}

    @contextmanager
    def _use_df(self, df: pd.DataFrame):
        old_df = self.df
        self.df = df
        try:
            yield
        finally:
            self.df = old_df

    def _one_vs_rest_df(self, group: str, focus_category: str, rest_label: str) -> pd.DataFrame:
        if focus_category not in set(self.df[group].dropna().unique()):
            raise ValueError(
                f"focus_category '{focus_category}' not found in column '{group}'."
            )
        df = self.df.copy()
        df[group] = np.where(df[group] == focus_category, focus_category, rest_label)
        return df

    # ---------- statistics ----------
    @staticmethod
    def _analytic_ci(x, y, x_grid):
        slope, intercept = np.polyfit(x, y, 1)
        y_fit = slope * x_grid + intercept

        resid = y - (slope * x + intercept)
        stderr = np.sqrt(resid.dot(resid) / (len(y) - 2))
        t_val = t.ppf(0.975, len(y) - 2)
        ci = t_val * stderr * np.sqrt(
            1 / len(x) + (x_grid - x.mean()) ** 2 / ((x - x.mean()) ** 2).sum()
        )
        return y_fit - ci, y_fit, y_fit + ci

    def bootstrap_correlations(self, x, y, group, n=1000, seed=None, focus_category=None, rest_label="all_other"):
        if focus_category is not None:
            df = self._one_vs_rest_df(group, focus_category, rest_label)
        else:
            df = self.df

        rng = np.random.default_rng(seed)
        for g, sub in df.groupby(group):
            print(f"Group '{g}' has {len(sub)} values.")
            r = np.empty(n)
            for i in range(n):
                boot = sub.sample(len(sub), replace=True, random_state=rng.integers(2**32))
                r[i] = boot[x].corr(boot[y])
            self.boot_corrs[g] = r
        return self.boot_corrs

    def permute_delta_r(self, x, y, group, pairs=None, n=10000, seed=None, absval=False,
                        focus_category=None, rest_label="all_other"):
        if focus_category is not None:
            df = self._one_vs_rest_df(group, focus_category, rest_label)
            pairs = [(focus_category, rest_label)]
        else:
            df = self.df

        if pairs is None:
            pairs = combinations(df[group].unique(), 2)
        rng = np.random.default_rng(seed)

        for a, b in pairs:
            sub_a = df[df[group] == a]
            sub_b = df[df[group] == b]
            n_a, n_b = len(sub_a), len(sub_b)

            if absval:
                obs = np.abs(sub_a[x].corr(sub_a[y])) - np.abs(sub_b[x].corr(sub_b[y]))
            else:
                obs = sub_a[x].corr(sub_a[y]) - sub_b[x].corr(sub_b[y])

            joined = pd.concat([sub_a, sub_b], ignore_index=True)
            diffs = np.empty(n)
            for i in range(n):
                shuffle = rng.permutation(n_a + n_b)
                pa = joined.iloc[shuffle[:n_a]]
                pb = joined.iloc[shuffle[n_a:]]
                if absval:
                    diffs[i] = np.abs(pa[x].corr(pa[y])) - np.abs(pb[x].corr(pb[y]))
                else:
                    diffs[i] = pa[x].corr(pa[y]) - pb[x].corr(pb[y])
            p_val = (np.abs(diffs) >= abs(obs)).mean()
            self.perm_results[(a, b)] = (obs, p_val)

        return self.perm_results

    # ---------- plotting ----------
    def scatter(self, x, y, group, palette="tab10", alpha=.3, figsize=(4, 5), point_hue=None, point_palette="tab20"):
        plt.rcParams.update({'font.family': 'Helvetica', 'font.size': 16})  # Set font to Helvetica 16
        fig, ax = plt.subplots(figsize=figsize)
        if point_hue is None:
            hue_vals = sorted(self.df[group].dropna().unique())
            hue_cols = sns.color_palette(palette, len(hue_vals))
            colour_map = dict(zip(hue_vals, hue_cols))

            for g, col in colour_map.items():
                sub = self.df[self.df[group] == g]
                ax.scatter(sub[x], sub[y], color=col, s=20, alpha=alpha, label=g)
        else:
            hue_vals = sorted(self.df[point_hue].dropna().unique())
            hue_cols = sns.color_palette(point_palette, len(hue_vals))
            colour_map = dict(zip(hue_vals, hue_cols))

            for h in hue_vals:
                sub = self.df[self.df[point_hue] == h]
                ax.scatter(sub[x], sub[y], color=colour_map[h], s=20, alpha=alpha, label=str(h), edgecolors="none")

        # ------------ 2) regression lines per GROUP -------------------
        grp_vals = sorted(self.df[group].dropna().unique())
        line_cols = sns.color_palette(palette, len(grp_vals))
        x_grid = np.linspace(self.df[x].min(), self.df[x].max(), 400)

        for g, col in zip(grp_vals, line_cols):
            sub = self.df[self.df[group] == g]
            y_lo, y_fit, y_hi = self._analytic_ci(sub[x], sub[y], x_grid)
            ax.plot(x_grid, y_fit, color=col, linewidth=2.0, label=f"fit {g}")
            ax.fill_between(x_grid, y_lo, y_hi, color=col, alpha=alpha / 4)

        # ------------ 3) permutation text (if already computed) -------
        if self.perm_results:
            txt = "\n".join(
                f"{a}–{b}: Δr={d:.2f}, p={p:.3f}"
                for (a, b), (d, p) in self.perm_results.items()
            )
            ax.set_title(txt, fontsize=9)

        ax.set_xlabel(x)
        ax.set_ylabel(y)
        sns.despine(ax=ax)
        ax.legend(frameon=False)
        return fig

    # ---------- plotting ----------
    def boxplot_boot(self, x: str, y: str, group: str, *,
        xlim=None, horiz: bool = True, figsize=(4, 4), out_path: str | None = None):
        """
        Box-plots of bootstrapped Pearson-r per group with observed r over-laid.
        Call `bootstrap_correlations` first (or `run`).

        Parameters
        ----------
        x, y        : column names whose correlation is analysed.
        group       : column name used to split the data.
        xlim        : limits for the correlation axis.
        horiz       : plot horizontally (True) or vertically (False).
        """
        if not self.boot_corrs:
            raise RuntimeError("Run bootstrap_correlations (or run) first.")

        # long-form DF for Seaborn
        df_long = (pd.DataFrame(self.boot_corrs).melt(var_name="Group", value_name="r").dropna())

        if xlim is not None: 
            xlim=xlim
        elif any (df_long.groupby("Group")["r"].mean() < 0):
            xlim=(-1, 1)
        else:
            xlim=(0, 1)

        fig = plt.figure(figsize=figsize)
        orient = "h" if horiz else "v"

        if horiz:
            sns.boxplot(data=df_long, x="r", y="Group", orient=orient, palette="coolwarm")
            plt.xlim(*xlim)
            plt.xlabel("Pearson r")
            plt.ylabel(group)
        else:
            sns.boxplot(data=df_long, x="Group", y="r", orient=orient, palette="coolwarm")
            plt.ylim(*xlim)
            plt.ylabel("Pearson r")
            plt.xlabel(group)
        plt.legend(frameon=False)
        sns.despine()

        if out_path:
            fig.savefig(out_path, bbox_inches="tight")
        return fig

    # ---------- orchestration function ----------
    def run(self, x: str, y: str, group: str, n_boot: int = 1000, n_perm: int = 10000, absval=True,
        palette: str = "tab10", alpha: float = 1, xlim: tuple[float, float] = (-1, 1),
        horiz_box: bool = True, out_dir: str | None = None, seed: int | None = None, figsize=(4, 4.5),
        point_hue: str | None = None, point_palette: str = "tab20",
        focus_category: str | None = None, rest_label: str = "all_other"):
        """
        One-liner entry: does bootstraps, permutations, scatter & box-plot.
        If focus_category is provided, all non-focus rows are grouped into `rest_label`.
        """
        # Print the number of unique values in the group
        unique_groups = self.df[group].nunique()
        print(f"Number of unique groups in '{group}': {unique_groups}")

        if focus_category is None:
            # 1) statistics
            self.bootstrap_correlations(x, y, group, n=n_boot, seed=seed)
            self.permute_delta_r(x, y, group, n=n_perm, seed=seed, absval=absval)

            # 2) figures
            sc_fig = self.scatter(x, y, group, palette=palette, alpha=alpha, figsize=figsize,
                                  point_hue=point_hue, point_palette=point_palette)
            bx_fig = self.boxplot_boot(x, y, group, xlim=xlim, horiz=horiz_box, figsize=figsize)
        else:
            temp_df = self._one_vs_rest_df(group, focus_category, rest_label)
            with self._use_df(temp_df):
                # 1) statistics
                self.bootstrap_correlations(x, y, group, n=n_boot, seed=seed)
                self.permute_delta_r(x, y, group, n=n_perm, seed=seed, absval=absval)

                # 2) figures
                sc_fig = self.scatter(x, y, group, palette=palette, alpha=alpha, figsize=figsize,
                                      point_hue=point_hue, point_palette=point_palette)
                bx_fig = self.boxplot_boot(x, y, group, xlim=xlim, horiz=horiz_box, figsize=figsize)

        # 3) save, if requested
        if out_dir:
            sc_fig.savefig(f"{out_dir}/scatter.svg", bbox_inches="tight")
            bx_fig.savefig(f"{out_dir}/boot_box.svg", bbox_inches="tight")

        return sc_fig, bx_fig