import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pingouin as pg
from itertools import combinations
import math
import matplotlib.patches as mpatches

class ICCAnalysis:
    
    def __init__(self, xlim=(0,1), xlabel='ICC', ylabel='Categories'):
        """
        Initialize the ICCAnalysis class with optional x-axis limits for plots.
        
        Parameters:
        xlim (tuple): Optional. A tuple specifying the x-axis limits (min, max) for the plots.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        """
        self.xlim = xlim
        self.xlabel = xlabel
        self.ylabel = ylabel

    def format_long_dataframe(self, series1, series2):
        """
        Format the series into a long-format DataFrame.
        
        Parameters:
        series1 (pd.Series): First series of ratings.
        series2 (pd.Series): Second series of ratings.
        
        Returns:
        pd.DataFrame: Long-format DataFrame suitable for ICC calculation.
        """
        df = pd.DataFrame()
        df['rating1'] = series1.values
        df['rating2'] = series2.values
        df['subject'] = df.index
        df_melted = df.melt(id_vars=['subject'], value_vars=['rating1', 'rating2'], var_name='rater', value_name='rating')
        return df_melted.dropna()
    
    def calculate_icc(self, series1, series2, icc):
        """
        Calculate the Intraclass Correlation Coefficient (ICC) between two series.
        
        Parameters:
        series1 (pd.Series): First series of ratings.
        series2 (pd.Series): Second series of ratings.
        icc (str): which icc to use. 
        
        Returns:
        float: ICC value.
        float: Lower bound of the 95% confidence interval.
        float: Upper bound of the 95% confidence interval.
        """
        df_melted = self.format_long_dataframe(series1, series2)
        if df_melted.shape[0] < 5:
            return None, None, None, None
        icc_result = pg.intraclass_corr(data=df_melted, targets='subject', raters='rater', ratings='rating', nan_policy='omit')
        icc_value = icc_result.set_index('Type').loc[icc, 'ICC']
        ci_lower = icc_result.set_index('Type').loc[icc, 'CI95'][0]
        ci_upper = icc_result.set_index('Type').loc[icc, 'CI95'][1]
        p_value = icc_result.set_index('Type').loc[icc, 'pval']
        return icc_value, ci_lower, ci_upper, p_value
    
    def plot_icc_forest(self, icc_results, ax, title):
        """
        Plot ICC forest plots for multiple comparisons.
        
        Parameters:
        icc_results (dict): Dictionary containing ICC results for each column comparison.
        ax (matplotlib.axes.Axes): Axes object to plot on.
        title (str): Title of the plot.
        
        Returns:
        plt.Figure: Matplotlib Figure object containing the ICC forest plot.
        """
        colors = sns.color_palette("tab10", len(icc_results))
        legend_patches = []

        for idx, (col, (icc_value, ci_lower, ci_upper, p_value)) in enumerate(icc_results.items()):
            if icc_value is not None:
                ax.errorbar(x=icc_value, y=idx, xerr=[[icc_value - ci_lower], [ci_upper - icc_value]], fmt='o', color=colors[idx], capsize=5)
                legend_patches.append(mpatches.Patch(color=colors[idx], label=f'{col} (p={p_value:.4f})'))

        ax.set_xlim(0, 1)
        ax.set_ylim(-1, len(icc_results))
        ax.set_yticks(range(len(icc_results)))
        ax.set_yticklabels(list(icc_results.keys()))
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.set_xlabel(self.xlabel)
        ax.set_title(title)
        ax.grid(axis='x', linestyle='--')
        ax.legend(handles=legend_patches, frameon=False, loc='upper right')
    
    def orchestrate_icc_analysis(self, df, category_col, columns_to_compare, outdir, filetype='svg', icc="ICC(1,1)"):
        """
        Orchestrate ICC analysis for multiple pairwise comparisons and columns.
        
        Parameters:
        df (pd.DataFrame): Original DataFrame.
        category_col (str): Column name containing categories to compare across.
        columns_to_compare (list): List of columns to compare in each subplot.
        outdir (str): Output directory for saving plots.
        filetype (str): File type for saving plots (e.g., 'svg', 'png'). 
        icc (string): which ICC to grab from pingouin. Options:  ICC(1,1) | ICC(A,1) | ICC(C,1) | ICC(1,k) | ICC(A,k) | ICC(C,k)

            Liljequist et al. (2019) recommend computing all three single-score ICCs
            together and comparing them, rather than selecting a single model upfront.s
            *1. Detecting bias (systematic errors):*

            Start by comparing ICC(1,1), ICC(A,1), and ICC(C,1). When they are
            approximately equal, systematic bias between raters is likely negligible.
            When ICC(C,1) is notably larger than ICC(A,1), non-negligible bias is
            likely present. In that case, ICC(1,1) is invalid and should not be
            reported. The F statistic and p-value in the output provide a formal test
            of whether rater means differ significantly.

            *2. Agreement vs. consistency:*

            When bias is present, both ICC(A,1) and ICC(C,1) should be reported
            together with their confidence intervals. ICC(A,1) reflects absolute
            agreement (do raters assign the same values?), while ICC(C,1) reflects
            consistency (do raters rank targets in the same order?).

            *3. Single vs. average ratings:*

            Use the single-score variants (ICC(1,1), ICC(A,1), ICC(C,1)) when
            reporting the reliability of one rating. Use the average-score variants
            (ICC(1,k), ICC(A,k), ICC(C,k)) when the final measurement will be the
            mean of :math:`k` ratings.
        
        Returns:
        dict: Dictionary containing ICC results for each comparison and category.
        """
        categories = df[category_col].unique()
        pairwise_comparisons = list(combinations(categories, 2))
        all_icc_results = {}

        n = len(pairwise_comparisons)
        cols = math.ceil(math.sqrt(n + 1))  # Add one for the final combined subplot
        rows = math.ceil((n + 1) / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 5.1))
        # Handle the case where there is only one plot
        if n == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, (cat1, cat2) in enumerate(pairwise_comparisons):
            icc_results = {}
            df_cat1 = df[df[category_col] == cat1]
            df_cat2 = df[df[category_col] == cat2]
            
            for col in columns_to_compare:
                icc_value, ci_lower, ci_upper, p_value = self.calculate_icc(df_cat1[col], df_cat2[col], icc)
                icc_results[col] = (icc_value, ci_lower, ci_upper, p_value)
            
            all_icc_results[(cat1, cat2)] = icc_results
            title = f'ICC Comparison: {cat1} vs {cat2}'
            self.plot_icc_forest(icc_results, axes[i], title)
        
        combined_icc_results = self.calculate_combined_icc(df, category_col, columns_to_compare, icc)
        self.plot_icc_forest(combined_icc_results, axes[-1], 'Combined ICC of All Raters')
        
        for j in range(i + 1, len(axes) - 1):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(f"{outdir}/ICC_forest_plots.{filetype}")
        plt.show()
        
        return all_icc_results

    def calculate_combined_icc(self, df, category_col, columns_to_compare, icc):
        """
        Calculate the ICC of all raters combined for each column to compare.
        
        Parameters:
        df (pd.DataFrame): Original DataFrame.
        category_col (str): Column name containing categories to compare across.
        columns_to_compare (list): List of columns to compare in the final ICC plot.
        
        Returns:
        dict: Dictionary containing combined ICC results for each column.
        """
        combined_icc_results = {}
        for col in columns_to_compare:
            ratings = []
            raters = []
            subjects = []

            for category in df[category_col].unique():
                series = df[df[category_col] == category][col]
                ratings.extend(series)
                raters.extend([category] * len(series))
                subjects.extend(range(len(series)))  # Subject IDs should be within the length of the series for each category
            
            df_combined = pd.DataFrame({'rating': ratings, 'rater': raters, 'subject': subjects}).dropna()
            if df_combined.shape[0] < 5:
                combined_icc_results[col] = (None, None, None, None)
                continue
            
            icc_result = pg.intraclass_corr(data=df_combined, targets='subject', raters='rater', ratings='rating', nan_policy='omit')
            icc_value = icc_result.set_index('Type').loc[icc, 'ICC']
            ci_lower = icc_result.set_index('Type').loc[icc, 'CI95'][0]
            ci_upper = icc_result.set_index('Type').loc[icc, 'CI95'][1]
            p_value = icc_result.set_index('Type').loc[icc, 'pval']
            combined_icc_results[col] = (icc_value, ci_lower, ci_upper, p_value)
        print('Combined ICC Results: \n ', combined_icc_results)
        return combined_icc_results