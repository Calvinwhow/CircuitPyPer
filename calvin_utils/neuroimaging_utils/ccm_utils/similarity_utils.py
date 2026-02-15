import os 
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from nilearn import image
from natsort import natsorted
import matplotlib.pyplot as plt
from scipy.stats import f_oneway, kruskal, ttest_1samp, pearsonr
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from calvin_utils.ccm_utils.stat_utils import CorrelationCalculator
from calvin_utils.plotting_utils.simple_heatmap import simple_heatmap

class SimilarityTester:
    def __init__(self, dep_var_df, indep_var_df, mask_path=None, out_dir=None, method_choice='pearson', align=False, reference=None, flip_list=[]):
        """
        Initialize the SimilarityTester class.

        Parameters:
        - dep_var_df (DataFrame): Dependent variable dataframe containing brain maps.
        - indep_var_df (DataFrame): Independent variable dataframe containing brain maps.
        - mask_path (str, optional): Path to the mask image to apply to the dataframes. Default is None.
        - out_dir (str, optional): Directory to save output figures and results. Default is None.
        - method_choice (str, optional): Correlation method to use ('pearson', 'spearman', or 'kendall'). Default is 'pearson'.
        - align (bool, optional): Whether to align the signs of the dataframes. Default is False.
        - reference (str, optional): Reference column name, 'sum', or None for alignment. Default is None.
        - flip_list (list, optional): List of column names to flip signs. Default is an empty list.
        """
        self.mask_path = mask_path  
        self.method_choice = method_choice
        self.out_dir = out_dir
        self.dep_var_df = self.prep_data(dep_var_df, align=align, reference=reference, flip_list=flip_list)
        self.indep_var_df = self.prep_data(indep_var_df, flip_list=flip_list)
        self.xlabel = self._setxlabel()
        
    def _setxlabel(self):
        return f'{self.method_choice.capitalize()} Similarity'
        
    def prep_data(self, df, align=False, reference=None, flip_list=[]):
        df = self.apply_mask_to_dataframe(df)
        df = self.sort_df(df)
        df = self.rename_cols(df)
        if align:
            df = self.align_signs(df, reference=reference)
        if flip_list:
            df = self.flip_networks(df, flip_list)
        return df
    
    def flip_networks(self, df, flip_list=[]):
        for col in df.columns:
            if col in flip_list:
                df[col] = -df[col]
        return df
    
    def rename_cols(self, df):
        df.columns = [os.path.basename(col) for col in df.columns]
        return df
    
    def sort_df(self, df):
        return df.reindex(columns=natsorted(df.columns))     

    def apply_mask_to_dataframe(self, df):
        """
        Apply a mask to a dataframe using either a provided mask or the default MNI ICBM152 mask.
        
        Parameters:
        - df (DataFrame): The dataframe to which the mask should be applied.
        - mask_path (str, optional): The path to the mask image. If not provided, the MNI ICBM152 mask will be used.
        
        Returns:
        - DataFrame: The masked dataframe containing only the rows specified by the mask.
        """
        if self.mask_path is not None:
            brain_indices = np.where(image.load_img(self.mask_path).get_fdata().flatten() > 0)[0]
            masked_df = df.iloc[brain_indices]
        else:
            masked_df = df
        return masked_df

    def align_signs(self, df, reference):
        """
        Align the signs of all brain images in `images_dict` so that the average
        pairwise Pearson's R is (roughly) maximized by flipping signs where needed.
        
        images_dict: dict[str, np.ndarray]
            Keys are dataset names, values are the corresponding brain images.
        
        Returns
        -------
        aligned_dict : dict[str, np.ndarray]
            A copy of `images_dict` with some values possibly flipped (multiplied by -1).
        """
        keys = df.columns
        
        # 1) Build a reference by summing all flattened images
        if reference is None: 
            reference = df.iloc[:, 0]
        elif reference=='sum':
            reference = np.sum(df.iloc[:, :], axis=0)
        else:
            try:
                reference = df.loc[:, [reference]].values.flatten()
            except Exception as e:
                if "None of [Index(['" in str(e):
                    raise ValueError("Invalid reference column. Please provide a valid column name from: \n{}".format(df.columns))
                else:
                    raise ValueError(e)
            
        # 2) Flip sign of each image if it negatively correlates with the reference
        for idx, k in enumerate(keys):
            r_val = np.corrcoef(df.loc[:, k], reference)[0, 1]
            if r_val < 0:
                df.loc[:, k] = -df.loc[:, k]
            else:
                df.loc[:, k] = df.loc[:, k]
        return df

    def calculate_similarity(self, similarity_method='spearman'):
        """
        Calculate the similarity between the dependent and independent variables using the specified method.
        
        Parameters:
        - similarity_method (str): The method to use for calculating similarity. Options are 'spearman' and 'pearson'.
        
        Returns:
        - DataFrame: A dataframe containing the similarity values for each voxel.
        """
        if similarity_method == 'spearman':
            similarity_values = self.calculate_spearman_similarity()
        elif similarity_method == 'pearson':
            similarity_values = self.calculate_pearson_similarity()
        else:
            raise ValueError("Invalid similarity method. Please choose either 'spearman' or 'pearson'.")
        return similarity_values
    
    def compute_correlation(self):
        """
        Compute correlations among columns of a DataFrame using a specified method.
        
        Returns:
        - DataFrame: The correlation matrix.
        """
        indep_var_columns = self.indep_var_df.columns
        dep_var_columns = self.dep_var_df.columns
        self.corr_matrix = np.zeros((len(indep_var_columns), len(dep_var_columns)))
        
        # Get the correlation matrix for each column pair
        for rows in range(0, len(indep_var_columns)):             # iterate over indep_vars
            for cols in range (0, len(dep_var_columns)):
                try: 
                    similarity = self.indep_var_df[indep_var_columns[rows]].corr(self.dep_var_df[dep_var_columns[cols]], method=self.method_choice)
                except Exception as e:
                    if "multiple values for argument 'method'" in str(e):
                        raise ValueError("DataFrame.corr() got multiple values for argument 'method'. \n it is likely your nifti files all have the same filename. Please rename them to unique names and try again. ")
                    else:
                        raise ValueError(e)
                self.corr_matrix[rows, cols] = similarity
        self.corr_df = pd.DataFrame(data=self.corr_matrix, columns=dep_var_columns, index=indep_var_columns)

    def show_heatmap(self):
        fig = plt.figure(figsize=(10, 10))
        ax = plt.gca()
        simple_heatmap(
            self.corr_df,
            dataset_name="",
            out_dir=None,
            output_name="correlation_heatmap.svg",
            ax=ax,
            palette="redblue",
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.5},
        )
        self.save_figure(fig, 'correlation_heatmap.svg')
    
    def show_raincloud_plot(self, prefix):
        """
        Show a raincloud plot (half violin, jittered points, and boxplot) where each row (index) is on the Y-axis,
        and the distribution of similarities for each index is visualized clearly.
        """
        # Sort the index by the highest average value across the row
        self.corr_df['mean_similarity'] = self.corr_df.mean(axis=1)
        sorted_corr_df = self.corr_df.sort_values(by='mean_similarity', ascending=False).drop(columns=['mean_similarity'])

        fig, ax = plt.subplots(figsize=(10, 10))

        melted_corr_df = sorted_corr_df.reset_index().melt(id_vars='index', var_name='Columns', value_name='Similarity')        
        # Half violin plot
        sns.violinplot(data=melted_corr_df, x='Similarity', y='index', ax=ax,
                    scale='width', inner='box', linewidth=1.5, cut=0, orient='h', split=True)

        # Jittered points
        sns.stripplot(data=melted_corr_df, x='Similarity', y='index', ax=ax,
                    size=3, color='black', alpha=0.5, jitter=0.2, orient='h')

        ax.set_title(f'Distribution of {self.xlabel} for Each Map')
        ax.set_xlabel(f'{self.xlabel}')
        ax.set_ylabel('Dataset')

        sns.despine(trim=True)

        self.save_figure(fig, f'{prefix}correlation_raincloud_plot.svg')
        
    def show_barplot_with_error(self, prefix):
        """
        Show a horizontal bar plot with mean and standard error for each row (index) in the correlation dataframe.
    
        Parameters:
        -----------
        prefix : str
            Prefix for the saved plot filename.
        """
        # Calculate mean and standard deviation for each row
        self.corr_df['mean_similarity'] = self.corr_df.mean(axis=1)
        self.corr_df['std_err'] = self.corr_df.sem(axis=1)
    
        # Sort the DataFrame by mean similarity
        sorted_corr_df = self.corr_df.sort_values(by='mean_similarity', ascending=False)
    
        # Create the horizontal bar plot
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.barh(
            sorted_corr_df.index,
            sorted_corr_df['mean_similarity'],
            xerr=sorted_corr_df['std_err'],
            capsize=5,
            color='skyblue',
            edgecolor='black'
        )
    
        # Customize the plot
        ax.set_title(f'Mean and Standard Error of {self.xlabel} for Each Map')
        ax.set_xlabel(f'{self.xlabel}')
        ax.set_ylabel('Index')
        ax.invert_yaxis()  # Invert y-axis to have the highest mean at the top
    
        sns.despine(trim=True)
    
        # Save the figure
        self.save_figure(fig, f'{prefix}correlation_barplot_with_error.svg')
        
    def show_barplot(self, prefix):
        """
        Show a bar plot if there is only one column in the correlation dataframe.
        """
        if self.corr_df.shape[1] != 1:
            raise ValueError("Bar plot can only be created if there is exactly one column in the correlation dataframe.")
        
        fig, ax = plt.subplots(figsize=(15, 13))
        self.corr_df.reset_index().plot.bar(x='index', y=self.corr_df.columns[0], ax=ax, legend=False)
        ax.set_title('Bar Plot of Similarities')
        ax.set_xlabel('Index')
        ax.set_ylabel('Similarity')
        self.save_figure(fig, f'{prefix}correlation_barplot.svg')
    
    def save_figure(self, fig, fname):
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)
            fig.savefig(os.path.join(self.out_dir, fname))

    def get_distribution_plot(self, prefix=''):
        """
        Choose between a violin plot and a bar plot based on the number of columns in the correlation dataframe.
        """
        if self.corr_df.shape[1] == 1:
            self.show_barplot(prefix)
        else:
            self.show_raincloud_plot(prefix)
            self.show_barplot_with_error(prefix)
    
    def get_statistics(self):
        if self.corr_df.shape[1] == 1:
            print("WARNING: Cannot perform ANOVA with only one column in the correlation dataframe. Use permutation test.")
            ptr = self.run_permutation_test()
            return (ptr)
        else:
            ai = self.anova_imbalanced()
            pi = self.posthoc_imbalanced()
            t1 = self.ttest_1samp()
            kw = self.run_kruskal_wallis()
            return (ai, pi, t1, kw)
            
    def anova_imbalanced(self):
        groups = [group.dropna().values for name, group in self.corr_df.iterrows()]
        valid_groups = [g for g in groups if len(g) > 1]

        if len(valid_groups) < 2:
            raise ValueError("Not enough valid groups (with >1 non-NaN values) for ANOVA.")

        f_stat, p_val = f_oneway(*valid_groups)
        print("ANOVA Results:")
        print('F-statistic:', f_stat, 'p-value:', p_val)
        return {'f_stat': f_stat, 'p_val': p_val}
        
    def posthoc_imbalanced(self):
        melted_corr_df = self.corr_df.reset_index().melt(id_vars='index', var_name='Columns', value_name='Similarity').dropna()
        if melted_corr_df['index'].nunique() < 2:
            print("Not enough groups for Post-hoc analysis.")
            return None
        posthoc = pairwise_tukeyhsd(endog=melted_corr_df['Similarity'],
                                    groups=melted_corr_df['index'],
                                    alpha=0.05)
        posthoc_results = pd.DataFrame(data=posthoc.summary().data[1:], columns=posthoc.summary().data[0])
        print("\nPost-hoc Test Results (Imbalanced Data):")
        print(posthoc_results)
        return {'posthoc_results': posthoc_results}

    def run_kruskal_wallis(self):
        """
        Perform Kruskal-Wallis H-test to compare similarities across rows, excluding NaN values.
        
        Returns:
        - kruskal_results (DataFrame): Kruskal-Wallis test results.
        """
        # Melt the correlation dataframe for Kruskal-Wallis test
        melted_corr_df = self.corr_df.reset_index().melt(id_vars='index', var_name='Columns', value_name='Similarity')

        # Perform Kruskal-Wallis test, excluding NaN values for each group
        grouped_data = [group['Similarity'].dropna().values for _, group in melted_corr_df.groupby('index')]
        h_stat, p_value = kruskal(*grouped_data)
        kruskal_results = pd.DataFrame({'H-statistic': [h_stat], 'p-value': [p_value]})
        
        # Print Kruskal-Wallis results
        print("Kruskal-Wallis Results:")
        print(kruskal_results)
        return {'kwt_resutls': kruskal_results}
    
    def run_permutation_test(self):
        """
        Perform a permutation test to compare similarities across rows.
        
        Returns:
        - perm_test_results (DataFrame): Permutation test results.
        """
        print("error: Permutation test not yet implemented.")
        self.statistics_results = (None, None)
        
    def ttest_1samp(self):
        """
        Perform a t-test to check if the similarities are significantly different from zero.
        
        Returns:
        - t_test_results (DataFrame): T-test results for each row.
        """

        # Melt the correlation dataframe for t-test
        melted_corr_df = self.corr_df.reset_index().melt(id_vars='index', var_name='Columns', value_name='Similarity')

        # Perform one-sample t-test for each row
        t_test_results = melted_corr_df.groupby('index').apply(
            lambda group: pd.Series(ttest_1samp(group['Similarity'].dropna(), 0), index=['t-statistic', 'p-value'])
        ).reset_index()

        # Print t-test results
        print("T-Test Results (Differences from Zero):")
        print(t_test_results)
        return {'1s_ttest_results': t_test_results}

    def run(self):
        self.compute_correlation()
        self.show_heatmap()
        self.get_distribution_plot()
        sr = self.get_statistics()
        return sr

class MapComparator(SimilarityTester):
    def __init__(self, data_loader, indep_var_df, mask_path=None, out_dir=None, 
                 method_choice='damage', 
                 align=False, reference=None, flip_list=[]):
        self.data_loader = data_loader
        self.mask_path = mask_path  
        self.method_choice = method_choice
        self.out_dir = out_dir
        self.xlabel = self._setxlabel()
        self.prep_dirs()
        self.indep_var_df = self.prep_data(indep_var_df, reference=reference, flip_list=flip_list)     
        
    def prep_dirs(self):
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)
            
    def get_results(self):
        '''the plotting function expects index to by IV maps, with cols as the observations for that IV, be it spcorr or dmg'''
        if self.method_choice=='damage':
            self.corr_df = self.dep_var_df
        else:
            return self.compute_correlation()
        
    def get_dv_df(self, data_loader):
        '''
        Returns a DF where each column is a map to plot or a series of observations
        '''
        if self.method_choice=='damage':
            return self.get_dmg_df(data_loader)
        else:
            return self.get_corr_df(data_loader)
    
    def _get_patient_map_pair(self, data, n_datasets):
        for dataset_map_idx in range(n_datasets):
            for pt_idx in range(data.shape[0]): #iterates over patients in the dataset
                yield data[pt_idx, :].flatten(), pt_idx, dataset_map_idx
    
    def get_dmg_df(self, data_loader):
        """
        Generate a damage dataframe where each row corresponds to a dataset and each column corresponds to a patient.
        The values represent the dot product between the independent variable map and the patient map.

        Parameters:
        - data_loader: DataLoader object to load datasets and patient maps.

        Returns:
        - dmg_df (DataFrame): A dataframe with datasets as rows and patients as columns.
        """
        # Initialize an empty dictionary to store damage values for each dataset
        dmg_dict = {}

        # Iterate through each dataset
        for dataset in data_loader.dataset_names_list:
            data_dict = data_loader.load_dataset(dataset)
            nifti_data = data_dict['niftis']

            # Initialize an empty list to store damage values for the current dataset
            damage_values = []

            # Iterate through patient maps and compute the dot product with the independent variable map
            for patient, pt_idx, dataset_map_idx in self._get_patient_map_pair(nifti_data, len(data_loader.dataset_names_list)):
                if self.similarity == 'spatial_correlation':
                    damage_values.append( ( pt_idx, (pearsonr(patient, self.iv_map)[0]) ))
                if self.similarity == 'cosine':
                    damage_values.append( ( pt_idx, ( patient.T @ self.iv_map ) / ( np.linalg.norm(patient) * np.linalg.norm(self.iv_map) ) ) )
                if self.similarity == 'sum':
                    damage_values.append( ( pt_idx, ( patient.T @ self.iv_map )) )
                elif self.similarity=='average_sub_in_target':
                    damage_values.append( ( pt_idx, ( patient.T @ self.iv_map ) / len(self.iv_map) ) )
                elif self.similarity=='average_target_in_subject':  
                    damage_values.append( ( pt_idx, ( patient.T @ self.iv_map ) / len(patient) ) )
                else:
                    raise ValueError(f"Invalid similarity method: {self.similarity}. Choose from 'spatial_correlation', 'cosine', 'average_sub_in_target', or 'average_target_in_subject'.")
            # Store the damage values in the dictionary with patient indices as keys
            dmg_dict[dataset] = {pt_idx: dmg for pt_idx, dmg in damage_values}
        # Create a dataframe from the dictionary, filling missing values with NaN
        dmg_df = pd.DataFrame.from_dict(dmg_dict, orient='index').sort_index(axis=1)
        return dmg_df
                    
    def get_corr_df(self, data_loader):
        correlation_calculator = CorrelationCalculator(method=self.method_choice, verbose=False)
        corr_map_dict = correlation_calculator.generate_correlation_maps(data_loader)
        return pd.DataFrame(corr_map_dict)

    def compute_dot_product(self, df1, df2):
        return df1.T.dot(df2)

    def run(self, similarity='average', log_file='stat_results.txt'):
        stats_dict = {}
        self.similarity=similarity
        # Redirect stdout to the log file
        original_stdout = sys.stdout
        log_file_path = os.path.join(self.out_dir, log_file) if self.out_dir else log_file
        
        with open(log_file_path, 'w') as f:
            sys.stdout = f  # Redirect print statements to the file
            try:
                for i in range(self.indep_var_df.shape[1]):
                    print("\n\n==========\nRunning for IV map: ", self.indep_var_df.columns[i])
                    self.iv_map = self.indep_var_df.iloc[:, i]
                    self.dep_var_df = self.get_dv_df(self.data_loader)
                    self.get_results()
                    self.show_heatmap()
                    self.get_distribution_plot(prefix=self.indep_var_df.columns[i] + '_')
                    stat_results_tuple = self.get_statistics()
                    stats_dict[self.indep_var_df.columns[i]] = stat_results_tuple
            finally:
                sys.stdout = original_stdout
                print(f"Log file saved to: {log_file_path}")
        return self.corr_df
