import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from math import log, exp, sqrt
from scipy.stats import spearmanr, pearsonr, rankdata
from calvin_utils.ccm_utils.stat_utils import CorrelationCalculator
from calvin_utils.ccm_utils.convergent_map import ConvergentMapGenerator

class LOOCVAnalyzer(ConvergentMapGenerator):
    def __init__(self, corr_map_dict, data_loader, mask_path=None, out_dir=None, weighting='unweighted', method='spearman', similarity='spatial_correl', n_bootstrap=1000, roi_path=None, group_dict={}, datasets_to_flip = [], align_all_maps=False, flip_axes=False, ylabel=None):
        """
        Initialize the LOOCVAnalyzer.

        Parameters
        ----------
        corr_map_dict : dict
            Dictionary of correlation maps for each dataset.
        data_loader : DataLoader
            Loader for dataset access.
        mask_path : str, optional
            Path to mask file.
        out_dir : str, optional
            Output directory for results.
        weighting : str, optional
            Weighting method: 'unweighted', 'weighted', or 'optimized'.
        method : str, optional
            Correlation method: 'spearman' or 'pearson'.
        similarity : str, optional
            Similarity metric: 'spatial_correl', 'cosine', 'avg_in_subject', or 'avg_in_target'.
        n_bootstrap : int, optional
            Number of bootstrap samples.
        roi_path : str, optional
            Path to ROI file for analysis.
        group_dict : dict, optional
            Maps dataset names to group labels.
        datasets_to_flip : list, optional
            List of dataset names to flip map sign.
        align_all_maps : bool, optional
            If True, aligns all maps to positive correlation.
        flip_axes : bool, optional
            If True, flips axes in scatterplots.
        """
        super().__init__(corr_map_dict, data_loader, mask_path, out_dir, weighting, align_all_maps=align_all_maps, group_dict=group_dict, datasets_to_flip=datasets_to_flip)
        self.method = method
        self.n_bootstrap = n_bootstrap
        self.similarity = similarity
        self.roi_path = roi_path
        self.flip_axes = flip_axes
        self.ylab = ylabel
        self.correlation_calculator = CorrelationCalculator(method=method, verbose=False, use_jax=False)
        self.label_dict  = {'cosine': 'Cosine Similarity', 'spatial_correl': 'Spatial Correlation', 'avg_in_subject': 'Average Damage', 'avg_in_target': 'Average Damage'}
        self.observed_x1 = None
        self.observed_x2 = None
        self.observed_y  = None
        
    ### Helpers ###
    def _cosine_similarity(self, a, b):
        """Calculate the cosine similarity between two vectors."""
        a = a.flatten()
        b = b.flatten()
        numerator = np.dot(a, b)
        denominator = np.sqrt(np.sum(a**2)) * np.sqrt(np.sum(b**2))
        similarity = numerator / denominator
        return similarity
    
    def _pearson(self, arr, convergent_map):
        return pearsonr(arr.flatten(), convergent_map.flatten())[0] 
    
    def _dot_in_arr(self, arr, convergent_map):
        return np.dot(arr.flatten(), convergent_map.flatten()) / np.count_nonzero(~np.isnan(arr.flatten()) & (arr.flatten() !=0))
    
    def _dot_in_cvgt(self, arr, convergent_map):
        return np.dot(arr.flatten(), convergent_map.flatten()) / np.count_nonzero(~np.isnan(convergent_map.flatten()) & (convergent_map.flatten() !=0))

    ### I/O ###
    def results_to_dataframe(self):
        """Convert the LOOCV results to a pandas DataFrame."""
        columns = ['Dataset', 'CI Lower', 'CI Upper', 'Mean R']
        data = []
        dataset_names = list(self.corr_map_dict.keys())
        for i, dataset_name in enumerate(dataset_names):
            dataset_name
            ci_lower, ci_upper, mean_r = self.results[i]
            data.append([dataset_name, ci_lower, ci_upper, mean_r])
        return pd.DataFrame(data, columns=columns)

    def generate_convergent_roi(self):
        """Load and prep the ROI file"""
        roi_data = self._load_nifti(self.roi_path)
        roi_data = self._mask_array(roi_data)
        return roi_data

    def _align_correlations(self, ci_lower, ci_upper, mean_r):
        '''Reorients the correlations to be positive if forcibly aligning maps'''
        if self.align_all_maps:
            if mean_r < 0:
                stored_ci = ci_upper
                ci_upper = ci_lower * -1
                ci_lower = stored_ci * -1
                mean_r = mean_r * -1
        return ci_lower, ci_upper, mean_r

    def generate_correlation_maps(self, dataset_names):
        """Generate correlation maps for the given dataset names"""
        correlation_maps = {}
        for dataset_name in dataset_names:
            data = self.data_loader.load_dataset(dataset_name)
            self.correlation_calculator._process_data(data)
            correlation_maps[dataset_name] = self.correlation_calculator.correlation_map
        return self._handle_datasets(correlation_maps) # ensure necessary maps are flipped as needed
                
    ### Plotting API ###
    def _generate_scatterplot(self, similarities, indep_var, dataset_name):
        """
        Generate and save a scatterplot of similarity vs. outcome with correlation.

        Parameters:
        -----------
        similarities : list of float
            List of cosine similarity values (X-axis).
        indep_var : np.array
            Array of independent variable values (Y-axis).
        dataset_name : str
            Name of the dataset (used for title and filename).
        """
        indep_var = indep_var.flatten()
        rho, p = spearmanr(similarities, indep_var)
        r, pr = pearsonr(similarities, indep_var)

        # Create DataFrame for Seaborn
        df = pd.DataFrame({"Similarity": similarities, "Outcome": indep_var})
        
        # Create LM plot
        plt.figure(figsize=(6, 6))
        if self.flip_axes:
            sns.lmplot(data=df, x="Outcome", y="Similarity", height=6, aspect=1, 
               scatter_kws={'alpha': 0.98, 'color': '#8E8E8E', 's': 150, 'edgecolors': 'white', 'linewidth': 2, 'zorder': 3}, 
               line_kws={'color': '#8E8E8E', 'zorder': 2})
        else:
            sns.lmplot(data=df, x="Similarity", y="Outcome", height=6, aspect=1, 
               scatter_kws={'alpha': 0.98, 'color': '#8E8E8E', 's': 150, 'edgecolors': 'white', 'linewidth': 2, 'zorder': 3}, 
               line_kws={'color': '#8E8E8E', 'zorder': 2})

        # Labels
        plt.title(f"{dataset_name}", fontsize=20)
        
        xlab = self.label_dict[self.similarity] if self.similarity in self.label_dict.keys() else 'Damage'
        ylab = 'Outcome' if self.ylab is None else self.ylab
        if self.flip_axes:
            ylab = xlab
            xlab = 'Outcome'
        plt.xlabel(xlab, fontsize=20)
        plt.ylabel(ylab, fontsize=20)

        # Dynamically place stats inside the plot
        x_pos = 0.05 if rho > 0 else 0.05
        y_pos = 0.95 if rho > 0 else 0.15
        plt.text(
            x_pos, y_pos, 
            f"Rho = {rho:.2f}, p = {p:.2e}\nR = {r:.2f}, p = {pr:.2e}",
            fontsize=16, 
            transform=plt.gca().transAxes, 
            verticalalignment='top', 
            bbox=dict(facecolor='white', alpha=0, edgecolor='none')
        )
        
        # Set font sizes and line thickness for the scatterplot
        ax = plt.gca()
        ax.tick_params(axis='both', labelsize=16)
        for spine in ax.spines.values():
            spine.set_linewidth(2)

        # Save plot
        os.makedirs(self.out_dir+'/scatterplots', exist_ok=True)
        plt.savefig(os.path.join(self.out_dir, f"scatterplots/{dataset_name}_scatterplot.svg"), bbox_inches="tight")
        plt.show()
        
    ### LOOCV Evaluation Loop ###
    def _prepare_train_dataset(self, i, dataset_names_list):
        '''Return datasets not part of test data'''
        return dataset_names_list[:i] + dataset_names_list[i+1:]    # Return groups that aren't the test group
            
    def _get_test_data(self, test_dataset_name):
        """TEST - Load test data"""
        test_data = self.data_loader.load_dataset(test_dataset_name)
        test_niftis = CorrelationCalculator._check_for_nans(test_data['niftis'], nanpolicy='remove', verbose=False)
        test_indep_var = CorrelationCalculator._check_for_nans(test_data['indep_var'], nanpolicy='remove', verbose=False)
        return test_niftis, test_indep_var

    def _train_map(self, train_dataset_names, test_dataset_name):
        '''TRAIN - Generate the convergent map using the training datasets (or an ROI)'''
        if self.roi_path is not None:
            print("No training required--loading user-defined ROI file")
            convergent_map = self.generate_convergent_roi()
        elif self.similarity == 'spatial_correl':
            print("Training on: ", train_dataset_names)
            print("Testing on held-out datset: ", test_dataset_name)
            local_corr_map_dict = self.generate_correlation_maps(train_dataset_names)
            convergent_map = self._select_weighted_map(local_corr_map_dict)
        elif self.similarity == 'cosine':
            print("Training on: ", train_dataset_names)
            print("Testing on held-out datset: ", test_dataset_name)
            local_corr_map_dict = self.generate_correlation_maps(train_dataset_names)
            convergent_map = self.generate_agreement_map(local_corr_map_dict)
        else:
            raise ValueError("Invalid similarity type (self.similarity). Please choose 'spatial_correl', 'cosine', or set path to a region of interest to test (self.roi_path).")
        return CorrelationCalculator._check_for_nans(convergent_map, nanpolicy='remove', verbose=False)

    def calculate_similarity(self, test_arr, convergent_map):
        """Calculate cosine similarity between patient maps and the convergent map."""        
        if self.similarity == 'cosine':
            similarities = [self._cosine_similarity(arr, convergent_map)for arr in test_arr]
        elif self.similarity == 'spatial_correl':
            similarities = [self._pearson(arr, convergent_map) for arr in test_arr]
        elif self.similarity == 'avg_in_subject':
            similarities = [self._dot_in_arr(arr, convergent_map) for arr in test_arr]
        elif self.similarity == 'avg_in_target':
            similarities = [self._dot_in_cvgt(arr, convergent_map) for arr in test_arr]
        else:
            raise ValueError("Invalid similarity measure (self.similarity). Please choose 'cosine', 'spatial_correl', 'avg_in_subject', 'avg_in_convergent'")
        return similarities
        
    ### Public API ###
    def perform_loocv(self, forest_plot):
        """Perform Leave-One-Out Cross-Validation (LOOCV) analysis and return scatterplot"""
        results = []
        dataset_names = list(self.corr_map_dict.keys())
        for i, test_dataset_name in enumerate(dataset_names):
            train_dataset_names = self._prepare_train_dataset(i, dataset_names)    
            test_niftis, test_indep_var = self._get_test_data(test_dataset_name)
            convergent_map = self._train_map(train_dataset_names, test_dataset_name)
            similarities = self.calculate_similarity(test_niftis, convergent_map)
            if not forest_plot: self._generate_scatterplot(similarities, test_indep_var, test_dataset_name)
                
            if forest_plot:
                ci_lower, ci_upper, mean_r = self.correlate_similarity_with_outcomes(similarities, test_indep_var, test_dataset_name)
                ci_lower, ci_upper, mean_r = self._align_correlations(ci_lower, ci_upper, mean_r)
                results.append((ci_lower, ci_upper, mean_r))
        return results
    
    def run(self):
        self.results = self.perform_loocv(forest_plot=False)
    
    ### Map/ROI Comparison Module ###
    def _shuffle_for_roi_comparison(self, niftis, indep_var, method):
        if method == "bootstrap":
            idx = np.random.choice(len(niftis), size=len(niftis), replace=True)
            return niftis[idx], indep_var[idx].flatten()
        elif method == "permutation":
            idx = np.random.permutation(len(indep_var))
            return niftis, indep_var[idx].flatten()
        elif method == "observed":
            return niftis, indep_var.flatten()
        else:
            raise ValueError("Invalid method.")
    
    def _compute_r_differences(self, roi1_map, roi2_map, dataset_names, method, n_iter, delta_r2, spearman=True):
        """If given multiple datasets, this will average each resample across datasets"""
        all_r_diffs = np.zeros((n_iter, len(dataset_names)))                                # prefille to enable insertion
        for idx, dataset_name in enumerate(dataset_names):
            r_diffs = []
            iter_count = 0
            with tqdm(total=n_iter) as pbar:
                while iter_count < n_iter:
                    data = self.data_loader.load_dataset(dataset_name)
                    test_niftis = CorrelationCalculator._check_for_nans(data['niftis'], nanpolicy='remove', verbose=False)
                    test_indep_var = CorrelationCalculator._check_for_nans(data['indep_var'], nanpolicy='remove', verbose=False)
                    sub_niftis, sub_indep_var = self._shuffle_for_roi_comparison(test_niftis, test_indep_var, method)
                    
                    if self.similarity == 'cosine':
                        sim1 = [self._cosine_similarity(n, roi1_map) for n in sub_niftis]
                        sim2 = [self._cosine_similarity(n, roi2_map) for n in sub_niftis]
                    else:
                        sim1 = [pearsonr(n.flatten(), roi1_map.flatten())[0] for n in sub_niftis]
                        sim2 = [pearsonr(n.flatten(), roi2_map.flatten())[0] for n in sub_niftis]
                        
                    if spearman:
                        stat1 = spearmanr(sim1, sub_indep_var, nan_policy='omit')[0]
                        stat2 = spearmanr(sim2, sub_indep_var, nan_policy='omit')[0]
                    else:
                        mask1 = ~np.isnan(sim1) & ~np.isnan(sub_indep_var)
                        mask2 = ~np.isnan(sim2) & ~np.isnan(sub_indep_var)
                        stat1 = pearsonr(np.array(sim1)[mask1], np.array(sub_indep_var)[mask1])[0]
                        stat2 = pearsonr(np.array(sim2)[mask2], np.array(sub_indep_var)[mask2])[0]
                        
                    if np.isnan(stat1) or np.isnan(stat2):
                        continue  # skip this iteration and do not increment iter_count
                    if delta_r2:
                        stat1 = stat1 ** 2
                        stat2 = stat2 ** 2
                    
                    delta = stat1 - stat2
                    r_diffs.append(delta)
                    for roi, stat in zip(['roi1', 'roi2'], [stat1, stat2]):
                        if method=='observed':      # careful not to overwrite existing self.r_values from resampling.
                            self.observed_r_values[roi].append(stat)
                            self.observed_x1 = np.array(sim1)
                            self.observed_x2 = np.array(sim2)
                            self.observed_y  = np.array(sub_indep_var)
                        else:
                            self.r_values[roi].append(stat)
                    iter_count += 1
                    pbar.update(1)
            all_r_diffs[:, idx] = r_diffs                                               # assign all the r_diffs to this this column
        return all_r_diffs                                                              # shape (resamples, datasets)
        
    def _calculate_probability(self, all_r_diffs, roi1_map, roi2_map, dataset_names, method, delta_r2):
        """Calculate the probability based on the method (bootstrap or permutation)"""
        print(f"Below results used delta explained variance (r-squared): {delta_r2}")
        observed_diffs = self._compute_r_differences(roi1_map, roi2_map, dataset_names, method='observed', n_iter=1, delta_r2=delta_r2) # shape: (1, datasets)
        if method == "bootstrap":                               # Overall bootstrap p-value and average delta ---
            overall_pval = np.mean(all_r_diffs > 0)             # shape (1, datasets), where value in each row corresponds to significance of that dataset
            print(f"Overall bootstrap:\n    avg delta R = {np.mean(all_r_diffs):.4f}, Probability ROI 1 is generally superior = {overall_pval:.4f}")
        elif method == "permutation":                           # Overall permutation p-value and average delta 
            overall_pval = np.mean(all_r_diffs > observed_diffs)        # shape (resamples, datasets)
            avg_observed = np.mean(observed_diffs)                      # shape (1, datasets), where value in each row corresponds to significance of that dataset
            print(f"Overall permutation:\n     observed avg delta R = {avg_observed:.4f}, p = {overall_pval:.4f}")
        return overall_pval, observed_diffs
    
    def compare_roi_correlations(self, roi1, roi2, method="bootstrap", n_iter=10000, delta_r2=True, seed=None):
        """Compare the correlation of two ROI maps with outcome variables using bootstrap or permutation test"""
        self.roi_path = roi1
        roi1_map = self.generate_convergent_roi()
        self.roi_path = roi2
        roi2_map = self.generate_convergent_roi()
        
        dataset_names = list(self.corr_map_dict.keys())
        self.r_values = {'roi1': [], 'roi2': []}
        self.observed_r_values = {'roi1': [], 'roi2': []}
        
        all_r_diffs = self._compute_r_differences(roi1_map, roi2_map, dataset_names, method, n_iter, delta_r2)
        self.prob = self._calculate_probability(all_r_diffs, roi1_map, roi2_map, dataset_names, method, delta_r2)
        
        
