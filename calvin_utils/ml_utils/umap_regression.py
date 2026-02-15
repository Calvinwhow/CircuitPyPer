import os
import numpy as np
import pandas as pd
from calvin_utils.ml_utils.brain_umap import BrainUmap
from calvin_utils.permutation_analysis_utils.voxelwise_regression import VoxelwiseRegression

class UmapRegression:
    def __init__(self, json_path, mask_path, formula, out_dir):
        self.out_dir = out_dir
        self.T = None
        self.cluster_persistence = None
        self.cluster_persistence_perm = None
        self.cluster_persistence_pvals = None
        self.regression = VoxelwiseRegression(json_path, mask_path=mask_path, out_dir=os.path.join(out_dir, "regression_maps"))
        self.umap_params = self._get_umap_params()
        self.dvars = self._get_dep_vars(formula)
        
    ### Setters and Getters ###
    def _get_dep_vars(self, formula):
        lhs = formula.split('~')[0].strip()
        return list(set(lhs.split(" + ")))
    
    def check_contrast_matrix(self):
        """Checks if the contrast matrix is valid for the regression."""
        if self.regression.contrast_matrix.shape[0] != 1:
            raise ValueError("The contrast matrix should only have one row for this analysis.")
        if self.regression.contrast_matrix.shape[1] != self.regression.design_matrix.shape[1]:
            raise ValueError("The contrast matrix should have the same number of columns as the design matrix.")
    
    def _get_umap_params(self):
        """Returns the parameters for the Umap. Abstracted to allow for easy changes for advanced users. See BrainUmap for more details."""
        return {
            'n_components': 3,
            'n_neighbors': 50,
            'min_dist': 0.05,
            'mask': None,
            'min_cluster_size': 10,
            'metric': 'euclidean',
            'projection': None,
            'cluster_voxels': False
        }

    def _get_max_stat(self, arr, pseudo_var_smooth=True, q=99.9):
        """max-stat for 1-D or 2-D input"""
        vals = np.asarray(arr)
        if pseudo_var_smooth:       # 99.9th percentile smooths out single-voxel spikes
            return np.nanpercentile(vals, q) if vals.ndim == 1 else np.nanpercentile(vals, q, axis=1)
        else:                       # raw max statistic
            return np.nanmax(vals) if vals.ndim == 1 else np.nanmax(vals, axis=1)
        
    ### Internal Helpers###
    def _generate_report(self):
        """Generates a report of the results."""
        map_ID = list(np.arange(self.T.shape[-1]))                                        # The map ID for each map, assuming maps are indexed from 0 to n_maps-1             
        cluster_label = list(self.cluster_labels)                                        # The cluster label for each map      
        persistence = [self.cluster_persistence[l] if l >= 0 else np.nan for l in cluster_label]      
        p = [self.cluster_persistence_pvals[l] if l >= 0 else np.nan for l in cluster_label]      
        path = [os.path.join(self.out_dir, f"contrast_1_tval_output_{i}.nii.gz") for i in map_ID]
        df = pd.DataFrame({
            'variable': self.dvars,
            'map_ID': map_ID,
            'cluster_label': cluster_label,
            'cluster_persistance': persistence,
            'cluster_persistence_pval': p,
            'path': path,
        })
        
        return df

    def _generate_umap_figs(self):
        """Generates the Umap figures and saves them to the output directory."""
        fig_full = self.umapper.plot_embedding(verbose=False)
        fig_full.write_html(os.path.join(self.out_dir, 'umap_embedding_full.html'))
        
        try:
            significant_clusters = np.array([
            label if self.cluster_persistence_pvals[label] < 0.05 else -1
            for label in self.cluster_labels
            ])
            override_probabilities = np.where(significant_clusters >= 0, 1.0, 0.1)          # Force significant cluster to full opacity
            fig_filtered = self.umapper.plot_embedding(verbose=False, override_probabilities=override_probabilities)
            fig_filtered.write_html(os.path.join(self.out_dir, 'umap_embedding_filtered.html'))
        except:
            print("Failed to generate filtered umap plot")

    ### Public Methods ###
    def run_umap(self, arr, permutation):
        """Runs the Umap on the entire array, then clusters with HDBSCAN, and returns persistence of the clusters."""
        arr = np.squeeze(arr) # (vox, obs) <- (vox, 1, obs)
        umapper = BrainUmap(arr, **self.umap_params)
        if permutation == False:
            self.umapper = umapper      # Store the true umapper instance for later use
        return umapper.cluster_persistence, umapper.cluster_labels
    
    def run_permutation(self, n_permutations):
        """Runs the regression and Umap on the permuted data, extracting max stat of interest (cluster persistence)."""
        self.cluster_persistence_perm = np.zeros((n_permutations, 1))
        if n_permutations < 1:
            print("Skipping permutations. None requested.")
            return 
        for i in range(n_permutations):
            _, T, _ = self.regression.run_single_multiout_regression(permutation=True)
            persistence, _ = self.run_umap(T, permutation=True)
            self.cluster_persistence_perm[i, :] = self._get_max_stat(persistence)       

    def calc_p_values(self):
        """Calculates one-sided p-values for the cluster persistence."""
        obs = self.cluster_persistence                     # (n_clusters,)
        perm = self.cluster_persistence_perm               # (n_perms, 1)
        self.cluster_persistence_pvals = (perm >= obs).mean(axis=0)
        return self.cluster_persistence_pvals

    def report_results(self):
        """Saves the observed T-maps, creates a CSV with each map's label, its cluster persistence, and persistence p-value."""
        self._generate_umap_figs()
        df = self._generate_report()
        df.to_csv(os.path.join(self.out_dir, 'umap_regression_results.csv'), index=False)
        self.regression._save_nifti_maps()
        print("Umap regression results saved to:", self.out_dir)
        print("CSV report saved to:", os.path.join(self.out_dir, 'umap_regression_results.csv'))
        print("Regression T-maps saved to:", self.out_dir)
        print("Umap embedding figures saved to:", os.path.join(self.out_dir, 'umap_embedding_full.html'))
        
    ### Orchestration Methods ###  
    def run(self, n_permutations=1000):
        '''
        Runs voxelwise regression and tests if there are specific clusters in the maps.
        Strongly suggest only having one row in the contrast matrix.
        '''
        # 1. Run observed
        _, self.T, _ = self.regression.run_single_multiout_regression(permutation=False)
        self.cluster_persistence, self.cluster_labels = self.run_umap(self.T, permutation=False)
        # 2. Run permutations
        self.run_permutation(n_permutations=n_permutations)
        # 3. Get p-values
        self.calc_p_values()
        # 4. Generate report
        self.report_results()