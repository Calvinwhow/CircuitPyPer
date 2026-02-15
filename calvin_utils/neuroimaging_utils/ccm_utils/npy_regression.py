import os
import numpy as np
from tqdm import tqdm
from calvin_utils.ccm_utils.npy_utils import DataLoader
import nibabel as nib
from nilearn import plotting


# New regression analysis class with FWE correction
class RegressionNPYAnalysis:
    def __init__(self, data_dict_path, n_permutations=1000, out_dir=None, fwe=False, max_stat_method=None, verbose=True, mask_path=None):
        """
        Parameters:
          data_dict_path: path to the JSON file produced by the NPY preparer.
          n_permutations: number of permutations to perform.
          out_dir: output directory (optional).
          fwe: bool, if True, p-values are computed using FWE correction (max-stat method).
          max_stat_method: str or None. If None, uses raw maximum statistic;
                           if "pseudo_var_smooth", uses the 99.99th percentile method;
                           if "var_smooth", would use a variance-smoothing method (not implemented).
        """
        self.data_dict_path = data_dict_path
        self.n_permutations = n_permutations
        self.out_dir = out_dir
        self.fwe = fwe
        self.max_stat_method = max_stat_method  # e.g., None or "pseudo_var_smooth"
        self.verbose = verbose
        self.data_loader = DataLoader(data_dict_path)
        self.mask_path = mask_path
        self._load_data()
        self._set_variables()
        self._prep_out_dir()
        self._readouts()
        
    def _load_data(self):
        self.dataset_name = list(self.data_loader.dataset_paths_dict.keys())[0]
        self.data = self.data_loader.load_regression_dataset(self.dataset_name)
        
    def _prep_out_dir(self):
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)
        
    def _set_variables(self):
        self.X = self.data['design_matrix']                     # shape: (n_subjects, n_predictors)
        self.Y = self.data['niftis']                            # shape: (n_voxels, n_subjects)
        self.contrast_matrix = self.data['contrast_matrix']     # shape: (n_contrasts, n_predictors)
        self.n_subjects = self.Y.shape[1]                       # shape: (n_subjects, )
        self.n_voxels = self.Y.shape[0]                         # shape: (voxels, )
        
    def _readouts(self):
        print(f'There are {self.n_subjects} subjects')
        print(f'There are {self.X.shape[1]} covariates')
        print(f'There are {self.n_voxels} voxels')
        print(f'X (observations, covariates): {self.X.shape}')
        print(f'Y (voxels, observations): {self.Y.shape}')
        print(f'\n Contrast matrix: \n f{self.contrast_matrix} \n')
        print(f"Using Family-wise error correction: {self.fwe} with method: {self.max_stat_method}")
        print(f"\n Saving results to {self.out_dir} \n")
    
    #### p value methods ####
    
    def raw_max_stat(self, arr):
        """Return the maximum absolute statistic from arr."""
        return np.max(np.abs(arr))
    
    def pseudo_var_smooth(self, arr):
        """Return the 99.99th percentile of the absolute values in arr."""
        return np.percentile(np.abs(arr), 99.99)
    
    def var_smooth(self, arr):
        """Not implemented."""
        raise ValueError("Function not yet complete.")
    
    def get_max_stat(self, arr):
        """Choose the max stat method based on self.max_stat_method."""
        # Remove NaNs (if any) from arr:
        arr = np.nan_to_num(arr)
        if self.max_stat_method is None:
            return self.raw_max_stat(arr)
        elif self.max_stat_method == "pseudo_var_smooth":
            return self.pseudo_var_smooth(arr)
        elif self.max_stat_method == "var_smooth":
            return self.var_smooth(arr)
        else:
            raise ValueError("Invalid max_stat_method.")

    def calculate_voxelwise_fwe_p_values(self):
        """
        Compute p-values for each contrast and each voxel using per-contrast null distributions.
        Defaults to 2-sample t test.
        """
        n_contrasts, n_voxels = self.contrast_tmaps.shape
        p_values = np.zeros((n_contrasts, n_voxels))

        for i in range(n_contrasts):  # Iterate per contrast
            for j in range(n_voxels):
                obs_val = np.abs(self.contrast_tmaps[i, j])
                p = np.mean(self.contrast_max_stats[i, :] >= obs_val)  # Compare to contrast-specific null
                p_values[i, j] = p

        self.voxelwise_p_values = p_values
        return p_values

    ### regression methods ###
    
    def run_regression(self, X, Y):
        """
        Runs voxelwise OLS regression:
          β = (XᵀX)⁻¹ Xᵀ Y
        Returns:
          beta: (n_predictors, n_voxels)
          t_values: (n_predictors, n_voxels)
          mse: (n_voxels,)
        """
        
        XtX_inv = np.linalg.pinv(X.T @ X)                               # (n_cov, n_cov) <- (n_cov, n_sub) @ (n_sub, n_cov)
        XtX_invY = X.T @ Y.T                                            # (n_cov, n_voxels) <- (n_cov, n_sub) @ (n_sub, n_voxels)
        beta = XtX_inv @ XtX_invY                                       # (n_cov, n_voxels) <- (n_cov, n_cov) @ (n_cov, n_voxels)
        Y_hat = X @ beta                                                # (n_sub, n_voxels) <- (n_sub, n_cov) @ (n_cov, n_voxels)
        residuals = Y.T - Y_hat                                         # (n_sub, n_voxels) <- (n_sub, n_voxels) - (n_sub, n_voxels)
        dof = self.n_subjects - X.shape[1]                              # scalar  <- (n_sub, ) - (n_cov, )
        mse = np.sum(residuals**2, axis=0) / dof                        # (n_voxels,) <- summed (n_sub, n_voxels) along n_sub
        se_beta = np.sqrt(np.outer(np.diag(XtX_inv), mse))              # (n_cov, n_voxels) <- (n_cov, ) @ (n_voxels, ) [tr(n_cov, n_cov) -> (n_cov, )]
        t_values = beta / se_beta                                       # (n_cov, n_voxels) <- (n_cov, n_voxels) / (n_cov, n_voxels)
        
        if self.verbose:
            print(f"XtX_inv shape: {XtX_inv.shape}")         # (p, p)
            print(f"beta shape: {beta.shape}")               # (p, n_voxels)
            print(f"Y_hat shape: {Y_hat.shape}")             # (n_subjects, n_voxels)
            print(f"residuals shape: {residuals.shape}")     # (n_subjects, n_voxels)
            print(f"dof shape: {dof}")                       # Scalar (not an array)
            print(f"mse shape: {mse.shape}")                 # (n_voxels,)
            print(f"se_beta shape: {se_beta.shape}")         # (p, n_voxels)
            print(f"t_values shape: {t_values.shape}")       # (p, n_voxels)
            print(f"self.XtX_inv shape: {XtX_inv.shape}")       # (p, p)
            print(f"self.beta shape: {beta.shape}")         # (p, n_voxels)
            print(f"self.t_values shape: {t_values.shape}") # (p, n_voxels)
            print(f"self.mse shape: {mse.shape}")           # (n_voxels,)
        return beta, t_values, mse, XtX_inv
    
    def run_permutation(self):
        """
        For each permutation, randomly shuffle the rows of X (breaking the link to Y),
        recompute regression and contrast t-maps, then record one maximum statistic (scalar)
        across all contrasts and voxels using get_max_stat.
        Returns:
          permuted_max: array of shape (n_permutations,)
        """
        permuted_max = np.zeros((self.contrast_matrix.shape[0], self.n_permutations))           # (n_contrasts, n_perms) <- stores the max contrast value for each contrast
        for i in tqdm(range(self.n_permutations), desc="Permutations"):
            perm_indices = np.random.permutation(self.n_subjects)
            X_perm = self.X[perm_indices, :]
            beta, t_values, mse, XtX_inv = self.run_regression(X_perm, self.Y)
            contrast_estimates, contrast_tmaps = self.apply_contrast(beta, XtX_inv, mse)
            
            for c in range(contrast_tmaps.shape[0]):
                permuted_max[c,i] = self.get_max_stat(contrast_tmaps[c,i])        # extract the max start per contrast map
        return permuted_max

    def apply_contrast(self, beta, XtX_inv, mse):
        """
        Applies the contrast matrix (each row is a contrast vector).
        For each contrast:
          contrast_est = c @ beta, and
          contrast_se = sqrt(c @ XtX_inv @ c.T * mse)
          t_map = contrast_est / contrast_se
        Returns:
          contrast_estimates: (n_contrasts, n_voxels)
          contrast_tmaps: (n_contrasts, n_voxels)
        """
        n_contrasts = self.contrast_matrix.shape[0]                 # (n_contrasts,)
        contrast_estimates = []                                     
        contrast_tmaps = []                                         
        for i in range(n_contrasts):
            c = self.contrast_matrix[i, :]                          # (n_cov, )
            contrast_est = c @ beta                                 # (n_voxels, ) <- (n_cov, 1) @ (n_cov, n_voxels)
            var_factor = c @ XtX_inv @ c.T                          # scalar       <- (n_cov, ) @ (n_cov, n_cov) @ (, n_cov)
            contrast_se = np.sqrt(var_factor * mse)                 # (n_voxels,)
            contrast_t = contrast_est / contrast_se                 # (n_voxels,)
            
            contrast_estimates.append(contrast_est)                 # standard error of the contrast
            contrast_tmaps.append(contrast_t)                       # t value of the contrast
            
        contrast_estimates = np.vstack(contrast_estimates)          # (n_contrasts, n_voxels)
        contrast_tmaps = np.vstack(contrast_tmaps)                  # (n_contrasts, n_voxels)
        
        if self.verbose:
            print(f"n_contrasts shape: {self.contrast_matrix.shape[0]}")  # Scalar (not an array)
            print(f"contrast_estimates (initial) shape: {len(contrast_estimates)}")  # Empty list
            print(f"contrast_tmaps (initial) shape: {len(contrast_tmaps)}")  # Empty list
            print(f"c shape (contrast vector {i}): {c.shape}")  # (n_cov,)
            print(f"contrast_est shape (contrast {i}): {contrast_est.shape}")  # (1, n_voxels)
            print(f"var_factor shape (contrast {i}): {var_factor.shape}")  # Scalar (not an array)
            print(f"contrast_se shape (contrast {i}): {contrast_se.shape}")  # (n_voxels,)
            print(f"contrast_t shape (contrast {i}): {contrast_t.shape}")  # (n_voxels,)
            print(f"contrast_estimates (final) shape: {np.vstack(contrast_estimates).shape}")  # (n_contrasts, n_voxels)
            print(f"contrast_tmaps (final) shape: {np.vstack(contrast_tmaps).shape}")  # (n_contrasts, n_voxels)
        return contrast_estimates, contrast_tmaps

    #### orchestration methods ####
    
    def run(self):
        """
        Run the full analysis:
         1. Run regression to get beta and voxelwise t-values.
         2. Apply contrasts to get contrast t-maps.
         3. If FWE correction is desired (self.fwe=True), run permutation testing
            to obtain a null distribution of max statistics.
         4. Compute voxelwise FWE-corrected p-values.
        Returns:
          results: dictionary containing beta, t_values, contrast estimates, contrast t-maps,
                   permutation null distribution, and voxelwise p-values.
        """
        self.beta, self.t_values, self.mse, self.XtX_inv = self.run_regression(self.X, self.Y)
        self.contrast_estimates, self.contrast_tmaps = self.apply_contrast(self.beta, self.XtX_inv, self.mse)
        self.contrast_max_stats = self.run_permutation()
        p_vals = self.calculate_voxelwise_fwe_p_values()
        
        results = {
        "beta": self.beta,
        "t_values": self.t_values,
        "contrast_estimates": self.contrast_estimates,
        "contrast_tmaps": self.contrast_tmaps,
        "permuted_max": self.contrast_max_stats if self.fwe else None,
        "voxelwise_p_values": p_vals,
        }
        return results
    
    #### Masking and Unmasking Methods ####
    
    def _unmask_array(self, data_array):
        """
        Unmasks a vectorized image to full-brain shape using self.mask_path.
        Returns:
            unmasked_array: full-brain NIfTI-like array
            mask_affine: affine transformation from mask
        """
        if self.mask_path is None:
            raise ValueError("Mask path is not provided. Provide the mask used in Notebook 05")
        else:
            mask = nib.load(self.mask_path)

            mask_data = mask.get_fdata()
            mask_indices = mask_data.flatten() > 0  # Assuming mask is binary
            
            unmasked_array = np.zeros(mask_indices.shape)
            unmasked_array[mask_indices] = data_array.flatten()
        return unmasked_array.reshape(mask_data.shape), mask.affine

    def _save_map(self, map_data, file_name):
        """
        Saves unmasked NIfTI image to disk.
        """
        if self.out_dir is None:
            return
        
        unmasked_map, mask_affine = self._unmask_array(map_data)
        img = nib.Nifti1Image(unmasked_map, affine=mask_affine)
        
        file_path = os.path.join(self.out_dir, file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        nib.save(img, file_path)
        return img

    def _visualize_map(self, img, title):
        """
        Opens a NIfTI image in a browser.
        """
        try:
            plotting.view_img(img, title=title).open_in_browser()
        except:
            pass  # Ignore failures in headless environments

    #### Save and Visualize Methods ####
    
    def save_and_visualize_results(self, verbose=False):
        """
        Saves and optionally visualizes:
        - Regression coefficients per predictor
        - Contrast t-maps
        - FWE-significant contrast t-maps (thresholded at p < 0.05)
        - P-value maps

        If self.data does not contain a 'predictor_names' key, default names will be generated.
        """
        if self.out_dir is None:
            return

        print("Saving results...")

        # Use predictor names if available, otherwise generate defaults.
        predictor_names = self.data.get('predictor_names', 
                                        [f"predictor_{i}" for i in range(self.X.shape[1])])
        # Save regression coefficients
        for i, predictor in enumerate(predictor_names):
            beta_img = self._save_map(self.beta[i, :], f'coefficients/{predictor}_beta.nii.gz')
            if verbose:
                self._visualize_map(beta_img, f"{predictor} Coefficient Map")

        # Save contrast t-maps, FWE-significant t-maps, and p-value maps
        for i in range(self.contrast_matrix.shape[0]):
            contrast_name = f"contrast_{i}"
            t_map_img = self._save_map(self.contrast_tmaps[i, :], f'tmaps/{contrast_name}_tmap.nii.gz')
            if verbose:
                self._visualize_map(t_map_img, f"{contrast_name} T-Map")

            # FWE-significant t-map (using a threshold of p < 0.05)
            fwe_mask = self.voxelwise_p_values[i, :] < 0.05
            significant_tmap = np.where(fwe_mask, self.contrast_tmaps[i, :], 0)
            fwe_tmap_img = self._save_map(significant_tmap, f'tmaps/{contrast_name}_tmap_FWE.nii.gz')
            if verbose:
                self._visualize_map(fwe_tmap_img, f"{contrast_name} FWE T-Map")

            # P-value map
            p_map_img = self._save_map(self.voxelwise_p_values[i, :], f'pvals/{contrast_name}_pval.nii.gz')
            if verbose:
                self._visualize_map(p_map_img, f"{contrast_name} P-Value Map")

        print("All results saved.")
