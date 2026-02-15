import os
import numpy as np
import json
from scipy.stats import pearsonr
from tqdm import tqdm
from calvin_utils.ccm_utils.npy_utils import DataLoader

# New class for regression analysis using npy files
class RegressionNPYAnalysis:
    def __init__(self, data_dict_path, n_permutations=1000, out_dir=None):
        """
        Parameters:
          data_dict_path: path to the JSON dictionary produced by your NPY preparer.
          n_permutations: number of permutations for FWE testing.
          out_dir: directory in which to save output maps (optional).
        """
        self.data_dict_path = data_dict_path
        self.n_permutations = n_permutations
        self.out_dir = out_dir
        self.data_loader = DataLoader(data_dict_path)
        # For simplicity, assume one dataset named "main"
        self.dataset_name = list(self.data_loader.dataset_paths_dict.keys())[0]
        data = self.data_loader.load_dataset(self.dataset_name)
        self.X = data['design_matrix']         # shape: (n_subjects, n_predictors)
        self.contrast_matrix = data['contrast_matrix']   # shape: (n_contrasts, n_predictors)
        self.Y = data['niftis']                 # shape: (x, y, z, n_subjects)
        self.n_subjects = self.X.shape[0]
        # Flatten spatial dimensions: reshape Y to (n_subjects, n_voxels)
        self.spatial_shape = self.Y.shape[:-1]
        self.n_voxels = np.prod(self.spatial_shape)
        self.Y_flat = self.Y.reshape(*self.spatial_shape, self.n_subjects)
        self.Y_flat = np.reshape(self.Y, (self.n_subjects, self.n_voxels))
        
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)
    
    def run_regression(self):
        """
        Run voxelwise OLS regression using the design matrix X and outcome Y.
        Returns:
          beta: array of shape (n_predictors, n_voxels)
          t_values: array of shape (n_predictors, n_voxels)
          mse: array of shape (n_voxels,) (mean squared error per voxel)
        """
        # Compute (XᵀX)⁻¹ Xᵀ once:
        XtX_inv = np.linalg.inv(self.X.T @ self.X)  # shape (p, p)
        beta = XtX_inv @ (self.X.T @ self.Y_flat.T)   # shape (p, n_voxels)
        
        # Compute predictions and residuals:
        Y_hat = self.X @ beta  # shape (n_subjects, n_voxels)
        residuals = self.Y_flat.T - Y_hat  # shape (n_subjects, n_voxels)
        dof = self.n_subjects - self.X.shape[1]
        mse = np.sum(residuals**2, axis=0) / dof  # shape (n_voxels,)
        
        # Standard error of beta: sqrt(mse * diag(XtX_inv))
        se_beta = np.sqrt(np.outer(np.diag(XtX_inv), mse))  # shape (p, n_voxels)
        t_values = beta / se_beta  # shape (p, n_voxels)
        
        self.XtX_inv = XtX_inv
        self.beta = beta
        self.t_values = t_values
        self.mse = mse
        return beta, t_values, mse

    def apply_contrast(self):
        """
        For each contrast vector (row of contrast_matrix), compute:
          - Contrast estimate: c @ beta, shape (n_voxels,)
          - Contrast standard error: sqrt(c @ XtX_inv @ c.T * mse)
          - Contrast t-map: contrast_estimate / contrast_se
        Returns:
          contrast_estimates: array of shape (n_contrasts, n_voxels)
          contrast_tmaps: array of shape (n_contrasts, n_voxels)
        """
        n_contrasts = self.contrast_matrix.shape[0]
        contrast_estimates = []
        contrast_tmaps = []
        for i in range(n_contrasts):
            c = self.contrast_matrix[i, :]  # shape (p,)
            contrast_est = c @ self.beta  # (n_voxels,)
            var_factor = c @ self.XtX_inv @ c.T  # scalar
            contrast_se = np.sqrt(var_factor * self.mse)  # (n_voxels,)
            contrast_t = contrast_est / contrast_se
            contrast_estimates.append(contrast_est)
            contrast_tmaps.append(contrast_t)
        contrast_estimates = np.vstack(contrast_estimates)
        contrast_tmaps = np.vstack(contrast_tmaps)
        self.contrast_estimates = contrast_estimates
        self.contrast_tmaps = contrast_tmaps
        return contrast_estimates, contrast_tmaps

    def run_permutation(self):
        """
        Permutation testing: for each permutation, randomly shuffle the rows of X,
        recompute regression and contrast t-values, then record the maximum absolute
        contrast t-value across all voxels and contrasts.
        Returns:
          permuted_max: array of length n_permutations containing the maximum contrast t-value.
        """
        permuted_max = []
        for i in tqdm(range(self.n_permutations), desc="Permutations"):
            perm_indices = np.random.permutation(self.n_subjects)
            X_perm = self.X[perm_indices, :]
            XtX_inv_perm = np.linalg.inv(X_perm.T @ X_perm)
            beta_perm = XtX_inv_perm @ (X_perm.T @ self.Y_flat.T)
            max_t = -np.inf
            for j in range(self.contrast_matrix.shape[0]):
                c = self.contrast_matrix[j, :]
                contrast_est_perm = c @ beta_perm
                var_factor_perm = c @ XtX_inv_perm @ c.T
                # Use original mse for simplicity:
                contrast_se_perm = np.sqrt(var_factor_perm * self.mse)
                contrast_t_perm = contrast_est_perm / contrast_se_perm
                max_t = max(max_t, np.max(np.abs(contrast_t_perm)))
            permuted_max.append(max_t)
        self.permuted_max = np.array(permuted_max)
        return self.permuted_max

    def calculate_p_values(self):
        """
        Compute FWE-corrected p-values for each contrast as the proportion of permutations
        where the maximum contrast t-value exceeds the observed contrast t-value.
        Returns:
          p_values: array of shape (n_contrasts,)
        """
        p_values = []
        for i in range(self.contrast_matrix.shape[0]):
            obs_t = np.abs(self.contrast_tmaps[i, :])
            # For each voxel in contrast i, p = proportion of permuted_max >= obs_t,
            # then average across voxels.
            p_voxel = np.array([np.mean(self.permuted_max >= t) for t in obs_t])
            p_values.append(np.mean(p_voxel))
        self.p_values = np.array(p_values)
        return self.p_values

    def reshape_maps(self):
        """
        Reshape the flattened maps back to the original 3D spatial dimensions.
        Returns:
          beta_maps: (n_predictors, x, y, z)
          t_maps: (n_predictors, x, y, z)
          contrast_tmaps_3d: (n_contrasts, x, y, z)
        """
        x, y, z = self.spatial_shape
        beta_maps = self.beta.reshape(self.X.shape[1], x, y, z)
        t_maps = self.t_values.reshape(self.X.shape[1], x, y, z)
        contrast_tmaps_3d = self.contrast_tmaps.reshape(self.contrast_matrix.shape[0], x, y, z)
        return beta_maps, t_maps, contrast_tmaps_3d

    def run(self):
        # Run the regression to compute beta and t-values for each voxel
        self.run_regression()
        # Apply the contrast matrix to obtain contrast t-maps
        self.apply_contrast()
        # Run permutation testing to obtain a null distribution for the maximum contrast statistic
        self.run_permutation()
        # Compute FWE-corrected p-values for each contrast
        p_vals = self.calculate_p_values()
        return {
            "beta": self.beta,
            "t_values": self.t_values,
            "contrast_estimates": self.contrast_estimates,
            "contrast_tmaps": self.contrast_tmaps,
            "permuted_max": self.permuted_max,
            "p_values": p_vals
        }
