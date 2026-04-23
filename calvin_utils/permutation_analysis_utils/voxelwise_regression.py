import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import nibabel as nib
from scipy.stats import t
import statsmodels.api as sm
from scipy.special import expit
from calvin_utils.file_utils.import_functions import GiiNiiFileImport
from calvin_utils.statistical_utils.scatterplot import simple_scatter
from calvin_utils.neuroimaging_utils.ccm_utils.npy_utils import DataLoader
from calvin_utils.neuroimaging_utils.nifti_utils.damage_score_utils import DamageScorer
from calvin_utils.neuroimaging_utils.output_functions import NeuroimageFileOutporter

class VoxelwiseRegression:
    """
    VoxelwiseRegression
    A class for performing voxelwise linear regression analysis on neuroimaging data, supporting permutation-based inference and NIfTI output.
    json_path : str
        Path to a JSON file specifying the locations of input data arrays (design matrix, outcome data, contrasts, weights, etc.).
    mask_path : str, optional
        Path to a NIfTI mask file used for unmasking and saving results in brain space.
    out_dir : str, optional
        Directory where output NIfTI images and results will be saved.
    Attributes
    json_path : str
        Path to the JSON configuration file.
    mask_path : str or None
        Path to the NIfTI mask file.
    out_dir : str or None
        Output directory for saving results.
    data_loader : DataLoader
        Loader for input data specified in the JSON file.
    design_tensor : np.ndarray
        Design matrix tensor (observations × predictors × voxels).
    outcome_tensor : np.ndarray
        Outcome data tensor (observations × regressions × voxels).
    contrast_matrix : np.ndarray
        Contrast matrix (contrasts × predictors).
    exchangeability_blocks : np.ndarray or None
        Exchangeability block labels for permutation testing.
    weight_vector : np.ndarray
        Weights for each observation.
    n_obs : int
        Number of observations.
    n_preds : int
        Number of predictors.
    dim3_X : int
        Number of voxels in the design tensor.
    dim3_Y : int
        Number of voxels in the outcome tensor.
    n_contrasts : int
        Number of contrasts.
    n_voxels : int
        Number of voxels (max of design and outcome).
    n_outputs : int
        Number of output channels in the outcome tensor.
    Methods
    -------
    load_data()
        Loads design, outcome, contrast, weights, and exchangeability block data from files.
    set_variables()
        Sets and returns key shape variables for the regression.
    _get_targets(permutation)
        Returns regressor, regressand, and weights, optionally permuting the outcome data.
    _prep_targets(regressor, regressand, weights, voxel_idx, regression_idx=0)
        Prepares X, Y, and W matrices for regression at a given voxel and output index.
    get_r2(Y, Y_HAT, W, e=1e-6)
        Computes R-squared for model fit.
    apply_contrasts(XtX_inv, BETA, MSE, e=1e-6, get_p=False)
        Applies contrast matrix to regression coefficients to compute t-values.
    _run_regression(X, Y, W)
        Runs weighted linear regression for a single voxel.
    voxelwise_regression(permutation=False)
        Performs voxelwise regression across all voxels, optionally with permutation.
    _get_max_stat(arr, pseudo_var_smooth=True, t=99.99)
        Computes the maximum (or high percentile) statistic for permutation testing.
    run_permutation(n_permutations)
        Runs permutation testing to compute FWE-corrected p-values for T and R2.
    _unmask_array(data_array)
        Unmasks a vectorized data array to full-brain NIfTI shape using the mask.
    _save_map(map_data, file_name)
        Saves a NIfTI image to disk after unmasking.
    _save_nifti_maps()
        Saves regression results (BETA, T, R2, and permutation-corrected maps) as NIfTI images.
    full_multiout_regression()
        Runs regression for all output channels in the outcome tensor.
    run_all_outputs(n_permutations=0)
        Runs regression and permutation testing for each output channel, saving results in subdirectories.
    run(n_permutations=0)
        Runs regression and permutation testing for the default output, saving results.
    Notes
    -----
    - This class is designed for neuroimaging applications where voxelwise regression and permutation-based inference are required.
    - Input data must be preprocessed and formatted as specified in the JSON configuration file.
    - NIfTI output requires a valid mask file for unmasking vectorized results.
    """
    def __init__(self, json_path, mask_path=None, out_dir=None, regression_type='linear', n_permutations=0):
        self.json_path = json_path
        self.out_dir = out_dir
        self.regression_type = regression_type
        self.n_permutations = n_permutations
        self.data_loader = DataLoader(self.json_path)
        self.design_tensor, self.outcome_tensor, self.contrast_matrix, self.exchangeability_blocks, self.weight_vector = self.load_data()
        self.n_obs, self.n_preds, self.dim3_X, self.dim3_Y, self.n_contrasts, self.n_voxels, self.n_outputs = self.set_variables()
        self._validate_regression_type()
        self.XTX_inv = np.zeros((self.n_preds, self.n_preds, self.n_voxels))    # (n_preds, n_preds, n_voxels)
        self.X_inv = None
        
        # For writing results
        info = self.data_loader.output_info()
        self.output_ftype = info["output_ftype"]
        self.mask_path = mask_path if mask_path is not None else info["mask_path"]
        self.output_handler = NeuroimageFileOutporter(output_ftype=self.output_ftype, mask_path=self.mask_path)

    #### Setter/Getter methods ####
    def load_data(self):
        with open(self.json_path, 'r') as f:
            paths = json.load(f)['neuroimaging_regression']
        design_tensor = np.load(paths['design_matrix'])                        # shape: (observations, predictors,  voxels)
        outcome_data = np.load(paths['outcome_data'])                          # shape: (observations, regressions, voxels)
        contrast_matrix = np.load(paths['contrast_matrix'])                    # shape: (contrasts, predictors)
        weight_vector = np.load(paths["weights_vector"])                       # shape: (observations, )
        exchangeability_blocks = np.load(paths["exchangeability_block"]) if "exchangeability_block" in paths else None
        return design_tensor, outcome_data, contrast_matrix, exchangeability_blocks, weight_vector
        
    def set_variables(self):
        n_obs,   n_preds,  n_voxels_X = self.design_tensor.shape
        n_obs_y, n_cols_Y, n_voxels_Y = self.outcome_tensor.shape
        n_contrasts, n_preds_cmx      = self.contrast_matrix.shape
        return n_obs, n_preds, n_voxels_X, n_voxels_Y, n_contrasts, max(n_voxels_X, n_voxels_Y), n_cols_Y
    
    def _validate_regression_type(self):
        '''Validates the chosen regression type based on the columns of Y'''
        if np.all(np.isin(self.outcome_tensor[...], [0,1])):
            print(f"Outcome is all binary. You should ensure regression_type='logistic'. Detected regression_type: {self.regression_type}")
        else:
            print(f"Outcome is not all binary. You should ensure regression_type='linear'. Detected regression_type: {self.regression_type}")

    def _is_voxelwise_outcome_constant_design(self):
        """
        Return True when Y is voxelwise and X is a single shared design matrix.

        This identifies models shaped like:
            voxelwise_outcome ~ scalar_covariates

        In tensor terms:
            design_tensor:  (n_obs, n_preds, 1)
            outcome_tensor: (n_obs, n_outputs, n_voxels)
        """
        return (
            self.regression_type == "linear"
            and self.design_tensor.ndim == 3
            and self.outcome_tensor.ndim == 3
            and self.design_tensor.shape[0] == self.outcome_tensor.shape[0]
            and self.design_tensor.shape[2] == 1
            and self.outcome_tensor.shape[2] > 1
        )

    ### REGRESSION HELPERS ###
    def _get_targets(self, permutation):
        """
        Returns the regressor (design tensor) and regressand (outcome data), 
        optionally permuting the outcome data if permutation is True.
        """
        regressor = self.design_tensor  # never permute regressor
        regressand = self.outcome_tensor
        weights = self.weight_vector
        if permutation:
            if self.exchangeability_blocks is None:
                resample_idx = np.random.permutation(regressand.shape[0])
            else:
                bl = self.exchangeability_blocks.ravel()
                resample_idx = np.arange(regressand.shape[0])
                for b in np.unique(bl):
                    m = np.where(bl == b)[0]                     # rows in block b
                    resample_idx[m] = np.random.permutation(m)   # permute only within b
            regressand = regressand[resample_idx, :, :]
            weights = weights[resample_idx]
        return regressor, regressand, weights  
    
    def _prep_targets(self, regressor, regressand, weights, voxel_idx="whole_brain", regression_idx=0):
        """
        Standardize X, Y, W for the four linear cases.

        whole_brain:
            1) voxelwise Y, constant X   -> X:(n_obs,n_preds),          Y:(n_obs,n_vox)
            2) voxelwise Y, voxelwise X  -> X:(n_obs,n_preds,n_vox),    Y:(n_obs,n_vox)
            3) constant Y, constant X    -> X:(n_obs,n_preds),          Y:(n_obs,1)
            4) constant Y, voxelwise X   -> X:(n_obs,n_preds,n_vox),    Y:(n_obs,n_vox)

        per-voxel:
            X:(n_obs,n_preds), Y:(n_obs,1)
        """
        x_vox = regressor.shape[2] == self.n_voxels
        y_vox = regressand.shape[2] == self.n_voxels

        def get_X():
            if voxel_idx == "whole_brain":
                return regressor if x_vox else regressor[:, :, 0]
            return regressor[:, :, voxel_idx] if x_vox else regressor[:, :, 0]

        def get_Y():
            if voxel_idx == "whole_brain":
                if y_vox:
                    return regressand[:, regression_idx, :]
                y = regressand[:, regression_idx, 0][:, None]
                return np.broadcast_to(y, (y.shape[0], self.n_voxels)) if x_vox else y

            if y_vox:
                return regressand[:, regression_idx, voxel_idx][:, None]
            return regressand[:, regression_idx, 0][:, None]

        return get_X(), get_Y(), weights
    
    def _prep_naive_bayes(self, X, store=False):
        """Precompute per-voxel pseudoinverse and (X^T X)^{-1}."""
        if X.ndim == 2:
            X_inv = np.linalg.pinv(X)                    # (p,n)
            XtX = X.T @ X                                # (p,p)
            XTX_inv = np.linalg.pinv(XtX)
            return X_inv, XTX_inv

        p, n, v = self.n_preds, self.n_obs, X.shape[2]
        X_inv = np.empty((p, n, v), float)               # pinv per voxel (p,n,v)
        XTX_inv = np.empty((p, p, v), float)             # (X^T X)^{-1} per voxel
        for vox in range(v):
            X_inv[:, :, vox] = np.linalg.pinv(X[:, :, vox])          # (p,n)
            XtX = X[:, :, vox].T @ X[:, :, vox]                      # (p,p)
            XTX_inv[:, :, vox] = np.linalg.pinv(XtX)
        if store:
            self.X_inv, self.XTX_inv = X_inv, XTX_inv
        return X_inv, XTX_inv
    
    #### Voxelwise Model Math ####
    def _gaussian_kernel(self, X, Y, k, h=2.0, block=250000, eps=1e-300):
        """
        Simple healper for Gaussian kernel for feature k. 
        
        X : (n_obs, n_features, n_voxels)
        Y : (n_obs,)
        k : the index of the current parameter
        h : (1,)
        returns F : (n_obs, n_vox)
        """
        if X.ndim == 2:
            X = X[:, :, None]
        n_obs, _, n_vox = X.shape
        idx = np.flatnonzero(Y)
        if idx.size == 0:
            return np.full((n_obs, n_vox), eps, float)
        term1 = 1 / (np.sum(Y) * h * np.sqrt(2*np.pi))
        inv2h2 = 1.0 / (2.0 * h * h)
        Xc = X[idx, k, :]                                    # (N_obs_, v) <- drops axis 1 corresponding to k. 
        
        F = np.empty((n_obs, n_vox), float)
        for obs_start in range(0, n_obs, block):
            obs_stop = min(obs_start + block, n_obs)              # prevent over-slicing
            D2 = (X[obs_start:obs_stop, k, :][:, None, :] - Xc[None, :, :])**2
            term2 = np.exp(-inv2h2 * D2).sum(axis=1)                                # (b-a, v) <- sum over axis 1, the feature axis, collapsing these into observation values at each voxel
            F[obs_start:obs_stop, :] = term1 * np.maximum(term2, eps)
        return F
        
    def _mahalanobis_norm_test(self, B_n, B, XtX, tol=1e-5):
        '''Mahalanobis norm is the natural fit for IRLS'''
        DB = B_n - B
        return np.sqrt(DB @ XtX @ DB) < tol
    
    def _infinity_norm_test(self, B_n, B, tol=1e-8): 
        '''Checks infinity norm'''
        return np.max(np.abs(B_n - B)) < tol
    
    def get_r2(self, Y, Y_HAT, W, eps=1e-12):
        """
        Weighted R^2 = 1 - SSE_w / TSS_w

        Y     : (n_obs,) or (n_obs, n_targets)
        Y_HAT : (n_obs,) or (n_obs, n_targets)
        W     : (n_obs,)
        """
        if Y.ndim == 1:
            Y = Y[:, None]
        if Y_HAT.ndim == 1:
            Y_HAT = Y_HAT[:, None]

        W = W[:, None]                                         # (n_obs, 1)
        wsum = np.sum(W, axis=0)                               # (1,)
        ybar_w = np.sum(W * Y, axis=0) / (wsum + eps)         # (n_targets,)

        sse = np.sum(W * (Y - Y_HAT) ** 2, axis=0)            # (n_targets,)
        tss = np.sum(W * (Y - ybar_w[None, :]) ** 2, axis=0)  # (n_targets,)

        R2 = 1.0 - (sse / (tss + eps))
        return R2[None, :]                                    # (1, n_targets)
    
    def _get_pseudo_r2(self, Y, W, P, eps=1e-9):
        '''
        Gets MacFadden Pseudo R^2 = 1 - (LL_o / LL)
        Y : (n_obs, )
        W : (n_obs, )
        P : (n_obs, n_vox)
        '''
        P_null = np.clip(np.average(Y, weights=W, axis=0), eps, 1 - eps)                # (1,) <- 
        ll_null = np.sum(W * (Y * np.log(P_null) + (1 - Y) * np.log(1 - P_null)))       # (n_obs, )

        if P.ndim == 1:  # single target
            ll_full = np.sum(W * (Y * np.log(P) + (1 - Y) * np.log(1 - P)))
            R2 = np.array([1.0 - (ll_full / ll_null)], dtype=float)                     # (1,)
        elif P.ndim == 2:  # voxelwise
            ll_full = np.sum(W[:, None] * (Y[:, None] * np.log(P) +
                                        (1 - Y)[:, None] * np.log(1 - P)), axis=0)      # (n_vox,)
            R2 = (1.0 - (ll_full / ll_null))[None, :]                                   # (1, n_vox)
        else:
            raise ValueError("P must be 1D or 2D.")
        return R2
    
    def apply_contrasts(self, XtX_inv, BETA, MSE, e=1e-6):
        """
        t = (C @ BETA) / sqrt(diag(C @ XtX_inv @ C.T) * MSE)
        When MSE is set to 1, this provides a wald-test equivalent. 
        For further reading, see https://www.jstor.org/stable/1912934.
        
        C : (n_contrasts, n_preds) design matrix
        BETA : (n_preds, ) or (n_preds, voxels)
        XtX_inv : (n_preds, n_preds) or (n_preds, n_preds, n_voxels)
        MSE : (1,) or (1, n_voxels)
        
        """
        C = self.contrast_matrix
        if XtX_inv.ndim == 2: # conventional 2d matrix
            NUM = C @ BETA                              # (n_contrasts, ) <- (n_contrasts, n_preds) @ (n_preds, )
            var_diag = np.diag(C @ XtX_inv @ C.T)       # (n_contrasts,n_contrasts) <- (n_contrasts, n_preds) @ (n_preds, n_preds) @ (n_preds, n_contrasts)
            DEN = np.sqrt(var_diag * MSE)      
        else:                 # tensor multiplcation
            NUM = np.einsum("cp,pv->cv", C, BETA)        # (n_contrasts, n_voxels) <- (n_contrasts, n_preds) @ (n_preds, n_voxels)
            DEN = np.einsum("cp,pqv,cq->cv", C, XtX_inv, C, optimize=True) # (n_contrasts, voxels) <- 
            DEN = np.sqrt((DEN * MSE)+e)                                  # (n_contrasts, voxels) <-
        return NUM / (DEN+e)
    
    def _run_naive_bayes_prediction(self, X, B, A):
        """
        params:
        X : np.array of (n_obs, n_preds, n_voxels), representing the design matrix for prediction. 
        B : np.array of (n_voxels, n_preds) representing the betas found from a fitted naive bayes. Imported from previously saved niftis.
        A : np.array of (n_voxels,), representing the log(p_1 / p_0), or the log of the prior probabilities of a binary class. 
        
        Notes:
        p =  1.0 / (1.0 + np.exp(-LL))   <-- formula used to convert log odds in binomial cases to probabilities
        """
        XB = np.einsum("opv,vp->ov", X, B)
        LL = A + XB
        return 1.0 / (1.0 + np.exp(-LL))                            # PROBABILITY (n_obs, n_vox) # This is the transform of log odds into probability. 
    
    def _run_linear_prediction(self, X, B):
        """
        params:
        X : np.array of (n_obs, n_preds, n_voxels), representing the design matrix for prediction. 
        B : np.array of (n_voxels, n_preds) representing the betas found from a fitted linear regression. Imported from previously saved niftis.
        """
        return np.einsum("opv,vp->ov", X, B)

    def _run_naive_bayes(self, X, Y, W, X_inv=None, XTX_inv=None, eps=1e-12):
        """
        Binomial case via closed form. 
        log-likelihood(yᵢ|X) = log(pᵢfᵢ(X)/p_jf_j(X)) = b + wTX]
        
        X : (n_obs, n_preds, n_voxels)
        Y : (n_obs, )
        W : (n_obs, )
        X_inv : (n_preds, n_obs, n_voxels)
        XTX_inv : (n_preds, n_preds, n_voxels)
        returns: WE (p,vox), T (n_contrasts,vox), R2 (McFadden) as (1,vox)
        """
        if X.ndim == 2:
            X = X[:, :, None]
        if X_inv is None or XTX_inv is None:
            X_inv, XTX_inv = self._prep_naive_bayes(X)
        j, J = (Y == 1), (Y == 0)
        p1 = j.sum() / self.n_obs
        p0 = 1.0 - p1
        LL = 0
        for k in range(self.n_preds):
            num = self._gaussian_kernel(X,j,k)
            den = self._gaussian_kernel(X,J,k)
            LL += np.log( (num + eps)/(den+eps) )                   # (n_obs, n_vox) <- (n_obs, n_vox) / (n_obs / n_vox)
        LL_adj = LL - np.log((p1 + eps)/(p0 + eps))
        B = np.einsum("pov,ov->pv", X_inv, LL_adj)                  # (n_obs, n_vox) <- (n_obs, n_preds, n_voxels) @ (n_obs, n_voxels)
        
        PR2 = np.zeros((1, X.shape[2])) # self._get_pseudo_r2(Y, W, P)   # Skipping pseudo-R2 for speed
        T = self.apply_contrasts(XTX_inv, B, MSE=1)                 # (n_contrast, n_voxels) <- setting MSE = 1 converts this to a Wald t-stat
        return B, T, PR2, np.log((p1 + eps)/(p0 + eps))                    
    
    def _run_precomputed_linear(self, X, Y, W, XtX_inv):
        """
        Runs a precomputed linear regression using known XTX and Y to speed up permutations. 
        X : (n_obs, n_preds, n_voxels)
        Y : (n_obs, n_voxels)
        W : (n_obs, )
        XtX_inv : (preds, preds, voxels) 
        """
        wsqrt = np.sqrt(W)                                          # (n_obs,)
        Xw = X * wsqrt[:, None, None]                               # (n_obs, n_preds, n_voxels) <- (n_obs, n_preds, n_voxels) * (n_obs, 1, 1)
        Yw = Y * wsqrt[:, None]                                     # (n_obs, )           <- (n_obs, n_voxels) * (n_obs, 1)
        XtY = np.einsum("opv,ov->pv", Xw, Yw)                       # (n_voxels, n_preds) <- (n_obs, n_preds n_voxels) @ (n_obs, )
        BETA = np.einsum("pqv,pv->qv", XtX_inv, XtY)                # (n_preds, n_voxels) <- (n_preds, n_preds, n_voxels) einsum (n_voxels, n_preds)
        Y_HAT = np.einsum("opv,pv->ov", X, BETA)                    # (obs, n_voxels) <- (n_obs, n_preds, n_voxels) einsum(n_preds, n_voxels)
        RES = Y - Y_HAT                                             # (n_obs, n_voxls) <- (n_obs, n_voxls) - (n_obs, n_voxls)
        dof = self.n_obs - self.n_preds                             # (1,)
        MSE = np.sum(W[:, None] * RES**2, axis=0) / dof             # (1, n_voxels) <- summed (n_obs, )^2
        T = self.apply_contrasts(XtX_inv, BETA, MSE)                # (n_contrasts, n_voxels) <- (n_cov, n_voxels) / (n_cov, n_voxels)
        R2 = self.get_r2(Y, Y_HAT, W)                               # (1, n_voxels)
        return BETA, T, R2
    
    def _linear_regression(self, X, Y, W, verbose=False):
        """
        Runs a regression with a NON-voxelwise design matrix.
        
        Parameters
        ----------
        X : (n_obs, n_preds)
        Y : (n_obs, n_vox)
        W : (n_obs,)

        Returns:
          beta: (n_predictors, n_voxels)
          t_values: (n_predictors, n_voxels)
          mse: (n_voxels,)
        """
        wsqrt = np.sqrt(W)                                              # (n_obs,)
        Xw = X * wsqrt[:, None]                                         # (n_obs, n_preds)    <- (n_obs, n_preds) * (n_obs, 1)
        Yw = Y * wsqrt[:, None]                                         # (n_obs, n_vox)      <- (n_obs, n_vox) * (n_obs, 1)
        
        XtX_inv = np.linalg.pinv(Xw.T @ Xw)                             # (n_preds, n_preds)  <- (n_preds, n_obs) @ (n_obs, n_preds)
        XtX_invY = Xw.T @ Yw                                            # (n_preds, n_voxels) <- (n_preds, n_obs) @ (n_obs, n_voxels)
        beta = XtX_inv @ XtX_invY                                       # (n_preds, n_voxels) <- (n_preds, n_preds) @ (n_preds, n_voxels)
        Y_hat = X @ beta                                                # (n_obs, n_voxels)   <- (n_obs, n_preds) @ (n_preds, n_voxels)
        residuals = Y - Y_hat                                           # (n_obs, n_voxels)   <- (n_obs, n_voxels) - (n_obs, n_voxels)
        dof = X.shape[0] - X.shape[1]                                   # (1, )               <- (n_obs, ) - (n_preds, )
        mse = np.sum(W[:, None] * residuals**2, axis=0) / dof           # (n_voxels,)         <- summed (n_obs, n_voxels) along n_obs
        t_values = self.apply_contrasts(XtX_inv, beta, mse)
        R2 = self.get_r2(Y, Y_hat, W)
        
        if verbose:
            print(f"XtX_inv shape: {XtX_inv.shape}")         # (p, p)
            print(f"Y_hat shape: {Y_hat.shape}")             # (n_obs, n_voxels)
            print(f"residuals shape: {residuals.shape}")     # (n_obs, n_voxels)
            print(f"dof shape: {dof}")                       # Scalar (not an array)
            print(f"mse shape: {mse.shape}")                 # (n_voxels,)
            print(f"t_values shape: {t_values.shape}")       # (p, n_voxels)
            print(f"beta shape: {beta.shape}")         # (p, n_voxels)
            print(f"R2 shape: {R2.shape}")                 # (n_voxels,)
        return beta, t_values, R2, XtX_inv
    
    def align_w(self, w, arr):
        if arr.shape[0] != w.shape[0]:
            raise ValueError(f"w has {w.shape[0]} but arr has {arr.shape[0]}")
        return w.reshape((w.shape[0],) + (1,) * (arr.ndim - 1))

    
    ### Switch Helper ###
    def _detect_switch(self):
        """Returns a string based on circumstances"""
        if self.regression_type == 'linear':
            dmatrix_is_voxelwise = self.design_tensor.shape[2] == self.n_voxels
            xtx_is_complete = not np.all(self.XTX_inv == 0)

            if not dmatrix_is_voxelwise:
                return "constant_dmatrix"
            if dmatrix_is_voxelwise and xtx_is_complete:
                return "voxelwise_dmatrix_xtx_complete"
            if dmatrix_is_voxelwise and (not xtx_is_complete):
                return "voxelwise_dmatrix_xtx_incomplete"

        if (self.regression_type=='naive_bayes') and (self.outcome_tensor.shape[2] == self.n_voxels):
            return "naive_bayes_voxelwise"
        if self.regression_type=='naive_bayes':
            return "naive_bayes"
        if self.regression_type=='logistic':
            return "logistic"
        return None
    
    #### Voxelwise Model Switching Methods ####
    def _run_voxelwise_model(self, regressor, regressand, weights, regression_idx, permutation):
        """Choose which regression to run, and how to run it"""
        X, Y, W = self._prep_targets(regressor, regressand, weights, 'whole_brain', regression_idx)
        B = np.zeros((self.n_preds, self.n_voxels)); 
        T = np.zeros((self.n_contrasts, self.n_voxels)); 
        R2 = np.zeros((1, self.n_voxels)) 
        LOG_PRIORS = np.zeros((1, self.n_voxels))
        
        # Linear regression
        switch = self._detect_switch()
        if switch=="voxelwise_dmatrix_xtx_complete":                                      # Linear regression, broadcast (XTX inv precalcualted)
            B, T, R2 = self._run_precomputed_linear(X,Y,W, self.XTX_inv)
        elif switch=="constant_dmatrix":                                                  # Linear regression, broadcast (XTX inv not yet calculated)
            B, T, R2, _ = self._linear_regression(X, Y, W)                                # defines self.XTX_inv for use in permutations
        elif switch=='voxelwise_dmatrix_xtx_incomplete':                                  # linear regression, voxelwise
            for voxel_idx in range(self.n_voxels):
                X, Y, W = self._prep_targets(regressor, regressand, weights, voxel_idx, regression_idx)
                b, t, r2, xtx = self._linear_regression(X, Y, W)
                B[:,voxel_idx] = b.flatten(); T[:,voxel_idx] = t.flatten(); R2[:,voxel_idx] = r2.flatten(); self.XTX_inv[:, :, voxel_idx] = xtx
        # Naive Bayes
        elif switch=='naive_bayes_voxelwise':                                                     # Naive bayes, voxelwise
            raise NotImplementedError("Voxelwise naive_bayes path removed; use non-voxelwise outcome or implement a fixed voxelwise path.")
        elif switch=='naive_bayes_voxelwise':                                                     # Naive bayes, batched
            Y = regressand[:, regression_idx, 0]
            for s, e in voxel_batches(self.n_voxels, batch_size=5000):
                if regressor.shape[2] == self.n_voxels: # build per-voxel design for batch
                    Xb = regressor[:, :, s:e]
                else:
                    Xb = np.broadcast_to(regressor[:, :, 0][:, :, None], (self.n_obs, self.n_preds, e - s)).copy()
                X_inv, XTX_inv = self._prep_naive_bayes(Xb)
                B[:, s:e], T[:, s:e], R2[:, s:e], LOG_PRIORS[:, s:e] = self._run_naive_bayes(Xb, Y, weights, X_inv, XTX_inv)
        
        # Logistic
        elif switch=='logistic':
            raise NotImplementedError("Logistic regression path removed; previous voxelwise path was broken.")
        else:
            raise ValueError(f"Regression type {self.regression_type} not implemented. Please set regression_type='linear' or 'logistic'.")
        return B, T, R2, LOG_PRIORS
        
    def voxelwise_regression(self, permutation=False, regression_idx=0):
        """
        Attemps to do a whole-brain broadcasting if possible. Otherwise defaults to standard looped regression.
        Anything that will be written to disk should be stored here
        """        
        regressor, regressand, weights = self._get_targets(permutation)  
        B, T, R2, LOG_PRIORS = self._run_voxelwise_model(regressor, regressand, weights, regression_idx, permutation)
        if (not permutation) and (self.regression_type=='naive_bayes'):
            self.LOG_PRIORS = LOG_PRIORS
        return B, T, R2
    
    ### P-VALUE METHODS ###
    def _get_max_stat(self, arr, pseudo_var_smooth=False, t=99.99):
        """Return the 99.9th percentile of the absolute values in arr. Or just the raw maximum if pseudo_var_smooth is false (this is subject to chaotic noise)."""
        if pseudo_var_smooth:        
            return np.nanpercentile(np.abs(arr), t, axis=1)  # Calculate along rows, ignoring NaNs
        else: 
            return np.nanmax(np.abs(arr), axis=1)  # Calculate along rows

    def run_permutation(self, n_permutations):
        if n_permutations < 1:
            print("No permutations requested.")
            return
        Tp = np.zeros_like(self.T)        
        R2p = np.zeros_like(self.R2) 
        for i in tqdm(range(n_permutations), desc='running permutations'):
            _, permT, permR2 = self.voxelwise_regression(permutation=True)
            max_statsT = self._get_max_stat(permT)
            max_statsR2 = self._get_max_stat(permR2)
            Tp += (max_statsT[:, None] > np.abs(self.T)).astype(int)  #max t is already absval. self.T must be set to absval for a 2-sample t test. 
            R2p += (max_statsR2 > self.R2).astype(int)                #R2 does not need to be absval. It is inherently 1-sided t test.
        self.Tp = Tp / n_permutations 
        self.R2p = R2p / n_permutations
        
    ### Neuroimaging File Saving Methods ####
    def _save_result_maps(self, visualize=False):
        """Method to orchestrate writing results to disk"""
        if not self.out_dir:
            return

        if hasattr(self, 'B_multi'):
            for j in range(self.n_outputs):
                for i in range(self.n_contrasts):
                    self.output_handler.save_map(self.B_multi[i, :, j], f"beta_predictor_{i}_output_{j}", self.out_dir, visualize=False)

        if hasattr(self, 'T_multi'):
            for j in range(self.n_outputs):
                for i in range(self.n_contrasts):
                    self.output_handler.save_map(self.T_multi[i, :, j], f"contrast_{i}_tval_output_{j}", self.out_dir, visualize=visualize)

        if hasattr(self, 'R2_multi'):
            for j in range(self.n_outputs):
                self.output_handler.save_map(self.R2_multi[0, :, j], f"R2_output_{j}", self.out_dir, visualize=visualize)

        if hasattr(self, 'BETA'):
            for i in range(self.n_preds):
                self.output_handler.save_map(self.BETA[i, :], f"beta_predictor_{i}", self.out_dir, visualize=False)

        if hasattr(self, 'T'):
            for c in range(self.n_contrasts):
                self.output_handler.save_map(self.T[c, :], f"contrast_tval_{c}", self.out_dir, visualize=visualize)

        if hasattr(self, 'R2'):
            self.output_handler.save_map(self.R2, "R2_vals", self.out_dir, visualize=visualize)

        if hasattr(self, 'LOG_PRIORS'):
            self.output_handler.save_map(self.LOG_PRIORS, "LOG_PRIORS", self.out_dir, visualize=visualize)

        if hasattr(self, 'Tp'):
            for c in range(self.n_contrasts):
                sig_tvals = np.where(self.Tp[c, :] < 0.05, self.T[c, :], np.nan)
                self.output_handler.save_map(sig_tvals, f"contrast_tval_FWE_{c}", self.out_dir, visualize=False)
                self.output_handler.save_map(self.Tp[c, :], f"contrast_pval_FWE_{c}", self.out_dir, visualize=False)

        if hasattr(self, 'R2p'):
            sig_r2vals = np.where(self.R2p < 0.05, self.R2, np.nan)
            self.output_handler.save_map(sig_r2vals, "R2_FWE", self.out_dir, visualize=False)
            self.output_handler.save_map(self.R2p, "R2_pval_FWE", self.out_dir, visualize=False)

        if hasattr(self, 'PREDICTIONS'):
            for o in range(self.PREDICTIONS.shape[0]):
                self.output_handler.save_map(self.PREDICTIONS[o, :], f"prediction_{o}", self.out_dir, visualize=False)

    #### Prediction Helpers #### ---TODO: MAKE NIFTI-AGNOSTIC.
    def _get_mask_indices(self):
        """
        Cached boolean mask (flattened) used for vectorizing/unvectorizing.
        """
        if self.mask_path is None:
            raise ValueError("Mask path is not provided. Provide the mask used to create the data_array.")

        cache = getattr(self, "_mask_cache", None)
        if cache is not None and cache.get("mask_path") == self.mask_path:
            return cache["mask_indices"]

        mask_img = nib.load(self.mask_path)
        mask_data = mask_img.get_fdata()
        mask_indices = mask_data.flatten() > 0
        self._mask_cache = {
            "mask_path": self.mask_path,
            "mask_indices": mask_indices,
        }
        return mask_indices

    def _mask_array(self, data_array):
        """
        Masks a full-brain array to a vector using self.mask_path.
        Returns:
            masked_array: vectorized data (n_vox,)
        """
        mask_indices = self._get_mask_indices()
        return data_array.flatten()[mask_indices]

    def _load_prediction_params(self, params_dir, files):
        """
        Generic loader for prediction parameters saved as nifti(s) in params_dir.
        files: list of strings. Each string is either:
            - a basename (e.g., "LOG_PRIORS") to load a single nifti
            - a prefix with '*' (e.g., "beta_predictor_*") to load a stack
        Returns:
            list of loaded arrays in the same order as files
        """
        if self.mask_path is None:
            raise ValueError("mask_path is required to load prediction params.")
        
        def _find_single_nifti(basename):
            candidates = [
                os.path.join(params_dir, f"{basename}.nii.gz"),
                os.path.join(params_dir, f"{basename}.nii"),
            ]
            return next((p for p in candidates if os.path.exists(p)), None)
        
        def _find_prefixed_niftis(prefix):
            files = []
            for ext in ("nii.gz", "nii"):
                files.extend([p for p in os.listdir(params_dir) if p.startswith(prefix) and p.endswith(ext)])
            return files
        
        def _beta_index(name, prefix):
            stem = name.replace(".nii.gz", "").replace(".nii", "")
            return int(stem.split(prefix)[1])

        loaded = []
        for item in files:
            if item.endswith("*"):
                prefix = item[:-1]
                files_found = _find_prefixed_niftis(prefix)
                if not files_found:
                    raise FileNotFoundError(f"No {prefix}*.nii(.gz) files found in prediction params directory.")
                files_found = sorted(set(files_found), key=lambda n: _beta_index(n, prefix))
                betas = []
                for bf in files_found:
                    b_full = nib.load(os.path.join(params_dir, bf)).get_fdata()
                    betas.append(self._mask_array(b_full))
                loaded.append(np.stack(betas, axis=1))       # (n_vox, n_preds)
            else:
                path = _find_single_nifti(item)
                if path is None:
                    raise FileNotFoundError(f"{item}.nii(.gz) not found in prediction params directory.")
                arr_full = nib.load(path).get_fdata()
                loaded.append(self._mask_array(arr_full))
        return loaded

    def _run_prediction_switch(self, temp_dir=None, *, B=None, A=None):
        """
        Dispatch prediction based on regression_type.
        """
        X = self.design_tensor
        X = X[:, :, None] if X.ndim == 2 else X
        X = np.broadcast_to(X, (self.n_obs, self.n_preds, self.n_voxels)).copy() if X.shape[2] == 1 else X 
        if self.regression_type == "naive_bayes":
            if B is None or A is None:
                if temp_dir is None:
                    raise ValueError("temp_dir is required when B/A are not provided.")
                B, A = self._load_prediction_params(temp_dir, ["beta_predictor_*", "LOG_PRIORS"])
            return self._run_naive_bayes_prediction(X, B, A)
        if self.regression_type == "linear":
            if B is None:
                if temp_dir is None:
                    raise ValueError("temp_dir is required when B is not provided.")
                (B,) = self._load_prediction_params(temp_dir, ["beta_predictor_*"])
            return self._run_linear_prediction(X, B)
        raise NotImplementedError(f"Prediction for regression_type='{self.regression_type}' is not yet implemented.")
    
    def _get_scalar_predictions(self, temp_dir, subject_arr, prediction_arr, *, T_all=None):
        """
        Takes voxelwise prediction arrays (or t-maps) and extracts an average value from them
        Returns array with all t-value predictions in the first rows, then the overall prediction in the last row.
        """
        preds = np.zeros((1, self.n_contrasts+1))                                   # (one for each contrast, then the regression prediction)
        if T_all is None:
            if temp_dir is None:
                raise ValueError("temp_dir is required when T_all is not provided.")
            (T_all,) = self._load_prediction_params(temp_dir, ["contrast_tval_*"])  # (n_vox, n_contrasts)
        
        # Get cosine similarity of subject's map with the T-map
        for i in range(self.n_contrasts):
            T = T_all[:, i]
            dmg_value_dict = DamageScorer._calculate_metrics(subject_arr, T, ["cosine"])
            preds[0, i] = dmg_value_dict["cosine"]
        
        # Get cosine similarity of subject's map with their prediction map
        dmg_value_dict = DamageScorer._calculate_metrics(subject_arr, prediction_arr, ["cosine"])
        preds[0, -1]    = dmg_value_dict["cosine"]
        return preds


    ### Cross Validated Predictions ###
    def _get_cv_splits(self, cv):
        """
        Returns a list of (train_idx, test_idx) tuples.
        cv can be:
            - "loocv"
            - "loo"
            - integer number of folds, e.g. 2, 5, 10
        """
        n = self.n_obs
        indices = np.arange(n)

        if isinstance(cv, str):
            cv = cv.lower()
            if cv in {"loocv", "loo"}:
                splits = []
                for i in range(n):
                    test_idx = np.array([i])
                    train_idx = indices[indices != i]
                    splits.append((train_idx, test_idx))
                return splits
            raise ValueError(f"Unrecognized cv string: {cv}")

        if isinstance(cv, int):
            if cv < 2:
                raise ValueError("cv must be 'loocv'/'loo' or an integer >= 2.")
            if cv > n:
                raise ValueError(f"cv={cv} cannot exceed n_obs={n}.")

            fold_indices = np.array_split(indices, cv)
            splits = []
            for k in range(cv):
                test_idx = fold_indices[k]
                train_idx = np.concatenate([fold_indices[j] for j in range(cv) if j != k])
                splits.append((train_idx, test_idx))
            return splits

        raise ValueError("cv must be 'loocv'/'loo' or an integer >= 2.")

    def run_prediction_cv(self, subject_arr, regression_idx=0, cv="loocv", *, save_fold_params=False):
        """
        Cross-validated predictions using in-sample data.
        cv can be 'loocv'/'loo' or an integer number of folds.
        Populates self.PREDICTIONS with per-observation predictions.
        """
        scalar_preds = np.zeros((self.n_obs, self.n_contrasts + 1))
        self.PREDICTIONS = np.zeros((self.n_obs, self.n_voxels))

        orig_design = self.design_tensor
        orig_outcome = self.outcome_tensor
        orig_weights = self.weight_vector
        orig_n_obs = self.n_obs
        orig_out_dir = self.out_dir
        # Avoid accidentally saving the final fold's fitted maps into orig_out_dir at the end.
        orig_maps = {}
        for name in ("B_multi", "T_multi", "R2_multi", "BETA", "T", "R2", "LOG_PRIORS", "Tp", "R2p"):
            if hasattr(self, name):
                orig_maps[name] = getattr(self, name)

        temp_dir = None
        if save_fold_params:
            temp_dir = os.path.join(self.out_dir or "/tmp", f"{cv}_prediction_params")
            os.makedirs(temp_dir, exist_ok=True)
            self.out_dir = temp_dir

        splits = self._get_cv_splits(cv)

        for fold_idx, (train_idx, test_idx) in enumerate(tqdm(splits, desc=f"{cv} predictions")):
            self.design_tensor = orig_design[train_idx, :, :]
            self.outcome_tensor = orig_outcome[train_idx, :, :]
            self.weight_vector = orig_weights[train_idx]
            self.n_obs = self.design_tensor.shape[0]

            self.BETA, self.T, self.R2 = self.voxelwise_regression(regression_idx=regression_idx)

            self.design_tensor = orig_design[test_idx, :, :]
            self.n_obs = self.design_tensor.shape[0]
            B = self.BETA.T  # (n_vox, n_preds)
            A = getattr(self, "LOG_PRIORS", None)
            A = np.asarray(A).squeeze() if A is not None else None
            prediction_arr = self._run_prediction_switch(temp_dir, B=B, A=A)
            self.PREDICTIONS[test_idx, :] = prediction_arr
            
            if save_fold_params:
                self._save_result_maps()

            for local_i, global_i in enumerate(test_idx):
                scalar_preds[global_i, :] = self._get_scalar_predictions(
                    temp_dir,
                    subject_arr[global_i, :],
                    prediction_arr[local_i, :],
                    T_all=self.T.T,
                )

        self.design_tensor = orig_design
        self.outcome_tensor = orig_outcome
        self.weight_vector = orig_weights
        self.n_obs = orig_n_obs
        self.out_dir = orig_out_dir
        # Restore any pre-existing fitted maps; otherwise drop fold artifacts.
        for name in ("B_multi", "T_multi", "R2_multi", "BETA", "T", "R2", "LOG_PRIORS", "Tp", "R2p"):
            if name in orig_maps:
                setattr(self, name, orig_maps[name])
            elif hasattr(self, name):
                delattr(self, name)
                
        return scalar_preds

    def _evaluate_map(self, y_true, subject_files, regression_idx=0, cv="loocv"):
        """
        Run CV prediction and generate scatterplots of predicted vs actual Y.
        Adds RMSE and MAE to the plot text.

        Parameters
        ----------
        y_true : pd.Series
            values of shape (n_obs, ) which individual correlations (n_obs, ) can be correlated to.
        subject_files : pd.Series
            values of shape (n_obs, ). Contains the files to import the subject files to evaluate.
        """
        if not self.out_dir:
            raise ValueError("out_dir must be set for cross-validation plotting.")
        
        importer = GiiNiiFileImport(import_path=subject_files, mask_path=self.mask_path, transpose=True)
        subject_arr = importer.run().values
        scalar_preds = self.run_prediction_cv(subject_arr=subject_arr, regression_idx=regression_idx, cv=cv)

        def _scatter(pred_col, y_true, *, name: str):
            rmse = float(np.sqrt(np.mean((pred_col - y_true) ** 2)))
            mae = float(np.mean(np.abs(pred_col - y_true)))
            df = pd.DataFrame({"pred": pred_col, "actual": y_true})
            ax = simple_scatter(
                df=df,
                x_col="pred",
                y_col="actual",
                dataset_name=name,
                out_dir=os.path.join(self.out_dir, "cross_validations"),
                x_label="Predicted",
                y_label="Actual",
                extra_lines=[f"RMSE = {rmse:.3g}", f"MAE = {mae:.3g}"],
                show=True,
            )
            # Avoid accumulating open figures when evaluating many models/columns.
            try:
                plt.close(ax.figure)
            except Exception:
                pass

        for j in range(scalar_preds.shape[1]):
            pred_col = scalar_preds[:, j]
            if j < self.n_contrasts:
                name = f"{cv}_contrast_{j}_correlation"
            else:
                name = f"{cv}_voxelwisemodel_aggregate-prediction"
            _scatter(pred_col, y_true, name=name)

    
    #### Public Code - Fitting ####
    def run_cross_validation(self, *, y_true, subject_files, regression_idx=0):
        """
        Orchestrates cross-validated relation of t-maps and voxelwise predictions to dependent variables
        
        Parameters
        ----------
        y_true : pd.Series
            values of shape (n_obs, ) which individual correlations (n_obs, ) can be correlated to.
        subject_files : pd.Series
            values of shape (n_obs, ). Contains the files to import the subject files to evaluate.
        regression_idx : int
            integer corresponding to which regression to use. Generally default is 0. 
        """
        self._evaluate_map(cv="loocv", y_true=y_true, subject_files=subject_files, regression_idx=regression_idx)
        self._evaluate_map(cv=2, y_true=y_true, subject_files=subject_files, regression_idx=regression_idx)
        self._evaluate_map(cv=5, y_true=y_true, subject_files=subject_files, regression_idx=regression_idx)
        self._evaluate_map(cv=10, y_true=y_true, subject_files=subject_files, regression_idx=regression_idx)

    def run_single_multiout_regression(self, permutation=False):
        """Runs regression across all outputs a single time and returns the associated arrays."""
        B_multi = np.zeros((self.n_contrasts, self.n_voxels, self.n_outputs))
        T_multi = np.zeros((self.n_contrasts, self.n_voxels, self.n_outputs))
        R2_multi = np.zeros((self.n_contrasts, self.n_voxels, self.n_outputs))
        for j in range(self.n_outputs):
            B_multi[:,:,j], T_multi[:,:,j], R2_multi[:,:,j] = self.voxelwise_regression(permutation=permutation, regression_idx=j)
        
        if permutation is False:            # Store the results in the class attributes for use later
            self.B_multi, self.T_multi, self.R2_multi = B_multi, T_multi, R2_multi
        return B_multi, T_multi, R2_multi

    def run_all_outputs(self, visualize=False):
        """
        Orchestrates full multi-output regression.
        For each output channel in outcome_tensor:
        - Runs regression
        - Runs permutation testing
        - Saves results into a separate subdirectory
        """
        base_out_dir = self.out_dir

        for j in range(self.n_outputs):
            print(f"\nRunning regression for output {j}")
            self.out_dir = os.path.join(base_out_dir, f"regression_{j}")
            os.makedirs(self.out_dir, exist_ok=True)

            # Run one regression for this output
            self.BETA, self.T, self.R2 = self.voxelwise_regression(regression_idx=j)
            self.run_permutation(self.n_permutations)
            self._save_result_maps(visualize=visualize)

    def run(self, visualize=False):
        """
        Executes the voxelwise regression analysis and optional permutation testing.
        This method performs the following steps:
            1. Runs the voxelwise regression and stores the resulting beta coefficients,
               t-statistics, and R-squared values.
            2. If `n_permutations` is greater than 0, performs permutation testing with
               the specified number of permutations.
            3. Saves the resulting statistical maps as NIfTI files.
        Args:
            n_permutations (int, optional): Number of permutations to run for permutation
                testing. Defaults to 0 (no permutation testing).
        """
        self.BETA, self.T, self.R2 = self.voxelwise_regression()
        self.run_permutation(self.n_permutations)
        self._save_result_maps(visualize=visualize)

# -----------------------
# batching / assembly
# -----------------------
def voxel_batches(n_vox, batch_size):
    for s in range(0, n_vox, batch_size):
        e = min(s + batch_size, n_vox)
        yield s, e

def build_X_batch(X_shared, voxel_cols, s, e):
    """
    Build per-voxel design for a slice [s:e].

    X_shared : (o, p0)         columns shared by all voxels
    voxel_cols : list of K arrays, each (o, V)  (per-voxel regressors)
                 pass [] if none
    returns Xb : (o, p0+K, vb)
    """
    o, p0 = X_shared.shape
    vb = e - s
    # broadcast shared X to 3D without tiling data in math ops; here we form a small 3D view
    Xb = np.broadcast_to(X_shared[:, :, None], (o, p0, vb)).copy()
    if voxel_cols:
        adds = [vc[:, s:e][..., None].transpose(0, 2, 1) for vc in voxel_cols]  # (o,1,vb) each
        Xb = np.concatenate([Xb] + adds, axis=1)  # (o, p0+K, vb)
    return Xb
