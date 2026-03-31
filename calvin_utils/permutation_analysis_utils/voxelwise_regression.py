import json
import numpy as np
from scipy.stats import t
from calvin_utils.neuroimaging_utils.ccm_utils.npy_utils import DataLoader
import os
from tqdm import tqdm
import nibabel as nib
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from scipy.special import expit

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
        self.mask_path = mask_path
        self.out_dir = out_dir
        self.regression_type = regression_type
        self.n_permutations = n_permutations
        self.data_loader = DataLoader(self.json_path)
        self.design_tensor, self.outcome_tensor, self.contrast_matrix, self.exchangeability_blocks, self.weight_vector = self.load_data()
        self.n_obs, self.n_preds, self.dim3_X, self.dim3_Y, self.n_contrasts, self.n_voxels, self.n_outputs = self.set_variables()
        self._validate_regression_type()
        self.XTX_inv = np.zeros((self.n_preds, self.n_preds, self.n_voxels))    # (n_preds, n_preds, n_voxels)
        self.X_inv = None

    #### Setter/Getter methods ####
    def load_data(self):
        with open(self.json_path, 'r') as f:
            paths = json.load(f)['voxelwise_regression']
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
    
    def _prep_targets(self, regressor, regressand, weights, voxel_idx, regression_idx=0):
        """
        Ensure shape of X, Y, and W matrices. 
        If a regression_idx is provided, it selects the corresponding output channel. This enables multi-output regression.
        """
        if voxel_idx=="whole_brain":                             # for whole-brain one-shot regression
            X = regressor[:, :, :]
        elif regressor.shape[2] == self.n_voxels:          # for voxel-wise design with potential multiple outputs
            X = regressor[:, :, voxel_idx]
        else:                                            # for broadcast design
            X = regressor[:, :, 0]                       

        if voxel_idx=="whole_brain":                             # for whole-brain one-shot regression
            Y = regressand[:, regression_idx, :]
        elif regressand.shape[2] == self.n_voxels:         # for voxel-wise outcome with potential multiple outputs
            Y = regressand[:, regression_idx, voxel_idx]
        else:                                            # for multi-output voxel-wise outcome
            Y = regressand[:, regression_idx, 0]
        return X, Y, weights
    
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
        
    def _voxelwise_logit(X, y, W):
        '''Simplified logistic that just works in a voxelwise looped manner'''
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        lr = LogisticRegression(random_state=0).fit(X, y, sample_weight=W)
        B = lr.coef_
        XTX_inv = np.linalg.pinv( X.T @ X )
        P = lr.predict_proba(X)[:, 1]
        return B, XTX_inv, P

    def _clipped_sigmoid(self, a, eps=1e-9):
        '''Returns a numerically stable sigmoid to get p(y|x)'''
        out = np.empty_like(a, dtype=float)
        pos = a >= 0
        out[pos]  = 1.0 / (1.0 + np.exp(-a[pos]))
        ea = np.exp(a[~pos])
        out[~pos] = ea / (1.0 + ea)
        return np.clip(out, eps, 1 - eps)

    def _irls(self, X, Y, B, P, W, eps=1e-9, ridge=1e-6):
        '''IRLS from Newton-raphson rewritten in OLS form'''
        V = P * (1.0 - P)                 # bernoulli probability variance function
        V = np.maximum(V, eps)            # add numeric stability
        s = np.sqrt(W*V)                  # sqrt of weights * bernoulli variance (effective weight). Scaled.

        Z = (X@B) + ((Y - P) / V)         # current response 
        Z_w = Z * s                       # gets Z weighed by ~scaled effective weights
        
        X_w = X * s[:, None]               # gets X weighed by ~scaled effective weights
        XtX = X_w.T @ X_w                  # XTWX = covariance matrix = observed fisher info = observed hessian!
        XtX = XtX + (ridge * np.eye(XtX.shape[0]))
        XtX_inv = np.linalg.pinv(XtX, rcond=1e-10)
        
        B_n = XtX_inv @ (X_w.T @ Z_w)
        return B_n, XtX
    
    def _mahalanobis_norm_test(self, B_n, B, XtX, tol=1e-5):
        '''Mahalanobis norm is the natural fit for IRLS'''
        DB = B_n - B
        return np.sqrt(DB @ XtX @ DB) < tol
    
    def _infinity_norm_test(self, B_n, B, tol=1e-8): 
        '''Checks infinity norm'''
        return np.max(np.abs(B_n - B)) < tol
    
    def _fit_logistic(self, X, Y, W, max_iter=50, ridge=1e-6, clip=20.0):
        '''
        IRLS for argmax(β) of:
            log-likelihood(β) = Σ_i  W[YβX - log(1 + e^βX)]
        which can be maximized with 
        Bn = (XTWX)^-1 XTW_eZ_e <- used to update W 
        z  = Xβ + ( (y - p) / p(1-p) )
        n  = Xβ
        '''
        B = np.zeros(self.n_preds)       # init B
        XtX = None
        for _ in range(max_iter):
            P = self._clipped_sigmoid(X@B)
            B_n, XtX = self._irls(X, Y, B, P, W, ridge=ridge)
            if self._mahalanobis_norm_test(B_n, B, XtX) or self._infinity_norm_test(B_n, B):
                B = B_n
                break
            B = B_n
        if clip is not None:
            B = np.clip(B, -clip, clip)
        if XtX is None:
            XtX = X.T @ X
        XtX_inv = np.linalg.pinv(XtX, rcond=1e-10)
        return B, XtX_inv
    
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

        return 1.0 - (sse / (tss + eps))
    
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
        p1 = j.sum() / self.n_obs; p0 = 1.0 - p1
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
    
    def _run_logistic(self, X, Y, W, vectorize=True):
        """
        Binomial logistic regression via IRLS (Y in {0,1}).
        Returns: BETA (p,), T (n_contrasts,), R2 (McFadden) as (1,)
        """
        try:
            if vectorize:
                B, XTX_inv = self._fit_logistic(X, Y, W)
                P = self._clipped_sigmoid(X @ B) # Gets probability, but clips for safety
            else:
                B, XTX_inv, P = self._voxelwise_logit(X, Y, W)
            PR2 = self._get_pseudo_r2(Y, W, P)
            T = self.apply_contrasts(XTX_inv, B, MSE=1)     # setting MSE = 1 converts this to a Wald t-stat
        except Exception as e:
            if 'SVD did not converge' in str(e):
                B  = np.zeros(self.n_preds)
                T  = np.zeros(self.n_contrasts)
                PR2= np.zeros((1,))
            else:
                print(f"Error in running logistic regression: {e}")
                B  = np.full(self.n_preds, np.nan)
                T  = np.full(self.n_contrasts, np.nan)
                PR2= np.full((1,), np.nan)
        return B, T, PR2

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
    
    def _run_linear(self, X, Y, W):
        """
        Weighted linear regression.

        Parameters
        ----------
        X : (n_obs, n_preds)
        Y : (n_obs,) or (n_obs, n_targets)
        W : (n_obs,)
        """
        if Y.ndim == 1:
            Y = Y[:, None]
        print(X.shape, Y.shape)
        wsqrt = np.sqrt(W)                              # (n_obs,)        
        Xw = X * self.align_w(wsqrt, X)                # (n_obs, n_preds)
        Yw = Y * self.align_w(wsqrt, Y)                # (n_obs, n_targets)
        XtX_inv = np.linalg.pinv(Xw.T @ Xw)             # (n_preds, n_preds)
        BETA = XtX_inv @ Xw.T @ Yw                      # (n_preds, n_targets)
        Y_HAT = X @ BETA                                # (n_obs, n_targets)
        residuals = Y - Y_HAT                           # (n_obs, n_targets)
        dof = X.shape[0] - X.shape[1]                   # (1,)
        mse = np.sum((residuals * self.align_w(wsqrt, residuals)  )**2, axis=0) / dof   # (n_targets,)
        T = self.apply_contrasts(XtX_inv, BETA, mse)    # should return (n_contrasts, n_targets)
        R2 = self.get_r2(Y, Y_HAT, W)                   # should return (n_targets,)

        return BETA, T, R2, XtX_inv
    
    def align_w(self, w, arr):
        if arr.shape[0] != w.shape[0]:
            raise ValueError(f"w has {w.shape[0]} but arr has {arr.shape[0]}")
        return w.reshape((w.shape[0],) + (1,) * (arr.ndim - 1))

    
    #### Voxelwise Model Switching Methods ####
    def _run_voxelwise_model(self, regressor, regressand, weights, regression_idx, permutation):
        """Choose which regression to run, and how to run it"""
        X, Y, W = self._prep_targets(regressor, regressand, weights, 'whole_brain', regression_idx)
        B = np.zeros((self.n_preds, self.n_voxels)); T = np.zeros((self.n_contrasts, self.n_voxels)); R2 = np.zeros((1, self.n_voxels)) 
        LOG_PRIORS = np.zeros((1, self.n_voxels))
        
        # Linear regression
        if (self.regression_type=='linear') and (not np.all(self.XTX_inv == 0)): # Linear regression, broadcast (XTX inv precalcualted)
            B, T, R2 = self._run_precomputed_linear(X,Y,W, self.XTX_inv)
        elif (self.regression_type=='linear') and (np.all(self.XTX_inv == 0)) and (self.design_tensor.shape[2] != self.n_voxels): # Linear regression, broadcast (XTX inv not yet calculated)
            B, T, R2, XTX_inv = self._run_linear(X, Y, W)                                       # defines self.XTX_inv for use in permutations
            if not permutation: self.XTX_inv = XTX_inv
        elif (self.regression_type=='linear'):                                                  # linear regression, voxelwise
            for idx in (range(self.n_voxels) if permutation else tqdm(range(self.n_voxels), desc='Running voxelwise regressions')):
                X, Y, W = self._prep_targets(regressor, regressand, weights, idx, regression_idx)
                B[:,idx], T[:,idx], R2[:,idx], XTX_inv = self._run_linear(X, Y, W)
                if not permutation: self.XTX_inv[:, :, idx] = XTX_inv
        
        # Naive Bayes
        elif (self.regression_type=='naive_bayes') and (self.outcome_tensor.shape[2] == self.n_voxels):      # Naive bayes, voxelwise
            for idx in tqdm(range(self.n_voxels), desc='Running voxelwise regressions'):
                X, Y, W = self._prep_targets(regressor, regressand, weights, idx, regression_idx)
                B[:, idx], T[:, idx], R2[:, idx], LOG_PRIORS[:, idx] = self._run_naive_bayes(X, Y, W)
        elif (self.regression_type=='naive_bayes'):                                                 # Naive bayes, batched
            Y = regressand[:, regression_idx, 0]
            for s, e in voxel_batches(self.n_voxels, batch_size=5000):
                if regressor.shape[2] == self.n_voxels: # build per-voxel design for batch
                    Xb = regressor[:, :, s:e]
                else:
                    Xb = np.broadcast_to(regressor[:, :, 0][:, :, None], (self.n_obs, self.n_preds, e - s)).copy()
                X_inv, XTX_inv = self._prep_naive_bayes(Xb)
                B[:, s:e], T[:, s:e], R2[:, s:e], LOG_PRIORS[:, s:e] = self._run_naive_bayes(Xb, Y, weights, X_inv, XTX_inv)
        
        # Logistic
        elif self.regression_type=='logistic':
            B, T, R2 = self._run_logistic(X, Y, W)
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
    def _get_max_stat(self, arr, pseudo_var_smooth=True, t=99.99):
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
        
    ### Nifti Saving Methods ####
    def _unmask_array(self, data_array):
        """
        Unmasks a vectorized image to full-brain shape using self.mask_path.
        Returns:
            unmasked_array: full-brain NIfTI-like array
            mask_affine: affine transformation from mask
        """
        if self.mask_path is None:
            raise ValueError("Mask path is not provided. Provide the mask used to create the data_array.")
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
    
    def _save_nifti_maps(self):
        """
        Example method that unmasks and saves NIfTI maps for BETA & T.
        Assumes you have self._unmask_array() and self._save_map() in place.
        """
        if not self.out_dir or not self.mask_path:
            return
        
        # Save multi-output regression results
        if hasattr(self, 'B_multi'):
            for j in range(self.n_outputs):
                for i in range(self.n_contrasts):
                    beta_name = f"beta_predictor_{i}_output_{j}.nii.gz"
                    self._save_map(self.B_multi[i, :, j], beta_name)
        if hasattr(self, 'T_multi'):
            for j in range(self.n_outputs):
                for i in range(self.n_contrasts):
                    t_name = f"contrast_{i}_tval_output_{j}.nii.gz"
                    self._save_map(self.T_multi[i, :, j], t_name)
        if hasattr(self, 'R2_multi'):
            for j in range(self.n_outputs):
                for i in range(self.n_contrasts):
                    r2_name = f"R2_output_{j}.nii.gz"
                    self._save_map(self.R2_multi[i, :, j], r2_name)
        
        # Save betas: shape (n_preds, n_voxels)
        if hasattr(self, 'BETA'):
            for i in range(self.n_preds):
                beta_name = f"beta_predictor_{i}.nii.gz"
                self._save_map(self.BETA[i, :], beta_name)
        # Save T-values for Betas: shape (n_preds, n_voxels)
        if hasattr(self, 'T'):
            for c in range(self.n_contrasts):
                self._save_map(self.T[c, :], f"contrast_{c}_tval.nii.gz")
        # Save overall R2 (measure of model overall model fit)
        if hasattr(self, 'R2'):
            self._save_map(self.R2, f"R2_vals.nii.gz")
        # Save log priors (naive bayes for log(p1 / p0))
        if hasattr(self, 'LOG_PRIORS'):
            self._save_map(self.LOG_PRIORS, f"LOG_PRIORS.nii.gz")

        # Save FWE-corrected significance masks if we have permutation results
        if hasattr(self, 'Tp'):
            for c in range(self.n_contrasts):
                sig_mask = (self.Tp[c, :] < 0.05)
                sig_tvals = np.where(sig_mask, self.T[c, :], np.nan)
                self._save_map(sig_tvals, f"contrast_{c}_tval_FWE.nii.gz")
                self._save_map(self.Tp[c, :], f"contrast_{c}_pval_FWE.nii.gz")
                
        # Save FWE-corrected significance masks if we have permutation results
        if hasattr(self, 'R2p'):
            sig_mask = (self.R2p < 0.05)
            sig_r2vals = np.where(sig_mask, self.R2, np.nan)
            self._save_map(sig_r2vals, f"R2_FWE.nii.gz")
            self._save_map(self.R2p, f"R2_pval_FWE.nii.gz")
            
        # Save predictions
        if hasattr(self, 'PREDICTIONS'):
            obs, vox = self.PREDICTIONS.shape
            for o in range(obs):
                self._save_map(self.PREDICTIONS[o, :], f"prediction_{o}.nii.gz")

    #### Prediction Helpers ####
    def _mask_array(self, data_array):
        """
        Masks a full-brain array to a vector using self.mask_path.
        Returns:
            masked_array: vectorized data (n_vox,)
        """
        if self.mask_path is None:
            raise ValueError("Mask path is not provided. Provide the mask used to create the data_array.")
        mask = nib.load(self.mask_path)
        mask_data = mask.get_fdata()
        mask_indices = mask_data.flatten() > 0
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

    def _run_prediction_switch(self, params_dir):
        """
        Dispatch prediction based on regression_type.
        """
        X = self.design_tensor
        if X.ndim == 2: X = X[:, :, None]
        if self.regression_type == "naive_bayes":
            B, A = self._load_prediction_params(params_dir, ["beta_predictor_*", "LOG_PRIORS"])
            if X.shape[2] == 1: X = np.broadcast_to(X, (self.n_obs, self.n_preds, B.shape[0])).copy()
            return self._run_naive_bayes_prediction(X, B, A)
        if self.regression_type == "linear":
            (B,) = self._load_prediction_params(params_dir, ["beta_predictor_*"])
            if X.shape[2] == 1: X = np.broadcast_to(X, (self.n_obs, self.n_preds, B.shape[0])).copy()
            return self._run_linear_prediction(X, B)
        raise NotImplementedError(f"Prediction for regression_type='{self.regression_type}' is not yet implemented.")
    
    #### Public Code ####
    def run_prediction(self, params_dir):
        """
        Loads prediction parameters from params_dir and runs prediction
        using the current regression_type.
        Returns:
            P : (n_obs, n_vox) predicted probabilities for naive_bayes or predictions for linear
        """
        self.PREDICTIONS = self._run_prediction_switch(params_dir)
        self._save_nifti_maps()
    
    def run_prediction_loocv(self, regression_idx=0, batch_size=5000):
        """
        Leave-one-out CV predictions using in-sample data.
        Populates self.PREDICTIONS with per-observation predictions.
        """
        preds = np.zeros((self.n_obs, self.n_voxels))
        orig_design = self.design_tensor
        orig_outcome = self.outcome_tensor
        orig_weights = self.weight_vector
        orig_n_obs = self.n_obs
        orig_out_dir = self.out_dir

        params_dir = os.path.join(self.out_dir or "/tmp", "loocv_prediction_params")
        os.makedirs(params_dir, exist_ok=True)

        for i in tqdm(range(self.n_obs), desc="LOOCV predictions"):
            train_idx = np.ones(self.n_obs, dtype=bool)
            train_idx[i] = False

            # Fit on N-1 using the established regression path
            self.design_tensor = orig_design[train_idx, :, :]
            self.outcome_tensor = orig_outcome[train_idx, :, :]
            self.weight_vector = orig_weights[train_idx]
            self.n_obs = self.design_tensor.shape[0]
            self.out_dir = params_dir

            self.BETA, self.T, self.R2 = self.voxelwise_regression(regression_idx=regression_idx)
            self._save_nifti_maps()

            # Predict on held-out using the established prediction loader
            self.design_tensor = orig_design[i:i+1, :, :]
            self.n_obs = 1
            self.PREDICTIONS = self._run_prediction_switch(params_dir)
            preds[i, :] = self.PREDICTIONS[0, :]

        # Restore original state
        self.design_tensor = orig_design
        self.outcome_tensor = orig_outcome
        self.weight_vector = orig_weights
        self.n_obs = orig_n_obs
        self.out_dir = orig_out_dir

        self.PREDICTIONS = preds
        self._save_nifti_maps()
    
    def run_single_multiout_regression(self, permutation=False):
        """Runs regression across all outputs a single time and returns the associated arrays."""
        B_multi = np.zeros((self.n_contrasts, self.n_voxels, self.n_outputs))
        T_multi = np.zeros((self.n_contrasts, self.n_voxels, self.n_outputs))
        R2_multi = np.zeros((self.n_contrasts, self.n_voxels, self.n_outputs))
        for j in range(self.n_outputs):
            B_multi[:,:,j], T_multi[:,:,j], R2_multi[:,:,j] = self.voxelwise_regression(permutation=permutation, regression_idx=j)
        
        if permutation == False:            # Store the results in the class attributes for use later
            self.B_multi, self.T_multi, self.R2_multi = B_multi, T_multi, R2_multi
        return B_multi, T_multi, R2_multi

    def run_all_outputs(self):
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
            self._save_nifti_maps()

    def run(self):
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
        self._save_nifti_maps()

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
