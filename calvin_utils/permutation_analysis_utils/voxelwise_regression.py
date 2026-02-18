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
    _get_max_stat(arr, pseudo_var_smooth=True, t=75)
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
        self.XTX_inv = np.zeros((self.n_preds, self.n_preds, self.n_voxels))
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
    def _check_broadcastable(self, permutation):
        '''Can only broadcast if in a permutation, XTX_inv is filled, and Y is not voxelwise'''
        if self.regression_type=='naive_bayes':
            return True
        if not permutation:
            return False
        if np.all(self.XTX_inv == 0):
            return False
        if self.outcome_tensor.shape[2] > 1:
            return False
        return True

    def _design_voxel_invariant(self, sample_voxels=8, atol=0.0):
        """Return True if design is identical across voxels (scalar-only IVs)."""
        X = self.design_tensor
        if X.ndim != 3:
            return False
        v = X.shape[2]
        if v == 1:
            return True
        idx = np.linspace(0, v - 1, min(sample_voxels, v), dtype=int)
        base = X[:, :, 0]
        for j in idx:
            if atol == 0.0:
                if not np.array_equal(X[:, :, j], base):
                    return False
            else:
                if not np.allclose(X[:, :, j], base, atol=atol, rtol=0.0):
                    return False
        return True

    def _weights_uniform(self, atol=0.0):
        """Return True if weights are all equal (or effectively equal)."""
        W = self.weight_vector
        if W is None:
            return True
        if W.size == 0:
            return True
        if atol == 0.0:
            return np.all(W == W[0])
        return np.allclose(W, W[0], atol=atol, rtol=0.0)

    def _can_use_ccm_fast(self):
        """Fast path if DV is voxelwise and IVs are voxel-invariant (scalar-only)."""
        if self.regression_type != 'linear':
            return False
        if self.outcome_tensor.shape[2] <= 1:
            return False
        if not self._design_voxel_invariant():
            return False
        if not self._weights_uniform():
            return False
        return True

    def _run_ccm_fast_linear(self, regressor, regressand, weights, regression_idx):
        """
        Use the fast vectorized regression from RegressionNPYAnalysis
        for the scalar-IV / voxelwise-DV case.
        """
        from calvin_utils.neuroimaging_utils.ccm_utils.npy_regression import RegressionNPYAnalysis

        # Shared X (n_obs, n_preds)
        X = regressor[:, :, 0]
        # Voxelwise Y (n_obs, n_vox)
        Y = regressand[:, regression_idx, :]

        # Build a lightweight RegressionNPYAnalysis instance without __init__
        reg = RegressionNPYAnalysis.__new__(RegressionNPYAnalysis)
        reg.verbose = False
        reg.n_subjects = X.shape[0]
        reg.contrast_matrix = self.contrast_matrix

        # RegressionNPYAnalysis expects Y as (n_vox, n_subj)
        beta, _, mse, XtX_inv = reg.run_regression(X, Y.T)
        _, contrast_tmaps = reg.apply_contrast(beta, XtX_inv, mse)

        # Compute R2 with current weight vector (uniform weights only)
        Y_hat = X @ beta  # (n_obs, n_vox)
        R2 = self.get_r2(Y, Y_hat, weights)
        return beta, contrast_tmaps, R2
    
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
    
    ### Statistical Helpers ###
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
        lr = LogisticRegression(random_sitate=0).fit(X, y, sample_weight=W)
        B = lr.coef_
        XTX_inv = np.linalg.pinv( X.T @ X )
        P = lr.predict(X)
        return B, XTX_inv, P

    def _clipped_sigmoid(self, a, eps=1e-9):
        '''Returns a numerically stable sigmoid to get p(y|x)'''
        out = np.empty_like(a, dtype=float)
        pos = a >= 0
        out[pos]  = 1.0 / (1.0 + np.exp(-a[pos]))
        ea = np.exp(a[~pos])
        out[~pos] = ea / (1.0 + ea)
        return np.clip(out, eps, 1 - eps)

    def _irls(self, X, Y, B, P, W, eps=1e-9):
        '''IRLS from Newton-raphson rewritten in OLS form'''
        V = P * (1.0 - P)                 # bernoulli probability variance function
        V = np.maximum(V, eps)            # add numeric stability
        s = np.sqrt(W*V)                  # sqrt of weights * bernoulli variance (effective weight). Scaled.

        Z = (X@B) + ((Y - P) / V)         # current response 
        Z_w = Z * s                       # gets Z weighed by ~scaled effective weights
        
        X_w = X * s[:, None]               # gets X weighed by ~scaled effective weights
        XtX = X_w.T @ X_w                  # XTWX = covariance matrix = observed fisher info = observed hessian!
        
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
    
    def _fit_logistic(self, X, Y, W, max_iter=50):
        '''
        IRLS for argmax(β) of:
            log-likelihood(β) = Σ_i  W[YβX - log(1 + e^βX)]
        which can be maximized with 
        Bn = (XTWX)^-1 XTW_eZ_e <- used to update W 
        z  = Xβ + ( (y - p) / p(1-p) )
        n  = Xβ
        '''
        B = np.zeros(self.n_preds)       # init B
        for _ in range(max_iter):
            P = self._clipped_sigmoid(X@B)
            B_n, XtX = self._irls(X, Y, B, P, W)
            if self._mahalanobis_norm_test(B_n, B, XtX) or self._infinity_norm_test(B_n, B):
                B = B_n
                break
            B = B_n
        return B, np.linalg.pinv( X.T @ X )
    
    #### Regression Methods ####
    def get_r2(self, Y, Y_HAT, W, e=1e-6):
        """
        Weighted R^2: 1 - SSE_w / TSS_w
        Y      : (o,)
        Y_HAT  : (o,) or (o,vox)
        W      : (o,)
        Return : scalar if Y_HAT is (o,), else (1,vox)
        """
        wsum = W.sum()
        ybar_w = (W * Y).sum() / wsum   # scalar weighted mean
        
        if Y_HAT.ndim == 1: # (o,)
            sse = np.sum(W * (Y - Y_HAT) ** 2)               # scalar
            tss = np.sum(W * (Y - ybar_w) ** 2)              # scalar
            return 1 - (sse / (tss + e))                     # scalar
        elif Y_HAT.ndim == 2: # (o, v)
            sse = np.sum(W[:, None] * (Y[:, None] - Y_HAT) ** 2, axis=0, keepdims=True)              # (1, v)
            tss = np.sum(W * (Y - ybar_w) ** 2)              # scalar
            return 1 - (sse / (tss + e))                     # (1, v)
    
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
            LL += np.log(p1*num/(p0*den+eps))                       # (n_obs, n_vox) <- (n_obs, n_vox) / (n_obs / n_vox)
        B = np.einsum("pov,ov->pv", X_inv, LL)                      # (n_obs, n_vox) <- (n_obs, n_preds, n_voxels) @ (n_obs, n_voxels)
        # P  = 1.0 / (1.0 + np.exp(-LL))                            # (n_obs, n_vox) # Skipping for speed
        PR2 = np.zeros((1, X.shape[2])) # self._get_pseudo_r2(Y, W, P)   # Skipping pseudo-R2 for speed
        T = self.apply_contrasts(XTX_inv, B, MSE=1)                 # (n_contrast, n_voxels) <- setting MSE = 1 converts this to a Wald t-stat
        return B, T, PR2                    
    
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
        Y : (n_obs, )
        W : (n_obs, )
        XtX_inv : (preds, preds, voxels) 
        """
        wsqrt = np.sqrt(W)                                          # (n_obs,)
        Xw = X * wsqrt[:, None, None]                               # (n_obs, n_preds, n_voxels) <- (n_obs, n_preds, n_voxels) * (n_obs, 1, 1)
        Yw = Y.flatten() * wsqrt.flatten()                          # (n_obs, )           <- (n_obs, ) * (n_obs, )
        XtY = np.einsum("opv,o->pv", Xw, Yw)                        # (n_voxels, n_preds) <- (n_obs, n_preds n_voxels) @ (n_obs, )
        BETA = np.einsum("ppv,pv->pv", XtX_inv, XtY)                # (n_preds, n_voxels) <- (n_preds, n_preds, n_voxels) einsum (n_voxels, n_preds)
        Y_HAT = np.einsum("opv,pv->ov", X, BETA)                    # (obs, n_voxels) <- (n_obs, n_preds, n_voxels) einsum(n_preds, n_voxels)
        RES = Y - Y_HAT                                             # (n_obs, n_voxls) <- (n_obs, n_voxls) - (n_obs, n_voxls)
        dof = self.n_obs - self.n_preds                             # (1,)
        MSE = np.sum(W[:, None] * RES**2, axis=0) / dof             # (1, n_voxels) <- summed (n_obs, )^2
        T = self.apply_contrasts(XtX_inv, BETA, MSE)                # (n_contrasts, n_voxels) <- (n_cov, n_voxels) / (n_cov, n_voxels)
        R2 = self.get_r2(Y.flatten(), Y_HAT, W.flatten())           # (1, n_voxels)
        return BETA, T, R2
    
    def _run_linear(self, X, Y, W):
        """
        Runs a standard linear regression. Gets Betas and T values.
            
        Parameters
        ----------
        X : (n_obs, n_preds) design matrix
        Y : (n_obs,) data (single voxel)
        W : (n_obs,) weights  (use w_i = sample_size_i  or any precision proxy)
        """
        wsqrt = np.sqrt(W)                                              # (n_obs,) <-- rooting w array enables its multiple multiplications to remain equivalet to X.T @ W @ X
        Xw = X * wsqrt[:, None]                                         # scale each row
        Yw = Y * wsqrt                                                  # scale each row
        XtX_inv = np.linalg.pinv(Xw.T @ Xw)                             # (preds, preds)
        BETA = XtX_inv @ Xw.T @ Yw                                      # (n_preds,) <- (n_preds,n_obs) @ (n_obs,)
        Y_HAT  = X @ BETA                                               # (obs,) <- (obs, preds) @ (preds,)
        residuals = Y - Y_HAT                                           # (n_obs, ) <- (n_obs, ) - (n_obs,)
        dof = self.n_obs - self.n_preds                                 # (1,)
        mse = np.sum(W * residuals**2, axis=0) / dof                    # (1,) <- summed (n_obs, )^2
        T = self.apply_contrasts(XtX_inv, BETA, mse)                    # (n_contrasts,)
        R2 = self.get_r2(Y, Y_HAT, W)                                   # (1,)
        return BETA, T, R2, XtX_inv                                     # beta is (predictors,), while T and P are (contrasts,)
    
    def _run_regression(self, X, Y, W, idx):
        if (self.regression_type=='linear') and (idx != "whole_brain"):
            B, T, R2, self.XTX_inv[:, :, idx] = self._run_linear(X,Y,W)
        elif (self.regression_type=='linear') and (idx == "whole_brain"):
            B, T, R2 = self._run_precomputed_linear(X,Y,W, self.XTX_inv)
        elif self.regression_type=='logistic':
            B, T, R2 = self._run_logistic(X, Y, W)
        elif self.regression_type=='naive_bayes':
            B, T, R2 = self._run_naive_bayes(X,Y,W)
        else:
            raise ValueError(f"Regression type {self.regression_type} not implemented. Please set regression_type='linear' or 'logistic'.")
        return B, T, R2

    def _run_naive_bayes_batched(self, regressor, regressand, weights, regression_idx, batch_size=5000):
        """
        Batch naive_bayes across voxels to avoid huge X_inv/XTX_inv allocations.
        Supports scalar outcome (Y is 1D). If outcome is voxelwise, falls back to loop.
        """
        # If outcome is voxelwise, fall back to looped path (Y differs by voxel).
        if regressand.shape[2] == self.n_voxels:
            BETA = np.zeros((self.n_preds, self.n_voxels))
            T = np.zeros((self.n_contrasts, self.n_voxels))
            R2 = np.zeros((1, self.n_voxels))
            for idx in tqdm(range(self.n_voxels), desc='Running voxelwise regressions'):
                X, Y, W = self._prep_targets(regressor, regressand, weights, idx, regression_idx)
                BETA[:, idx], T[:, idx], R2[:, idx] = self._run_regression(X, Y, W, idx)
            return BETA, T, R2

        # Scalar outcome: batch across voxels
        Y = regressand[:, regression_idx, 0]
        BETA = np.zeros((self.n_preds, self.n_voxels))
        T = np.zeros((self.n_contrasts, self.n_voxels))
        R2 = np.zeros((1, self.n_voxels))

        for s, e in voxel_batches(self.n_voxels, batch_size):
            # build per-voxel design for batch
            if regressor.shape[2] == self.n_voxels:
                Xb = regressor[:, :, s:e]
            else:
                Xb = np.broadcast_to(regressor[:, :, 0][:, :, None], (self.n_obs, self.n_preds, e - s)).copy()

            X_inv_b, XTX_inv_b = self._prep_naive_bayes(Xb)
            B, Tb, R2b = self._run_naive_bayes(Xb, Y, weights, X_inv_b, XTX_inv_b)
            BETA[:, s:e] = B
            T[:, s:e] = Tb
            R2[:, s:e] = R2b
        return BETA, T, R2

    def _looped_vs_broadcast_regression(self, regressor, regressand, weights, regression_idx, permutation):
        """Attemps to do a whole-brain broadcasting if possible. Otherwise defaults to standard looped regression."""
        broadcastable = self._check_broadcastable(permutation)

        if self.regression_type == 'naive_bayes':
            return self._run_naive_bayes_batched(regressor, regressand, weights, regression_idx)

        if self._can_use_ccm_fast():
            return self._run_ccm_fast_linear(regressor, regressand, weights, regression_idx)

        if not broadcastable:
            BETA = np.zeros((self.n_preds, self.n_voxels))
            T = np.zeros((self.n_contrasts, self.n_voxels))
            R2 = np.zeros((1, self.n_voxels))  
            for idx in (range(self.n_voxels) if permutation else tqdm(range(self.n_voxels), desc='Running voxelwise regressions')):
                X, Y, W = self._prep_targets(regressor, regressand, weights, idx, regression_idx)
                BETA[:,idx], T[:,idx], R2[:,idx] = self._run_regression(X, Y, W, idx)
        else:
            X, Y, W = self._prep_targets(regressor, regressand, weights, "whole_brain", regression_idx)
            BETA, T, R2 = self._run_regression(X, Y, W, "whole_brain")
        return BETA, T, R2

    def voxelwise_regression(self, permutation=False, regression_idx=0):
        '''Relies on hat matrix (X'@(X'X)^-1@X')@Y to calculate beta, t-values, and p-values'''      
        regressor, regressand, weights = self._get_targets(permutation)    
        return self._looped_vs_broadcast_regression(regressor, regressand, weights, regression_idx, permutation)
    
    ### P-VALUE METHODS ###
    def _get_max_stat(self, arr, pseudo_var_smooth=True, t=95):
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

    #### Public Code ####
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

# -----------------------
# core linear solver
# -----------------------
def run_linear_batched(X, Y, W, C, ridge=1e-10, eps=1e-9):
    """
    Weighted linear regression, vectorized across voxels.

    X : (o,p) shared   OR (o,p,v) voxelwise for this slice
    Y : (o,v)
    W : (o,)   (broadcasted)
    C : (c,p)

    Returns:
      BETA : (p,v)
      T    : (c,v)     Wald t using per-voxel XtX_inv and MSE
      R2   : (v,)
    """
    o = X.shape[0]
    if X.ndim == 2:
        # --- fast shared-X path ---
        o, p = X.shape
        v = Y.shape[1]

        wsqrt = np.sqrt(W)                     # (o,)
        Xw = X * wsqrt[:, None]                # (o,p)
        WY = W[:, None] * Y                    # (o,v)

        XtX = Xw.T @ Xw                        # (p,p)
        XtX += ridge * np.eye(p)
        XtX_inv = np.linalg.pinv(XtX)          # (p,p)

        XwT_Yw = Xw.T @ WY                     # (p,v)
        BETA = XtX_inv @ XwT_Yw                # (p,v)

        Y_hat = X @ BETA                        # (o,v)
        resid = Y - Y_hat                       # (o,v)
        dof = max(o - p, 1)
        mse = (W[:, None] * resid**2).sum(axis=0) / dof   # (v,)

        # contrasts
        NUM = C @ BETA                          # (c,v)
        G = C @ XtX_inv @ C.T                   # (c,c)
        var_c = np.diag(G)                      # (c,)
        DEN = np.sqrt(var_c[:, None] * mse[None, :] + eps)
        T = NUM / DEN

        # weighted R^2
        ybar = (W[:, None] * Y).sum(axis=0) / W.sum()
        sst  = (W[:, None] * (Y - ybar[None, :])**2).sum(axis=0)
        sse  = (W[:, None] * resid**2).sum(axis=0)
        R2   = 1.0 - sse / (sst + eps)
        return BETA, T, R2

    # --- voxelwise-X path (o,p,v) ---
    o, p, v = X.shape

    wsqrt = np.sqrt(W)[:, None, None]          # (o,1,1)  pure broadcasting
    Xw = X * wsqrt                              # (o,p,v)
    Yw = Y * np.sqrt(W)[:, None]                # (o,v)

    # XtX per voxel
    # (p,p,v) <= sum over obs: Xw[o,p,v] * Xw[o,q,v]
    XtX = np.einsum('opv,oqv->pqv', Xw, Xw)     # (p,p,v)
    XtX += ridge * np.eye(p)[:, :, None]

    # RHS per voxel
    XwT_Yw = np.einsum('opv,ov->pv', Xw, Yw)    # (p,v)

    # invert batched
    XtX_inv = np.linalg.pinv(XtX.transpose(2,0,1)).transpose(1,2,0)  # (p,p,v)

    # betas, fits, residuals
    BETA  = np.einsum('pqv,pv->qv', XtX_inv, XwT_Yw)   # (p,v)
    Y_hat = np.einsum('opv,pv->ov', X, BETA)           # (o,v)
    resid = Y - Y_hat                                  # (o,v)

    dof = np.maximum(o - p, 1)
    mse = (W[:, None] * resid**2).sum(axis=0) / dof    # (v,)

    # contrasts (per-voxel variance from XtX_inv)
    NUM   = np.einsum('cp,pv->cv', C, BETA)            # (c,v)
    CXinv = np.einsum('cp,pqv->cqv', C, XtX_inv)       # (c,p,v)
    G     = np.einsum('cqv,dq->cdv', CXinv, C)         # (c,c,v)
    var_c = np.einsum('ccv->cv', G)                    # diag for each v
    DEN   = np.sqrt(var_c * mse[None, :] + eps)
    T     = NUM / DEN

    # weighted R^2
    ybar = (W[:, None] * Y).sum(axis=0) / W.sum()
    sst  = (W[:, None] * (Y - ybar[None, :])**2).sum(axis=0)
    sse  = (W[:, None] * resid**2).sum(axis=0)
    R2   = 1.0 - sse / (sst + eps)

    return BETA, T, R2
