import numpy as np
from typing import Tuple
from scipy.stats import pearsonr, spearmanr

def check_arrs(X:np.ndarray, Y:np.ndarray) -> Tuple[int, int]:
    """Checks arrays and outputs variable counts"""
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"The number of rows in X ({X.shape}) must match the number of rows in Y ({Y.shape}).")
    return X.shape[1], Y.shape[1]  

def efficient_rank(arr:np.ndarray) -> np.ndarray:
    """
    Runs a broadcast ranking approach. Expects an array of (Observations, Variables).
    Forces sorting along observations to rank observations. 
    Args:
        arr (np.ndarray): Array to be ranked.
        axis (int): Axis along which to rank the data.
    
    Returns:
        np.ndarray: Ranked data.
    """
    ranks = np.empty_like(arr, dtype=float) # (Observations, Variables)
    n_obs, n_vars = arr.shape[0], arr.shape[1]
    sorter = np.argsort(arr, axis=0)
    ranks[sorter, np.arange(n_vars)] = np.arange(n_obs)[:, np.newaxis] + 1 # range [1, n_obs], shape (n_obs, 1)
    return ranks

def run_spearman(X:np.array, Y:np.array, vectorize:bool = False, debug:bool = False) -> np.ndarray:
    """
    Will run a vectorized or looping spearman Rho.
    Args:
        X: arr of shape (Observations, Variables).
        Y: arr of shape (Observations, Variables).
        Vectorize: choose to use faster custom implementation of Spearman. If false, uses scipy.stats.spearmanr.
    Returns:
        Rho: arr of shape (Indepvars, DepVars)
    """
    n_indep_vars, n_dep_vars = check_arrs(X, Y)
    if not vectorize:
        RHO = np.zeros((n_indep_vars, n_dep_vars))
        for i in range(n_indep_vars):
            for j in range(n_dep_vars):
                RHO[i, j], _ = spearmanr(X[:,i], Y[:,j]) # (Indepvars, Depvars)
    else:
        X, Y = efficient_rank(X), efficient_rank(Y)
        RHO = run_pearson(X, Y, vectorize=True, debug=False)
    mask = np.isfinite(RHO)
    RHO = np.nan_to_num(RHO, nan=0, posinf=np.max(RHO[mask]),  neginf=np.min(RHO[mask]))
    return RHO

def run_pearson(X:np.array, Y:np.array, vectorize:bool = True, debug:bool = False) -> np.ndarray:
    """
    Will run a vectorized or looping pearson R.
    
    Formula (Linalg Formulation):
        r = XTY / (sqrt(XTX) @ sqrt(YTY).T)
         
    However, need to be aware the XTX gives the Gram Matix. To recover SST for a the X variables, take the diagonal.
    Further, the SSTX and SSTY need to be out-producted to scale the deviances down with an appropriately sized matrix.  
    
    Args:
        X: arr of shape (Observations, Variables).
        Y: arr of shape (Observations, Variables).
        Vectorize: choose to use faster custom implementation of Pearson. If false, uses scipy.stats.pearsonr.
    Returns:
        Rho: arr of shape (Indepvars, DepVars)
    """
    n_indep_vars, n_dep_vars = check_arrs(X, Y)
    
    if not vectorize:    
        R = np.zeros((n_indep_vars, n_dep_vars))
        for i in range(n_indep_vars):
            for j in range(n_dep_vars):
                R[i, j], _ = pearsonr(X[:,i], Y[:,j]) # (Indepvars, Depvars)
    else:
        # Calculate Numerator
        X_C = X - X.mean(axis=0)
        Y_C = Y - Y.mean(axis=0)
        NUMERATOR = np.dot(X_C.T, Y_C) # Transposing to facilitate matmul (IndepVars, DepVars) <- (IndepVars, Vox) @ (Vox, DepVars)
        # Calculate Denominator
        GRAM_X = X_C.T @ X_C # dotproduct sums over the inner dimension (the shared dimension). (IndepVars, IndepVars) <- (IndepVars, Vox) @ (Vox, IndepVars)
        GRAM_Y = Y_C.T @ Y_C # dotproduct sums over the inner dimension (the shared dimension) (DepVars, DepVars) <- (DepVars, Vox) @ (Vox, DepVars)
        SST_X = np.diag(GRAM_X) # Trace gives the diagonal, which is the total sum of squares for each depvar. Off-diagonal is nonsense and is SST for one depvar to a separate depvar. (IndepVars,) <- Diag(IndepVars, IndepVars)
        SST_Y = np.diag(GRAM_Y) # Trace gives the diagonal, which is the total sum of squares for each indepvar. Off-diagonal is nonsense and is SST for one depvar to a separate depvar. (DepVars,) <- Diag(DepVars, DepVars)
        DENOMINATOR = np.sqrt(SST_X)[:, np.newaxis] @ np.sqrt(SST_Y)[np.newaxis, :] # Pairwise (Indepvars, DepVars) <- (Indepvars, 1) @ (1, Depvars)

        # Pearson
        R = NUMERATOR / DENOMINATOR # (Indepvars, DepVars) <- (Indepvars, DepVars) / (Indepvars, DepVars)
        
        if debug:
            print("X: ", X.shape, " Y: ", Y.shape, " X_BAR :", X_BAR.shape, " Y_BAR ", Y_BAR.shape, " Y_C: ", Y_C.shape, " X_C: ", X_C.shape, "NUMERATOR :", NUMERATOR.shape, "DENOMINATOR: ", DENOMINATOR.shape, "R: ", r.shape)
            print(np.max(X_C), np.max(Y_C), NUMERATOR, DENOMINATOR)
        if debug:
            print('Correlation matrix shape: ', R.shape)
    
    # Remove NaNs and Infs. 
    mask = np.isfinite(R)
    R = np.nan_to_num(R, nan=0, posinf=np.max(R[mask]),  neginf=np.min(R[mask]))
    return R