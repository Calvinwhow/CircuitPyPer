import jax.numpy as jnp
from jax import jit
import time

@jit
def _rankdata_jax(array):
    """Vectorized ranking function using JAX. Handles ties sloppily."""
    ranks = jnp.empty_like(array, dtype=jnp.float64)  # shape (voxels, n) initialized array
    rows = jnp.arange(array.shape[0])[:, None]        # shape (voxels, 1) Row indices, ascending in order.
    cols = jnp.arange(array.shape[1])                 # shape (1, n) Col indices, ascending in order.

    idx_sorted = jnp.argsort(array, axis=0)           # shape (voxels, n) Row indices, ascending in rank. For each column.
    ranks = ranks.at[idx_sorted, cols].set(rows)      # shape (voxels, n) Assigns row indices by ascending rank. For each column.
    return ranks                                          # returns the rank matrix, not the sorted data.

@jit
def calculate_spearman_r_map_jax(niftis, indep_var):
    """
    Calculate the Spearman rank-order correlation coefficient for each voxel
    in a fully vectorized manner using JAX.

    Parameters:
    -----------
    niftis : jnp.array
        2D array where each row represents a subject and each column represents a voxel.
    indep_var : jnp.array
        1D array representing the independent variable for each subject.

    Returns:
    --------
    rho : jnp.array
        1D array of Spearman's rank correlation coefficients for each voxel.
    """
    ranked_niftis = _rankdata_jax(niftis)
    ranked_indep_var = _rankdata_jax(indep_var)[:, jnp.newaxis]
    rho = _calculate_pearson_r_map_jax(ranked_niftis, ranked_indep_var)
    return rho

@jit
def _calculate_pearson_r_map_jax(niftis, indep_var):
    X = indep_var
    Y = niftis
    X_BAR = X.mean(axis=0)[:, jnp.newaxis]
    Y_BAR = Y.mean(axis=0)[jnp.newaxis, :]
    X_C = X - X_BAR
    Y_C = Y - Y_BAR
    NUMERATOR = jnp.dot(X_C.T, Y_C)
    SST_X = jnp.sum((X - X_BAR)**2, axis=0)
    SST_Y = jnp.sum((Y - Y_BAR)**2, axis=0)
    DENOMINATOR = jnp.sqrt(SST_X * SST_Y)
    r = NUMERATOR / DENOMINATOR
    return r