import os
import numpy as np
import pandas as pd
import forestplot as fp
from jax import jit
import jax.numpy as jnp
from tqdm import tqdm
from scipy.stats import rankdata, spearmanr
from calvin_utils.ccm_utils.npy_utils import DataLoader
from calvin_utils.ccm_utils.stat_utils_jax import _rankdata_jax, calculate_spearman_r_map_jax, _calculate_pearson_r_map_jax

class CorrelationCalculator:
    def __init__(self, method='pearson', verbose=False, use_jax=False, datasets_to_flip = []):
        '''
        Params
        datasets_to_flip : list, optional
            A list of keys corresponding to datasets which should have their correlation maps multiplied by -1. 
            This is typically performed to align maps to enable testing of topology, controlling for sign. 
        '''
        self.method = method
        self.verbose = verbose
        self.use_jax = use_jax
        self.datasets_to_flip = datasets_to_flip

    @staticmethod
    def _check_for_nans(array, nanpolicy='remove', verbose=False):
        '''Check array for nan and raise error if present'''
        if (np.isnan(array).any()) or (np.any(~np.isfinite(array))):
            if verbose:
                print("NaN detected in the array.")
                print(f"Array shape: {array.shape}")
                print(f"Array with NaNs: {array}")
                nan_indices = np.argwhere(np.isnan(array))
                print(f"Indices of NaNs: {nan_indices}")
                for index in nan_indices:
                    print(f"NaN at index: {index}, value: {array[tuple(index)]}")
                inf_indices = np.argwhere(np.isinf(array))
                if inf_indices.size > 0:
                    print(f"Indices of Infs: {inf_indices}")
                    for index in inf_indices:
                        print(f"Inf at index: {index}, value: {array[tuple(index)]}")
            if nanpolicy=='remove':
                max_val = np.nanmax(array[np.isfinite(array)])
                min_val = np.nanmin(array[np.isfinite(array)])
                array = np.nan_to_num(array, nan=0, posinf=max_val, neginf=min_val)
            elif nanpolicy=='permit':
                array = array
            elif nanpolicy=='stop':
                raise ValueError("The array contains NaNs.")
            else:
                raise ValueError("Selected nanpolicy does not exist. choose 'stop' | 'permit' | 'remove' ")
        return array
            
    def _rankdata(self, array, vectorize=True):
        """Vectorized ranking function using NumPy. Handles ties sloppily."""
        if self.use_jax:
            return _rankdata_jax(array, vectorize)
        
        if vectorize:       #very fast, but no tie handling.
            ranks = np.empty_like(array, dtype=float)       #shape (voxels, n) initialized array
            rows = np.arange(array.shape[0])[:, None]       #shape (voxels, 1) Row indices, ascending in order. 
            cols = np.arange(array.shape[1])                #shape (1,      n) Col indices, ascending in order. 
            
            idx_sorted = np.argsort(array, axis=0)          #shape (voxels, n) Row indices, ascending in rank. For each column. 
            ranks[idx_sorted, cols] = rows                  #shape (voxels, n) Assigns row indices by ascending rank. For each column. 
        else:               #very slow, but handles ties better.
            ranks = np.apply_along_axis(rankdata, 0, array) 
        return ranks                                        #returns the rank matrix, not the sorted data. 

    def _calculate_spearman_r_scipy_map(self, niftis, indep_var):
        """
        Runs the voxelwise Spearman R, which has a cleaner distribution than the vectorized version.
        Used for figure-grade maps, and for smaller analysis. Unsuited for permutations. 
        """
        X = indep_var
        Y = niftis
        RHO = np.zeros((1,Y.shape[1]))
        for i in tqdm(range(Y.shape[1]), desc='Voxelwise Spearman'):
            RHO[0, i], _ = spearmanr(X[:], Y[:,i])
        return RHO

    def efficient_rankdata(self, arr, axis=0):
        """
        Efficiently rank data using numpy.
        
        Args:
            arr (np.ndarray): Array to be ranked.
            axis (int): Axis along which to rank the data.
        
        Returns:
            np.ndarray: Ranked data.
        """
        arr = np.asarray(arr)
        ranks = np.empty_like(arr, dtype=float)
        if axis == 0:
            sorter = np.argsort(arr, axis=axis)
            ranks[sorter, np.arange(arr.shape[1])] = np.arange(arr.shape[0])[:, np.newaxis] + 1
        else:
            sorter = np.argsort(arr, axis=axis)
            ranks[np.arange(arr.shape[0])[:, np.newaxis], sorter] = np.arange(arr.shape[1]) + 1
        return ranks

    # Rank the data
    def _calculate_spearman_manual(self, X, Y):
        X_ranked = self.efficient_rankdata(X, axis=0)
        Y_ranked = self.efficient_rankdata(Y, axis=0)

        if X_ranked.shape[0] != Y_ranked.shape[0]:
            raise ValueError(f"The number of rows in X ({X_ranked.shape}) must match the number of rows in Y ({Y_ranked.shape}).")

        # Calculate differences in ranks
        D = np.square(X_ranked - Y_ranked)

        # Sum the squared differences across patients
        SIGMA_D = np.sum(D, axis=0)

        # Calculate Spearman correlation
        N = X_ranked.shape[0]
        RHO = 1 - ( (6 * SIGMA_D) / (N * (N**2 - 1)) )
        return RHO

    def _calculate_spearman_r_map(self, niftis, indep_var):
        """Calculate the Spearman rank-order correlation coefficient for each voxel in a fully vectorized manner."""
        if self.use_jax:
            return calculate_spearman_r_map_jax(niftis, indep_var)
        
        ranked_indep_var = rankdata(indep_var)[:, np.newaxis] #scipy.stats.rankdata  handles ties well
        try:
            rho = self._calculate_pearson_r_map(niftis, ranked_indep_var)
        except Exception as e:
            if ("shapes" in str(e)) and ("not aligned: " in str(e)):
                raise RuntimeError("Shape alginment error. You have a mismatched number of niftis and observations. Repeat Notebook 01 ensuring same number of observations in each matrix.")
            else:
                raise RuntimeError(f"Error: {e}")
        return rho

    def _calculate_pearson_r_map(self, niftis, indep_var):
        if self.use_jax:
            return _calculate_pearson_r_map_jax(niftis, indep_var)
        
        X = indep_var
        Y = niftis
        X_BAR = X.mean(axis=0)[:, np.newaxis]
        Y_BAR = Y.mean(axis=0)[np.newaxis, :]
        X_C = X - X_BAR
        Y_C = Y - Y_BAR
        NUMERATOR = np.dot(X_C.T, Y_C)
        SST_X = np.sum((X - X_BAR)**2, axis=0)
        SST_Y = np.sum((Y - Y_BAR)**2, axis=0)
        DENOMINATOR = np.sqrt(SST_X * SST_Y)
        r = NUMERATOR / DENOMINATOR
        
        if self.verbose:
            print(f"Shape of X: {X.shape}")
            print(f"Shape of Y: {Y.shape}")
            print(f"Shape of X_BAR: {X_BAR.shape}")
            print(f"Shape of Y_BAR: {Y_BAR.shape}")
            print(f"Shape of X_C: {X_C.shape}")
            print(f"Shape of Y_C: {Y_C.shape}")
            print(f"Shape of NUMERATOR: {NUMERATOR.shape}")
            print(f"Shape of DENOMINATOR: {DENOMINATOR.shape}")
            print("Shape of r: ", r.shape)
        return r

    def _calculate_average(self, data):
        return data.mean(axis=0).flatten()

    def _process_data(self, data):
        data['niftis'] = self._check_for_nans(data['niftis'])
        data['indep_var'] = self._check_for_nans(data['indep_var'])
        if self.method == 'pearson':
            self.correlation_map = self._calculate_pearson_r_map(data['niftis'], data['indep_var'])
        elif self.method == 'spearman':
            self.correlation_map = self._calculate_spearman_r_map(data['niftis'], data['indep_var'])
        elif self.method == 'spearman_scipy':
            self.correlation_map = self._calculate_spearman_r_scipy_map(data['niftis'], data['indep_var'])
        elif self.method == 'spearman_manual':
            self.correlation_map = self._calculate_spearman_manual(data['niftis'], data['indep_var'])
        else:
            self.correlation_map = self._calculate_average(data['niftis'])
        return self.correlation_map
    
    def generate_correlation_maps(self, data_loader):
        corr_map_dict = {}
        for dataset_name in data_loader.dataset_paths_dict.keys():
            print(f"Evaluating {dataset_name}")
            if self.method == 'pearson':
                data = data_loader.load_dataset(dataset_name)
            elif self.method =='spearman':
                data = data_loader.load_dataset(dataset_name, nifti_type='niftis_ranked') 
            else:
                data = data_loader.load_dataset(dataset_name) 
            corr_map_dict[dataset_name] = self._process_data(data)
            
            if dataset_name in self.datasets_to_flip:           # manually flip dataset correlation for topology test
                print('flipping')
                corr_map_dict[dataset_name] = corr_map_dict[dataset_name] * -1
        return corr_map_dict
    
class MetaConvergenceForestPlot:
    """
    Class for generating and saving a forest plot visualizing meta-convergence data.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with columns: 'Dataset', 'CI Lower', 'CI Upper', and 'Mean R'.
    sig_digits : int, optional
        Number of significant digits displayed (default: 2).
    out_dir : str, optional
        Directory path for saving the plot (default: None).
    table : bool, optional
        Whether to format the plot as a table (default: False).
    """

    def __init__(self, data, sig_digits=2, out_dir=None, table=False):
        self.data = data.copy()
        self.sig_digits = sig_digits
        self.out_dir = out_dir
        self.table = table

    def _prepare_table_data(self, min_rows=2):
        """
        Ensures DataFrame has a minimum number of rows by padding with NaNs if necessary.
        """
        current_row_count = len(self.data)
        if self.table and current_row_count < min_rows:
            rows_to_add = min_rows - current_row_count
            padding_df = pd.DataFrame({
                'Mean R': [1] * rows_to_add,
                'CI Lower': [1] * rows_to_add,
                'CI Upper': [1] * rows_to_add,
                'Dataset': ['(placeholder)'] * rows_to_add
            })
            self.data = pd.concat([self.data, padding_df], ignore_index=True)

    def create_forest_plot(self, x_label="Mean R", estimate_col="Mean R",
                            varlabel_col="Dataset", ll_col="CI Lower", hl_col="CI Upper"):
        """
        Creates and displays the forest plot.
        """
        ax = fp.forestplot(
            dataframe=self.data,
            estimate=estimate_col,
            varlabel=varlabel_col,
            ll=ll_col,
            hl=hl_col,
            xlabel=x_label,
            ylabel='Est. (95% Conf. Int.)',
            decimal_precision=self.sig_digits,
            capitalize='capitalize',
            color_alt_rows=False,
            table=self.table,
            flush=False,
            **{
                "marker": "D",
                "markersize": 150,
                "xlinestyle": (0, (10, 5)),
                "xlinecolor": "#808080",
                "xtick_size": 12,
                "fontfamily": "helvetica"
            }
        )
        self.fig = ax.figure
        self.fig.show()

    def save_plot(self, filename="meta_convergence_forest_plot"):
        """
        Saves the forest plot in PNG and SVG formats.
        """
        if self.out_dir:
            output_path = os.path.join(self.out_dir, 'forest_plots')
            os.makedirs(output_path, exist_ok=True)
            for ext in ['png', 'svg']:
                full_path = os.path.join(output_path, f"{filename}.{ext}")
                self.fig.savefig(full_path, bbox_inches='tight')
            print(f'Saved plot as {filename}.png and {filename}.svg in {output_path}')

    def run(self):
        """
        Executes the full pipeline for generating and saving the forest plot.
        """
        self._prepare_table_data()
        self.create_forest_plot()
        self.save_plot()

        