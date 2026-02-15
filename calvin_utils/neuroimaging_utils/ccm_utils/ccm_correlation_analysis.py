import os
import numpy as np
import pandas as pd
import nibabel as nib
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from nilearn import plotting
from scipy.stats import spearmanr, pearsonr, rankdata
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from calvin_utils.ccm_utils.npy_utils import DataLoader
from calvin_utils.ccm_utils.stat_utils import CorrelationCalculator
from calvin_utils.plotting_utils.simple_heatmap import simple_heatmap

class CorrelationAnalysis:
    def __init__(self, data_dict_path, method='pearson', similarity='pearson', n_permutations=1000, out_dir=None, use_jax=False, datasets_to_flip=[], topology_command=None):
        """
        Initializes the CCMCorrelationAnalysis object with the specified parameters.

        Args:
            data_dict_path (str): Path to the data dictionary file.
            method (str, optional): Method used for correlation calculation. Defaults to 'pearson'.
            similarity (str, optional): Similarity metric to use. Defaults to 'pearson'.
            n_permutations (int, optional): Number of permutations for statistical analysis. Defaults to 1000.
            out_dir (str, optional): Output directory for results. Defaults to None.
            use_jax (bool, optional): Whether to use JAX for computations. Defaults to False.
            datasets_to_flip (list, optional): List of dataset names to flip during analysis. Defaults to [].
            topology_command (Any, optional): Command or configuration for topology analysis. Defaults to None.
        """
        self.data_dict_path = data_dict_path
        self.method = method
        self.similarity = similarity
        self.n_permutations = n_permutations
        self.out_dir = out_dir
        self.corr_map_dict = None
        self.data_loader = None
        self.original_similarity_matrix = None
        self.permuted_similarity_tensor = None
        self.use_jax = use_jax
        self.datasets_to_flip = datasets_to_flip
        self.topology_command = topology_command
        self.correlation_calculator = CorrelationCalculator(method=method, verbose=False, use_jax=self.use_jax)
        self.data_loader = DataLoader(self.data_dict_path)
    
    ### I/O ###
    def save_results(self, prefix=''):
        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)
            np.save(f'{self.out_dir}/{prefix}original_similarity_matrix.npy', self.original_similarity_matrix)
            np.save(f'{self.out_dir}/{prefix}permuted_similarity_tensor.npy', self.permuted_similarity_tensor)
    
    ### Statistical Measurements ###
    def _prepare_map_dict(self, permute=False, datasets_list=[]):
        """
        Generates a dictionary of arrays
        Args:
            permute (bool, optional): If True, permutes the independent variable before 
                calculating correlations. Defaults to False.
            dataset_list (list, optional): List of dataset names to process. If empty, 
                all datasets in `data_loader.dataset_paths_dict` are used. Defaults to [].
        Returns:
            dict: A dictionary where keys are dataset names and values are the 
                corresponding correlation maps.
        """
        
        if datasets_list == []:
            datasets_list = self.data_loader.dataset_paths_dict.keys()
    
        map_dict = {}
        for dataset_name in datasets_list:
            if self.method =='spearman':
                data = self.data_loader.load_dataset(dataset_name, nifti_type='niftis_ranked')
            else: 
                data = self.data_loader.load_dataset(dataset_name)
            
            if permute: np.random.shuffle(data['indep_var'])  # permute the independent variable
            self.correlation_calculator._process_data(data)         # Condense all patient data into a single map 
            if dataset_name in self.datasets_to_flip:  self.correlation_calculator.correlation_map *= -1
            map_dict[dataset_name] = self.correlation_calculator.correlation_map
        return map_dict
    
    def align_spatial_correlation(self, matrix_1, matrix_2):
        """
        Align the spatial correlation values of each r,c pair in the given array
        to the value at index (0,0). If the correlation is negative relative to (0,0),
        multiply it by -1.
        """
        r = pearsonr(x=matrix_1, y=matrix_2)[0]
        if r < 0:
            matrix_2 = matrix_2 * -1
        return matrix_1, matrix_2
    
    def _measure_similarity(self, arr1, arr2, similarity="pearson"):
        """Measures similarity between 2 arrays"""
        if similarity=="pearson":
            similarity, p = pearsonr(arr1, arr2)
        elif similarity=="euclidean":
            similarity = np.linalg.norm(arr1 - arr2)
            p = None
        return similarity, p
    
    def calculate_similarity_matrix(self, corr_map_dict):
        dataset_names = list(corr_map_dict.keys())
        n = len(dataset_names)
        similarity_matrix = np.full((n, n), np.nan)
        
        for i in range(n):
            for j in range(i, n):
                mx_1 = corr_map_dict[dataset_names[i]].flatten()
                mx_2 = corr_map_dict[dataset_names[j]].flatten()

                mx_1 = self.correlation_calculator._check_for_nans(mx_1, nanpolicy='remove', verbose=False)
                mx_2 = self.correlation_calculator._check_for_nans(mx_2, nanpolicy='remove', verbose=False)

                if self.topology_command is None:
                    similarity, _ = self._measure_similarity(mx_1, mx_2) 
                elif self.topology_command == 'r2':
                    similarity, _ = self._measure_similarity(np.square(mx_1), np.square(mx_2)) 
                elif self.topology_command == 'absval':
                    similarity, _ = self._measure_similarity(np.abs(mx_1), np.abs(mx_2)) 
                elif self.topology_command == 'aligned':
                    mx_1, mx_2 = self.align_spatial_correlation(mx_1, mx_2)
                    similarity, _ = self._measure_similarity(mx_1, mx_2)
                else:
                    raise ValueError(f"Error: {self.topology_command} not recognized. Please set topology_command to one of: None | 'r2' | 'absval'.")
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

        return similarity_matrix
    
    ### P-value Methods ###
    def calculate_p_value(self, method='two_tail', absolute_similarity=True):
        """
        method : str
            if true, will take the absolute value of the average similarity before computing. 
        absolute_similarity : bool
            if true, will take the absolute value of the similarity matrices before computing.
            This tests if the topology was similar.
            If false, will test if there was a significant similarity, considering signs. 
        """
        # Remove the diagonal from the comparison
        diag_mask = ~np.eye(self.original_similarity_matrix.shape[0], dtype=bool)  # False on diagonal, True elsewhere
        local_obsv = self.original_similarity_matrix * diag_mask 
        local_perm = self.permuted_similarity_tensor * diag_mask  # Apply across all permutations
        
        # check the absolute correlation.
        if absolute_similarity:
            local_obsv = np.abs(local_obsv)
            local_perm = np.abs(local_perm)
        
        # get averages
        observed_avg = np.nanmean(local_obsv)
        permuted_avg = np.nanmean(local_perm, axis=(1, 2))

        # Get p values
        if method == 'two_tail':
            p_value = np.mean(np.abs(permuted_avg) > np.abs(observed_avg))
        elif method == 'one_tail':
            p_value = np.mean(permuted_avg > observed_avg)
        else:
            raise ValueError('Invalid method. Please choose either "one_tail" or "two_tail".')
        print(f"The observed average similarity is: {observed_avg}")
        print(f'The {method} p-value is: {p_value}. This is the proportion of permuted averages that are greater than the observed average.')
        return p_value, observed_avg

    def calculate_max_stat_p_values(self, method):
        n = self.original_similarity_matrix.shape[0]
        p_value_matrix = np.full((n, n), np.nan)
        max_permuted_values = np.max(np.abs(self.permuted_similarity_tensor), axis=(1, 2)) if method == 'two_tail' else np.max(self.permuted_similarity_tensor, axis=(1, 2))

        for i in range(n):
            for j in range(i, n):
                observed_value = np.abs(self.original_similarity_matrix[i, j]) if method == 'two_tail' else self.original_similarity_matrix[i, j]
                p_value = np.mean(max_permuted_values > observed_value)
                p_value_matrix[i, j] = p_value
                p_value_matrix[j, i] = p_value
        return p_value_matrix

    def calculate_pairwise_p_values_uncorrected(self, method):
        n = self.original_similarity_matrix.shape[0]
        p_value_matrix = np.full((n, n), np.nan)

        for i in range(n):
            for j in range(i, n):
                observed_value = self.original_similarity_matrix[i, j]
                permuted_values = self.permuted_similarity_tensor[:, i, j]
                if method == 'two_tail':
                    p_value = np.mean(np.abs(permuted_values) > np.abs(observed_value))
                elif method == 'one_tail':
                    p_value = np.mean(permuted_values > observed_value)
                p_value_matrix[i,j] = p_value
                p_value_matrix[j,i] = p_value
        return p_value_matrix

    def calculate_pairwise_p_values(self, method='two_tail', max_stat=False, verbose=False):
        if max_stat:
            p_val_mx =  self.calculate_max_stat_p_values(method)
        else:
            p_val_mx =  self.calculate_pairwise_p_values_uncorrected(method)
        if verbose:
            print("Pairwise P-values")
            print(p_val_mx)
        return p_val_mx
    
    def perform_permutation_testing(self, datasets_list=[]):
        if datasets_list == []:
            datasets_list = self.data_loader.dataset_paths_dict.keys()
            
        self.corr_map_dict = self._prepare_map_dict(permute=False, datasets_list=datasets_list)
        self.original_similarity_matrix = self.calculate_similarity_matrix(self.corr_map_dict)
        
        n = len(self.corr_map_dict)
        self.permuted_similarity_tensor = np.zeros((self.n_permutations, n, n))
        
        for i in tqdm(range(self.n_permutations), desc='Running permutations'):
            permuted_corr_map_dict = self._prepare_map_dict(permute=True, datasets_list=datasets_list)
            permuted_similarity_matrix = self.calculate_similarity_matrix(permuted_corr_map_dict)
            self.permuted_similarity_tensor[i] = permuted_similarity_matrix

    ### Orchestration API ###
    def run(self, tails='two_tail', fwe=False, verbose=False):
        self.perform_permutation_testing()
        self.save_results()
        
        if tails == 'two_tail':
            print('Calculating two-tailed p-values')
        elif tails == 'one_tail':
            print('Calculating one-tailed p-values')
        else:
            raise ValueError('Invalid method. Please choose either "one_tail" or "two_tail".')
        
        p_value, observed_avg = self.calculate_p_value(tails)
        pairwise_p_values = self.calculate_pairwise_p_values(tails, fwe)
        if verbose:
            print('Similarity Matrix: ')
            print(self.original_similarity_matrix)
        print(f"Overall p-value: {p_value}")
        self.p_value = p_value; self.observed_average = observed_avg
        return p_value, pairwise_p_values
    
    def run_lofo(self, folds, tails='two_tail', verbose=False):
        """
        Perform Leave-One-Out Cross-Validation (LOOCV) using the provided array of lists.
    
        Parameters:
        -----------
        folds : list of lists
            An array of lists, where each list contains dataset names to fold out.
            Example: [['dataset1'], ['dataset2'], ['dataset3']] for LOOCV.
        tails : {'one_tail', 'two_tail'}, optional
            Method to calculate p-values. Default is 'two_tail'.
        verbose : bool, optional
            Whether to print detailed logs. Default is False.
    
        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the LOOCV results, including R-values and confidence intervals.
        """    
        for fold_idx, test_datasets in enumerate(folds):
            if verbose:
                print(f"Running LOOCV fold {fold_idx + 1}/{len(folds)} with test datasets: {test_datasets}")
    
            # Step 1: Split datasets into training and testing sets
            train_datasets = [ds for ds in self.data_loader.dataset_paths_dict.keys() if ds not in test_datasets]
    
            # Step 2: Run Permutation Process on Training Datasets
            self.perform_permutation_testing(datasets_list=train_datasets)
            
            if tails == 'two_tail':
                print('Calculating two-tailed p-values')
            elif tails == 'one_tail':
                print('Calculating one-tailed p-values')
            else:
                raise ValueError('Invalid method. Please choose either "one_tail" or "two_tail".')
            
            p_value, observed_avg = self.calculate_p_value(method=tails)
            pairwise_p_values = self.calculate_pairwise_p_values(method=tails, max_stat=False)

            print(f"  Left Out Dataset: {test_datasets}")
            print(f"    P-value: {p_value:.4f}")
            print(f"    Observed Average Similarity: {observed_avg:.4f}")
            print(f"    Pairwise P-values:")
            for j, train_dataset in enumerate(train_datasets):
                print(f"      {train_dataset}: {pairwise_p_values[j, j]:.4f}")
            prefix = '_'.join(train_datasets) + '_'
            self.save_results(prefix=prefix)
    
    ### Plotting API ###
    def matrix_heatmap(self, similarity_matrix, type='similarity', mask_half=False, output_path=None, limit=None, colorscale='RedBlackGreen'):
        output_file = "heatmap.svg"
        palette = "similarity"
        if type == "similarity":
            output_file = "heatmap_similarity.svg"
            palette = "similarity"
        elif type == "pvals":
            output_file = "heatmap_pvals.svg"
            palette = "pvals"
        elif type == "misc":
            output_file = "heatmap.svg"
            palette = colorscale
        else:
            raise ValueError("Invalid input. Please choose either 'similarity', 'misc', or 'pvals'.")

        if output_path and bool(os.path.splitext(output_path)[1]):
            output_file = os.path.basename(output_path)
            output_path = os.path.dirname(output_path)

        limit = simple_heatmap(
            similarity_matrix,
            dataset_name="",
            out_dir=output_path,
            output_name=f"similarity-{self.observed_average}_p-{self.p_value}_{output_file}",
            palette=palette,
            mask_half=mask_half,
            limit=limit,
            labels=list(self.corr_map_dict.keys()),
        )
        plt.show()
        return limit
    
    ### Specific Comparison for P-Value Calculations ###
    @staticmethod
    def calculate_slice_p_value(observed_matrix, permuted_tensor, row_indices, col_indices, method='two_tail', absolute_similarity=True, remove_diagonal=True):
        """
        Compute the p-value for a submatrix (slice) of the similarity matrix.

        Parameters
        ----------
        observed_matrix : np.ndarray
            The original NxN similarity matrix (2D).
        permuted_tensor : np.ndarray
            The permuted similarity tensor of shape (num_permutations, N, N).
        row_indices : array-like
            Indices of rows to include in the slice.
        col_indices : array-like
            Indices of columns to include in the slice.
        method : {'two_tail','one_tail'}, optional
            - 'two_tail': checks if the absolute permuted average is greater than the absolute observed average.
            - 'one_tail': checks if the permuted average is greater than the observed average.
        absolute_similarity : bool, optional
            If True, takes the absolute value of the similarities before computing.
        remove_diagonal : bool, optional
            If True and row_indices == col_indices, sets the diagonal in that sub-slice to NaN before computing.

        Returns
        -------
        p_value : float
            The proportion of permutations whose average in the specified slice
            exceeds the observed average (two-tail uses absolute values).
        observed_avg : float
            The mean similarity value in this slice of the observed matrix.
        """

        # 1) Extract submatrices
        sub_obsv = observed_matrix[np.ix_(row_indices, col_indices)]
        sub_perm = permuted_tensor[:, row_indices][:, :, col_indices]

        # 2) Optionally remove diagonal if slice is square
        if remove_diagonal and (set(row_indices) == set(col_indices)):
            diag_mask_sub = ~np.eye(len(row_indices), dtype=bool)
            sub_obsv = sub_obsv * diag_mask_sub
            sub_perm = sub_perm * diag_mask_sub[None, :, :]

        # 3) Apply absolute value if needed
        if absolute_similarity:
            sub_obsv = np.abs(sub_obsv)
            sub_perm = np.abs(sub_perm)

        # 4) Compute averages
        observed_avg = np.nanmean(sub_obsv)
        permuted_avg = np.nanmean(sub_perm, axis=(1, 2))

        # 5) Calculate p-value
        if method == 'two_tail':
            p_value = np.mean(np.abs(permuted_avg) > np.abs(observed_avg))
        elif method == 'one_tail':
            p_value = np.mean(permuted_avg > observed_avg)
        else:
            raise ValueError('method must be either "one_tail" or "two_tail"')

        return p_value, observed_avg
    
    @staticmethod
    def example_identity_contrast(observed_matrix):
        """
        Given an observed_matrix (NxN), print an example contrast matrix that is just an identity matrix.
        """
        n = observed_matrix.shape[0]
        contrast_matrix = np.eye(n)
        print("Example identity contrast matrix:")
        print(contrast_matrix)
        return contrast_matrix
    
    @staticmethod
    def contrast_outer_summation(observed_matrix, permuted_tensor, contrasts, tails='two_tail'):
        """
        observed_matrix : (N,N)
        permuted_tensor : (B,N,N)
        contrasts       : list of lists (K), each length N (e.g., [[1,1,1,1,1,1], [-1,0,0,0,1,...]])
        tails           : 'two_tail' or 'one_tail'
        zero_diagonal   : if True, diagonal of each outer-product weight is zeroed
        return_pvals    : if True, return empirical p-values
        """
        # Prep Data
        C = np.asarray(contrasts, dtype=float)      # (K,N)
        if C.ndim == 1: C = C[None, :]              # allow single contrast
        
        W = C[:, :, None] * C[:, None, :]                                           # (K,N,N) <- W_k = c_k ⊗ c_k  
        T_obs  = np.einsum('ij,kij->k', observed_matrix, W, optimize=True)          # T_obs[k] = Σ_ij W_k[i,j] * S[i,j]
        T_perm = np.einsum('bij,kij->bk', permuted_tensor, W, optimize=True)        # T_perm[b,k] = Σ_ij W_k[i,j] * S_b[i,j]

        if tails == 'two_tail':
            pvals = np.mean(np.abs(T_perm) >= np.abs(T_obs)[None, :], axis=0)
        elif tails == 'one_tail':
            pvals = np.mean(T_perm >= T_obs[None, :], axis=0)
        else:
            raise ValueError("tails must be 'two_tail' or 'one_tail'")

        return T_obs, pvals

    
### GENERAL NIFTI FUNCTIONS ###
                    
def mask_array(mask_path, data_array, threshold=0):
    if mask_path is None:
        from nimlab import datasets as nimds
        mask = nimds.get_img("mni_icbm152")
    else:
        mask = nib.load(mask_path)
    mask_data = mask.get_fdata()
    mask_indices = mask_data.flatten() > threshold
    masked_array = data_array.flatten()[mask_indices]
    return masked_array

def unmask_array(mask_path, data_array, threshold=0):
    if mask_path is None:
        from nimlab import datasets as nimds
        mask = nimds.get_img("mni_icbm152")
    else:
        mask = nib.load(mask_path)
    mask_data = mask.get_fdata()
    mask_indices = mask_data.flatten() > threshold
    unmasked_array = np.zeros(mask_indices.shape)
    unmasked_array[mask_indices] = data_array.flatten()
    return unmasked_array.reshape(mask_data.shape), mask.affine

def save_map(map_data, out_dir, file_name, mask_path=None):
    unmasked_map, mask_affine = unmask_array(data_array=map_data, mask_path=mask_path)
    img = nib.Nifti1Image(unmasked_map, affine=mask_affine)
    file_path = os.path.join(out_dir, file_name)
    if out_dir is not None:
        if bool(os.path.splitext(file_path)[1]):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        else: 
            os.makedirs(out_dir, exist_ok=True)
        nib.save(img, file_path)
        
    return img

def visualize_map(img, title):
    plotting.view_img(img, title=title).open_in_browser()

def save_individual_r_maps(data, verbose=False, out_dir='', filename='generated_map.nii.gz', mask_path=None):
    r_img = save_map(map_data=data, out_dir=out_dir, file_name=filename, mask_path=mask_path)
    if verbose: visualize_map(img=r_img, title=filename)
