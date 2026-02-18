import os
import json
import numpy as np
import nibabel as nib
from tqdm import tqdm
from nilearn import image
from calvin_utils.neuroimaging_utils.nifti_utils.generate_mask import GenerateMask

class RegressionPrep:
    def __init__(self, design_matrix, contrast_matrix, outcome_df, out_dir,
                 voxelwise_variables=None, voxelwise_interactions=None,
                 mask_path=None, exchangeability_block=None, weights=None,
                 data_transform_method='standardize'):
        """
        Initializes the RegressionPrep class.

        Parameters:
        - design_matrix (pd.DataFrame): DataFrame containing scalar and voxelwise variable columns for each subject.
        - contrast_matrix (np.ndarray or list): Array or list specifying the contrasts of interest.
        - outcome_df (pd.DataFrame): DataFrame containing the outcome variable(s), either scalar or voxelwise.
        - out_dir (str): Output directory for saving processed data.
        - voxelwise_variables (list of str, optional): Names of variables in design_matrix that are voxelwise (i.e., NIfTI paths per subject). Default is None.
        - voxelwise_interactions (list of str, optional): List of interaction terms (e.g., 'age:voxelwise_var') involving voxelwise variables. Default is None.
        - mask_path (str, optional): Path to a NIfTI mask file to restrict analysis to specific voxels. Default is None.
        - exchangeability_block (np.ndarray, optional): 1D array of integers indicating exchangeability blocks for permutation testing. Default is None.
        - weights (np.ndarray or list, optional): 1D array or list of positive floats specifying regression weights for each subject. Default is None (equal weights).
        - data_transform_method (str, optional): Method for data transformation; options are 'standardize', 'rank', or None. Default is 'standardize'.
        """
        self.design_matrix = design_matrix
        self.contrast_matrix = contrast_matrix
        self.outcome_df = outcome_df
        self.voxelwise_variables = voxelwise_variables or []
        self.voxelwise_interactions = [interaction.replace(' ', '') for interaction in (voxelwise_interactions or [])]
        self.out_dir = out_dir
        self.mask_path = mask_path
        self.exchangeability_block = exchangeability_block
        self.data_transform_method = data_transform_method
        self.weights_vector = self.get_weights(weights)
        os.makedirs(self.out_dir, exist_ok=True)
        if self.mask_path is None:
            self._generate_and_save_mask()
        self.mask = self.get_mask()
        self.voxelwise_regressors = self._get_voxelwise_regressors()
        self.design_tensor = self._get_design_tensor()
        self.outcome_data = self._get_outcome_data()
        
    ### setters and getters ###
    def get_mask(self):
        if not self.mask_path:
            return None
        mask_img = self._load_mask_img()
        return mask_img.get_fdata().flatten() > 0

    def _load_mask_img(self):
        if getattr(self, "_mask_img", None) is None:
            self._mask_img = nib.load(self.mask_path)
        return self._mask_img

    def _generate_and_save_mask(self):
        paths = self._collect_all_nifti_paths()
        if not paths:
            return None
        gen = GenerateMask(paths, threshold=None, verbose=False)
        mask_img, _ = gen.run()
        out_path = os.path.join(self.out_dir, "mask.nii.gz")
        nib.save(mask_img, out_path)
        self.mask_path = out_path
        self._mask_img = mask_img
        return out_path
    
    def get_weights(self, values):
        n_obs = self.design_matrix.shape[0] if hasattr(self, 'design_matrix') else None
        if values is None:
            _weights = np.ones(n_obs, dtype=float)
        else:
            _weights = np.array(values, dtype=float)
            _weights = _weights / np.sum(_weights)
            if n_obs is not None and _weights.shape[0] != n_obs:
                raise ValueError(f"weights must have same length as number of observations ({n_obs}), got {_weights.shape[0]}")
            if np.any(np.isnan(_weights)):
                raise ValueError("weights must not contain NaNs")
            if np.any(_weights <= 0):
                raise ValueError("weights must be positive")
        return _weights
    
    def _get_voxelwise_regressors(self):
        voxelwise_regressors = self._prepare_voxelwise_terms()
        return self._apply_interactions(voxelwise_regressors)
         
    def _get_design_tensor(self):
        design_tensor = self._prepare_design_matrix(self.voxelwise_regressors)
        return self._clean(design_tensor)
    
    def _get_outcome_data(self):
        outcome_data = self._prepare_outcome_data()
        return self._clean(outcome_data)
    
   ### I/O ###
    def _prep_paths(self, df, term):
        """Ensure the result is a flat list of strings (paths)"""
        paths = df[term].values
        print("paths are: ", len(paths))
        if isinstance(paths, np.ndarray):
            paths = paths.flatten().tolist()
        elif hasattr(paths, 'tolist'):
            paths = paths.tolist()
        paths = [str(p) for p in paths]
        return paths

    def _collect_all_nifti_paths(self):
        """Collect all voxelwise NIfTI paths from design and outcome."""
        all_paths = []
        for term in self.voxelwise_variables:
            if term in self.outcome_df.columns:
                continue
            all_paths.extend(self._prep_paths(self.design_matrix, term))

        outcome_colname = self.outcome_df.columns[0]
        if outcome_colname in self.voxelwise_variables:
            all_paths.extend(self._prep_paths(self.outcome_df, outcome_colname))
        return [p for p in all_paths if isinstance(p, str) and p]
    
    def _load_nifti_stack(self, paths, dtype=np.float32):
        """Return (n_subj, n_vox) float32 array, masked and z-scored."""
        # ----- allocate once -----
        first = nib.load(paths[0]).get_fdata(dtype=dtype).ravel()
        vox_in_mask = self.mask.sum() if self.mask is not None else first.size
        stack = np.empty((len(paths), vox_in_mask), dtype=dtype)   # subj × vox

        # ----- fill -----
        for k, p in enumerate(tqdm(paths, desc="Loading & masking")):
            img = nib.load(p)
            if self.mask is not None:
                mask_img = self._load_mask_img()
                if img.shape != mask_img.shape:
                    img = image.resample_to_img(img, mask_img, interpolation="continuous")
                arr = img.get_fdata(dtype=dtype).ravel()
                arr = arr[self.mask]            # 1-D masked
            else:
                arr = img.get_fdata(dtype=dtype).ravel()
            stack[k, :] = arr                   # single write
        return stack 
    
    ### Data Preprocessing ###      
    def _clean(self, arr, verbose=False):
        '''Process the data'''
        if verbose: print(arr.shape)
        
        arr = self._handle_nans(arr)
        if self.data_transform_method=='standardize':
            arr = self._standardize(arr)
        if self.data_transform_method=='rank':
            arr = self._rank_across_subjects(arr)
        return arr
    
    def _handle_nans(self, arr, value=0):
        """Handles NaNs by replacing them (and pos/neg inf) with finite values."""
        max_val = np.nanmax(arr)
        min_val = np.nanmin(arr)
        return np.nan_to_num(arr, nan=value, posinf=max_val, neginf=min_val)

    def _standardize(self, data: np.ndarray, axis: int = 0, skip_ordinals: bool = True, max_unique: int = 10):
        """
        Z-score along `axis` (default = subjects) for *any* rank.

        * Ordinal-column skipping is applied **only** when `data.ndim == 2`.
        * For 3-D tensors (subj, out, vox) every (out, vox) column is scaled.

        Returns array of same shape.
        """
        # ---- constant columns check (all ranks) ----
        std = data.std(axis=axis, keepdims=True)
        scale_mask = std > 1e-12            # broadcastable mask

        # ---- optional ordinal skip (only for 2-D matrices) ----
        if skip_ordinals and data.ndim == 2:
            uniq = np.apply_along_axis(lambda c: len(np.unique(c)), axis, data)
            ordinal = uniq <= max_unique
            scale_mask = scale_mask & ~ordinal[:, None]   # keepdims alignment

        mean = data.mean(axis=axis, keepdims=True)

        z = data.copy()
        z = np.where(scale_mask, (data - mean) / (std + 1e-8), data)
        return z

    def _rank_across_subjects(self, arr: np.ndarray) -> np.ndarray:
        """
        Replace each column with its ranks (1..n_subj).
        Works for 2-D (n_subj, n_feat) or 3-D (n_subj, k, n_vox).

        Constant columns are returned unchanged.
        """
        subj = arr.shape[0]
        flat = arr.reshape(subj, -1)                     # 2-D view (subj, Ncol)

        # Vectorised two-pass argsort (fast, but ignores ties)
        idx = np.argsort(flat, axis=0, kind='mergesort')
        ranks = np.empty_like(flat, dtype=np.float32)
        rows = np.arange(subj, dtype=np.float32)[:, None]
        ranks[idx, np.arange(flat.shape[1])] = rows + 1  # 1-based ranks

        # Identify constant columns and leave them unchanged
        const_mask = flat.ptp(axis=0) == 0               # range == 0
        if const_mask.any():
            ranks[:, const_mask] = flat[:, const_mask]

        return ranks.reshape(arr.shape)
    
    ### INTERNAL REGRESSION MATRIX PREP ####
    def _apply_interactions(self, voxelwise_data):
        '''Grabs the voxelwise data and multiplies it by the scalar term'''
        for col in self.voxelwise_interactions:
            term1, term2 = [x.strip() for x in (col.split(':') if ':' in col else col.split('*'))]
            voxel_term = term1 if term1 in self.voxelwise_variables else term2
            scalar_term = term2 if voxel_term == term1 else term1            
            interaction_values = self.design_matrix[scalar_term].values.astype(float)
            voxelwise_data[col] = voxelwise_data[voxel_term] * interaction_values
        return voxelwise_data
    
    def _prepare_voxelwise_terms(self):
        '''Prepares voxelwise regressors'''
        voxelwise_data = {}
        for term in self.voxelwise_variables:
            if term in self.outcome_df.columns: # don't use the dependent variable
                continue
            paths = self._prep_paths(self.design_matrix, term)
            stacked = self._load_nifti_stack(paths)
            voxelwise_data[term] = stacked
        return voxelwise_data
    
    def _prepare_design_matrix(self, voxelwise_regressors: dict[str, np.ndarray]):
        """Build a design tensor of shape (n_obs, n_params, n_voxels)."""
        n_obs, n_params = self.design_matrix.shape
        vox = next(iter(voxelwise_regressors.values())).shape[1] if voxelwise_regressors else 1                         
        tensor = np.empty((n_obs, n_params, vox), dtype=np.float32)
        for j, col in enumerate(self.design_matrix.columns):
            if col in voxelwise_regressors or col in self.voxelwise_interactions:
                tensor[:, j, :] = voxelwise_regressors[col]
            else:                                                     
                tensor[:, j, :] = self.design_matrix[col].values[:, None]
        return tensor
    
    def _prepare_outcome_data(self):
        """
        Preps outcome data as shape (N_subj, N_outcomes, N_voxels).
        Prepare outcome data from niftis (if voxelwise outcome) or dataframe (if not voxelwise). Enables multiple outcomes.
        TODO: handle multiple voxelwise outcomes.
        """
        outcome_colname = self.outcome_df.columns[0]
        if outcome_colname in self.voxelwise_variables:
            paths = self._prep_paths(self.outcome_df, outcome_colname)
            outcome_data = self._load_nifti_stack(paths)
            outcome_data = outcome_data[:, None, :]                             # shape (n_obs, 1, n_voxels)
        else:
            outcome_data = self.outcome_df.values.astype(float)[:, :, None]     # shape (n_obs, n_outcomes, 1)
        return outcome_data
    
    ### PUBLIC API ####
    def save_dataset(self):
        dataset_dict = {
            'voxelwise_regression': {
                "design_matrix": f"{self.out_dir}/design_matrix.npy",
                "contrast_matrix": f"{self.out_dir}/contrast_matrix.npy",
                "outcome_data": f"{self.out_dir}/outcome_data.npy"
            }
        }
        
        np.save(f"{self.out_dir}/design_matrix.npy", self.design_tensor)
        np.save(f"{self.out_dir}/contrast_matrix.npy", self.contrast_matrix)
        np.save(f"{self.out_dir}/outcome_data.npy", self.outcome_data)
        
        if self.exchangeability_block is not None:
            dataset_dict['voxelwise_regression']["exchangeability_block"] = f"{self.out_dir}/exchangeability_block.npy"
            np.save(f"{self.out_dir}/exchangeability_block.npy", self.exchangeability_block)
        if self.weights_vector is not None:
            dataset_dict['voxelwise_regression']["weights_vector"] = f"{self.out_dir}/weights_vector.npy"
            np.save(f"{self.out_dir}/weights_vector.npy", self.weights_vector)
        if self.mask_path is not None:
            dataset_dict['voxelwise_regression']["mask_path"] = self.mask_path
        
        with open(f"{self.out_dir}/dataset_dict.json", "w") as f:
            json.dump(dataset_dict, f, indent=4)
        return dataset_dict, f"{self.out_dir}/dataset_dict.json"
    
    def run(self):
        dataset_dict, json_path = self.save_dataset()
        print("design_tensor shape:", self.design_tensor.shape)
        print("outcome_data shape:", self.outcome_data.shape)
        return dataset_dict, json_path
