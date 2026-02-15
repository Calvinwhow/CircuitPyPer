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

from calvin_utils.ccm_utils.npy_utils import DataLoader
from calvin_utils.ccm_utils.stat_utils import CorrelationCalculator
from calvin_utils.ccm_utils.optimization.convergent_map_optimizer import NiftiOptimizer

class ConvergentMapGenerator:
    """
    ConvergentMapGenerator is a utility class for generating, processing, and saving convergent correlation maps from multiple datasets, typically in neuroimaging applications.
    This class provides methods for:
    - Preprocessing correlation maps (handling NaNs, flipping signs, aligning maps).
    - Applying spatial masks to data arrays.
    - Saving and visualizing NIfTI images.
    - Generating weighted or unweighted average correlation maps.
    - Creating agreement maps indicating regions with consistent correlation sign across datasets.
    - Optionally optimizing the average correlation map using a custom optimizer.
    - Grouping datasets for group-wise map generation and saving.
    
    Notes
    -----
    - Requires numpy, nibabel, nilearn, and other domain-specific dependencies.
    - Designed for use in neuroimaging workflows involving group-level correlation analysis.
    """
    def __init__(self, corr_map_dict, data_loader, mask_path=None, out_dir=None, weighting='unweighted', align_all_maps=False, datasets_to_flip = [], group_dict = {}) :
        """
        Initialize the ConvergentMapGenerator with configuration options.
        Args:
            corr_map_dict (dict): Correlation maps for each dataset.
            data_loader (object): Loader for dataset-specific information.
            mask_path (str, optional): Path to NIfTI mask file.
            out_dir (str, optional): Output directory for results.
            weighting (str, optional): 'unweighted', 'weighted', or 'optimized'. Defaults to 'unweighted'.
            align_all_maps (bool, optional): Align all maps to a reference. Defaults to False.
            datasets_to_flip (list, optional): Datasets whose maps should be sign-flipped.
            group_dict (dict, optional): Mapping of dataset names to group names.
        """
        
        self.corr_map_dict = corr_map_dict
        self.data_loader = data_loader
        self.mask_path = mask_path
        self.out_dir = out_dir
        self.weighting = weighting
        self.align_all_maps = align_all_maps
        self.datasets_to_flip = datasets_to_flip
        self.group_dict = self._prep_group_dict(group_dict)
        self.corr_map_dict = self._handle_datasets()
        
        self._handle_nans()
        self._prep_dirs(out_dir)
            
    ### Preprocessing ###
    def _prep_group_dict(self, group_dict):
        if group_dict == {}:
            group_dict = {ds: 'all_datasets' for ds in self.corr_map_dict.keys()}
        return group_dict
        
    def _prep_dirs(self, out_dir):
        if out_dir is not None: 
            os.makedirs(out_dir, exist_ok=True)

    def _handle_nans(self):
        print("Removing NaNS from self.coor_map_dict.")
        for dataset, matrix in self.corr_map_dict.items():
            self.corr_map_dict[dataset] = CorrelationCalculator._check_for_nans(matrix, nanpolicy='remove', verbose=False)
            
    def _handle_datasets(self, override_corr_map_dict=None):
        '''Flip Specified Maps'''
        if override_corr_map_dict is not None: 
            local_corr_map_dict = override_corr_map_dict
        else:
            local_corr_map_dict = self.corr_map_dict
            
        # Orient Dataset to Account for Sign Flips
        for ds in self.datasets_to_flip:
            if ds in local_corr_map_dict:
                self.corr_map_dict[ds] *= -1
        if self.align_all_maps: 
            local_corr_map_dict = self._align_signs(local_corr_map_dict)
        return local_corr_map_dict
            
    def _align_signs(self, images_dict, reference=None):
        """
        Flip signs of images so all correlate positively with the reference.
        """
        keys = list(images_dict.keys())
        if reference is None:
            ref = images_dict[keys[0]].ravel()
        else:
            ref = images_dict[reference].ravel()
        return {k: (img if np.corrcoef(img.ravel(), ref)[0, 1] >= 0 else -img)
                for k, img in images_dict.items()}

    ### Masking Functions ###
    def _mask_array(self, data_array, threshold=0):
        if self.mask_path is None:
            masked_array = data_array.flatten()
        else:
            mask = nib.load(self.mask_path)
            mask_data = mask.get_fdata()
            mask_indices = mask_data.flatten() > threshold
            masked_array = data_array.flatten()[mask_indices]
        return masked_array
    
    def _unmask_array(self, data_array, threshold=0):
        if self.mask_path is None:
            unmasked_array = data_array.flatten()
        else:
            mask = nib.load(self.mask_path)
            mask_data = mask.get_fdata()
            mask_indices = mask_data.flatten() > threshold
            unmasked_array = np.zeros(mask_indices.shape)            
            unmasked_array[mask_indices] = data_array.flatten()
        return unmasked_array.reshape(mask_data.shape), mask.affine

    ### I/O ###
    def _save_map(self, map_data, file_name):
        unmasked_map, mask_affine = self._unmask_array(map_data)
        img = nib.Nifti1Image(unmasked_map, affine=mask_affine)
        file_path = os.path.join(self.out_dir, file_name)
        if self.out_dir is not None:
            if bool(os.path.splitext(file_path)[1]):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
            else: 
                os.makedirs(self.out_dir, exist_ok=True)
            nib.save(img, file_path)
        return img
    
    def _load_nifti(self, path):
        img = nib.load(path)
        return img.get_fdata()
    
    def _visualize_map(self, img, title):
        try:
            plotting.view_img(img, title=title).open_in_browser()
        except:
            pass
    
    def save_individual_r_maps(self, verbose=False, subdir=''):
        for dataset_name, r_map in self.corr_map_dict.items():
            r_img = self._save_map(r_map, f'{subdir}{dataset_name}_correlation_map.nii.gz')
            if verbose: self._visualize_map(r_img, f'{dataset_name} Correlation Map')
            
    ### Processing ###
    def _select_weighted_map(self, override_corr_map_dict):
        if hasattr(self, 'weighting'):
            if self.weighting == 'unweighted':
                return self.generate_weighted_average_r_map(override_corr_map_dict, weight=False)
            elif self.weighting == 'weighted':
                return self.generate_weighted_average_r_map(override_corr_map_dict, weight=True)
            elif self.weighting == 'optimized':
                return self.generate_optimal_average_r_map(override_corr_map_dict)
            else:
                raise ValueError(f"Invalid weighting: {self.weighting}. Must be 'weighted', 'unweighted', or 'optimized'")
        else:
            raise ValueError("Attribute 'weighting' not found in the class.")
            
    def generate_weighted_average_r_map(self, override_corr_map_dict=None, weight=False):
        '''Gets either weighted or unweighted average R map depending on self.weight'''
        if override_corr_map_dict is not None:
            local_corr_map_dict = override_corr_map_dict  
        else:
            local_corr_map_dict = self.corr_map_dict  
               
        r_maps = np.array(list(local_corr_map_dict.values()))
        if weight:
            weights = []
            for dataset_name in local_corr_map_dict.keys():
                data = self.data_loader.load_dataset(dataset_name)
                weights.append(data['niftis'].shape[0]) # get sample size
            weights = np.array(weights)
            return np.average(r_maps, axis=0, weights=weights)
        else:
            return np.average(r_maps, axis=0)
        
    def generate_optimal_average_r_map(self, override_corr_map_dict=None):
        '''Calls optimization function'''
        local_corr_map_dict = self.corr_map_dict
        if override_corr_map_dict is not None:
            local_corr_map_dict = override_corr_map_dict
        optimizer = NiftiOptimizer(local_corr_map_dict, self.data_loader, load_in_time=True)
        optimal_average_r_map = optimizer.optimize()
        return optimal_average_r_map

    def generate_agreement_map(self, override_corr_map_dict=None):
        '''Generates mask of regions that share signs among all maps in set of [-1, 0, 1]'''
        local_corr_map_dict = self.corr_map_dict
        if override_corr_map_dict is not None:
            local_corr_map_dict = override_corr_map_dict
        r_maps = np.array(list(local_corr_map_dict.values()))
        signs = np.sign(r_maps)
        agreement = np.all(signs == signs[0], axis=0)
        return agreement.astype(int) * signs[0] 
    
    def _get_local_corr_map_dict(self, group):
        '''Retrieve the R maps belonging to a particular group, as noted by self.group_dict'''
        datasets_in_group = [key for key, val in self.group_dict.items() if val == group]
        return {dataset: self.corr_map_dict[dataset] for dataset in datasets_in_group}
    
    def _gen_weighted_r_map(self, local_corr_map_dict, subdir, group, verbose):
        '''Orchestrates making, saving, and viewing the weighted map'''
        weighted_avg_map = self._select_weighted_map(local_corr_map_dict)
        weighted_avg_img = self._save_map(weighted_avg_map, f'{subdir}{group}_{self.weighting}_r_map.nii.gz')
        if verbose: self._visualize_map(weighted_avg_img, f'{group} {self.weighting} R Map')
    
    def _gen_agreement_map(self, local_corr_map_dict, subdir, group, verbose):
        '''Orchestrates making, saving, and viewing the agreement map'''
        agreement_map = self.generate_agreement_map(local_corr_map_dict)
        agreement_img = self._save_map(agreement_map, f'{subdir}{group}_agreement_map.nii.gz')
        if verbose: self._visualize_map(agreement_img, f'{group} Agreement Map')

    ### Public API ###
    def generate_and_save_maps(self, verbose=False, subdir=''):
        """
        Generates and saves various maps for each unique group in the dataset.
        For each group identified in `self.group_dict`, this method:
            1. Retrieves a local correlation map dictionary for the group.
            2. Generates and saves a weighted R map.
            3. Generates and saves an agreement map.
            4. Generates and saves an optimal map.
        Args:
            verbose (bool, optional): If True, enables verbose output during map generation. Defaults to False.
            subdir (str, optional): Subdirectory path where the maps will be saved. Defaults to '' (current directory).
        Returns:
            None
        """
        for group in set(self.group_dict.values()):
            local_corr_map_dict = self._get_local_corr_map_dict(group)    
            self._gen_weighted_r_map(local_corr_map_dict, subdir, group, verbose)        
            self._gen_agreement_map(local_corr_map_dict, subdir, group, verbose)