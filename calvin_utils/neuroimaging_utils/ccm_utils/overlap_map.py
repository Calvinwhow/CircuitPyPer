import os
import numpy as np
import nibabel as nib
from nilearn import plotting
from calvin_utils.ccm_utils.npy_utils import DataLoader

class OverlapMap:
    """
    Class to compute a voxelwise percent-overlap map across each dataset loaded by DataLoader.

    The user can define how to binarize each dataset's NIfTI volumes via thresholding rules
    (e.g. 't' => 7, 'r' => 0.1, 'rfz' => 5, 'ROI' => 1.0, or a user-defined float).

    Steps:
    -------
    1) For each dataset in DataLoader, load its 4D NIfTI stack (shape: [n_subjects, ...]).
    2) Binarize each subject's volume based on abs(value) >= threshold: 1, <= -threshold: -1, else: 0.
    3) Sum these binarized maps over subjects → overlap_count.
    4) Convert overlap_count to percentage by dividing by n_subjects and multiplying by 100.
    5) Generate a stepwise map (floored at step_size% increments).
    6) Save results as new NIfTI files, if desired.

    Example:
    --------
    omap = OverlapMap(
        data_loader=my_loader,
        mask_path='mask.nii.gz',
        map_type='t',         # 't', 'r', 'rfz', 'ROI' or custom
        manual_threshold=None # e.g. 2.5. If None, uses default per map_type
    )
    overlap_dict = omap.generate_overlap_maps()
    stepwise_dict = omap.generate_stepwise_maps()
    omap.save_maps(overlap_dict, suffix='_overlap')
    omap.save_maps(stepwise_dict, suffix='_stepwise')
    """

    def __init__(
        self,
        data_loader: DataLoader,
        mask_path: str,
        out_dir: str = None,
        map_type: str = 'ROI',      # 'ROI', 't', 'r', 'rfz' or 'custom'
        manual_threshold: float = None,
        save_absval : bool = True,
        verbose: bool = True
    ):
        """
        Parameters
        ----------
        data_loader : DataLoader
            DataLoader instance providing access to datasets and their NIfTI arrays.
        mask_path : str
            Path to a brain mask NIfTI file for masking/unmasking arrays.
        out_dir : str, optional
            Directory to save output overlap maps. If None, maps are not saved.
        map_type : str, optional
            One of {'ROI', 't', 'r', 'rfz'} or 'custom'. Determines the default threshold if manual_threshold is not set.
        manual_threshold : float, optional
            If provided, overrides the default threshold for binarization.
        save_absval : bool, optional
            If True, also saves the absolute value of overlap maps. Default is True.
        verbose : bool, optional
            If True, displays maps in a web browser using NiLearn. Default is True.
        """
        self.data_loader = data_loader
        self.mask_path = mask_path
        self.out_dir = out_dir
        self.map_type = map_type
        self.manual_threshold = manual_threshold
        self.verbose = verbose
        self.save_absval = save_absval
        self.threshold = self._get_threshold()
        self.step_size = 5  # steps for percentage overlap maps
        if out_dir is not None:
            self.out_dir = os.path.join(self.out_dir, f'{self.map_type}_overlap_maps')
            os.makedirs(out_dir, exist_ok=True)
    
    def _get_threshold(self):
        if self.manual_threshold is None:
            if self.map_type == 'ROI':
                threshold = 1.0
            elif self.map_type == 't':
                threshold = 7.0
            elif self.map_type == 'r':
                threshold = 0.1
            elif self.map_type == 'rfz':
                threshold = 5.0
            elif self.map_type == 'z':
                threshold = 2.0
            elif self.map_type == 'w':
                threshold = 2.0
            else:
                raise ValueError("For 'custom' map_type, you must provide manual_threshold.")
        else:
            threshold = float(self.manual_threshold)
        return threshold

    def generate_overlap_maps(self):
        """
        Generate the voxelwise overlap map (percentage) for each dataset in the DataLoader.
        
        Returns
        -------
        dict
            A dictionary of {dataset_name: overlap_map}, where overlap_map is a 1D array
            (masked) containing the percent overlap at each voxel.
        """
        overlap_map_dict = {}
        for dataset_name in self.data_loader.dataset_paths_dict.keys():
            data = self.data_loader.load_dataset(dataset_name)
            niftis = data["niftis"]  # shape: (n_subj, x, y, z) or (n_subj, n_voxels)
            
            # Ensure 2D shape: (n_subj, n_voxels)
            n_subj = niftis.shape[0]
            flatten_niftis = niftis.reshape(n_subj, -1)
            
            # Binarize each subject
            binarized = self._binarize(flatten_niftis, self.threshold)
            
            # Sum binarized maps across subjects, resulting in a map ranging from 0-N_subjects
            overlap_map_dict[dataset_name] = np.nansum(binarized, axis=0).astype(np.float32)
        
        return overlap_map_dict

    def generate_stepwise_maps(self):
        """
        Floor the overlap maps to integer bins of self.step_size. E.g. step_size=5 => 0,5,10,...
        
        Returns
        -------
        dict
            A dictionary of {dataset_name: stepwise_map}, where stepwise_map is the binned map.
        """
        stepwise_dict = {}
        for dataset_name in self.data_loader.dataset_paths_dict.keys():
            data = self.data_loader.load_dataset(dataset_name)
            niftis = data["niftis"]  # shape: (n_subj, x, y, z) or (n_subj, n_voxels)
            
            # Ensure 2D shape: (n_subj, n_voxels)
            n_subj = niftis.shape[0]
            flatten_niftis = niftis.reshape(n_subj, -1)
            
            # Binarize each subject
            binarized = self._binarize(flatten_niftis, self.threshold)
            
            # Sum binarized maps across subjects, resulting in a map ranging from 0-N_subjects
            percent_overlap_map = np.sum(binarized, axis=0) / n_subj * 100.0
            stepwise = (np.floor(percent_overlap_map / self.step_size) * self.step_size).astype(np.float32)
            stepwise_dict[dataset_name] = stepwise
        return stepwise_dict

    def _binarize(self, flatten_niftis, threshold):
        """Binarize the flattened NIfTI data according to absolute thresholding. """
        binary_map = np.zeros_like(flatten_niftis, dtype=int)
        binary_map[flatten_niftis >= threshold] = 1
        binary_map[flatten_niftis <= -threshold] = -1
        return binary_map
    
    def _load_nifti(self, path):
        img = nib.load(path)
        return img.get_fdata(), img.affine

    def _mask_array(self, data_array, threshold=0):
        """
        Flatten and return only voxels within mask > threshold.
        """
        mask_data, _ = self._get_mask_data()
        mask_indices = mask_data.flatten() > threshold
        return data_array.flatten()[mask_indices]

    def _unmask_array(self, data_array, threshold=0):
        """
        Reshape a 1D array back into mask space.
        """
        mask_data, mask_affine = self._get_mask_data()
        mask_indices = mask_data.flatten() > threshold
        
        out_array = np.zeros(mask_data.size, dtype=data_array.dtype)
        out_array[mask_indices] = data_array
        return out_array.reshape(mask_data.shape), mask_affine

    def _get_mask_data(self):
        """
        Load mask data. If none specified, load MNI default.
        """
        mdata = nib.load(self.mask_path)
        mask_data = mdata.get_fdata()
        mask_affine = mdata.affine
            
        return mask_data, mask_affine

    def _save_map(self, map_data, file_name):
        """
        Unmask (if needed) and save a NIfTI file to self.out_dir.
        """
        unmasked_map, mask_affine = self._unmask_array(map_data)
        img = nib.Nifti1Image(unmasked_map, affine=mask_affine)
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)
            out_path = os.path.join(self.out_dir, f'threshold_{(self.threshold)}_{file_name}')
            nib.save(img, out_path)
        return img

    def _visualize_map(self, img, title):
        """
        Quick HTML view in-browser via NiLearn's view_img.
        """
        plotting.view_img(img, title=title).open_in_browser()
    
    def save_maps(self, overlap_map_dict, suffix='_overlap'):
        """
        Save each overlap map in `overlap_map_dict` as a NIfTI file, unmasking as needed.
        
        Parameters
        ----------
        overlap_map_dict : dict
            Dictionary of {dataset_name: overlap_1d_array}.
        suffix : str, optional
            Filename suffix, e.g. '_overlap' or '_stepwise'.
        """
        for dataset_name, overlap_map in overlap_map_dict.items():
            img = self._save_map(overlap_map, dataset_name + suffix + '.nii.gz')
            _ = self._save_map(np.abs(overlap_map), dataset_name + suffix + '_absval.nii.gz')
            if self.verbose:
                try:
                    self._visualize_map(img, f"{dataset_name}{suffix}")
                except:
                    pass

    def run(self):
        """
        Orchestrate the overlap computation pipeline for all datasets in the DataLoader.

        Steps:
        1) Generate continuous percent overlap maps.
        2) Generate stepwise (floored) maps.
        3) Save both sets of maps to disk (if out_dir is specified).

        Returns:
        --------
        tuple of dict:
            (overlap_map_dict, stepwise_map_dict)
            Where each is {dataset_name: 1D_array_of_voxelwise_values}
        """
        # Generate continuous overlap maps
        overlap_map_dict = self.generate_overlap_maps()
        
        # Generate stepwise maps
        stepwise_map_dict = self.generate_stepwise_maps()
        
        # Save them if out_dir is set
        if self.out_dir:
            self.save_maps(overlap_map_dict, suffix='_n_overlap')
            self.save_maps(stepwise_map_dict, suffix='_percent_overlap_stepwise')
        
        return overlap_map_dict, stepwise_map_dict

