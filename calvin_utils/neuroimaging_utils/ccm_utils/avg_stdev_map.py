import os
import numpy as np
import nibabel as nib
from nilearn import plotting
from calvin_utils.ccm_utils.npy_utils import DataLoader

class AvgStdevMap:
    """
    Class to compute voxelwise average and standard deviation maps across each dataset loaded by DataLoader.

    Steps:
    -------
    1) For each dataset in DataLoader, load its 4D NIfTI stack (shape: [n_subjects, ...]).
    2) Get average across the stack of patients (across n_subjects dimension).
    3) Get the standard deviation across the stack of patients (across n_subjects dimension).
    4) Save results as new NIfTI files, if desired.

    Example:
    --------
    asmap = AvgStdevMap(
        data_loader=my_loader,
        mask_path='mask.nii.gz',
        manual_threshold=None,
        out_dir='output_maps'
    )
    asmap.run()
    """

    def __init__(
        self,
        data_loader: DataLoader,
        mask_path: str,
        out_dir: str = None,
        manual_threshold: float = None,
        save_absval : bool = False,
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
        manual_threshold : float, optional
        save_absval : bool, optional
            If True, also saves the absolute value of overlap maps. Default is True.
        verbose : bool, optional
            If True, displays maps in a web browser using NiLearn. Default is True.
        """
        self.data_loader = data_loader
        self.mask_path = mask_path
        self.out_dir = out_dir
        self.manual_threshold = manual_threshold
        self.verbose = verbose
        self.save_absval = save_absval
        self._prep_dirs()
    
    def _prep_dirs(self):
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)

    def generate_avg_maps(self):
        """Generate the voxelwise average map for each dataset in the DataLoader."""
        avg_map_dict = {}
        for dataset_name in self.data_loader.dataset_paths_dict.keys():
            data = self.data_loader.load_dataset(dataset_name)
            niftis = data["niftis"]  # shape: (n_subj, x, y, z) or (n_subj, n_voxels)
            
            # Ensure 2D shape: (n_subj, n_voxels)
            n_subj = niftis.shape[0]
            flatten_niftis = niftis.reshape(n_subj, -1)
            
            # Calc avg.
            sum_niftis = np.nansum(flatten_niftis, axis=0)
            avg_map = sum_niftis / n_subj
            avg_map_dict[dataset_name] = avg_map.astype(np.float32)
        
        return avg_map_dict

    def generate_stdev_maps(self):
        """Generate the voxelwise stdev map for each dataset in the DataLoader."""
        stdev_dict = {}
        for dataset_name in self.data_loader.dataset_paths_dict.keys():
            data = self.data_loader.load_dataset(dataset_name)
            niftis = data["niftis"]  # shape: (n_subj, x, y, z) or (n_subj, n_voxels)
            
            # Ensure 2D shape: (n_subj, n_voxels)
            n_subj = niftis.shape[0]
            flatten_niftis = niftis.reshape(n_subj, -1)
            
            # Calc standard deviation.
            stdev = np.nanstd(flatten_niftis, axis=0)
            stdev_dict[dataset_name] = stdev.astype(np.float32)
        return stdev_dict
    
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
            out_path = os.path.join(self.out_dir, file_name)
            nib.save(img, out_path)
        return img

    def _visualize_map(self, img, title):
        """Quick HTML view in-browser via NiLearn's view_img."""
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

        Returns:
        --------
        tuple of dict:
            (avg_map_dict, stdev_map_dict)
            Where each is {dataset_name: 1D_array_of_voxelwise_values}
        """
        avg_map_dict = self.generate_avg_maps()
        stdev_map_dict = self.generate_stdev_maps()
        if self.out_dir:
            self.save_maps(avg_map_dict, suffix='_avg_map')
            self.save_maps(stdev_map_dict, suffix='_stdev_map')
        return avg_map_dict, stdev_map_dict

