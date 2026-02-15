import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.spatial import KDTree
import glob

class ImageDownSampler:
    """
    A robust image downsampling utility. Works either 
    relative to a template or by a direct approach.
    
    Note:
    This class gets the grid of values in the origin space and calculates the real (mm) location of each voxel. 
    Then, it sets up a grid of values in the destingation space and calculates the real (mm) location of each voxel. 
    It then takes each point value in the origin space and allocates it to the closest neighbouring voxel in destination space.
    This is done by choosing the destination voxel as the index with the lowest pairwise distance. 
    Each time a value is entered into a destination voxel, it is added to it. 
    """

    def __init__(self, origin_img_path, target_img_path):
        self.origin_img = origin_img_path
        self.target_img = target_img_path
        self.output_path = origin_img_path
        
    ###INTERNAL METHODS###
    def _load_and_validate_image(self, path):
        if not os.path.exists(path):
            raise ValueError(f"File does not exist: {path}")
        img = nib.load(path)
        if not isinstance(img.affine, np.ndarray) or img.affine.shape != (4, 4):
            raise ValueError(f"Affine matrix is corrupt or invalid in file: {path}")
        return img

    def _find_closest_coordinates_kdtree(self, origin_xyz, target_xyz):
        """
        Build a KDTree from 'target_xyz' and find the closest neighbor index in 'target_xyz'
        for each point in 'origin_xyz'. Prevents need for pairwise euclidean distance calculation. 
        """
        tree = KDTree(target_xyz)
        distances, indexes = tree.query(origin_xyz, k=1)
        return indexes
    
    def _get_grid_of_ijk_coords_from_img(self, img):
        img_shape = img.get_fdata().shape
        x_indices = np.arange(0, img_shape[0])
        y_indices = np.arange(0, img_shape[1])
        z_indices = np.arange(0, img_shape[2])
        i_plane, j_plane, k_plane = np.meshgrid(x_indices, y_indices, z_indices, indexing='ij')
        ijk_grid = np.stack([i_plane, j_plane, k_plane], axis=-1)
        return ijk_grid
    
    def _get_grid_of_real_coords_from_img(self, img):
        ijk_grid = self._get_grid_of_ijk_coords_from_img(img)
        ijk_flat = ijk_grid.reshape(-1,3)
        xyz_flat = nib.affines.apply_affine(img.affine, ijk_flat)
        return xyz_flat
            
    ###PROPERTY METHODS###
    @property
    def origin_img(self):
        return self._origin_img

    @origin_img.setter
    def origin_img(self, path):
        self._origin_img = self._load_and_validate_image(path)

    @property
    def target_img(self):
        return self._target_img

    @target_img.setter
    def target_img(self, path):
        self._target_img = self._load_and_validate_image(path)
    
    @property
    def output_path(self):
        return self._output_path

    @output_path.setter
    def output_path(self, path):
        base, ext = os.path.splitext(path)
        if ext == '.gz':
            base, ext2 = os.path.splitext(base)
            ext = ext2 + ext
        output_path = base + '_resampled' + ext
        print("Output path set to: ", output_path)
        self._output_path = output_path
        
    ###BUSINESS LOGIC###

    def map_origin_grid_to_target_grid(self):
        '''Find location of origin data in target space'''
        origin_xyz = self._get_grid_of_real_coords_from_img(self.origin_img)
        target_xyz = self._get_grid_of_real_coords_from_img(self.target_img)
        origin_to_target_idx = self._find_closest_coordinates_kdtree(origin_xyz, target_xyz)
        return origin_to_target_idx
            
    def assign_origin_to_target_location(self, origin_to_target_idx):
        '''Will add the origin data to its location in the target, preventing drops'''
        origin_data_flat = self.origin_img.get_fdata().ravel()
        target_shape = self.target_img.shape
        target_data_flat = np.zeros(target_shape, dtype=origin_data_flat.dtype).ravel()
        target_data_flat[origin_to_target_idx] += origin_data_flat
        target_data_resampled = target_data_flat.reshape(target_shape)
        return target_data_resampled

    def generate_nifti(self, target_data_resampled):
        '''Create a nifti modelled after the target'''
        out_img = nib.Nifti1Image(target_data_resampled, self.target_img.affine, self.target_img.header)
        nib.save(out_img, self.output_path)
    
    def resample_img(self):
        '''Orchestration method'''
        origin_to_target_idx = self.map_origin_grid_to_target_grid()
        resampled_data = self.assign_origin_to_target_location(origin_to_target_idx)
        self.generate_nifti(resampled_data)