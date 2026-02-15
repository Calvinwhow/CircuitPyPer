import os
import numpy as np
import nibabel as nib
from calvin_utils.nifti_utils.generate_nifti import view_and_save_nifti
from calvin_utils.nifti_utils.matrix_utilities import threshold_matrix, import_nifti_to_numpy_array, unmask_matrix, apply_mask_to_dataframe

class ConjunctionMap:
    def __init__(self, nift_path_1, nift_path_2, threshold1, threshold2, direction1, direction2, mask_path=None, 
                 output_dir='generated_nifti', output_name='conjunction_map.nii', method='unsigned'):
        """
        Initialize the class with two data frames, two threshold values, and two directions for thresholding.

        Parameters:
        nift_path_1, nift_path_2 : path
            Path to NIFTI files and rows represent voxels.
        threshold1, threshold2 : float
            Threshold values for the data frames.
        direction1, direction2 : str
            Directions for thresholding ('keep_above', 'keep_below', 'keep_between', or 'exclude_between').
            If you want to apply a keep_between or exclude_between threshold, you must provide a tuple of two values for the corresponding threshold
        mask_path : str, optional
            Shared path to the NIFTI mask file.
        output_dir : str
            The path to save the output NIfTI file.
        output_name : str, optional
            The name to use when saving the NIfTI file.
        method : str, optional
            If unsigned, returns regions meeting threshold and directional criteria. Map is [0,1]
            If signed, returns regions meeting threshold, direction criteria, and sharing signs. Map is [-1,0,1] 
            If agreement, 
            Defaults to unsigned
        """
        self.df1 = import_nifti_to_numpy_array(nift_path_1).flatten()
        self.df2 = import_nifti_to_numpy_array(nift_path_2).flatten()
        self.threshold1 = threshold1
        self.threshold2 = threshold2
        self.direction1 = direction1
        self.direction2 = direction2
        self.mask_path = mask_path
        self.output_dir = output_dir
        self.output_name = output_name
        self.method = method
        
        if mask_path:
            self.mask = import_nifti_to_numpy_array(mask_path).astype(bool)
        else:
            raise ValueError("Mask path is required. Please provide a valid mask path.")
    @staticmethod
    def _thresh(arr, thr, direction):
        if direction == "keep_above":
            return arr >= thr
        if direction == "keep_below":
            return arr <= thr
        if direction in ("keep_between", "exclude_between"):
            lo, hi = sorted(thr)
            inside = (arr >= lo) & (arr <= hi)
            return inside if direction == "keep_between" else ~inside
        raise ValueError(f"Bad direction: {direction}")

    @staticmethod
    def _clean(arr, fill=0.0):
        vmax, vmin = np.nanmax(arr), np.nanmin(arr)
        return np.nan_to_num(arr, nan=fill, posinf=vmax, neginf=vmin)

    ### Helpers ###
    def mask_matrices(self):
        mflat = self.mask.flatten()
        self.df1 = self.df1[mflat]
        self.df2 = self.df2[mflat]

    def clean_matrices(self):
        self.df1 = self._clean(self.df1)
        self.df2 = self._clean(self.df2)

    def threshold_matrices(self):
        self.keep1 = self._thresh(self.df1, self.threshold1, self.direction1)
        self.keep2 = self._thresh(self.df2, self.threshold2, self.direction2)

    def perform_conjunction(self):
        base = self.keep1 & self.keep2
        if self.method=='unsigned':
            self.keep_conj = base.astype(np.int8)
        elif self.method=='signed':
            pos = base & (self.df1 > 0) & (self.df2 > 0)
            neg = base & (self.df1 < 0) & (self.df2 < 0)
            conj = np.zeros_like(self.df1, dtype=np.int8)
            conj[pos] = 1
            conj[neg] = -1
            self.keep_conj = conj
        elif self.method=='agreement':
            pos = (self.df1 > 0) & (self.df2 > 0)
            neg = (self.df1 < 0) & (self.df2 < 0)
            conj = np.zeros_like(self.df1, dtype=np.int8)
            conj[pos] = 1
            conj[neg] = -1
            self.keep_conj = conj 
        else:
            raise ValueError("Invalid method specified for conjunction analysis. Supported methods are 'unsigned', 'signed', and 'agreement'.")
        self.indices = np.where(self.keep_conj != 0)[0]

    ### Plotting ###
    def generate_save_and_view_nifti(self):
        full = np.zeros(self.mask.size, dtype=np.int8)
        full[self.mask.flatten()] = self.keep_conj.astype(np.int8)
        full = full.reshape(self.mask.shape)
        self.img = nib.Nifti1Image(full, affine=nib.load(self.mask_path).affine)
        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, self.output_name)
        nib.save(self.img, out_path)
    
    def get_html(self):
        full = np.zeros(self.mask.size, dtype=np.int8)
        full[self.mask.flatten()] = self.keep_conj.astype(np.int8)
        full = full.reshape(self.mask.shape)
        
        html = view_and_save_nifti(
            full,
            out_dir=self.output_dir,
            output_name=self.output_name,
            ref_file=self.mask_path,
        )
        return html

    ### Public ###
    def run(self):
        self.mask_matrices()
        self.clean_matrices()
        self.threshold_matrices()
        self.perform_conjunction()
        self.generate_save_and_view_nifti()
        return self.get_html()