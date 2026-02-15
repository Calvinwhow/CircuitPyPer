import nibabel as nib
import numpy as np
import os
class ThresholdNifti:
    def __init__(self, path, t,  command='keep_above'):
        """
        Initialize the object with the path to NIFTI files, threshold values, and a thresholding command.
            path (str): Path to the NIFTI files.
            t (list): List of threshold values.
            command (str, optional): Thresholding command ('keep_above', 'keep_below', 'keep_between', or 'exclude_between'). Default is 'keep_above'.
        """
        self.path = path
        self.t = t
        self.command = command
        self._validate_thresholds()
        self._prep_thresholds()
        self.arr, self.img = self._load_nifti()
        
    def _load_nifti(self):
        """Load the NIFTI file and return the numpy array."""
        img = nib.load(self.path)
        arr = img.get_fdata()
        arr = np.asarray(arr)
        arr = np.nan_to_num(arr, nan=0.0)
        if np.isposinf(arr).any():
            arr[np.isposinf(arr)] = np.finfo(arr.dtype).max
        if np.isneginf(arr).any():
            arr[np.isneginf(arr)] = np.finfo(arr.dtype).min
        return arr, img
    
    def _validate_thresholds(self):
        if self.t is None:
            raise ValueError("Threshold values (t) must not be None.")
        if not isinstance(self.t, (list, tuple)):
            raise TypeError("Threshold values (t) must be a list or tuple.")
        if len(self.t) < 1 or len(self.t) > 2:
            raise ValueError("Threshold values (t) must have 1 or 2 elements.")
        if not all(isinstance(x, (int, float)) for x in self.t):
            raise TypeError("All threshold values must be int or float.")
        if len(self.t) == 1 and self.command not in ("keep_above", "keep_below"):
            raise ValueError("If one threshold value is provided, command must be 'keep_above' or 'keep_below'.")
        if len(self.t) == 2 and self.command not in ("keep_between", "exclude_between"):
            raise ValueError("If two threshold values are provided, command must be 'keep_between' or 'exclude_between'.")

    def _prep_thresholds(self):
        if len(self.t) == 2:
            self.t = sorted(self.t)

    def _get_thresh(self, arr, thr, direction):
        if direction == "keep_above":
            return arr >= thr[0]
        if direction == "keep_below":
            return arr <= thr[0]
        if direction in ("keep_between", "exclude_between"):
            lo, hi = thr
            inside = (arr >= lo) & (arr <= hi)
            return inside if direction == "keep_between" else ~inside
        raise ValueError(f"Entered: {direction}. Expected one of: keep_above, keep_below, keep_between, exclude_between")
    
    def _thresh_img(self):
        t_mask = self._get_thresh(self.arr, self.t, self.command)
        return self.arr * t_mask
    
    def _save_masked_image(self, arr):
        if self.path.endswith('.nii.gz'):
            fname = self.path[:-7]
            ext = '.nii.gz'
        else:
            fname, ext = os.path.splitext(self.path)
        fname = f"{fname}_thresh{ext}"
        nib.save(nib.Nifti1Image(arr, self.img.affine, self.img.header), fname)
    
    def run(self):
        """Threshold the NIFTI file based on the specified thresholds and command."""
        arr = self._thresh_img()
        self._save_masked_image(arr)