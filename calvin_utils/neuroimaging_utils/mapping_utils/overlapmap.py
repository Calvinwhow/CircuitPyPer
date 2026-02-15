import os
import numpy as np
import pandas as pd
import nibabel as nib
from calvin_utils.ccm_utils.overlap_map import OverlapMap

class OverlapMapDF(OverlapMap):
    def __init__(self, df_dict, mask_path: str | None = None, out_dir: str | None = None, **kwargs):
        '''
        Basic overwrite of OverlapMap that allows a dataframe [shape: (observations, voxels)] to be passed. 
        Read in from calvin_utils.ccm_utils.overlap_map import OverlapMap for more info. 
        '''
        super().__init__(data_loader=None, mask_path=mask_path, out_dir=out_dir, **kwargs)
        self.df_dict = self._get_dict_data(df_dict)
        
    ### Setter/Getter ###
    def _get_dict_data(self, data):
        if isinstance(data, pd.DataFrame):
            return {"df": data}
        else:
            return data
            
    ### Core Logic Overwrite ###
    def generate_overlap_maps(self):
        out = {}
        for name, df in self.df_dict.items():
            bin_ = self._binarize(df.values.astype(np.float32), self.threshold)
            out[name] = bin_.sum(0).astype(np.float32)
        return out

    def generate_stepwise_maps(self):
        out = {}
        for name, df in self.df_dict.items():
            n_subj = len(df)
            bin_ = self._binarize(df.values.astype(np.float32), self.threshold)
            pct = bin_.sum(0) / n_subj * 100
            out[name] = (np.floor(pct / self.step_size) * self.step_size).astype(np.float32)
        return out

    ### i/o ###
    def _save_map(self, arr, file_name):
        """Overwritten to work with df data"""
        mask_img = nib.load(self.mask_path)
        arr = arr.reshape(mask_img.get_fdata().shape)
        img = nib.Nifti1Image(arr, affine=mask_img.affine)
        self._visualize_map(img,title=file_name)
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)
            out_path = os.path.join(self.out_dir, f'threshold_{int(self.threshold)}_{file_name}')
            nib.save(img, out_path)
        return img
    
    ### Public API ###
    def run(self):
        '''Overridden run function'''
        overlap = self.generate_overlap_maps()
        stepwise = self.generate_stepwise_maps()

        if self.out_dir and self.mask_path:
            self.save_maps(overlap,   suffix='_n_overlap')
            self.save_maps(stepwise, suffix='_percent_overlap_stepwise')
        elif self.out_dir and not self.mask_path:
            print("‣ No mask_path supplied → skipping NIfTI export.")

        return overlap, stepwise
