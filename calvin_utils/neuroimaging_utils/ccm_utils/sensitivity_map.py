import os
import numpy as np
import pandas as pd
import nibabel as nib
from calvin_utils.neuroimaging_utils.ccm_utils.overlap_map import OverlapMap


class SensitivityMap(OverlapMap):
    def __init__(self, df, mask_path: str | None = None, out_dir: str | None = None, **kwargs):
        """Generate overlap, stepwise, and continuous average maps.

        ``df`` must have shape ``(observations, voxels)``. A mapping of dataset
        names to DataFrames is also accepted when several cohorts should be
        processed together.
        """
        super().__init__(data_loader=None, mask_path=mask_path, out_dir=out_dir, **kwargs)
        self.df = self._get_dict_data(df)

    ### Setter/Getter ###
    @staticmethod
    def _get_dict_data(data):
        if isinstance(data, pd.DataFrame):
            return {"df": data}
        if isinstance(data, dict) and data and all(
            isinstance(value, pd.DataFrame) for value in data.values()
        ):
            return data
        raise TypeError("df must be a pandas DataFrame or a non-empty mapping of DataFrames")

    @staticmethod
    def _as_observation_matrix(name, df):
        values = df.to_numpy(dtype=np.float32, copy=False)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError(
                f"Dataset {name!r} must have shape (observations, voxels) "
                "and contain at least one observation"
            )
        return values

    ### Core Logic ###
    def generate_average_maps(self):
        """Return each dataset's continuous voxelwise mean across observations."""
        out = {}
        for name, df in self.df.items():
            values = self._as_observation_matrix(name, df)
            out[name] = np.nanmean(values, axis=0).astype(np.float32)
        return out

    def generate_overlap_maps(self):
        out = {}
        for name, df in self.df.items():
            values = self._as_observation_matrix(name, df)
            bin_ = self._binarize(values, self.threshold)
            out[name] = np.nansum(bin_, axis=0).astype(np.float32)
        return out

    def generate_stepwise_maps(self):
        out = {}
        for name, df in self.df.items():
            values = self._as_observation_matrix(name, df)
            n_subj = values.shape[0]
            bin_ = self._binarize(values, self.threshold)
            pct = bin_.sum(0) / n_subj * 100
            out[name] = (np.floor(pct / self.step_size) * self.step_size).astype(np.float32)
        return out

    ### i/o ###
    def _save_map(self, arr, file_name):
        """Save either a full-volume or mask-vectorized DataFrame result."""
        mask_img = nib.load(self.mask_path)
        mask_data = np.asarray(mask_img.dataobj)
        flat_arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        if flat_arr.size == mask_data.size:
            volume = flat_arr.reshape(mask_data.shape)
        else:
            mask_indices = mask_data.reshape(-1) > 0
            if flat_arr.size != np.count_nonzero(mask_indices):
                raise ValueError(
                    f"Map has {flat_arr.size} voxels, but mask expects either "
                    f"{mask_data.size} full-volume or "
                    f"{np.count_nonzero(mask_indices)} masked voxels"
                )
            volume = np.zeros(mask_data.size, dtype=np.float32)
            volume[mask_indices] = flat_arr
            volume = volume.reshape(mask_data.shape)

        img = nib.Nifti1Image(volume, affine=mask_img.affine)
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)
            threshold = f"{self.threshold:g}"
            out_path = os.path.join(self.out_dir, f'threshold_{threshold}_{file_name}')
            nib.save(img, out_path)
        return img

    ### Public API ###
    def run(self, return_average: bool = False):
        """Generate sensitivity maps.

        The default two-item return preserves the historical public API.
        Pass ``return_average=True`` to also receive continuous averages.
        """
        overlap = self.generate_overlap_maps()
        stepwise = self.generate_stepwise_maps()
        average = self.generate_average_maps()
        self.average_maps_ = average

        if self.out_dir and self.mask_path:
            self.save_maps(overlap,   suffix='_n_overlap')
            self.save_maps(stepwise, suffix='_percent_overlap_stepwise')
            self.save_maps(average, suffix='_average')
        elif self.out_dir and not self.mask_path:
            print("No mask_path supplied. Skipping NIfTI export.")

        if return_average:
            return overlap, stepwise, average
        return overlap, stepwise
