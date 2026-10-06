"""VBM postprocessing."""

import pandas as pd
import numpy as np
import nibabel as nib
from tqdm import tqdm
import os

class PostProcessing:
    def __init__(self, mask_path=None):
        """Initialize the PostProcessing class."""
        self.mask_path = mask_path
        self.brain_indices = self._load_brain_indices()

    def _load_brain_indices(self):
        """Load brain indices from the mask file."""
        try:
            mask_data = nib.load(self.mask_path).get_fdata().flatten()
            return np.where(mask_data > 0)[0]
        except Exception as e:
            raise FileNotFoundError("Error loading brain mask: " + str(e))

    def _initialize_damage_df(self, thresholded_df):
        """Initialize an empty damage dataframe."""
        return pd.DataFrame(index=thresholded_df.columns)

    def _calculate_metrics(self, damage_df, thresholded_df, region_of_interest_df, roi, subject, metrics):
        """Calculate metrics for damage scores."""
        subject_array = thresholded_df[subject].values
        roi_array = region_of_interest_df[roi].values

        if 'spatial_correlation' in metrics:
            damage_df.loc[subject, f'{roi}_spatial_corr'] = self._calculate_spatial_correlation(subject_array, roi_array)
        if 'cosine' in metrics:
            damage_df.loc[subject, f'{roi}_cosine'] = self._calculate_cosine_similarity(subject_array, roi_array)
        if 'sum' in metrics:
            damage_df.loc[subject, f'{roi}_sum'] = self._calculate_dot_product(subject_array, roi_array)
        if 'average' in metrics:
            damage_df.loc[subject, f'{roi}_average'] = self._calculate_normalized_dot_product(subject_array, roi_array)
        if 'num_atrophic' in metrics:
            damage_df.loc[subject, f'{roi}_num_atrophic'] = self._count_voxels_greater_than_threshold(subject_array, mask=roi_array, threshold=2)

    def _calculate_spatial_correlation(self, array1, array2):
        """Calculate pearson correlation of two arrays."""
        mean1 = np.mean(array1)
        mean2 = np.mean(array2)
        numerator = np.sum((array1 - mean1) * (array2 - mean2))
        denominator = np.sqrt(np.sum((array1 - mean1) ** 2) * np.sum((array2 - mean2) ** 2))
        if denominator == 0:
            return 0  # Avoid division by zero
        return numerator / denominator

    def _calculate_cosine_similarity(self, array1, array2):
        """Calculate cosine similarity between two arrays."""
        dot_product = np.dot(array1, array2)
        norm1 = np.linalg.norm(array1)
        norm2 = np.linalg.norm(array2)
        if norm1 == 0 or norm2 == 0:
            return 0  # Avoid division by zero
        return dot_product / (norm1 * norm2)

    def _calculate_dot_product(self, array1, array2):
        """Calculate the dot product of two arrays."""
        return np.dot(array1, array2)

    def _calculate_normalized_dot_product(self, array1, array2):
        """Calculate the normalized dot product (average of the dot product)."""
        dot_product = np.dot(array1, array2)
        non_zero_elements = np.count_nonzero(array2)
        if non_zero_elements == 0:
            return 0  # Avoid division by zero
        return dot_product / non_zero_elements

    def _count_voxels_greater_than_threshold(self, array, mask=None, threshold=2):
        """Count the number of voxels where the value is greater than the threshold within a given mask."""
        if mask is None:
            mask = self.brain_indices
        if mask is not None:
            mask_indices = np.where(mask > 0)[0] # Ensure mask provides binary indices
            array = array[mask_indices]
        return np.sum(array > threshold)

    def sort_dataframes_by_index(self, damage_df_dict):
        """Sort dataframes by their index."""
        sorted_df_dict = {}
        for key, df in damage_df_dict.items():
            try:
                df.index = df.index.astype(int)
                sorted_df_dict[key] = df.sort_index(ascending=True)
            except ValueError:
                sorted_df_dict[key] = df
        return sorted_df_dict
    
    def calculate_damage_scores(self, thresholded_atrophy_df_dict, region_of_interest_df, metrics=['cosine', 'sum', 'average', 'num_atrophic', 'total']):
        """Calculate damage scores for given metrics."""
        damage_df_dict = {}
        for key, thresholded_df in tqdm(thresholded_atrophy_df_dict.items()):
            damage_df = self._initialize_damage_df(thresholded_df)
            for roi in region_of_interest_df.columns:
                for subject in thresholded_df.columns:
                    self._calculate_metrics(damage_df, thresholded_df, region_of_interest_df, roi, subject, metrics)
            damage_df_dict[key] = damage_df
        return damage_df_dict
    
    def save_csv_to_metadata(self, dataframe, root_dir, analysis='atrophy_results', ses=None, dry_run=True):
        """Save a single dataframe to a CSV file."""
        sub_no = 'all'
        ses_no = ses if ses else '01'
        filename = f'{analysis}_ses-{ses_no}_sub-{sub_no}.csv'
        out_dir = os.path.join(root_dir, 'metadata')
        os.makedirs(out_dir, exist_ok=True)
        file_path = os.path.join(out_dir, filename)
        if dry_run:
            print(file_path)
        else:
            dataframe.to_csv(file_path)
            print('Saved to: ', file_path)
