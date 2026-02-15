import pandas as pd
import numpy as np
import nibabel as nib
from tqdm import tqdm
import os
from scipy.stats import dirichlet
from scipy.spatial.distance import dice

class DamageScorer:
    def __init__(self, mask_path=None, dv_df=None, roi_df=None):
        """
        Initializes the class with the given mask path, dv_df, and roi_df, and loads brain indices.
        Args:
            mask_path (str, optional): The file path to the mask file. Defaults to None.
            dv_df (pd.DataFrame, optional): A DataFrame containing thresholded nifti data
                for each subject. Defaults to None.
            roi_df (pd.DataFrame, optional): A DataFrame containing regions of interest or map data.
                Defaults to None.
        Attributes:
            mask_path (str): Stores the provided mask file path.
            brain_indices (Any): Stores the brain indices loaded by the `_load_brain_indices` method.
            dv_df (pd.DataFrame): A DataFrame containing thresholded nifti data for each subject.
            roi_df (pd.DataFrame): A DataFrame containing regions of interest or map data.
        """
        
        self.mask_path = mask_path
        self.brain_indices = self._load_brain_indices()
        self._dv_df = None
        self._roi_df = None
        self.dv_df = dv_df
        self.roi_df = roi_df

    @property
    def dv_df(self):
        """Getter for dv_df."""
        return self._dv_df

    @dv_df.setter
    def dv_df(self, df):
        """Setter for dv_df, cleans up column names."""
        self._dv_df = df

    @property
    def roi_df(self):
        """Getter for roi_df."""
        return self._roi_df

    @roi_df.setter
    def roi_df(self, df):
        """Setter for roi_df, cleans up column names."""
        if df is not None:
            df.columns = [
            os.path.basename(col).replace('.nii', '').replace('.gz', '').strip().replace(' ', '_') 
            for col in df.columns
            ]
        self._roi_df = df

    def _load_brain_indices(self):
        try:
            mask_data = nib.load(self.mask_path).get_fdata().flatten()
            return np.where(mask_data > 0)[0]
        except Exception as e:
            raise FileNotFoundError("Error loading brain mask: " + str(e))

    def _initialize_damage_df(self):
        """Initialize an empty damage dataframe."""
        return pd.DataFrame(index=self.dv_df.columns)

    def _calculate_metrics(self, thresholded_df, region_of_interest_df, roi, subject, metrics):
        '''Gets metric of damage between each independent variable nifti and dependent variable nifti'''
        subject_array = thresholded_df[subject].values
        roi_array = region_of_interest_df[roi].values
        subject_array = np.nan_to_num(subject_array, nan=0.0, posinf=np.nanmax(subject_array[np.isfinite(subject_array)]) if np.isfinite(subject_array).any() else 0, neginf=np.nanmin(subject_array[np.isfinite(subject_array)]) if np.isfinite(subject_array).any() else 0)
        roi_array = np.nan_to_num(roi_array, nan=0.0, posinf=np.nanmax(roi_array[np.isfinite(roi_array)]) if np.isfinite(roi_array).any() else 0, neginf=np.nanmin(roi_array[np.isfinite(roi_array)]) if np.isfinite(roi_array).any() else 0)
        
        if 'spatial_correlation' in metrics:
            self.damage_df.loc[subject, f'{roi}_spatial_corr'] = self._calculate_spatial_correlation(subject_array, roi_array)
        if 'cosine' in metrics:
            self.damage_df.loc[subject, f'{roi}_cosine'] = self._calculate_cosine_similarity(subject_array, roi_array)
        if 'sum' in metrics:
            self.damage_df.loc[subject, f'{roi}_sum'] = self._calculate_dot_product(subject_array, roi_array)
        if 'avg_in_target' in metrics:
            self.damage_df.loc[subject, f'{roi}_average_subject_in_target'] = self._calculate_normalized_dot_product(subject_array, roi_array, denominator='avg_in_target')
        if 'avg_in_subject' in metrics:
            self.damage_df.loc[subject, f'{roi}_average_target_in_subject'] = self._calculate_normalized_dot_product(subject_array, roi_array, denominator='avg_in_subject')
        if 'num_in_roi' in metrics:
            self.damage_df.loc[subject, f'{roi}_num_in_roi'] = self._count_voxels_greater_than_threshold(subject_array, mask=roi_array, threshold=2)
        if 'dice' in metrics:
            self.damage_df.loc[subject, f'{roi}_dice_coeff'] = self._calculate_dice(subject_array, roi_array)
        return self.damage_df

    def _calculate_spatial_correlation(self, array1, array2):
        '''Calculates pearson correlation of 2 arrays'''
        mean1 = np.mean(array1)
        mean2 = np.mean(array2)
        numerator = np.sum((array1 - mean1) * (array2 - mean2))
        denominator = np.sqrt(np.sum((array1 - mean1) ** 2) * np.sum((array2 - mean2) ** 2))
        if denominator == 0:
            return 0  # Avoid division by zero
        return numerator / denominator

    def _calculate_cosine_similarity(self, array1, array2):
        '''Calculate cosine similarity between two arrays'''
        dot_product = np.dot(array1, array2)
        norm1 = np.linalg.norm(array1)
        norm2 = np.linalg.norm(array2)
        if norm1 == 0 or norm2 == 0:
            return 0  # Avoid division by zero
        return dot_product / (norm1 * norm2)

    def _calculate_dice(self, array1, array2):
        '''Calculates dice coefficient'''
        return dice(array1, array2)
        
    def _calculate_dot_product(self, array1, array2):
        '''Calculate the dot product of two arrays'''
        return np.dot(array1, array2)

    def _calculate_normalized_dot_product(self, subj_arr, roi_arr, denominator='array_1'):
        '''Calculate the normalized dot product (average of the dot product)'''
        dot_product = np.dot(subj_arr, roi_arr)
        if denominator == 'avg_in_target':
            non_zero_elements = np.count_nonzero( ~np.isnan(subj_arr) & (roi_arr > 0))
        elif denominator == 'avg_in_subject':
            non_zero_elements = np.count_nonzero( ~np.isnan(roi_arr) & (subj_arr > 0))
        else:
            raise ValueError(f"Value {denominator} not supported.")
        if non_zero_elements == 0:
            return 0  # Avoid division by zero
        return dot_product / non_zero_elements

    def _count_voxels_greater_than_threshold(self, array, mask=None, threshold=2):
        '''Count the number of voxels where the value is greater than the threshold within a given mask'''
        if mask is None:
            mask = self.brain_indices
        if mask is not None:
            mask_indices = np.where(mask > 0)[0] # Ensure mask provides binary indices
            array = array[mask_indices]
        return np.sum(array > threshold)

    def sort_dataframes_by_index(self, df):
        try:
            df.index = df.index.astype(int)
            return df.sort_index(ascending=True)
        except:
            return df
    
    def calculate_damage_scores(self, metrics=['spatial_correlation', 'cosine', 'sum', 'avg_in_target', 'avg_in_subject', 'num_in_roi'], trace=False):
        """
        Calculate damage scores for dv_df and roi_df based on specified metrics.
        This function computes damage scores by iterating through regions of interest and subjects,
        applying the specified metrics to evaluate the atrophy data.
        Args:
            metrics (list of str, optional): A list of metrics to calculate damage scores. 
                Default is ['spatial_correlation', 'cosine', 'sum', 'average', 'num_in_roi'].
                Supported metrics include:
                    - 'spatial_correlation': Measures spatial correlation between niftis.
                    - 'cosine': Computes cosine similarity.
                    - 'sum': Calculates the sum of values.
                    - 'avg_in_roi': Computes the average of subject within target, treating the target as an ROI. i.e. calculate connectivity within a region
                    - 'avg_in_subject': Computes average of target within the subject, treating the subject as an ROI. i.e. calculate how connected some constant map (target) is to a subject's VTA 
                    - 'num_in_roi': Counts the number of suprathreshold voxels inside the mmask.
                    - 'dice': Takes the dice coefficient 
            trace (bool): determines if a trace matrix should be created or not
        Returns:
            pd.DataFrame: A DataFrame containing the calculated damage scores for each subject
            and region of interest. Columns represent subjects, and rows represent regions.
        """
        self.damage_df = self._initialize_damage_df()            # (n_subjects, n_rois)
        for roi_idx, roi in tqdm(enumerate(self.roi_df.columns)):               # iterate over each roi
            for sub_idx, subject in enumerate(self.dv_df.columns):              # iterate over each subject
                if not trace:
                    self._calculate_metrics(self.dv_df, self.roi_df, roi, subject, metrics)
                elif (trace) and (roi_idx == sub_idx):
                    self._calculate_metrics(self.dv_df, self.roi_df, roi, subject, metrics)
        if trace:
            self.damage_df['diagonal'] = np.diag(self.damage_df)
            self.damage_df = self.damage_df.loc[:,['diagonal']]
        self.damage_df.index.name = 'subject'
        return self.damage_df
    
    def save_csv_to_metadata(self, df, root_dir, analysis='atrophy_results', ses=None, dry_run=True):
        """
        Save a DataFrame as a CSV file in the metadata directory.
        This method saves the given DataFrame to a CSV file in a specified root directory
        under a subdirectory named 'metadata'. The filename is generated based on the 
        analysis type, session number, and subject number. If `dry_run` is True, the 
        method will only print the file path without saving the file.
        Args:
            df (pandas.DataFrame): The DataFrame to be saved as a CSV file.
            root_dir (str): The root directory where the 'metadata' folder will be created.
            analysis (str, optional): The type of analysis. Defaults to 'atrophy_results'.
            ses (str, optional): The session number. Defaults to None, which is replaced by '01'.
            dry_run (bool, optional): If True, only prints the file path without saving. Defaults to True.
        Returns:
            None
        """
        sub_no = 'all'
        ses_no = ses if ses else '01'
        filename = f'{analysis}_ses-{ses_no}_sub-{sub_no}.csv'
        out_dir = os.path.join(root_dir, 'metadata')
        os.makedirs(out_dir, exist_ok=True)
        file_path = os.path.join(out_dir, filename)
        if dry_run:
            print(file_path)
        else:
            df.to_csv(file_path)
            print('Saved to: ', file_path)
