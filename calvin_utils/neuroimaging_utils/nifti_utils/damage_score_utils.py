import pandas as pd
import numpy as np
import nibabel as nib
from nilearn import image
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
        if self.mask_path is None:
            return None
        try:
            mask_data = nib.load(self.mask_path).get_fdata().flatten()
            return np.where(mask_data > 0)[0]
        except Exception as e:
            raise FileNotFoundError("Error loading brain mask: " + str(e))

    def _initialize_damage_df(self):
        """Initialize an empty damage dataframe."""
        return pd.DataFrame(index=self.dv_df.columns)

    @staticmethod
    def _calculate_metrics(subject_array, roi_array, metrics, score_nonzero_only=False):
        '''Gets metric of damage between each independent variable nifti and dependent variable nifti'''
        subject_array = np.asarray(subject_array, dtype=float).copy()
        roi_array = np.asarray(roi_array, dtype=float).copy()
        if score_nonzero_only:
            zero_mask = np.isfinite(subject_array) & (subject_array == 0)
            roi_array[zero_mask] = 0
        subject_array = np.nan_to_num(subject_array, nan=0.0, posinf=np.nanmax(subject_array[np.isfinite(subject_array)]) if np.isfinite(subject_array).any() else 0, neginf=np.nanmin(subject_array[np.isfinite(subject_array)]) if np.isfinite(subject_array).any() else 0)
        roi_array = np.nan_to_num(roi_array, nan=0.0, posinf=np.nanmax(roi_array[np.isfinite(roi_array)]) if np.isfinite(roi_array).any() else 0, neginf=np.nanmin(roi_array[np.isfinite(roi_array)]) if np.isfinite(roi_array).any() else 0)
        
        results = {}
        if 'spatial_correlation' in metrics:
            results['spatial_correlation'] = DamageScorer._calculate_spatial_correlation(subject_array, roi_array)
        if 'cosine' in metrics:
            results['cosine'] = DamageScorer._calculate_cosine_similarity(subject_array, roi_array)
        if 'sum' in metrics:
            results['sum'] = DamageScorer._calculate_dot_product(subject_array, roi_array)
        if 'max_in_roi' in metrics:
            results['max_in_roi'] = DamageScorer._calculate_max_in_roi(subject_array, roi_array)
        if 'min_in_roi' in metrics:
            results['min_in_roi'] = DamageScorer._calculate_min_in_roi(subject_array, roi_array)
        if 'avg_in_target' in metrics:
            results['avg_in_target'] = DamageScorer._calculate_normalized_dot_product(subject_array, roi_array, denominator='avg_in_target')
        if 'avg_in_subject' in metrics:
            results['avg_in_subject'] = DamageScorer._calculate_normalized_dot_product(subject_array, roi_array, denominator='avg_in_subject')
        if 'num_in_roi' in metrics:
            results['num_in_roi'] = DamageScorer._count_voxels_greater_than_threshold(subject_array, roi_array, threshold=2)
        if 'dice' in metrics:
            results['dice'] = DamageScorer._calculate_dice(subject_array, roi_array)
        return results

    @staticmethod
    def _normalize_metric_name(metric):
        """Normalize public metric aliases to the internal DamageScorer names."""
        aliases = {
            "spatial_correl": "spatial_correlation",
            "spatial_corr": "spatial_correlation",
            "correlation": "spatial_correlation",
        }
        return aliases.get(metric, metric)

    @staticmethod
    def _output_metric_name(metric):
        """Return the user-facing score column stem for a metric."""
        output_names = {
            "spatial_correlation": "spatial_correl",
        }
        return output_names.get(metric, metric)

    @staticmethod
    def _score_column_name(metric, target_suffix=None):
        """Build a score column name without forcing a target suffix."""
        metric_name = DamageScorer._output_metric_name(metric)
        if target_suffix is None or str(target_suffix).strip() == "":
            return metric_name
        return f"{metric_name}_{target_suffix}"

    @staticmethod
    def _calculate_max_in_roi(array1, roi_arr):
        '''Expects a binary'''
        return np.nanmax(array1[roi_arr>0])
        
    @staticmethod
    def _calculate_min_in_roi(array1, roi_arr):
        '''Expects a binary'''
        return np.nanmin(array1[roi_arr>0])

    @staticmethod
    def _calculate_spatial_correlation(array1, array2):
        '''Calculates pearson correlation of 2 arrays'''
        mean1 = np.mean(array1)
        mean2 = np.mean(array2)
        numerator = np.sum((array1 - mean1) * (array2 - mean2))
        denominator = np.sqrt(np.sum((array1 - mean1) ** 2) * np.sum((array2 - mean2) ** 2))
        if denominator == 0:
            return 0  # Avoid division by zero
        return numerator / denominator

    @staticmethod
    def _calculate_cosine_similarity(array1, array2):
        '''Calculate cosine similarity between two arrays'''
        dot_product = np.dot(array1, array2)
        norm1 = np.linalg.norm(array1)
        norm2 = np.linalg.norm(array2)
        if norm1 == 0 or norm2 == 0:
            return 0  # Avoid division by zero
        return dot_product / (norm1 * norm2)

    @staticmethod
    def _calculate_dice(array1, array2):
        '''Calculates dice coefficient'''
        return dice(array1, array2)
        
    @staticmethod
    def _calculate_dot_product(array1, array2):
        '''Calculate the dot product of two arrays'''
        return np.dot(array1, array2)

    @staticmethod
    def _calculate_normalized_dot_product(subj_arr, roi_arr, denominator='array_1'):
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

    @staticmethod
    def _count_voxels_greater_than_threshold(array, roi_arr, threshold=2):
        '''Count the number of voxels where the value is greater than the threshold within a given mask'''
        mask_indices = np.where(roi_arr != 0)[0] # Ensure mask provides binary indices
        array = array[mask_indices]
        return np.sum(array > threshold)

    def score_csv_against_target(
        self,
        csv_path: str,
        *,
        path_col: str,
        target_path: str,
        selected_damage: str | list[str] = 'avg_in_target',
        target_suffix: str | None = None,
        threshold: float | None = None,
        out_path: str | None = None,
        resample_interpolation: str = "nearest",
        resample_to_target: bool = True,
        target_threshold: float = 0.0,
        score_nonzero_only: bool = False,
        verbose: bool = False,
        log_resample: bool = True,
    ) -> pd.DataFrame:
        """
        Load a CSV, score each NIfTI in `path_col` against a target image,
        and save the results back to CSV.

        If resample_to_target is True, each subject image is resampled into the
        target image space. If False, the target image is resampled into each
        subject image space before scoring.
        `target_suffix` is optional and retained only for backwards-compatible
        column naming; omit it to write columns named by metric only.
        """
        df = pd.read_csv(csv_path)
        if path_col not in df.columns:
            raise ValueError(f"Missing path column: {path_col}")

        target_img = image.load_img(target_path)
        mask_img = None
        if self.mask_path is not None:
            mask_img = image.load_img(self.mask_path)

        if verbose or log_resample:
            print(
                f"Target image: shape={target_img.shape}, "
                f"zooms={target_img.header.get_zooms()[:3]}"
            )
            print("Resample mode: subjects -> target" if resample_to_target else "Resample mode: target -> subjects")
            if mask_img is not None:
                print(
                    f"Mask image: shape={mask_img.shape}, "
                    f"zooms={mask_img.header.get_zooms()[:3]}"
                )

        raw_metrics = [selected_damage] if isinstance(selected_damage, str) else list(selected_damage)
        metrics = [self._normalize_metric_name(metric) for metric in raw_metrics]
        valid_metrics = {
            "spatial_correlation",
            "cosine",
            "sum",
            "max_in_roi",
            "min_in_roi",
            "avg_in_target",
            "avg_in_subject",
            "num_in_roi",
            "dice",
        }
        invalid_metrics = sorted(set(metrics) - valid_metrics)
        if invalid_metrics:
            raise ValueError(f"Unsupported selected_damage metric(s): {invalid_metrics}")

        def same_space(img_a, img_b):
            return img_a.shape == img_b.shape and np.allclose(img_a.affine, img_b.affine)

        def data_in_space(src_img, space_img, label, interpolation):
            if same_space(src_img, space_img):
                if log_resample or verbose:
                    print(f"[no resample] {label} -> scoring space")
                return src_img.get_fdata()
            if log_resample or verbose:
                print(
                    f"[resample] {label} shape={src_img.shape}, "
                    f"zooms={src_img.header.get_zooms()[:3]} "
                    f"-> space shape={space_img.shape}, "
                    f"zooms={space_img.header.get_zooms()[:3]}"
                )
            resampled = image.resample_to_img(
                src_img,
                space_img,
                interpolation=interpolation,
                force_resample=True,
                copy_header=True,
            )
            return resampled.get_fdata()

        def target_and_mask_in_space(space_img):
            target_data = data_in_space(target_img, space_img, "target_path", "nearest")
            target_data = np.nan_to_num(target_data, nan=0.0, posinf=0.0, neginf=0.0)

            if mask_img is None:
                target_mask = target_data > target_threshold
            else:
                mask_data = data_in_space(mask_img, space_img, "mask_path", "nearest")
                mask_data = np.nan_to_num(mask_data, nan=0.0, posinf=0.0, neginf=0.0)
                target_mask = mask_data > 0
            return target_data, target_mask

        fixed_target_vec = None
        fixed_target_mask = None
        if resample_to_target:
            fixed_target_data, fixed_target_mask = target_and_mask_in_space(target_img)
            fixed_target_vec = fixed_target_data.flatten()[fixed_target_mask.flatten()]

        subject_paths = df[path_col].tolist()
        score_rows = []
        last_target_vec = fixed_target_vec
        for path in tqdm(subject_paths, desc="Scoring damage", disable=log_resample):
            if pd.isna(path) or not isinstance(path, str) or not path:
                score_rows.append({self._score_column_name(metric, target_suffix): np.nan for metric in metrics})
                continue
            try:
                subj_img = image.load_img(path)
                if resample_to_target:
                    subj_data = data_in_space(subj_img, target_img, path, resample_interpolation)
                    target_vec = fixed_target_vec
                    target_mask = fixed_target_mask
                else:
                    subj_data = subj_img.get_fdata()
                    target_data, target_mask = target_and_mask_in_space(subj_img)
                    target_vec = target_data.flatten()[target_mask.flatten()]
                    last_target_vec = target_vec
                subj_data = np.nan_to_num(subj_data, nan=0.0, posinf=0.0, neginf=0.0)
                subj_vec = subj_data.flatten()[target_mask.flatten()]
                results = self._calculate_metrics(
                    subj_vec,
                    target_vec,
                    metrics,
                    score_nonzero_only=score_nonzero_only,
                )
                score_rows.append(
                    {
                        self._score_column_name(metric, target_suffix): results.get(metric, np.nan)
                        for metric in metrics
                    }
                )
            except Exception:
                if verbose:
                    print(f"Failed to score {path}")
                score_rows.append({self._score_column_name(metric, target_suffix): np.nan for metric in metrics})

        self.dv_df = None
        self.roi_df = pd.DataFrame({"target": last_target_vec if last_target_vec is not None else []})
        self.damage_df = pd.DataFrame(score_rows, index=subject_paths)
        self.damage_df.index.name = path_col
        for col in self.damage_df.columns:
            df[col] = self.damage_df[col].to_numpy()

        out_path = out_path or csv_path
        df.to_csv(out_path, index=False)
        return df

    def sort_dataframes_by_index(self, df):
        try:
            df.index = df.index.astype(int)
            return df.sort_index(ascending=True)
        except:
            return df
    
    def calculate_damage_scores(self, metrics=['spatial_correlation', 'cosine', 'sum', 'avg_in_target', 'avg_in_subject', 'num_in_roi', 'max_in_roi', 'min_in_roi'], trace=False, score_nonzero_only=False):
        """
        Calculate damage scores for dv_df and roi_df based on specified metrics.
        This function computes damage scores by iterating through regions of interest and subjects,
        applying the specified metrics to evaluate the atrophy data.

        Returns:
            pd.DataFrame: A DataFrame containing the calculated damage scores for each subject
            and region of interest. Columns represent subjects, and rows represent regions.
        """
        self.damage_df = self._initialize_damage_df()            # (n_subjects, n_rois)
        for roi_idx, roi in tqdm(enumerate(self.roi_df.columns)):               # iterate over each roi
            for sub_idx, subject in enumerate(self.dv_df.columns):  
                subject_array = self.dv_df[subject].values
                roi_array = self.roi_df[roi].values# iterate over each subject
                results = self._calculate_metrics(
                    subject_array,
                    roi_array,
                    metrics,
                    score_nonzero_only=score_nonzero_only,
                ) if (not trace or roi_idx == sub_idx) else {}
                for metric, value in results.items():
                    self.damage_df.loc[subject, f"{roi}_{metric}"] = value
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
