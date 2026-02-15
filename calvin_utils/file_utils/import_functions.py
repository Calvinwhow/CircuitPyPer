## Paths Input Here
import warnings
warnings.filterwarnings('ignore')
import os 
import re
import numpy as np
import pandas as pd
import nibabel as nib
from glob import glob
from tqdm import tqdm
from nilearn import image
from calvin_utils.ccm_utils.bounding_box import NiftiBoundingBox
from calvin_utils.nifti_utils.generate_nifti import view_and_save_nifti
from pathlib import Path
from tqdm import tqdm
import random

class GiiNiiFileImport:
    """
    A versatile class for importing and processing NIFTI and GIFTI files into NumPy arrays.

    This class is designed to handle the import of NIFTI and GIFTI files, convert them into NumPy arrays,
    and provide options for customizing the import process, handling special values, and generating column names.

    Parameters:
    -----------
    import_path : str
        The path to the directory containing the files to be imported or the path to a CSV file with file paths.
    subject_pattern : str, optional
        A regular expression pattern indicating the part of the file path to be used as the subject ID.
    process_special_values : bool, optional
        Whether to handle NaNs and infinities in the data without significantly biasing the distribution.
    file_column : str, optional
        The name of the column in the CSV file that stores file paths (required if importing from CSV).
    file_pattern : str, optional
        A file pattern to filter specific files in a folder (e.g., '*.nii.gz').

    Attributes:
    -----------
    import_path : str
        The provided import path.
    subject_pattern : str
        The subject ID pattern.
    process_special_values : bool
        Indicates whether special values should be processed.
    file_column : str
        The name of the column in the CSV file storing file paths (if applicable).
    file_pattern : str
        The file pattern for filtering files in a folder.
    matrix_df : pandas.DataFrame
        A DataFrame to store imported data.
    seen_names : set
        A set to track unique column names.
    pattern : re.Pattern
        A compiled regular expression pattern for extracting subject IDs.
        
    Note:
    -----
    This class provides a flexible way to import and process neuroimaging data in NIFTI and GIFTI formats,
    making it suitable for various data analysis tasks.

    """
    def __init__(self, import_path, subject_pattern='', process_special_values=True, file_column: str=None, file_pattern: str=None, transpose=False):
        self.import_path = import_path
        self.file_pattern = file_pattern
        self.file_column = file_column
        self.transpose = transpose
        self.subject_pattern = re.compile(subject_pattern) if subject_pattern is not None else None
        self.process_special_values = process_special_values
        self.matrix_df = pd.DataFrame({})
        self.seen_names = set()
        self.pattern = re.compile(f"{self.subject_pattern}(\d+)")
    
    def generate_unique_column_name(self, file_path: str) -> str:
        base_name = os.path.basename(file_path)
        if base_name in self.seen_names:
            name = file_path
        else:
            name = base_name
            self.seen_names.add(base_name)
        return name
    
    def match_id(self, file_path):
        match = self.subject_pattern.search(file_path)
        if match:
            start_index = match.start()
            subject_id = file_path[start_index:]
            return subject_id
        else:
            return None
        
    def generate_name(self, file_path: str):
        """
        Generates a name based on the file path and an optional subject ID pattern.

        Parameters:
        -----------
        file_path : str
            The file path from which to extract the base name or subject ID.
        subject_id_pattern : str, optional
            A pattern indicating the part of the file_path to be used as the subject ID.

        Returns:
        --------
        str
            A generated name based on the file path and the specified pattern.
        """

        if self.subject_pattern is not None:
            name = self.match_id(file_path)
        else:
            name = self.generate_unique_column_name(file_path)
        return name 
    
    def handle_special_values(self, data):
        """
        Handles NaNs and infinities in the data without significantly biasing the distribution.

        Args:
            data: NumPy array with the data.

        Returns:
            Updated data with NaNs and infinities handled.
        # """
        max_val = np.nanmax(data)
        min_val = np.nanmin(data)
        data = np.nan_to_num(data, nan=0.0, posinf=max_val, neginf=min_val)
        return data
    
    @property
    def affines(self):
        if not hasattr(self, '_affines'):
            self._affines = set()
        return self._affines

    @affines.setter
    def affines(self, value):
        if not hasattr(self, '_affines'):
            self._affines = set()
        self._affines.add(value)

    def import_npy_to_numpy_array(self, file_path):
        '''Loads a numpy array from a .npy file. Calvin formats these as (subjects, voxels), which are the entire dataframe.'''
        data = np.load(file_path)
        data = data.T  # Transpose to (voxels, observations)
        column_names = [f"sub_{i}" for i in range(data.shape[1])]  # Generate column names
        return pd.DataFrame(data, columns=column_names)  # Assign to DataFrame with column names

    def import_nifti_to_numpy_array(self, file_path):
        '''
        Does what it says. Just provide the absolute filepath.
        Args:
            filepath: absolute path to the file to import
        Returns:
            nifti_data: nifti_data as a numpy array
        '''
        try:
            nifti_img = image.load_img(file_path)
            # print(nifti_img.affine)
            self.affines = tuple(nifti_img.affine.flatten())
            nifti_data = nifti_img.get_fdata().flatten()
            return nifti_data
        except Exception as e:
            print("Error:", e)
            return None

    def import_gifti_to_numpy_array(self, file_path):
        """Imports a GIFTI file and converts it to a NumPy array."""
        try:
            gifti_img = nib.load(file_path)
            gifti_data = gifti_img.darrays[0].data.flatten()
            return gifti_data
        except Exception as e:
            print("Error:", e)
            return None
        
    def identify_file_type(self, file_path: str) -> str:
        """Identifies file extension"""
        try:
            p = Path(file_path)
            ext = ''.join(p.suffixes).lower()
            if any(e in ext for e in ['.nii', '.nii.gz']):
                return 'nii'
            elif any(e in ext for e in ['.gii', '.gii.gz']):
                return 'gii'
            elif any(e in ext for e in ['.npy']):
                return 'npy'
            else:
                return ext
        except Exception:
            return 'unrecognized'

    def align_imported_matrices(self, file_paths):
        '''Using the affines of the imported matrices, we will create a bounding box that encompasses all the matrices and place them in it.'''
        self.bbox = NiftiBoundingBox(file_paths)
        self.bbox.generate_bounding_box()
        self.bbox.add_niftis_to_bounding_box()
        self.bbox.collapse_bbox_to_3d()
        self.bbox_mask = self.bbox.collapsed_bbox_to_mask()
        self.bbox_4d = self.bbox._stacked_data
    
    def import_matrices(self, file_paths, duplicate_cols=True):
        '''
        Given a list of file paths, import the data and return a dataframe with the imported files, flattened such that each column is a file.
        
        Params
        duplicate_cols (bool): if true, allows duplicate columns. Will change their names to allow this. 
        '''
        for file_path in tqdm(file_paths, desc='Importing niftis'):
            path = self.identify_file_type(file_path)
            if path == 'nii':
                data = self.import_nifti_to_numpy_array(file_path)
            elif path == 'gii':
                data = self.import_gifti_to_numpy_array(file_path)
            elif path == 'npy':
                data = self.import_npy_to_numpy_array(file_path)
                
            elif path == 'unrecognized':    
                continue # Skip unrecognized files
            else:
                raise RuntimeError(f"Failed to import file: {file_path}. Error: path type {path} is not yet implemented")

            if path == 'npy':                   # The data is a whole dataframe in the numpy version, by virtue of standard NPY packaging with Calvin Howard's code.
                self.matrix_df = data
            else:                  # Assign to the matrix in (voxels, observations) form for when voxels are observations. 
                if file_path in self.matrix_df.columns:
                    id = random.randint(1, 1000000)
                    name = file_path + str(id)
                    print("WARNING: DUPLICATE NIFTIS ARE BEING IMPORTED INTO YOUR DATA. THIS CAN SEVERELY CONFOUND ANALYSES IF UNINTENTIONAL. \n\tCAUSE: ", file_path)
                else:
                    name = file_path
                self.matrix_df[name] = data


        if len(self.affines) > 1:
            print("Warning: Multiple affines detected. Aligning to common space. Mask is available as self.bbox_mask and 4d data as self.bbox_4d.")
            self.matrix_df = pd.DataFrame({})               # Reset the matrix_df to empty
            self.align_imported_matrices(file_paths)
            for i, file_path in tqdm(enumerate(file_paths), desc='Aligning misaligned files'):
                data = self.bbox_4d[:, :, :, i]
                self.matrix_df[file_path] = data.flatten()
                
        for file_path, data in self.matrix_df.items():
            if self.process_special_values:
                try:
                    data = self.handle_special_values(data)
                except Exception as e:
                    data = 0
            new_name = self.generate_name(file_path)
            self.matrix_df[new_name] = data

        # drop the original file paths so we only have the processed filenames
        self.matrix_df = self.matrix_df.drop(columns=[file_path for file_path in self.matrix_df.keys() if file_path not in self.matrix_df.columns])
        return self.matrix_df

    def import_from_csv(self):
        print(f'Attempting to import from: {os.path.basename(self.import_path)}')
        if self.import_path is None:
            raise ValueError ("Argument file_column is None. Please specify file_column='column_storing_file_paths.")
        self.paths = pd.read_csv(self.import_path)[self.file_column].tolist()
        return self.import_matrices(self.paths)

    def import_from_folder(self):
        print(f'Attempting to import from: {self.import_path}/{self.file_pattern}')
        if self.file_pattern == '':
            raise ValueError ("Argument file_pattern is empty. Please specify file_pattern='*my_file*patter.nii.gz'")
        glob_path = os.path.join(self.import_path, self.file_pattern)
        file_paths = glob(glob_path)
        self.file_paths = file_paths
        return self.import_matrices(file_paths)
    
    def import_from_series(self):
        print('Attempting to import from pandas series. Will fail unless a series is provided using df["path_column"]')
        file_paths = list(self.import_path.to_numpy())
        return self.import_matrices(file_paths)

    def detect_input_type(self):
        """
        Detects whether the input_path is a CSV file or a folder.

        Parameters:
        -----------
        input_path : str
            The input path to be checked.

        Returns:
        --------
        str
            'csv' if the input is a CSV file, 'folder' if it's a folder, or 'unsupported' if neither.
        """
        if isinstance(self.import_path, pd.Series):
            self.import_type = 'pd_series'
        elif isinstance(self.import_path, (str, Path)) and str(self.import_path).lower().endswith('.csv'):
            self.import_type = 'csv'
        elif isinstance(self.import_path, (str, Path)):
            self.import_type = 'folder'
        else:
            raise ValueError(f"Unrecognized import_path type: {type(self.import_path)}")
    
    def import_data_based_on_type(self):
        self.detect_input_type()
        if self.import_type == 'pd_series':
            return self.import_from_series()
        elif self.import_type == 'csv':
            # Input is a CSV file
            return self.import_from_csv()
        elif self.import_type == 'folder':
            # Input is a folder
            return self.import_from_folder()
        else:
            raise ValueError("Invalid input type")
        
    @staticmethod
    def save_files(dataframe, file_paths, dry_run=True, file_suffix=None):
        """
        Convenience saving function. Allows saving files after acting upon them. 
        """
        for i, file_path in tqdm(enumerate(file_paths), desc='Saving files'):
            out_dir = os.path.dirname(file_path)
            nifti_name = os.path.splitext(os.path.basename(file_path))[0] + (file_suffix if file_suffix is not None else '')
            if dry_run:
                print(f"Saving to: {os.path.join(out_dir, nifti_name)}")
            else:
                view_and_save_nifti(dataframe.iloc[:, i], out_dir=out_dir, output_name=nifti_name, silent=True)
    
    @staticmethod
    def mask_dataframe(df: pd.DataFrame, mask_path: str=None, threshold: float=0):
        """
        Simple masking function.
        """
        if mask_path is None:
            mask = np.ones(df.shape[0], dtype=bool)
            mask_indices = np.arange(df.shape[0])
            masked_df = df
        else:
            mask = nib.load(mask_path)
            mask = mask.get_fdata().flatten()
            mask_indices = mask > threshold
            masked_df = df.loc[mask_indices, :]
        return mask, mask_indices, masked_df
    
    @staticmethod
    def unmask_dataframe(df: pd.DataFrame, mask_path: str=None, threshold: float=0):
        """
        Simple unmasking function.
        """
        if mask_path is None:
            unmasked_df = df
        else:
            mask = nib.load(mask_path)
            mask = mask.get_fdata().flatten()
            mask_indices = mask > threshold
            unmasked_df = pd.DataFrame(index=mask, columns=df.columns, data=0)
            unmasked_df.iloc[mask_indices, :] = df
        return unmasked_df
    
    @staticmethod
    def mask_array(arr: np.array, mask_path: str=None, threshold: float=0):
        """
        Simple masking function.
        """
        if mask_path is None:
            mask = np.ones(df.shape[0], dtype=bool)
            mask_indices = np.arange(df.shape[0])
            masked_arr = arr
        else:
            mask = nib.load(mask_path)
            mask = mask.get_fdata().flatten()
            mask_indices = mask > threshold
            masked_arr = arr[mask_indices]
        return mask, mask_indices, masked_arr
    
    @staticmethod
    def splice_colnames(df, pre, post):
        raw_names = df.columns
        name_mapping = {}
        # For each column name in the dataframe
        for name in raw_names:
            new_name = name  # Default to the original name in case it doesn't match any split command
            if pre != '':
                new_name = new_name.split(pre)[1]
            if post != '':
                new_name = new_name.split(post)[0]
            
            # Add the original and new name to the mapping
            name_mapping[name] = new_name
        return df.rename(columns=name_mapping)
    
    def return_function(self):
        if not self.transpose:
            return self.matrix_df
        return self.matrix_df.transpose(copy=False)
    
    def run(self):
        self.import_data_based_on_type()
        df = self.return_function()
        return df

class ImportDatasetsToDict(GiiNiiFileImport):
    def __init__(self, df, dataset_col, nifti_col, indep_var_col, covariate_cols):
        self.df = df
        self.dataset_col = dataset_col
        self.nifti_col = nifti_col
        self.indep_var_col = indep_var_col
        self.covariate_cols = covariate_cols
        self.data_dict = {}
        self._prepare_data_dict()

    def _prepare_data_dict(self):
        # Iterate over each unique dataset
        for dataset in self.df[self.dataset_col].unique():
            print("Importing dataset: ", dataset)
            dataset_df = self.df[self.df[self.dataset_col] == dataset]
            dataset_df.columns = self.df.columns

            # Initialize sub-dictionary for the dataset
            self.data_dict[dataset] = {
                'niftis': pd.DataFrame(),
                'indep_var': pd.DataFrame(),
                'covariates': pd.DataFrame()
            }

            # Extract NIFTI file paths and import them using GiiNiiFileImport
            nifti_paths = dataset_df[self.nifti_col].tolist()
            nifti_importer = GiiNiiFileImport(import_path=None, file_column=None, file_pattern=None)
            self.data_dict[dataset]['niftis'] = nifti_importer.import_matrices(nifti_paths)

            # Extract independent variable and covariates
            self.data_dict[dataset]['indep_var'] = dataset_df.loc[:, [self.indep_var_col]]
            self.data_dict[dataset]['covariates'] = dataset_df.loc[:, self.covariate_cols]
