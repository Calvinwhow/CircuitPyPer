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
from calvin_utils.neuroimaging_utils.ccm_utils.bounding_box import NiftiBoundingBox
from calvin_utils.neuroimaging_utils.nifti_utils.generate_nifti import view_and_save_nifti
from pathlib import Path
from tqdm import tqdm
import random
from uuid import uuid4
from calvin_utils.neuroimaging_utils.surface_utils.surface_io import SurfaceIO
from calvin_utils.neuroimaging_utils.nifti_utils.volume_io import NiftiIO, VolumetricTimeSeriesIO
from calvin_utils.neuroimaging_utils.tract_utils.fiber_io import FiberIO


class GiiNiiFileImport:
    """
    Import NIFTI, GIFTI, or NPY files into a pandas DataFrame.

    Supports three input styles:
    1) A folder path plus a glob `file_pattern`
    2) A CSV path plus a `file_column` that contains file paths
    3) A pandas Series of file paths

    Column naming can be customized via regex extraction or string splicing.

    Parameters:
    -----------
    import_path : str
        Path to a folder or CSV file, or a pandas Series of file paths.
    file_column : str, optional
        Column in the CSV that contains file paths (required when `import_path` is a CSV).
    file_pattern : str, optional
        Glob pattern for files within a folder (e.g., '*.nii.gz').
    pre_splice : str, optional
        String used to split file paths for column name generation.
    post_splice : str, optional
        String used to split file paths for column name generation.
    subject_pattern : str, optional
        Regex pattern used to extract subject IDs from file paths.
    process_special_values : bool, optional
        Whether to replace NaNs/Infs with finite values (NaN->0, +Inf->max, -Inf->min).
    transpose : bool, optional
        If True, returns a transposed DataFrame (files as rows).

    Attributes:
    -----------
    seen_names : set
        A set to track unique column names.
    """
    def __init__(self, import_path, file_column: str=None, file_pattern: str=None, pre_splice='sub', post_splice="", subject_pattern=None, process_special_values=True, transpose=False, mask_path='default'):
        self.import_path = import_path
        self.file_pattern = file_pattern
        self.file_column = file_column
        self.transpose = transpose
        self.process_special_values = process_special_values
        self.seen_names = set()
        self.pre_splice  = pre_splice
        self.post_splice = post_splice
        self.subject_pattern = re.compile(subject_pattern) if subject_pattern is not None else None
        self.mask_path = mask_path
        self.output_ftype = "nifti"
        
    ### Column Name Generation ###
    def _generate_unique_column_name(self, file_path: str) -> str:
        base_name = os.path.basename(file_path)
        if base_name in self.seen_names:
            name = file_path
        else:
            name = base_name
            self.seen_names.add(base_name)
        return name
    
    def _match_id(self, file_paths):
        """Use regex to extract subject ID"""
        names_list = []
        for file_path in file_paths:
            match = self.subject_pattern.search(file_path)
            if match:
                start_index = match.start()
                subject_id = file_path[start_index:]
                names_list.append(subject_id)
            else:
                names_list.append(None)
        return names_list
    
    def _splice_colnames(self, file_list):
        """Generates column names by string splicing"""
        names_list = []
        for name in file_list:
            new_name = name
            if self.pre_splice not in (None, ""):
                if self.pre_splice in new_name:
                    new_name = new_name.split(self.pre_splice, 1)[1]
            if self.post_splice not in (None, ""):
                if self.post_splice in new_name:
                    new_name = new_name.split(self.post_splice, 1)[0]
            names_list.append(new_name)
        return names_list
    
    @staticmethod
    def splice_colnames(df, pre, post):
        names_list = []
        for name in df.columns:
            new_name = name
            if pre not in (None, ""):
                if pre in new_name:
                    new_name = new_name.split(pre, 1)[1]
            if post not in (None, ""):
                if post in new_name:
                    new_name = new_name.split(post, 1)[0]
            names_list.append(new_name)
        df = df.rename(columns=dict(zip(df.columns, names_list)))
        return df
    
    def _coerce_duplicate_names(self, names_list):
        for idx, name in enumerate(names_list):
            if names_list.count(name) > 1:
                names_list[idx] = f"{name}_{uuid4()}"
                print("WARNING: DUPLICATE FILES ARE BEING IMPORTED INTO YOUR DATA. THIS CAN SEVERELY CONFOUND ANALYSES IF UNINTENTIONAL.\n\tCAUSE:", name)
        return names_list
    
    def _generate_colnames(self, file_paths):
        """Switching function for different ways of getting colnames from filenames"""
        ftype = self._import_type_switch(file_paths)

        if ftype == 'npy':
            data = np.load(file_paths[0]).T
            names_list = [f"sub_{i}" for i in range(data.shape[1])]
        elif self.subject_pattern is not None:
            names_list = self._match_id(file_paths)
        elif (self.pre_splice is not None) or (self.post_splice is not None):
            names_list = self._splice_colnames(file_paths)
        else:
            names_list = [self._generate_unique_column_name(file_path) for file_path in file_paths]

        names_list = self._coerce_duplicate_names(names_list)
        return names_list
    
    def _handle_special_values(self, arr):
        """Handles NaNs and infinities in the data without significantly biasing the distribution."""
        max_val = np.nanmax(arr)
        min_val = np.nanmin(arr)
        arr = np.nan_to_num(arr, nan=0.0, posinf=max_val, neginf=min_val)
        return arr

    ### Dataframe Creation Methods ###
    def _initialize_dataframe(self, arr, file_paths):
        cols = self._generate_colnames(file_paths)
        df = pd.DataFrame(data=arr, columns=cols)
        return df
    
    def _transpose_dataframe(self, df, transpose):
        if transpose:
            return df.transpose(copy=False)
        return df
    
    def prepare_dataframe(self, arr, file_paths):
        """Orchestrator for dataframe quality control"""
        if self.process_special_values:
            arr = self._handle_special_values(arr)
        df  = self._initialize_dataframe(arr, file_paths)
        df  = self._transpose_dataframe(df, self.transpose)
        return df
    
    ### Import Functions ###
    def _identify_file_type(self, path):
        """Detects file type"""
        p = Path(path)
        suffixes = [s.lower() for s in p.suffixes]
        basename = p.name.lower()
        
        if suffixes[-2:] == ['.nii', '.gz'] or suffixes[-1:] == ['.nii']:
            img = nib.load(str(path))
            shape = img.shape
            if len(shape) == 4 and shape[3] > 1:
                return 'nii_timeseries'
            return 'nii'
        if suffixes[-2:] == ['.gii', '.gz'] or suffixes[-1:] == ['.gii']:
            return 'gii'
        if suffixes[-2:] == ['.fib', '.npy']:
            return 'fiber'
        if suffixes[-1:] == ['.npy']:
            return 'npy'
        freesurfer_suffixes = {
            '.thickness',
            '.pial',
            '.volume',
            '.white',
            '.curv',
            '.sulc',
            '.sphere',
        }
        if any(basename.endswith(s) for s in freesurfer_suffixes):
            return 'freesurfer'
        return 'unknown'

    def _import_type_switch(self, file_paths):
        """Chooses import type to return"""
        path_types = {self._identify_file_type(path) for path in file_paths}
        if len(path_types) != 1:
            raise ValueError(f"Multiple file types detected: {path_types}")
        return next(iter(path_types))

    def _import_matrices(self, file_paths):
        """Orchestrates import"""
        self.output_ftype = self._import_type_switch(file_paths)
        if self.output_ftype == "npy":
            arr = self.import_npy_to_numpy_array(file_paths)
        elif self.output_ftype == "nii":
            arr = self.import_nifti_to_numpy_array(file_paths)
        elif self.output_ftype == "nii_timeseries":
            arr = self.import_nifti_timeseries_to_numpy_array(file_paths)
        elif self.output_ftype in {"gii", "freesurfer"}:
            arr = self.import_surface_to_numpy_array(file_paths)
        elif self.output_ftype == "fiber":
            arr = self.import_fiber_to_numpy_array(file_paths)
        else:
            raise RuntimeError(f"Unknown file type ({self.output_ftype}) imported. Did not detect nifti, gifti, fiber, or npy. Please provide one of these files for import.")
        return arr

    def import_nifti_timeseries_to_numpy_array(self, file_paths):
        """Loads 4D volumetric timeseries niftis"""
        self.timeseries_io = VolumetricTimeSeriesIO(mask_path=self.mask_path)
        data = self.timeseries_io.import_nifti_to_numpy_array(file_paths)
        return data

    def import_npy_to_numpy_array(self, file_paths):
        '''Loads a numpy array from a single .npy file. Calvin formats these as (subjects, voxels), which are the entire dataframe.'''
        if len(file_paths) != 1:
            raise ValueError(f"NPY import expects a single file, but received {len(file_paths)}.")
        data = np.load(file_paths[0])
        return data.T  # Transpose to (voxels, observations)

    def import_nifti_to_numpy_array(self, file_path):
        """Loads niftis"""
        self.nifti_io = NiftiIO(mask_path=self.mask_path)
        data = self.nifti_io.import_nifti_to_numpy_array(file_path)
        return data

    def import_surface_to_numpy_array(self, file_paths):
        """Imports surface files (GIFTI/FreeSurfer) and converts them to a NumPy array."""
        mask_path = None if self.mask_path == 'default' else self.mask_path
        self.surface_io = SurfaceIO(mask_path=mask_path)
        arr = self.surface_io.import_surface_to_numpy_array(file_paths)
        return arr
    
    def import_fiber_to_numpy_array(self, file_paths):
        mask_path = None if self.mask_path == 'default' else self.mask_path
        self.fiber_io = FiberIO(mask_path=mask_path)
        arr = self.fiber_io.import_fiber_to_numpy_array(file_paths)
        return arr
        
    def import_from_csv(self):
        print("Attempting to import from CSV. Expecting CSV full of filenames.")
        print(f'Attempting to import from: {os.path.basename(self.import_path)}')
        if self.file_column is None:
            raise ValueError ("Argument file_column is None. Please specify file_column='column_storing_file_paths.")
        file_paths = pd.read_csv(self.import_path)[self.file_column].tolist()
        return self._import_matrices(file_paths), file_paths

    def import_from_folder(self):
        print("Attempting to import from globbable path. Expecting globbable filenames.")
        print(f'Attempting to import from: {self.import_path}/{self.file_pattern}')
        if self.file_pattern in (None, ''):
            raise ValueError ("Argument file_pattern is empty. Please specify file_pattern='*my_file*patter.nii.gz'")
        glob_path = os.path.join(self.import_path, self.file_pattern)
        file_paths = glob(glob_path)
        self.file_paths = file_paths
        return self._import_matrices(file_paths), file_paths
    
    def import_from_series(self):
        print('Attempting to import from pandas series. expecting PD series full of filenames. Will fail unless a series is provided using df["path_column"]')
        file_paths = list(self.import_path.to_numpy())
        return self._import_matrices(file_paths), file_paths

    def detect_input_type(self):
        """Detects whether the input_path is a Pandas DataFrame, a CSV, or a Folder."""
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
            return self.import_from_csv()
        elif self.import_type == 'folder':
            return self.import_from_folder()
        else:
            raise ValueError("Invalid input type")
    
    ### Public API ###
    def run(self):
        arr, file_paths = self.import_data_based_on_type()
        df = self.prepare_dataframe(arr, file_paths)
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
            self.data_dict[dataset]['niftis'] = nifti_importer._import_matrices(nifti_paths)

            # Extract independent variable and covariates
            self.data_dict[dataset]['indep_var'] = dataset_df.loc[:, [self.indep_var_col]]
            self.data_dict[dataset]['covariates'] = dataset_df.loc[:, self.covariate_cols]
