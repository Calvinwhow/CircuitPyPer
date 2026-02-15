import os
import json
import glob
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorly as tl
import statsmodels.api as sm
from tensorly.regression.cp_regression import CPRegressor
from calvin_utils.file_utils.import_functions import GiiNiiFileImport

class DatasetNiftiImporter(GiiNiiFileImport):
    def __init__(self, df, dataset_col, nifti_col, indep_var_col, covariate_cols, out_dir, mask_path=None, regression_method='tensor', data_transform_method='standardize'):
        """
        Initializes the utility class for handling NIfTI data processing and regression analysis.
        This class is designed to facilitate the processing of neuroimaging data stored in NIfTI format.
        It prepares the data for regression analysis, applies transformations, and organizes the data
        into a structured format for further processing. The class supports multiple regression methods
        and data transformation techniques, making it flexible for various use cases in neuroimaging
        research.
        Args:
            df (pd.DataFrame): A pandas DataFrame containing metadata and file paths for the NIfTI files.
            dataset_col (str): The name of the column in `df` that specifies the dataset identifier.
            nifti_col (str): The name of the column in `df` that contains the file paths to the NIfTI files.
            indep_var_col (str): The name of the column in `df` that specifies the independent variable for regression.
            covariate_cols (list of str): A list of column names in `df` that specify covariates to include in the regression.
            out_dir (str): The output directory where processed data and results will be saved.
            mask_path (str, optional): The file path to a NIfTI mask file. If provided, the mask will be applied to the data.
                                       Defaults to None.
            regression_method (str, optional): The regression method to use. Options include 'tensor', 'ols' or None. 
            data_transform_method (str, optional): The method for transforming the data. Options include 'standardize' and others.
                                                   Defaults to 'standardize'.
        Attributes:
            data_dict (dict): A dictionary to store processed data for each dataset.
            regression_method (str): The regression method specified during initialization.
            data_transform_method (str): The data transformation method specified during initialization.
        Methods:
            _prepare_data_dict(): Prepares the internal data dictionary for storing processed data.
            _create_dataset_dict(): Organizes the data into a structured format for each dataset.
        Notes:
            This class is part of a larger utility module designed for neuroimaging data analysis.
            It assumes that the input DataFrame (`df`) is properly formatted and contains all necessary
            columns specified in the arguments.
        """
        
        self.df = df
        self.dataset_col = dataset_col
        self.nifti_col = nifti_col
        self.indep_var_col = indep_var_col
        self.covariate_cols = covariate_cols
        self.out_dir = out_dir
        self.mask_path = mask_path
        self.data_dict = {}
        self.regression_method = regression_method
        self.data_transform_method = data_transform_method
    
    def _prep_dataset_dir(self, dataset):
        dataset_dir = os.path.join(self.out_dir, 'tmp', dataset)
        os.makedirs(dataset_dir, exist_ok=True)
        return dataset_dir

    def _prepare_data_dict(self):
        if self.dataset_col not in self.df.columns or self.df[self.dataset_col].empty:
            self.df[self.dataset_col] = 'dataset'
        
        for dataset in self.df[self.dataset_col].unique():
            print("Importing dataset: ", dataset)
            dataset_df = self.df[self.df[self.dataset_col] == dataset]
            dataset_df.columns = self.df.columns
            dataset_dir = self._prep_dataset_dir(dataset)

            self.data_dict[dataset] = {
                'niftis': pd.DataFrame(),
                'indep_var': pd.DataFrame(),
                'covariates': pd.DataFrame()
            }

            nifti_paths = dataset_df[self.nifti_col].tolist()
            nifti_importer = GiiNiiFileImport(import_path=None, file_column=None, file_pattern=None)
            self.data_dict[dataset]['niftis'] = nifti_importer.import_matrices(nifti_paths)
            self.data_dict[dataset]['indep_var'] = self.prep_dmatrix(dataset_df.loc[:, [self.indep_var_col]])
            self.data_dict[dataset]['covariates'] = self.prep_dmatrix(dataset_df.loc[:, self.covariate_cols])
        
            if hasattr(nifti_importer, 'bbox_mask') and (nifti_importer.bbox_mask is not None) and (self.mask_path is None):
                print("----- \n MASK NOT FOUND NOR PROVIDED. GENERATING MASK AT BELOW PATH. RERUN THIS NOTEBOOK USING THE MASK BELOW: \m")
                print("\t Use this mask for all subsequent analyses: ", os.path.join(self.out_dir, 'mask.nii.gz'))
                nifti_importer.bbox.save_nifti(nifti_importer.bbox_mask, os.path.join(self.out_dir, 'mask.nii.gz'))
            self._process_dataset(dataset, dataset_dir)
            
    def prep_dmatrix(self, df):
        """
        Converts columns in the DataFrame to numeric. If conversion fails, converts them to categorical.

        Parameters:
        -----------
        df : pandas.DataFrame
            The DataFrame whose columns need to be converted.

        Returns:
        --------
        pandas.DataFrame
            The DataFrame with columns converted to numeric or categorical.
        """
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                df[col] = df[col].astype('category').cat.codes
        return df
    
    def _add_intercept(self, covariates_arr):
        intercept = np.ones((covariates_arr.shape[0], 1)) # shape (samples, 1)
        return np.hstack((intercept, covariates_arr)) # shape (samples, N_covariates + 1)
    
    def _tensor_regression(self, endog, exog):
        # Ensure endog and exog are tensors
        endog = tl.tensor(endog)
        exog = tl.tensor(exog)

        # Perform tensor regression using CP decomposition
        model = CPRegressor(weight_rank=1)
        model.fit(exog, endog)
        residuals = endog - model.predict(exog)
        return residuals
        
    def _standardize(self, arr):
        """Standardize the input array by subtracting the mean and dividing by the standard deviation."""
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        std_arr = (arr - mean) / std
        return std_arr
    
    def _run_transform(self, niftis_arr, indep_var_arr, covariates_arr):
        """
        Apply a specified transformation to the input arrays.
    
        Parameters:
        -----------
        niftis_arr : np.array
            Array of NIfTI data (samples, voxels).
        indep_var_arr : np.array
            Array of independent variables (samples, 1).
        covariates_arr : np.array
            Array of covariates (samples, N_covariates).
    
        Returns:
        --------
        std_niftis_arr : np.array
            Transformed NIfTI data.
        std_indep_var_arr : np.array
            Transformed independent variables.
        std_covariates_arr : np.array
            Transformed covariates.
        """
        if self.data_transform_method == 'standardize':
            niftis_arr = self._standardize(niftis_arr)
            indep_var_arr = self._standardize(indep_var_arr)
            covariates_arr = self._standardize(covariates_arr)
        elif self.data_transform_method is None: 
            pass
        else:
            raise ValueError("Invalid transformation method. Choose 'standardize' or None.")

        return niftis_arr, indep_var_arr, covariates_arr
    
    def _handle_nans(self, *arrs, value=0):
        "handles NaNs, setting posinf and neginf to max in-array values."
        p_arrs = []
        for arr in arrs:
            max_val = np.nanmax(arr)
            min_val = np.nanmin(arr)
            arr = np.nan_to_num(arr, nan=value, posinf=max_val, neginf=min_val)
            if np.isnan(arr).any():
                raise ValueError(f"NaN persists in array {arr}.")
            p_arrs.append(arr)
        return tuple(p_arrs)
    
    def _ols_regression(self, endog, exog):
        residuals = np.zeros_like(endog)
        for i in tqdm(range(endog.shape[1]), desc='OLS Regression'):
            model = sm.OLS(endog[:, i], exog).fit()
            residuals[:, i] = model.resid
        return residuals
    
    def _run_regression(self, masked_niftis_array, indep_var_array, covariates_array):
        print(masked_niftis_array.shape, indep_var_array.shape, covariates_array.shape)
        if self.regression_method == 'tensor':
            # Perform tensor regression
            nifti_residuals = self._tensor_regression(masked_niftis_array, covariates_array)
            indep_residuals = self._tensor_regression(indep_var_array, covariates_array)
        elif self.regression_method == 'ols':
            # Perform OLS regression
            nifti_residuals = self._ols_regression(masked_niftis_array, covariates_array)
            indep_residuals = self._ols_regression(indep_var_array, covariates_array)
        elif self.regression_method == None:
            # Return as-is
            nifti_residuals = masked_niftis_array
            indep_residuals = indep_var_array
        else:
            raise ValueError("Invalid regression method. Choose 'tensor', 'ols', or None.")
        return nifti_residuals, indep_residuals
    
    def _save_residuals(self, dataset_dir, nifti_residuals_arr, indep_residuals_arr, covariates_arr):
        for fname, arr in zip(['niftis.npy', 'indep_var.npy', 'covariates.npy'],
                      [nifti_residuals_arr, indep_residuals_arr, covariates_arr]):
            fpath = os.path.join(dataset_dir, fname)
            if os.path.exists(fpath):
                os.remove(fpath)
            np.save(fpath, arr)
        
    def _process_dataset(self, dataset, dataset_dir):
        niftis = self.data_dict[dataset]['niftis'] # Get the nifti
        _, _, masked_niftis = self.mask_dataframe(niftis, mask_path=self.mask_path) # mask the nifti 
        masked_niftis_arr = masked_niftis.values.T                          # shape (samples, voxels)
        
        indep_var_arr = self.data_dict[dataset]['indep_var'].values         # shape (samples, 1)
        covariates_arr = self.data_dict[dataset]['covariates'].values       # shape (samples, N_covariates)
        covariates_arr = self._add_intercept(covariates_arr)                # shape (samples, N_covariates+1)
        
        masked_niftis_arr, indep_var_arr, covariates_arr = self._handle_nans(masked_niftis_arr, indep_var_arr, covariates_arr) 
        masked_niftis_arr, indep_var_arr, covariates_arr = self._run_transform(masked_niftis_arr, indep_var_arr, covariates_arr) 
        nifti_residuals, indep_residuals = self._run_regression(masked_niftis_arr, indep_var_arr, covariates_arr)

        self._save_residuals(dataset_dir, nifti_residuals, indep_residuals, covariates_arr)

    def cleanup(self):
        tmp_dir = os.path.join(self.out_dir, 'tmp')
        shutil.rmtree(tmp_dir)
        
    def _create_dataset_dict(self):
        dataset_dict = {}
        for dataset in self.data_dict.keys():
            dataset_dir = self._prep_dataset_dir(dataset)
            dataset_dict[dataset] = {
                'niftis': os.path.join(dataset_dir, 'niftis.npy'),
                'indep_var': os.path.join(dataset_dir, 'indep_var.npy'),
                'covariates': os.path.join(dataset_dir, 'covariates.npy')
            }
        dict_path = os.path.join(self.out_dir, 'tmp', 'dataset_dict.json')
        with open(dict_path, 'w') as f:
            json.dump(dataset_dict, f)
        print(f"Dataset dictionary saved to: {dict_path}")
        return dataset_dict, dict_path
    
    @staticmethod
    def import_niftis_to_dict(mask_path, nifti_glob):
        """
        Discover NIfTI files via a glob pattern, apply the current mask (self.mask_path),
        and return a dictionary where:
        - keys = basenames of each file
        - values = masked data arrays, shape (1, number_of_voxels)
        """
        out_dict = {}
        nifti_list = sorted(glob.glob(nifti_glob))
        if not nifti_list:
            raise ValueError(f"No files found for glob pattern: {nifti_glob}")
        print(f"Found {len(nifti_list)} NIfTI files.")

        for file_path in nifti_list:
            importer = GiiNiiFileImport(file_path, subject_pattern='', process_special_values=False)
            nifti_arr = importer.import_nifti_to_numpy_array(file_path)
            _, _, masked_arr = GiiNiiFileImport.mask_array(nifti_arr, mask_path=mask_path)
            if masked_arr.ndim != 2 and masked_arr.shape[0] != 1:
                masked_arr = masked_arr.reshape(1, -1)
            out_dict[os.path.basename(file_path)] = masked_arr
        return out_dict
    
    ### Public API ###
    def package_data(self):
        self._prepare_data_dict()
        _, dict_path = self._create_dataset_dict()
        return dict_path
