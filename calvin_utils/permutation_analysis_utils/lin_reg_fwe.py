import numpy as np
import pandas as pd
from tqdm import tqdm
import nibabel as nib
from typing import Tuple
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from calvin_utils.neuroimaging_utils.nifti_utils.generate_nifti import view_and_save_nifti

class CalvinFWEMap():
    """
    This is a class to orchestrate a simple association between some Y variable of interest and voxelwise data (X variable).
    It will run FWE correction via the Maximum Statistic Correction method.

    Notes:
    ------
    - Running max_stat_method = pseudo_var_smooth will reduce the risk of the maximum stat
        being a numerically unstable result of a bad permutation.
    - Linear regression is implemented using sklearn's LinearRegression, allowing for efficient 
        and vectorized computations across large datasets.
    - Empiric and theoretical evaluation has revealed:
        - R2(voxel ~ outcome) == R2(outcome ~ voxel)
            - Spatial correlation of 100%
        - T_outcome(voxel ~ outcome) != T_voxel(outcome ~ voxel)
            - Spatial correlation ~49%
        - This can be leveraged, to test the overall fit with vectorization of voxel ~ outcome
           and then using a single evaluation of T_voxel(outcome ~ voxel) to get the relationship.
           We take advantage of this trick to permute the overall R2 for significance testing
           Then, the individual regressors can be calculated properly on a voxelwise manner and masked by the
           regions with a significant fit. 
    
    Attributes:
    -----------
    neuroimaging_dataframe : pd.DataFrame
        DataFrame with neuroimaging data where each column represents a subject and each row represents a voxel.
    variable_dataframe : pd.DataFrame
        DataFrame where each column represents a subject and each row represents the variable to regress upon.
    mask_path : str or None
        The path to the mask to use. If None, will threshold the voxelwise image itself by mask_threshold.
    mask_threshold : float
        The threshold to mask the neuroimaging data at.
    out_dir : str
        Output directory to save results.
    max_stat_method : str or None
        Method for maximum statistic correction. Options: None | 'pseudo_var_smooth' | 'var_smooth'.
    vectorize : bool
        Whether to use vectorized implementation for correlation calculation.

    Methods:
    --------
    sort_dataframes(voxel_df: pd.DataFrame, covariate_df: pd.DataFrame, debug: bool=False) -> Tuple[pd.DataFrame, pd.DataFrame]
        Sorts the rows of the voxelwise and covariate DataFrames to ensure they are identically organized.
    
    threshold_probabilities(df: pd.DataFrame, debug: bool=False) -> pd.Series
        Applies a threshold to mask raw voxelwise data.
    
    mask_dataframe(neuroimaging_df: pd.DataFrame) -> Tuple[pd.Index, pd.Series, pd.DataFrame]
        Applies a mask to the neuroimaging DataFrame based on nonzero voxels.
    
    unmask_dataframe(df: pd.DataFrame) -> pd.DataFrame
        Simple unmasking function to restore original dimensions.
    
    mask_by_p_values(results_df: pd.DataFrame, p_values_df: pd.DataFrame) -> pd.DataFrame
        Thresholds results by FWE corrected p-values.
    
    permute_covariates() -> pd.DataFrame
        Permutes the patient data by randomly assigning patient data to new patients.
    
    run_lin_reg(X: np.ndarray, Y: np.ndarray, use_intercept: bool=True, debug: bool=False) -> pd.DataFrame
        Calculates voxelwise relationship to Y variable with linear regression using sklearn's LinearRegression.
    
    orchestrate_linear_regression(permuted_variable_df: pd.DataFrame=None, debug: bool=False) -> pd.DataFrame
        Orchestrates the linear regression analysis for voxelwise data.
    
    var_smooth(df: pd.DataFrame)
        Takes the 95th percentile of the permuted data as the 'maximum stat' as a proxy for variance smoothed max stat.
    
    pseudo_var_smooth(df: pd.DataFrame) -> np.ndarray
        Takes the 99th percentile of the permuted data as the 'maximum stat' as a proxy for variance smoothed max stat.
    
    raw_max_stat(df: pd.DataFrame) -> float
        Returns the max statistic in the data.
    
    get_max_stat(df: pd.DataFrame) -> float
        Chooses the max stat method and returns the max stat.
    
    maximum_stat_fwe(n_permutations: int=100, debug: bool=False) -> list
        Performs maximum statistic Family-Wise Error (FWE) correction using permutation testing.
    
    p_value_calculation(uncorrected_df: pd.DataFrame, max_stat_dist: list, debug: bool=False) -> Tuple[pd.DataFrame, pd.DataFrame]
        Calculates p-values for the uncorrected statistic values using the distribution of maximum statistics.
    
    save_single_nifti(nifti_df: pd.DataFrame, out_dir: str, name: str='generated_nifti', silent: bool=True)
        Saves NIFTI images to the specified directory.
    
    save_results(voxelwise_results: pd.DataFrame, unmasked_p_values: pd.DataFrame, voxelwise_results_fwe: pd.DataFrame)
        Saves the generated result files.
    
    run(n_permutations: int=100, debug: bool=False)
        Orchestration method to run the entire analysis and save the results.
    """
    def __init__(self, neuroimaging_dataframe: pd.DataFrame, variable_dataframe: pd.DataFrame, outcome_row: str = None, mask_path=None, mask_threshold: int=0.0, out_dir='', max_stat_method=None, vectorize=True):
        """
        Need to provide the dataframe dictionaries and dataframes of importance. 
        
        Args:
        - neuroimaging_dataframe (df): DF with neuroimaging data (voxelwise dataframe) column represents represents a subject,
                                        and each row represents a voxel.
        - variable_dataframe (pd.DataFrame): DataFrame where each column represents represents a subject,
                                        and each row represents the variable to regress upon. 
        - outcome_col (str): The column in variable_dataframe which holds the outcome variable.
                                        This will cause a voxelwise regression to occur. 
        - mask_path (str): the path to the mask you want to use. 
                                        If None, will threshold the voxelwise image itself by mask_threshold.
        - mask_threshold (int): The threshold to mask the neuroimaging data at.
        """
        self.mask_path = mask_path
        self.mask_threshold = mask_threshold
        self.out_dir = out_dir
        self.max_stat_method = max_stat_method
        self.vectorize = vectorize
        neuroimaging_dataframe, self.variable_dataframe = self.sort_dataframes(covariate_df=variable_dataframe, voxel_df=neuroimaging_dataframe)
        self.original_mask, self.nonzero_mask, self.neuroimaging_dataframe = self.mask_dataframe(neuroimaging_dataframe)
        self.coerce_df_to_numeric()
        self.outcome_df = self.get_outcome_df(outcome_row)
        self.R2_trick = True
    
    def get_outcome_df(self, outcome_row:str = None):
        """A simple method to get the outcome of interest from self.variable_df"""
        if outcome_row is not None:
            outcome_df = self.variable_dataframe.loc[[outcome_row], :]
        else:
            outcome_df = None
        return outcome_df
    
    def coerce_df_to_numeric(self):
        self.variable_dataframe = self.variable_dataframe.apply(pd.to_numeric, errors='coerce')
        self.neuroimaging_dataframe = self.neuroimaging_dataframe.apply(pd.to_numeric, errors='coerce')

    def sort_dataframes(self, voxel_df: pd.DataFrame, covariate_df: pd.DataFrame, debug: bool=False, dedup: bool=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Will sort the rows of the voxelwise DF and the covariate DF to make sure they are identically organized.
        Then will check that the columns are equivalent.
        """
        try:
            voxel_cols = voxel_df.columns.astype(int).values
            covariate_cols = covariate_df.columns.astype(int).values
            
            voxel_df.columns = voxel_cols
            covariate_df.columns = covariate_cols
            
            shared_columns = list(set(voxel_cols).intersection(set(covariate_cols)))
            
        except:
            # Force Columns to Match
            voxel_cols = voxel_df.columns.astype(str).values
            covariate_cols = covariate_df.columns.astype(str).values
            
            voxel_df.columns = voxel_cols
            covariate_df.columns = covariate_cols
                        
            shared_columns = list(set(voxel_cols).intersection(set(covariate_cols)))
        
        if debug:
            # Print shared columns for debugging
            print("Shared Columns:", shared_columns)
            
            # Identify and print dropped columns
            dropped_voxel_cols = set(voxel_df.columns) - set(shared_columns)
            dropped_covariate_cols = set(covariate_df.columns) - set(shared_columns)
            
            if dropped_voxel_cols:
                print("Dropped Voxel Columns:", dropped_voxel_cols)
            if dropped_covariate_cols:
                print("Dropped Covariate Columns:", dropped_covariate_cols)
                
        if dedup:
            # Detect and print duplicate columns
            voxel_duplicates = voxel_df.columns[voxel_df.columns.duplicated()].tolist()
            covariate_duplicates = covariate_df.columns[covariate_df.columns.duplicated()].tolist()

            if voxel_duplicates:
                print("WARNING: duplicate columns in voxel_df:", voxel_duplicates)
            if covariate_duplicates:
                print("WARNING: duplicate columns in covariate_df:", covariate_duplicates)
                
            # Remove any potential duplicate columns
            voxel_df = voxel_df.loc[:, ~voxel_df.columns.duplicated()]
            covariate_df = covariate_df.loc[:, ~covariate_df.columns.duplicated()]
        
        # Align dataframes to shared columns
        aligned_voxel_df = voxel_df.loc[:, shared_columns]
        aligned_covariate_df = covariate_df.loc[:, shared_columns]
        return aligned_voxel_df, aligned_covariate_df
    
    def threshold_probabilities(self, df: pd.DataFrame, debug=False) -> pd.Series:
        """
        Apply a threshold to mask raw voxelwise data. 
        Finds all voxels which are nonzero across all rows and create a mask from them. 
        
        Parameters:
        df (pd.DataFrame): DataFrame with voxelwise data.
        
        Returns:
        pd.Series: Mask of nonzero voxels.
        """
        if self.mask_path is not None: 
            mask_data = nib.load(self.mask_path).get_fdata().flatten()
            mask_data = pd.DataFrame(mask_data, index=df.index, columns=['mask_data'])
            if len(mask_data) != len(df):
                raise ValueError("Length of mask data does not match the length of the DataFrame. Resolution error suspected")
            mask_data_thr = mask_data.where(mask_data > self.mask_threshold, 0)
        else:
            mask_data_thr = df.where(df > self.mask_threshold, 0)

        mask_indices = mask_data_thr.sum(axis=1) > 0
        if debug:
            print(mask_indices.shape, np.max(mask_indices))
        return mask_indices
    
    def mask_dataframe(self, neuroimaging_df: pd.DataFrame):
        """
        Apply a mask to the neuroimaging DataFrame based on nonzero voxels.
        
        Parameters:
        neuroimaging_df (pd.DataFrame): DataFrame with neuroimaging data.
        
        Returns:
        pd.Index: Index of the whole DataFrame.
        pd.Series: Mask of nonzero voxels.
        pd.DataFrame: Masked neuroimaging DataFrame.
        """
        # Now you can use the function to apply a threshold to patient_df and control_df
        mask = self.threshold_probabilities(neuroimaging_df)
        
        original_mask = neuroimaging_df.index
        masked_neuroimaging_df = neuroimaging_df.loc[mask, :]
        return original_mask, mask, masked_neuroimaging_df
    
    def unmask_dataframe(self, df:pd.DataFrame):
        """
        Simple unmasking function.
        """
        # Initialize a new DF
        empty_mask = pd.DataFrame(index=self.original_mask, columns=['voxels'], data=0)

        # Insert data into the DF 
        empty_mask.loc[self.nonzero_mask, :] = df.values.reshape(-1, 1)
        return empty_mask
    
    def mask_by_p_values(self, results_df:pd.DataFrame, p_values_df:pd.DataFrame):
        """Simple function to perform the thresholding by FWE corrected p-values"""
        unmasked_df = results_df.copy()
        
        mask = p_values_df.where(p_values_df < 0.05, 0)
        mask = mask.sum(axis=1) == 0
        
        unmasked_df.loc[mask, :] = 0
        return unmasked_df
    
    def permute_covariates(self):
        """Permute the patient data by randomly assigning patient data (columnar data) to new patients (columns)"""
        return self.variable_dataframe.sample(frac=1, axis=1, random_state=None)
    
    def prepare_dmatrix(self, use_intercept, permuted_variable_df, debug):
        """
        Params
        -----
        use_intercept (bool): whether to add an intercept. if true, numpy will add intercept to start of covariates
        permuted_variable_df (pd.DataFrame): the permuted data from either vectorized or voxelwise path. If detected, will use this for Y. 
        
        Returns
        -------
        X: covariates. in vectorized method, returns (observations, covariates) | In voxelwise method, voxels need to be added later 
        Y: Dep var. in vectorized method, returns (observations, voxels) | In voxelwise method, is (observations, 1)
        
        Notes
        -----
        Will prepare the design matrix and dependent variables for either vectorized or voxelwise regression
        """
        Y = self.neuroimaging_dataframe.values # Y is a an array of niftis
        
        # Get X variables
        if permuted_variable_df is not None: 
            X = permuted_variable_df.values
        else:
            X = self.variable_dataframe.values
                
        # Prepend the intercept to the X variables
        if use_intercept:
            intercept = np.ones((1, X.shape[1]))  # Create a row of ones for the intercept
            X = np.vstack((intercept, X))  # Add the intercept term to the covariates
        else:
            X = X 
        self.used_intercept = use_intercept
        
        if debug:
            print("X PRE RESHAPE: ", X.shape, "Y PRE RESHAPE: ", Y.shape)
        
        return X.T, Y.T # Reshaped so shape is: X: (observations, covariates) and Y: (observations, voxels)
    
    def prepare_voxelwise_dmatrix(self, use_intercept=True):
        Y = self.outcome_df.values # Y, an array of observation variables in shape (1, observations)
        
        # Get X variables
        if self.variable_dataframe.shape[0] > 0: # Check if there are any covariates
            outcome_row = self.outcome_df.index[0]
            covariates = self.variable_dataframe.drop(labels=outcome_row, axis=0)
        
        # Get the covariates
        if covariates.shape[1] > 0:
            X = covariates.values # covariates in shape (covariates, observation)
        else:
            X = None # if empty, set to None

        # Prepend the intercept to the X variables
        if use_intercept and X is not None:
            intercept = np.ones((1, X.shape[1]))  # Create a row of ones for the intercept
            X = np.vstack((intercept, X))  # Add the intercept term to the covariates
        # If no X variables, only use intercept
        elif use_intercept and X is None:
            X = np.ones((1, Y.shape[1]))
        # impossible condition
        else:
            raise NotImplementedError("It is impossible to regress something upon No intercept and no covariates. Fix design matrix or add intercept.")
        
        return X.T, Y.T # Reshaped so shape is: X: (observations, covariates) and Y: (observations, voxels)
        
    def get_t_values(self, model, X, Y, SSE=None, debug=False, vectorize=False):
        """
        Params:
        -------
        model: the model fit by sklearn.LinearRegression
        X: the matrix of regressors
        Y: matrix of outcomes
        SSE: sum of squared errors
        vectorize: use reliable regression method on voxelwise basis or uses math to broadcast.
        
        Returns:
        --------
        Numpy array of the T-values for each regressor. 
        
        Notes:
        ------
        TODO: The vectorized version is failing to calculate the coefficients appropriately. Fix this. 
        
        T = beta/std. err(beta)
        betas can be achieved from model._coef
        Std. err(beta) is MSE * Inverse(variance-covariance matrix of X). 
            The variance of each beta is along the diagonal. 
        MSE is SSE / DF
            DF of error is N - P
        """
        # Generate vectorized T-Values. Is unstable.
        if vectorize:
            print("Broadcasting manual calculation of T-values of voxel ~ covariates")
            INT = model.intercept_[:, np.newaxis] #convert to 2d array for concat
            COEF = model.coef_[:, 1:] # When fitting an intercept, SKLEARN will overwrite ours. It's a bitch, but what can you do.
            BETA = np.hstack((INT, COEF))
            
            N, P = X.shape
            DF = N - P
            MSE = SSE / DF 
            MSE = MSE.reshape(-1, 1)# Shaped to (voxels, observation)
            
            XTX = np.dot(X.T, X)
            XTX_I = np.linalg.inv(XTX)  
            DIAG_XTX_I = np.diag(XTX_I) #taking diagonal early to avoid dealing with covariance matrix
            DIAG_XTX_I = DIAG_XTX_I.reshape(1, -1) # Shaped to (voxel, covariates)
            VAR_BETA = MSE * DIAG_XTX_I
            
            STD_BETA = np.sqrt((VAR_BETA))
            
            T = BETA/STD_BETA # reshaping to enable broadcasting to shape: (n_voxels, n_covariates)

            if debug:
                print("BETA SHAPE: ", BETA.shape, "MSE SHAPE: ", MSE.shape, "XTX SHAPE: ", XTX.shape, "XTX_I SHAPE: ", XTX_I.shape, "DIAG_XTX_I SHAPE: ", DIAG_XTX_I.shape, "VAR_BETA SHAPE: ", VAR_BETA.shape, " STD BETA SHAPE: ", STD_BETA.shape)
        
        # Generate T-values for regression WITHOUT voxelwise variables.
        elif not vectorize and self.outcome_df is None:
            T = []
            for voxel in tqdm(range(Y.shape[1]), desc='Using statsmodels for voxelwise calculation of T-values of voxel ~ covariates'):
                y_voxel = Y[:, voxel]
                sm_model = sm.OLS(y_voxel, X).fit()
                T.append(sm_model.tvalues)
            
            T = np.array(T)  # Transpose to match expected shape
            if debug:
                print("Statsmodels T-values shape:", T.shape)
        
        # Generate T-values for regression WITH voxelwise variables       
        elif self.outcome_df is not None:
            T = []
            X, Y = self.prepare_voxelwise_dmatrix()
            for voxel_i in tqdm(range(self.neuroimaging_dataframe.shape[0]), desc="Using statsmodels for voxelwise calculation of T-values of Outcome ~ voxel + covariates"): #neuroimaging DF is in shape (voxels, observations)
                voxel = self.neuroimaging_dataframe.iloc[voxel_i, :].values.T.reshape(-1, 1) #reshaping to (observation, voxels) for compat. w/ X
                X_V = np.hstack((X, voxel)) # Finalize the design matrix
                
                model = sm.OLS(Y, X_V).fit() # fit model on a single voxel
                T.append(model.tvalues)
            T = np.array(T)
        else:
            raise NotImplementedError("Unimplemented t-values request.")
        return T
            
    def run_lin_reg(self, X, Y, debug: bool=False):
        """Will run regression on X and Y arrays, entering as shape (observations, variable)"""
        # Voxelwise regression, suitable for regression of voxels on variables.
        if (not self.vectorize) and (self.outcome_df is not None): 
            R2 = []
            X, Y = self.prepare_voxelwise_dmatrix()
            for voxel_i in tqdm(range(self.neuroimaging_dataframe.shape[0]), desc='Using statsmodels for voxelwise calculation of R2 of voxel ~ covariates'): #neuroimaging DF is in shape (voxels, observations)
                voxel = self.neuroimaging_dataframe.iloc[voxel_i, :].values.T.reshape(-1, 1) #reshaping to (observation, voxels) for compat. w/ X
                X_V = np.hstack((X, voxel)) # Finalize the design matrix
                model = sm.OLS(Y, X_V).fit() # fit model on a single voxel
                R2.append(model.rsquared) 
            
            R2 = np.array(R2)  # Transpose to match expected shape
            SSE = None
            
        # Voxelwise regression, suitable for regression of variables on voxel.
        elif (not self.vectorize) and (self.outcome_df is None): 
            R2 = []
            for voxel_i in tqdm(range(Y.shape[1]), desc='Using statsmodels for voxelwise calculation of R2 of voxel ~ covariates'): #neuroimaging DF is in shape (voxels, observations)
                y_voxel = Y[:, voxel_i]
                model = sm.OLS(y_voxel, X).fit()
                R2.append(model.rsquared) 
            
            R2 = np.array(R2)  # Transpose to match expected shape
            SSE = None
        
        # Vectorized regression suitable for regression of Cov. on an entire nifti. 
        else: 
            # Fit model on control data across all voxels
            model = LinearRegression(fit_intercept=True)
            model.fit(X, Y)

            # Predict on experimental group and calculate R-squared
            Y_HAT = model.predict(X)
            Y_BAR = np.mean(Y, axis=0, keepdims=True)

            SSE = np.sum( (Y_HAT - Y_BAR)**2, axis=0)
            SST = np.sum( (Y     - Y_BAR)**2, axis=0)
            R2 = SSE/SST

            if debug:
                print("Y HAT SHAPE: ", Y_HAT.shape, "Y BAR SHAPE: ",  Y_BAR.shape, "SSE SHAPE: ", SSE.shape, "SST SHAPE: ", SST.shape)
        
        if debug:
            print("X SHAPE: ", X.shape, "Y SHAPE: ", Y.shape, 'Observed R2 max: ', np.max(R2), "R2 SHAPE: ", R2.shape)
                    
        # Reshape R2 to DataFrame format
        return pd.DataFrame(R2.T, index=self.neuroimaging_dataframe.index, columns=['R2']), model, SSE
    
    def orchestrate_linear_regression(self, permuted_variable_df: pd.DataFrame=None, use_intercept: bool=True, debug: bool=False) -> pd.DataFrame:
        """
        Calculate voxelwise relationship to Y variable with linear regression.
        It is STRONGLY advised to set mask=True when running this.

        This function performs a linear regression using sklearn's LinearRegression
        The regression is done once across all voxels simultaneously, 
        treating each voxel's values across subjects as independent responses. 
        This vectorized approach efficiently handles the calculations by leveraging matrix operations, 
        which are computationally optimized in libraries like numpy and sklearn.

        Args:
            use_intercept (bool): if true, will add intercept to the regression
            debug (bool): if true, prints out summary metrics

        Returns:
            pd.DataFrame:
        """ 
        X, Y = self.prepare_dmatrix(use_intercept=use_intercept, permuted_variable_df=permuted_variable_df, debug=debug)
        
        R2_df, model, SSE = self.run_lin_reg(X, Y, debug=debug)
        
        if permuted_variable_df is None: # Only get t-values on the first loop. Are automatically calulcated when using voxelwise method.
            self.T_arr = self.get_t_values(model, X, Y, SSE, debug=debug)
        return R2_df

    def var_smooth(self, df):
        """Will take the 95th percentile of the permuted data as the 'maximum stat' as a proxy for variance smoothed max stat."""
        raise ValueError("Function not yet complete.")
    
    def pseudo_var_smooth(self, df):
        """Will take the 99.5th percentile of the permuted data as the 'maximum stat' as a proxy for variance smoothed max stat."""
        return np.array([[np.percentile(df, 99.5, axis=None)]])
    
    def raw_max_stat(self, df):
        """Will simply return the max statistic in the data"""
        return np.max(df)
    
    def get_max_stat(self,df):
        """Will choose the max stat methoda and return the max stat"""
        if self.max_stat_method is None:
            max_stat = self.raw_max_stat(df)
        elif self.max_stat_method == 'pseudo_var_smooth':
            max_stat = self.pseudo_var_smooth(df)
        elif self.max_stat_method == 'var_smooth':
            max_stat = self.var_smooth(df)
        else:
            raise ValueError("Invalid max_stat_method arg. Choose None | 'pseudo_var_smooth | 'var_smooth")
        return max_stat
    
    def maximum_stat_fwe(self, n_permutations=100, debug=False):
        """
        Perform maximum statistic Family-Wise Error (FWE) correction using permutation testing.

        This method calculates the maximum voxelwise R-squared values across multiple permutations
        of the covariates. It then uses these maximum statistics to correct for multiple comparisons,
        ensuring robust and conservative statistical inference.

        Args:
            n_permutations (int): Number of permutations to perform. Defaults to 100.

        Returns:
            list: A list of maximum R-squared values from each permutation.
        """
        max_stats = []
        for i in tqdm(range(0, n_permutations), desc='Permuting'):
            permuted_covariates = self.permute_covariates()
            permuted_df = self.orchestrate_linear_regression(permuted_covariates, debug=False)
            max_stat = self.get_max_stat(permuted_df)
            max_stats.append(max_stat)
            if debug:
                print("Permutation max stat: ", max_stat)
                print("Max stat shape: ", max_stat.shape)
        print('95th percentile of permuted statistic: ', np.percentile(max_stats, 95))
        if debug:
            print('5th percentile of permuted statistic: ', np.percentile(max_stats, 5))
        return max_stats
            
    def p_value_calculation(self, uncorrected_df, max_stat_list, debug=False):
        """
        Calculate p-values for the uncorrected statistic values using the distribution of maximum statistics.

        Args:
            uncorrected_df (pd.DataFrame): DataFrame of uncorrected statistic values.
            max_stat_list (list): Distribution of maximum statistic values from each permutation.

        Returns:
            np.ndarray: Array of p-values corresponding to the uncorrected statistic values.
        """
        # Calculate P-Values
        max_stat_dist = np.array(max_stat_list).reshape(1,-1)
        if debug:
            print("Max Stat Dist Shape: ",max_stat_dist.shape, "DF Shape: ", uncorrected_df.values.shape)
    
        # Calulcate P Values
        boolean_dist = max_stat_dist >= uncorrected_df.values
        p_values = np.mean(boolean_dist, axis=1, keepdims=True)
        if debug:
            print("Boolean Shape: ", boolean_dist.shape, "P Values shape: ", p_values.shape)
            
        # insert P values
        p_values_df = uncorrected_df.copy()
        p_values_df.loc[:,:] = p_values
        
        # Threshold by 95th Percentile of Max Status
        threshold = np.percentile(max_stat_dist, 95)
        corrected_df = uncorrected_df.where(uncorrected_df > threshold, 0)

        if debug:
            print(p_values_df.shape, f'\n Max in uncorrected DF: {np.max(uncorrected_df)} \n', f'Threshold: {threshold} \n', f'Max in corrected DF: {np.max(corrected_df)}')
        return p_values_df, corrected_df

    def save_single_nifti(self, nifti_df, out_dir, name='generated_nifti', silent=True):
        """Saves NIFTI images to directory."""
        preview = view_and_save_nifti(matrix=nifti_df,
                            out_dir=out_dir,
                            output_name=name,
                            silent=silent)
        return preview
        
    def save_results(self, voxelwise_results, unmasked_p_values, voxelwise_results_fwe):
        """
        Saves the generated files. 
        """
        self.uncorrected_img = self.save_single_nifti(nifti_df=voxelwise_results, out_dir=self.out_dir, name='R2_uncorrected', silent=False)
        self.p_img = self.save_single_nifti(nifti_df=unmasked_p_values, out_dir=self.out_dir, name='R2_uncorrected_p_values', silent=False)
        self.corrected_img = self.save_single_nifti(nifti_df=voxelwise_results_fwe, out_dir=self.out_dir, name='R2_fwe_corrected', silent=False)
    
    def save_each_covariate(self):
        """
        Save each covariate's t-values into separate NIFTI files after unmasking.
        
        Params:
        -------
        self.t_values_array: np.ndarray
            Array of t-values with shape (n_covariates, n_voxels)
        self.out_dir: str
            Directory to save the NIFTI files.
        """
        n_covariates = self.T_arr.shape[1]

        # Prepare the covariates, in order of appearance. 
        if self.outcome_df is not None:  # Remove outcomes from the covariates if used as regressand
            outcome_row = self.outcome_df.index[0]
            covariates_df = self.variable_dataframe.drop(labels=outcome_row, axis=0)
            covariate_names = list(covariates_df.index.values)
        else: 
            covariate_names = list(self.variable_dataframe.index.values)
            
        # Extend covariate name list based on intercept
        if self.used_intercept: 
            covariate_names = ['intercept'] + covariate_names
            n_covariates += 1
           
        # Extend covariate name list based on regression setup 
        if self.outcome_df is not None:
            covariate_names = covariate_names + ['voxelwise']
            n_covariates += 1
            
        for i in range(0, len(covariate_names)): # Generate Name for each covariates
            covariate_t_values = self.T_arr[:, i]
            name = covariate_names[i]   

            # Convert to DataFrame
            covariate_df = pd.DataFrame(covariate_t_values, index=self.neuroimaging_dataframe.index)

            # Unmask the DataFrame
            unmasked_covariate_df = self.unmask_dataframe(covariate_df)

            # Save the NIFTI file
            self.final_t_value = self.save_single_nifti(nifti_df=unmasked_covariate_df, out_dir=self.out_dir, name=f'beta_{name}_t_values', silent=False)

    def run(self, n_permutations=100, debug=False):
        """
        Orchestration method. 
        """
        # Can be abstracted to run the analysis of choice and return it and the p-values
        voxelwise_results = self.orchestrate_linear_regression(debug=debug)
        max_stat_list = self.maximum_stat_fwe(n_permutations=n_permutations, debug=debug)
        p_values, voxelwise_results_fwe = self.p_value_calculation(voxelwise_results, max_stat_list, debug=debug)
        
        # Unmask
        voxelwise_results = self.unmask_dataframe(voxelwise_results)
        unmasked_p_values = self.unmask_dataframe(p_values)
        voxelwise_results_fwe = self.unmask_dataframe(voxelwise_results_fwe)
        
        # Save
        self.save_results(voxelwise_results, unmasked_p_values, voxelwise_results_fwe)
        self.save_each_covariate()
        if debug:
            print(np.max(voxelwise_results), np.max(unmasked_p_values), np.max(voxelwise_results_fwe))
            print(voxelwise_results.shape, unmasked_p_values.shape, voxelwise_results_fwe.shape)