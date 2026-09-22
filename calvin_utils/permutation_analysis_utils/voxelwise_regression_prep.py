import warnings

warnings.filterwarnings("ignore")

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from calvin_utils.file_utils.import_functions import GiiNiiFileImport

class RegressionPrep:
    def __init__(
        self,
        design_matrix,
        contrast_matrix,
        outcome_df,
        out_dir,
        neuroimaging_variables=None,
        neuroimaging_interactions=None,
        mask_path=None,
        exchangeability_block=None,
        weights=None,
        data_transform_method='standardize',
        voxelwise_variables=None,          # backward compatibility
        voxelwise_interactions=None,       # backward compatibility
        formula=None,
    ):
        """
        Initializes the RegressionPrep class.

        Parameters:
        - design_matrix (pd.DataFrame): DataFrame containing scalar and neuroimaging variable columns for each subject.
        - contrast_matrix (np.ndarray or list): Array or list specifying the contrasts of interest.
        - outcome_df (pd.DataFrame): DataFrame containing the outcome variable(s), either scalar or neuroimaging-backed.
        - out_dir (str): Output directory for saving processed data.
        - neuroimaging_variables (list of str, optional): Names of columns containing subject-aligned neuroimaging paths.
        - neuroimaging_interactions (list of str, optional): Interaction terms involving neuroimaging variables.
        - mask_path (str, optional): Optional backend-specific mask argument passed through to importers.
        - exchangeability_block (np.ndarray, optional): 1D array of integers indicating exchangeability blocks.
        - weights (np.ndarray or list, optional): Nonzero regression weights. A negative
          weight flips that observation's outcome (x -1) and contributes |w|.
        - data_transform_method (str, optional): 'standardize', 'rank', or None.

        Notes:
        - `voxelwise_variables` and `voxelwise_interactions` are still accepted for backward compatibility.
        - Automatic mask generation has been removed here because it is NIfTI-specific.
        """
        self.design_matrix = design_matrix
        self.contrast_matrix = contrast_matrix
        self.outcome_df = outcome_df
        self.out_dir = out_dir
        # ``mask_path`` is backend-specific: a NIfTI mask for volumes, a
        # surface mask for GIFTI/FreeSurfer, or the canonical ``.npz`` atlas
        # for fibers. Preserve it until the importer resolves the backend.
        self.mask_path = str(mask_path) if mask_path is not None else None
        self.output_ftype = None
        
        self.exchangeability_block = exchangeability_block
        self.data_transform_method = data_transform_method
        self.weights_vector, self.outcome_signs = self.get_weights(weights)
        self.formula=formula

        if neuroimaging_variables is None:
            neuroimaging_variables = voxelwise_variables
        if neuroimaging_interactions is None:
            neuroimaging_interactions = voxelwise_interactions

        self.neuroimaging_variables = neuroimaging_variables or []
        self.neuroimaging_interactions = [
            interaction.replace(' ', '') for interaction in (neuroimaging_interactions or [])
        ]

        os.makedirs(self.out_dir, exist_ok=True)

        # Backward-compatible aliases so external code does not immediately break.
        self.voxelwise_variables = self.neuroimaging_variables
        self.voxelwise_interactions = self.neuroimaging_interactions

        # Mask semantics now depend on backend. Do not auto-generate volume masks here.
        self.mask = self.get_mask()

        self.neuroimaging_regressors = self._get_neuroimaging_regressors()
        self.voxelwise_regressors = self.neuroimaging_regressors  # backward-compatible alias

        self.design_tensor = self._get_design_tensor()
        self.outcome_data = self._get_outcome_data()

    ### setters and getters ###
    def get_mask(self):
        """
        Return the raw mask specification.
        For NIfTI workflows this may still be a path to a mask image.
        For surfaces this may be ignored by the backend.
        """
        return self.mask_path

    def get_weights(self, values):
        n_obs = self.design_matrix.shape[0] if hasattr(self, 'design_matrix') else None
        if values is None:
            return np.ones(n_obs, dtype=float), None
        _weights = np.array(values, dtype=float)
        if n_obs is not None and _weights.shape[0] != n_obs:
            raise ValueError(
                f"weights must have same length as number of observations ({n_obs}), got {_weights.shape[0]}"
            )
        if np.any(np.isnan(_weights)):
            raise ValueError("weights must not contain NaNs")
        if np.any(_weights == 0):
            raise ValueError("weights must be nonzero")
        # A negative weight means "use this observation flipped": the outcome is
        # multiplied by -1 in _get_outcome_data and the regression weight is |w|.
        signs = np.sign(_weights) if np.any(_weights < 0) else None
        _weights = np.abs(_weights)
        return _weights / np.sum(_weights), signs

    def _get_neuroimaging_regressors(self):
        neuroimaging_regressors = self._prepare_neuroimaging_terms()
        return self._apply_interactions(neuroimaging_regressors)

    def _get_design_tensor(self):
        design_tensor = self._prepare_design_matrix(self.neuroimaging_regressors)
        return self._clean(design_tensor, keep_categorical=True)

    def _get_outcome_data(self):
        outcome_data = self._prepare_outcome_data()
        if self.outcome_signs is not None:
            outcome_data = outcome_data * self.outcome_signs[:, None, None]
        return self._clean(outcome_data)

    ### I/O ###
    def _prep_paths(self, df, term):
        """Ensure the result is a flat list of strings (paths)."""
        paths = df[term].values
        if isinstance(paths, np.ndarray):
            paths = paths.flatten().tolist()
        elif hasattr(paths, 'tolist'):
            paths = paths.tolist()
        paths = [str(p) for p in paths]
        return paths

    def _load_neuroimaging_array(self, df, term):
        """
        Load a neuroimaging-backed column through GiiNiiFileImport and return
        an array of shape (n_obs, n_locations).

        Backend importers return arrays as (n_locations, n_files), and
        GiiNiiFileImport preserves that orientation in the DataFrame
        as rows = spatial locations, columns = files/subjects.
        RegressionPrep needs subject-major orientation, so we transpose.
        """
        importer = GiiNiiFileImport(
            import_path=df[term],
            mask_path=self.mask_path if self.mask_path is not None else 'default',
            transpose=False,
        )
        loaded = importer.run()
                
        if not isinstance(loaded, pd.DataFrame):
            raise TypeError(f"Expected GiiNiiFileImport.run() to return a DataFrame, got {type(loaded)}")

        arr = loaded.to_numpy(dtype=np.float32).T

        if arr.shape[0] != df.shape[0]:
            raise ValueError(
                f"Imported neuroimaging data for '{term}' has {arr.shape[0]} observations, "
                f"but dataframe has {df.shape[0]}"
            )

        if self.output_ftype is not None and importer.output_ftype != self.output_ftype:
            raise ValueError(
                "All neuroimaging terms in one regression must use the same IO backend: "
                f"already loaded '{self.output_ftype}', but '{term}' resolved to "
                f"'{importer.output_ftype}'."
            )
        self.output_ftype = importer.output_ftype
        self.mask_path = importer.mask_path
        return arr

    ### Data Preprocessing ###
    def _clean(self, arr, verbose=False, keep_categorical=False, max_unique=2):
        """Handle NaNs, then transform across observations.

        keep_categorical (design only): regressor columns with <= max_unique
        distinct values across observations -- 0/1 indicators such as cluster
        membership -- are passed through untransformed. Ranking or z-scoring an
        indicator centres it, and a set of centred indicators without an
        intercept is rank-deficient, so per-group contrasts become meaningless.
        """
        if verbose:
            print(arr.shape)

        return self.transform_array(
            arr,
            self.data_transform_method,
            keep_categorical=keep_categorical,
            max_unique=max_unique,
        )

    @classmethod
    def transform_array(cls, arr, data_transform_method, *,
                        keep_categorical=False, max_unique=2):
        """Apply the canonical RegressionPrep transform to an array."""
        if data_transform_method not in {'standardize', 'rank', None}:
            raise ValueError(
                "data_transform_method must be 'standardize', 'rank', or None."
            )
        arr = cls._handle_nans(arr)
        original = arr
        if data_transform_method == 'standardize':
            arr = cls._standardize(arr)
        if data_transform_method == 'rank':
            arr = cls._rank_across_subjects(arr)
        if keep_categorical and arr is not original and original.ndim == 3:
            # Only tabular regressors (identical at every location); imaging
            # regressors are always transformed, whatever their values.
            for j in range(original.shape[1]):
                column = original[:, j, :]
                if (np.all(column == column[:, :1])
                        and np.unique(column[:, 0]).size <= max_unique):
                    arr[:, j, :] = column
        return arr

    @staticmethod
    def _handle_nans(arr, value=0):
        max_val = np.nanmax(arr)
        min_val = np.nanmin(arr)
        return np.nan_to_num(arr, nan=value, posinf=max_val, neginf=min_val)

    @staticmethod
    def _standardize(data: np.ndarray, axis: int = 0, skip_ordinals: bool = True, max_unique: int = 10):
        std = data.std(axis=axis, keepdims=True)
        scale_mask = std > 1e-12

        if skip_ordinals and data.ndim == 2:
            uniq = np.apply_along_axis(lambda c: len(np.unique(c)), axis, data)
            ordinal = uniq <= max_unique
            scale_mask = scale_mask & ~ordinal[:, None]

        mean = data.mean(axis=axis, keepdims=True)
        z = data.copy()
        z = np.where(scale_mask, (data - mean) / (std + 1e-8), data)
        return z

    @staticmethod
    def _rank_across_subjects(arr: np.ndarray, handle_ties: bool = True) -> np.ndarray:
        """
        Efficient rank. If perfect tie handling is needed, use scipy.rankdata
        (slower). For maximum speed, use handle_ties=False--it is perfectly equivalent to the
        scipy.rankdata algorithm in the absence of ties. In presence of ties, it is approximate.
        """
        subj = arr.shape[0]
        flat = arr.reshape(subj, -1)

        if handle_ties:
            ranks = rankdata(flat, axis=0, method="average").astype(np.float32)

        else:
            idx = np.argsort(flat, axis=0, kind="mergesort")
            ranks = np.empty_like(flat, dtype=np.float32)
            rows = np.arange(subj, dtype=np.float32)[:, None]
            ranks[idx, np.arange(flat.shape[1])] = rows + 1

        ranks -= ranks.mean(axis=0, keepdims=True)

        if not handle_ties:
            const_mask = np.ptp(flat, axis=0) == 0
            if const_mask.any():
                ranks[:, const_mask] = 0

        return ranks.reshape(arr.shape)

    ### INTERNAL REGRESSION MATRIX PREP ###
    def _interaction_case(self, term1: str, term2: str) -> str:
        """Classify interaction type: neuroimaging-neuroimaging, neuroimaging-scalar, scalar-scalar."""
        in_img1 = term1 in self.neuroimaging_variables
        in_img2 = term2 in self.neuroimaging_variables
        if in_img1 and in_img2:
            return "neuroimaging_neuroimaging"
        if in_img1 or in_img2:
            return "neuroimaging_scalar"
        return "scalar_scalar"

    def _apply_interactions(self, neuroimaging_data):
        """Apply interactions involving neuroimaging terms."""
        for col in self.neuroimaging_interactions:
            term1, term2 = [x.strip() for x in (col.split(':') if ':' in col else col.split('*'))]
            case = self._interaction_case(term1, term2)

            if case == "neuroimaging_neuroimaging":
                neuroimaging_data[col] = neuroimaging_data[term1] * neuroimaging_data[term2]
                continue

            if case == "neuroimaging_scalar":
                neuro_term = term1 if term1 in self.neuroimaging_variables else term2
                scalar_term = term2 if neuro_term == term1 else term1
                interaction_values = self.design_matrix[scalar_term].values.astype(float)[:, None]
                neuroimaging_data[col] = neuroimaging_data[neuro_term] * interaction_values
                continue

            raise ValueError(
                f"Interaction '{col}' has no neuroimaging term. "
                "Use scalar interactions outside neuroimaging_interactions."
            )

        return neuroimaging_data

    def _prepare_neuroimaging_terms(self):
        """Prepare neuroimaging regressors from any supported backend handled by GiiNiiFileImport."""
        neuroimaging_data = {}
        for term in self.neuroimaging_variables:
            if term in self.outcome_df.columns:
                continue
            stacked = self._load_neuroimaging_array(self.design_matrix, term)
            neuroimaging_data[term] = stacked
        return neuroimaging_data

    def _prepare_design_matrix(self, neuroimaging_regressors: dict[str, np.ndarray]):
        """Build a design tensor of shape (n_obs, n_params, n_locations)."""
        n_obs, n_params = self.design_matrix.shape
        n_loc = next(iter(neuroimaging_regressors.values())).shape[1] if neuroimaging_regressors else 1

        tensor = np.empty((n_obs, n_params, n_loc), dtype=np.float32)

        for j, col in enumerate(self.design_matrix.columns):
            if col in neuroimaging_regressors or col in self.neuroimaging_interactions:
                tensor[:, j, :] = neuroimaging_regressors[col]
            else:
                tensor[:, j, :] = self.design_matrix[col].values.astype(np.float32)[:, None]

        return tensor

    def _prepare_outcome_data(self):
        """
        Prepare outcome data as shape (n_obs, n_outcomes, n_locations).

        If the first outcome column is neuroimaging-backed, currently assumes a single
        neuroimaging outcome column. Scalar outcomes may be multi-column.
        """
        outcome_colname = self.outcome_df.columns[0]

        if outcome_colname in self.neuroimaging_variables:
            outcome_data = self._load_neuroimaging_array(self.outcome_df, outcome_colname)
            outcome_data = outcome_data[:, None, :]
        else:
            outcome_data = self.outcome_df.values.astype(float)[:, :, None]

        return outcome_data

    ### PUBLIC API ###
    def save_dataset(self):
        dataset_dict = {
            'neuroimaging_regression': {
                "design_matrix": f"{self.out_dir}/design_matrix.npy",
                "contrast_matrix": f"{self.out_dir}/contrast_matrix.npy",
                "outcome_data": f"{self.out_dir}/outcome_data.npy",
                "output_ftype": self.output_ftype,
                "mask_path": self.mask_path,
                "formula": self.formula,
            }
        }

        np.save(f"{self.out_dir}/design_matrix.npy", self.design_tensor)
        np.save(f"{self.out_dir}/contrast_matrix.npy", self.contrast_matrix)
        np.save(f"{self.out_dir}/outcome_data.npy", self.outcome_data)

        if self.exchangeability_block is not None:
            dataset_dict['neuroimaging_regression']["exchangeability_block"] = f"{self.out_dir}/exchangeability_block.npy"
            np.save(f"{self.out_dir}/exchangeability_block.npy", self.exchangeability_block)

        if self.weights_vector is not None:
            dataset_dict['neuroimaging_regression']["weights_vector"] = f"{self.out_dir}/weights_vector.npy"
            np.save(f"{self.out_dir}/weights_vector.npy", self.weights_vector)

        if self.mask_path is not None:
            dataset_dict['neuroimaging_regression']["mask_path"] = self.mask_path

        with open(f"{self.out_dir}/dataset_dict.json", "w") as f:
            json.dump(dataset_dict, f, indent=4)

        return dataset_dict, f"{self.out_dir}/dataset_dict.json"

    def run(self):
        dataset_dict, json_path = self.save_dataset()
        print("design_tensor shape:", self.design_tensor.shape)
        print("outcome_data shape:", self.outcome_data.shape)
        return dataset_dict, json_path
