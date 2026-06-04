import json
import os
from pathlib import Path
from tempfile import mkdtemp
from glob import glob

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


from calvin_utils.file_utils.import_functions import GiiNiiFileImport
from calvin_utils.neuroimaging_utils.nifti_utils.damage_score_utils import DamageScorer
from calvin_utils.neuroimaging_utils.output_functions import NeuroimageFileOutporter
from calvin_utils.permutation_analysis_utils.statsmodels_palm import CalvinStatsmodelsPalm
from calvin_utils.permutation_analysis_utils.voxelwise_regression import VoxelwiseRegression
from calvin_utils.permutation_analysis_utils.voxelwise_regression_prep import RegressionPrep


def load_parcel_df(parcel_path=None, parcel_df=None, mask_path=None):
    if parcel_df is not None:
        return parcel_df.copy()
    if parcel_path is None:
        return None

    if isinstance(parcel_path, (pd.Series, list, tuple)):
        raise TypeError("parcel_path must be a glob path to parcel NIfTI files.")

    parcel_glob = str(Path(parcel_path).expanduser())
    matches = sorted(glob(parcel_glob))
    if not matches:
        raise FileNotFoundError(
            f"No parcel files matched parcel_path glob: {parcel_glob}"
        )
    parcel_dir = os.path.dirname(matches[0])
    file_pattern = os.path.basename(parcel_glob)
    importer = GiiNiiFileImport(
        import_path=parcel_dir,
        file_pattern=file_pattern,
        mask_path=mask_path if mask_path is not None else "default",
        transpose=False,
    )
    return importer.run()


def calculate_parcel_damage_scores(
    map_data,
    parcel_df,
    mask_path,
    selected_damage,
    score_nonzero_only=False,
):
    if parcel_df is None:
        return pd.Series(np.asarray(map_data, dtype=float).ravel())
    if mask_path is None:
        raise ValueError("mask_path is required when using DamageScorer for parcel reduction.")
    if isinstance(selected_damage, (list, tuple)):
        if len(selected_damage) != 1:
            raise ValueError("selected_damage must be a single metric for parcelwise map building.")
        selected_damage = selected_damage[0]

    map_data = np.asarray(map_data, dtype=float).ravel()
    scorer = DamageScorer(
        mask_path=mask_path,
        dv_df=pd.DataFrame({"map": map_data}),
        roi_df=parcel_df.copy(),
    )
    scores = scorer.calculate_damage_scores(
        metrics=[selected_damage],
        trace=False,
        score_nonzero_only=score_nonzero_only,
    )
    values = {
        parcel: scores.loc["map", f"{parcel}_{selected_damage}"]
        for parcel in scorer.roi_df.columns
    }
    return pd.Series(values, dtype=float)


class ParcelwiseDamageMap:
    """
    Build a single float NIfTI by filling each parcel with its damage score.

    This class composes the existing parcel importer, DamageScorer, and NIfTI
    output/viewer primitives used by the voxelwise regression workflow.
    """

    def __init__(
        self,
        target_map,
        parcel_path=None,
        parcel_df=None,
        mask_path=None,
        out_dir=None,
        output_name="parcelwise_damage_map",
        selected_damage="avg_in_target",
        fill_value=np.nan,
        score_nonzero_only=True,
    ):
        self.target_map = target_map
        self.parcel_path = parcel_path
        self.parcel_df = parcel_df
        self.out_dir = out_dir
        self.output_name = output_name
        self.selected_damage = selected_damage
        self.fill_value = fill_value
        self.score_nonzero_only = score_nonzero_only

        output_mask_path = mask_path if mask_path is not None else "default"
        self.output_handler = NeuroimageFileOutporter(
            output_ftype="nii",
            mask_path=output_mask_path,
        )
        self.mask_path = self.output_handler.io.resolved_mask_path
        self.parcel_scores = None
        self.output_map = None
        self.saved_path = None

    def _load_target_map(self):
        if isinstance(self.target_map, (str, Path)):
            target_arr = self.output_handler.io.import_nifti_to_numpy_array(
                [str(Path(self.target_map).expanduser())]
            )
            return target_arr[:, 0]
        return np.asarray(self.target_map, dtype=float).ravel()

    def _load_parcel_df(self):
        parcel_df = load_parcel_df(
            parcel_path=self.parcel_path,
            parcel_df=self.parcel_df,
            mask_path=self.mask_path,
        )
        if parcel_df is None:
            raise ValueError("Either parcel_path or parcel_df must be provided.")
        return parcel_df

    @staticmethod
    def _score_to_parcel_map(parcel_df, parcel_scores, fill_value):
        out = np.full(parcel_df.shape[0], fill_value, dtype=np.float32)
        for parcel, value in parcel_scores.items():
            if parcel not in parcel_df.columns:
                raise ValueError(f"Parcel score {parcel!r} is missing from parcel_df.")
            parcel_mask = np.asarray(parcel_df[parcel]) > 0
            out[parcel_mask] = value
        return out

    def build_map(self):
        target_vec = self._load_target_map()
        parcel_df = self._load_parcel_df()
        if target_vec.shape[0] != parcel_df.shape[0]:
            raise ValueError(
                "Target map and parcels are not in the same vector space: "
                f"target length={target_vec.shape[0]}, parcel length={parcel_df.shape[0]}."
            )

        self.parcel_scores = calculate_parcel_damage_scores(
            target_vec,
            parcel_df,
            self.mask_path,
            self.selected_damage,
            score_nonzero_only=self.score_nonzero_only,
        )
        parcel_df = parcel_df.copy()
        parcel_df.columns = self.parcel_scores.index
        self.output_map = self._score_to_parcel_map(
            parcel_df,
            self.parcel_scores,
            self.fill_value,
        )
        return self.output_map

    def save_map(self):
        if self.out_dir is None:
            raise ValueError("out_dir is required to save the parcelwise damage map.")
        if self.output_map is None:
            self.build_map()

        self.output_handler.save_map(
            self.output_map,
            self.output_name,
            self.out_dir,
            visualize=False,
        )
        self.saved_path = os.path.join(self.out_dir, f"{self.output_name}.nii.gz")
        return self.saved_path

    def view_map(self):
        if self.output_map is None:
            self.build_map()
        return self.output_handler.view_map(self.output_map, self.output_name)

    def run(self):
        self.save_map()
        return self.view_map()


class ParcelwiseRegressionSimilarity:
    """
    Compare voxelwise-regression contrast maps after reducing them to parcel scores.

    The regression path intentionally mirrors 05b:
    table/formula -> CalvinStatsmodelsPalm -> RegressionPrep -> VoxelwiseRegression.
    Each requested contrast map is then summarized against parcels using
    DamageScorer.calculate_damage_scores(metric='avg_in_target').
    """

    def __init__(
        self,
        input_path=None,
        formula_1=None,
        formula_2=None,
        contrast_matrix=None,
        parcel_path=None,
        parcel_df=None,
        mask_path=None,
        out_dir=None,
        sheet=None,
        df=None,
        voxelwise_vars=None,
        voxelwise_interactions=None,
        add_intercept=True,
        drop_nans=None,
        drop_rows=None,
        one_hot=None,
        exchangeability_col=None,
        weights_col=None,
        data_transform_method="standardize",
        regression_type="linear",
        n_permutations=1000,
        similarity="pearson",
        absval=False,
        selected_damage="avg_in_target",
        random_state=None,
        verbose=False,
    ):
        self.input_path = input_path
        self.formula_1 = self._as_list(formula_1, "formula_1")
        self.formula_2 = self._as_list(formula_2, "formula_2")
        self.contrast_matrix = contrast_matrix
        self.parcel_path = parcel_path
        self.parcel_df = parcel_df
        self.mask_path = mask_path
        self.out_dir = out_dir
        self.sheet = sheet
        self.df = df
        self.voxelwise_vars = list(voxelwise_vars or [])
        self.voxelwise_interactions = list(voxelwise_interactions or [])
        self.add_intercept = add_intercept
        self.drop_nans = list(drop_nans or [])
        self.drop_rows = list(drop_rows or [])
        self.one_hot = list(one_hot or [])
        self.exchangeability_col = exchangeability_col
        self.weights_col = weights_col
        self.data_transform_method = data_transform_method
        self.regression_type = regression_type
        self.n_permutations = int(n_permutations)
        self.similarity = similarity
        self.absval = absval
        self.selected_damage = selected_damage
        self.random_state = random_state
        self.verbose = verbose

        self.formula_jobs = []
        self.left_vectors = None
        self.right_vectors = None
        self.observed_similarity_matrix = None
        self.permuted_similarity_tensor = None
        self.p_value = None
        self.observed_average = None
        self.pairwise_p_values = None
        self._work_dir = None

    @staticmethod
    def _as_list(value, name):
        if value is None:
            raise ValueError(f"{name} is required.")
        if isinstance(value, str):
            return [value]
        return list(value)

    @staticmethod
    def _formula_lhs(formula):
        return formula.split("~", 1)[0].strip()

    def _formula_columns(self, formula, df):
        columns = set(self.drop_nans)
        for col in df.columns:
            if col in formula:
                columns.add(col)
        if self.exchangeability_col is not None:
            columns.add(self.exchangeability_col)
        if self.weights_col is not None:
            columns.add(self.weights_col)
        return [col for col in df.columns if col in columns]

    def _drop_formula_nans(self, df, formula):
        columns = self._formula_columns(formula, df)
        if not columns:
            return df
        before = len(df)
        df = df.dropna(subset=columns).copy()
        if self.verbose and before != len(df):
            print(f"[{formula}] dropped {before - len(df)} rows with NaNs in {columns}")
        return df

    @staticmethod
    def _coerce_value(raw):
        try:
            return int(raw)
        except Exception:
            pass
        try:
            return float(raw)
        except Exception:
            return raw

    def _get_cal_palm(self):
        return CalvinStatsmodelsPalm(
            input_csv_path=self.input_path or "",
            output_dir=self.out_dir,
            sheet=self.sheet,
        )

    def _load_dataframe(self):
        cal_palm = self._get_cal_palm()
        if self.df is None:
            if self.input_path is None:
                raise ValueError("Either input_path or df must be provided.")
            df = cal_palm.read_and_display_data()
        else:
            df = self.df.copy()

        if self.drop_nans:
            df = df.dropna(subset=self.drop_nans)

        for col, cond, raw in self.drop_rows:
            val = self._coerce_value(raw)
            if cond in {"==", "eq"}:
                df = df.loc[df[col] == val]
            elif cond in {"!=", "ne"}:
                df = df.loc[df[col] != val]
            elif cond == ">":
                df = df.loc[df[col] > val]
            elif cond == ">=":
                df = df.loc[df[col] >= val]
            elif cond == "<":
                df = df.loc[df[col] < val]
            elif cond == "<=":
                df = df.loc[df[col] <= val]
            else:
                raise ValueError(f"Unsupported drop_rows condition: {cond}")

        for col in self.one_hot:
            dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
            df = df.join(dummies)
        return df

    def _load_contrast_matrix(self):
        if self.contrast_matrix is None:
            return None
        if isinstance(self.contrast_matrix, pd.DataFrame):
            return self.contrast_matrix.to_numpy(dtype=float)
        if isinstance(self.contrast_matrix, (str, Path)):
            path = Path(self.contrast_matrix)
            if path.suffix.lower() == ".json":
                return np.asarray(json.loads(path.read_text()), dtype=float)
            return pd.read_csv(path, header=None).to_numpy(dtype=float)
        return np.asarray(self.contrast_matrix, dtype=float)

    def _load_parcel_df(self):
        return load_parcel_df(
            parcel_path=self.parcel_path,
            parcel_df=self.parcel_df,
            mask_path=self.mask_path,
        )

    def _prepare_formula_job(self, formula, group, formula_idx, df, parcel_df):
        cal_palm = self._get_cal_palm()
        formula_df = self._drop_formula_nans(df.copy(), formula)
        out_dir = None
        if self.out_dir is not None:
            out_dir = os.path.join(self.out_dir, group, f"formula_{formula_idx}")
            os.makedirs(out_dir, exist_ok=True)
        else:
            if self._work_dir is None:
                self._work_dir = mkdtemp(prefix="parcelwise_regression_similarity_")
            out_dir = os.path.join(self._work_dir, group, f"formula_{formula_idx}")
            os.makedirs(out_dir, exist_ok=True)

        outcome_df, design_matrix = cal_palm.define_design_matrix(
            formula,
            formula_df,
            add_intercept=self.add_intercept,
            voxelwise_variable_list=self.voxelwise_vars,
            voxelwise_interaction_terms=self.voxelwise_interactions,
        )

        contrast_matrix = self._load_contrast_matrix()
        if contrast_matrix is None:
            contrast_matrix = cal_palm.generate_basic_contrast_matrix(design_matrix)
        try:
            contrast_matrix_df = cal_palm.finalize_contrast_matrix(
                design_matrix=design_matrix,
                contrast_matrix=contrast_matrix,
            )
        except ValueError as exc:
            contrast_shape = np.asarray(contrast_matrix).shape
            raise ValueError(
                "Could not finalize contrast_matrix for formula "
                f"{formula!r}. The contrast must have one value per design-matrix "
                f"column. Got contrast shape {contrast_shape}; design columns are "
                f"{list(design_matrix.columns)}. Original error: {exc}"
            ) from exc

        exchangeability_block = None
        if self.exchangeability_col is not None:
            exchangeability_block = pd.to_numeric(formula_df[self.exchangeability_col], errors="raise").astype(int).to_numpy()

        weights = None
        if self.weights_col is not None:
            weights = pd.to_numeric(formula_df[self.weights_col], errors="raise").astype(float).to_numpy()

        preparer = RegressionPrep(
            design_matrix=design_matrix,
            contrast_matrix=contrast_matrix_df,
            outcome_df=outcome_df,
            out_dir=out_dir,
            voxelwise_variables=self.voxelwise_vars,
            voxelwise_interactions=self.voxelwise_interactions,
            mask_path=self.mask_path,
            exchangeability_block=exchangeability_block,
            data_transform_method=self.data_transform_method,
            weights=weights,
            formula=formula,
        )
        _, json_path = preparer.run()
        regression = VoxelwiseRegression(
            json_path,
            mask_path=self.mask_path,
            out_dir=None,
            regression_type=self.regression_type,
            n_permutations=0,
        )
        return {
            "group": group,
            "formula_idx": formula_idx,
            "formula": formula,
            "lhs": self._formula_lhs(formula),
            "row_index": tuple(formula_df.index.tolist()),
            "regression": regression,
            "parcel_df": parcel_df,
        }

    def prepare(self):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        df = self._load_dataframe()
        parcel_df = self._load_parcel_df()
        self.formula_jobs = []
        for i, formula in enumerate(self.formula_1):
            self.formula_jobs.append(self._prepare_formula_job(formula, "formula_1", i, df, parcel_df))
        for i, formula in enumerate(self.formula_2):
            self.formula_jobs.append(self._prepare_formula_job(formula, "formula_2", i, df, parcel_df))
        return self

    def _map_to_parcel_vector(self, map_data, parcel_df):
        if parcel_df is None:
            return np.asarray(map_data, dtype=float).ravel()
        return calculate_parcel_damage_scores(
            map_data,
            parcel_df,
            self.mask_path,
            self.selected_damage,
        ).to_numpy(dtype=float)

    def _run_job(self, job, permutation_index=None):
        reg = job["regression"]
        original_y = reg.outcome_tensor
        original_w = reg.weight_vector
        try:
            if permutation_index is not None:
                reg.outcome_tensor = original_y[permutation_index, :, :]
                reg.weight_vector = original_w[permutation_index]
            _, t_values, _ = reg.voxelwise_regression(permutation=False)
        finally:
            reg.outcome_tensor = original_y
            reg.weight_vector = original_w

        vectors = {}
        for contrast_idx in range(reg.n_contrasts):
            name = f"{job['group']}_formula_{job['formula_idx']}_contrast_{contrast_idx}"
            vectors[name] = self._map_to_parcel_vector(t_values[contrast_idx, :], job["parcel_df"])
        return vectors

    def _run_all_jobs(self, permutation_indices=None):
        vectors = {}
        for job in self.formula_jobs:
            perm_idx = None
            if permutation_indices is not None:
                perm_idx = permutation_indices[self._permutation_key(job)]
            vectors.update(self._run_job(job, permutation_index=perm_idx))
        left = {k: v for k, v in vectors.items() if k.startswith("formula_1_")}
        right = {k: v for k, v in vectors.items() if k.startswith("formula_2_")}
        return left, right

    def _permutation_key(self, job):
        return job["lhs"], job["row_index"]

    def _new_permutation_indices(self):
        indices = {}
        for job in self.formula_jobs:
            key = self._permutation_key(job)
            if key not in indices:
                indices[key] = job["regression"]._get_permutation_index()
        return indices

    def _measure_similarity(self, arr1, arr2):
        arr1 = np.asarray(arr1, dtype=float).ravel()
        arr2 = np.asarray(arr2, dtype=float).ravel()
        finite = np.isfinite(arr1) & np.isfinite(arr2)
        arr1 = arr1[finite]
        arr2 = arr2[finite]
        if self.absval:
            arr1 = np.abs(arr1)
            arr2 = np.abs(arr2)
        if arr1.size < 2:
            return np.nan
        if self.similarity == "pearson":
            return pearsonr(arr1, arr2)[0]
        if self.similarity == "spearman":
            return spearmanr(arr1, arr2)[0]
        if self.similarity == "cosine":
            return DamageScorer._calculate_cosine_similarity(arr1, arr2)
        if self.similarity == "spatial_correlation":
            return DamageScorer._calculate_spatial_correlation(arr1, arr2)
        raise ValueError("similarity must be one of: pearson, spearman, cosine, spatial_correlation")

    def calculate_similarity_matrix(self, left_vectors, right_vectors):
        left_names = list(left_vectors.keys())
        right_names = list(right_vectors.keys())
        matrix = np.full((len(left_names), len(right_names)), np.nan)
        for i, left_name in enumerate(left_names):
            for j, right_name in enumerate(right_names):
                matrix[i, j] = self._measure_similarity(left_vectors[left_name], right_vectors[right_name])
        return matrix

    @staticmethod
    def _cross_to_symmetric(cross_matrix, left_names, right_names):
        names = list(left_names) + list(right_names)
        n_left = len(left_names)
        n_total = len(names)
        matrix = np.full((n_total, n_total), np.nan)
        matrix[:n_left, n_left:] = cross_matrix
        matrix[n_left:, :n_left] = cross_matrix.T
        return matrix, names

    def perform_permutation_testing(self):
        if not self.formula_jobs:
            self.prepare()

        self.left_vectors, self.right_vectors = self._run_all_jobs()
        self.left_names = list(self.left_vectors.keys())
        self.right_names = list(self.right_vectors.keys())
        cross_observed = self.calculate_similarity_matrix(self.left_vectors, self.right_vectors)
        self.observed_similarity_matrix, self.matrix_names = self._cross_to_symmetric(
            cross_observed,
            self.left_names,
            self.right_names,
        )
        self.permuted_similarity_tensor = np.full(
            (self.n_permutations, len(self.matrix_names), len(self.matrix_names)),
            np.nan,
            dtype=float,
        )

        for i in tqdm(range(self.n_permutations), desc="Running parcelwise similarity permutations"):
            perm_indices = self._new_permutation_indices()
            perm_left, perm_right = self._run_all_jobs(permutation_indices=perm_indices)
            cross_permuted = self.calculate_similarity_matrix(perm_left, perm_right)
            self.permuted_similarity_tensor[i], _ = self._cross_to_symmetric(
                cross_permuted,
                self.left_names,
                self.right_names,
            )
        return self

    def calculate_p_value(self, tails="two_tail"):
        observed = np.nanmean(self.observed_similarity_matrix)
        permuted = np.nanmean(self.permuted_similarity_tensor, axis=(1, 2))
        if tails == "two_tail":
            p_value = np.mean(np.abs(permuted) >= np.abs(observed))
        elif tails == "one_tail":
            p_value = np.mean(permuted >= observed)
        else:
            raise ValueError("tails must be 'two_tail' or 'one_tail'")
        self.p_value = p_value
        self.observed_average = observed
        return p_value, observed

    def calculate_pairwise_p_values(self, tails="two_tail", max_stat=False):
        p_values = np.full_like(self.observed_similarity_matrix, np.nan, dtype=float)
        if max_stat:
            if tails == "two_tail":
                null = np.nanmax(np.abs(self.permuted_similarity_tensor), axis=(1, 2))
            else:
                null = np.nanmax(self.permuted_similarity_tensor, axis=(1, 2))
        for i in range(p_values.shape[0]):
            for j in range(p_values.shape[1]):
                observed = self.observed_similarity_matrix[i, j]
                if not np.isfinite(observed):
                    continue
                if max_stat:
                    p_values[i, j] = np.mean(null >= (abs(observed) if tails == "two_tail" else observed))
                    continue
                permuted = self.permuted_similarity_tensor[:, i, j]
                if tails == "two_tail":
                    p_values[i, j] = np.mean(np.abs(permuted) >= np.abs(observed))
                elif tails == "one_tail":
                    p_values[i, j] = np.mean(permuted >= observed)
                else:
                    raise ValueError("tails must be 'two_tail' or 'one_tail'")
        self.pairwise_p_values = pd.DataFrame(p_values, index=self.matrix_names, columns=self.matrix_names)
        return self.pairwise_p_values

    @staticmethod
    def contrast_outer_summation(observed_matrix, permuted_tensor, row_contrasts, col_contrasts=None, tails="two_tail"):
        row_c = np.asarray(row_contrasts, dtype=float)
        if row_c.ndim == 1:
            row_c = row_c[None, :]
        col_c = row_c if col_contrasts is None else np.asarray(col_contrasts, dtype=float)
        if col_c.ndim == 1:
            col_c = col_c[None, :]

        weights = row_c[:, None, :, None] * col_c[None, :, None, :]
        weights = weights.reshape(row_c.shape[0] * col_c.shape[0], observed_matrix.shape[0], observed_matrix.shape[1])
        observed = np.einsum("ij,kij->k", observed_matrix, weights, optimize=True)
        permuted = np.einsum("bij,kij->bk", permuted_tensor, weights, optimize=True)
        if tails == "two_tail":
            p_values = np.mean(np.abs(permuted) >= np.abs(observed)[None, :], axis=0)
        elif tails == "one_tail":
            p_values = np.mean(permuted >= observed[None, :], axis=0)
        else:
            raise ValueError("tails must be 'two_tail' or 'one_tail'")
        return observed, p_values

    def save_results(self, prefix=""):
        if self.out_dir is None:
            return
        os.makedirs(self.out_dir, exist_ok=True)
        np.save(os.path.join(self.out_dir, f"{prefix}observed_similarity_matrix.npy"), self.observed_similarity_matrix)
        np.save(os.path.join(self.out_dir, f"{prefix}permuted_similarity_tensor.npy"), self.permuted_similarity_tensor)
        pd.DataFrame(self.observed_similarity_matrix, index=self.matrix_names, columns=self.matrix_names).to_csv(
            os.path.join(self.out_dir, f"{prefix}observed_similarity_matrix.csv")
        )
        if self.pairwise_p_values is not None:
            self.pairwise_p_values.to_csv(os.path.join(self.out_dir, f"{prefix}pairwise_p_values.csv"))

    def run(self, tails="two_tail", fwe=False, save=True):
        self.perform_permutation_testing()
        p_value, observed = self.calculate_p_value(tails=tails)
        pairwise = self.calculate_pairwise_p_values(tails=tails, max_stat=fwe)
        if save:
            self.save_results()
        return {
            "p_value": p_value,
            "observed_average": observed,
            "pairwise_p_values": pairwise,
            "observed_similarity_matrix": pd.DataFrame(
                self.observed_similarity_matrix,
                index=self.matrix_names,
                columns=self.matrix_names,
            ),
            "permuted_similarity_tensor": self.permuted_similarity_tensor,
        }
