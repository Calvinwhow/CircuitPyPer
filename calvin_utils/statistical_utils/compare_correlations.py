import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from calvin_utils.plotting_utils.simple_heatmap import simple_heatmap
from itertools import combinations
from scipy.stats import norm, pearsonr, spearmanr, ttest_ind
from calvin_utils.plotting_utils.pair_superiority_plot import PairSuperiorityPlot

class CompareCorrelations:
    def __init__(self, dv_path: str,  iv_paths: list[str] = None, n_bootstraps: int = 1000, seed: int = 42, method: str = 'spearman'):
        """
        Initializes the CompareCorrelations object.
        Args:
            dv_path (str): File path to a CSV with observed outcomes in a column, observations in rows
                Should share same order as CSVs chosen in DV_paths
            iv_paths (list[str], optional): List of file paths containing dependent variables in columns. Defaults to None.
                Expects a single column, with observations in rows. Should have same order across CSVs. 
                Names for the plot are derived from the basename of these CSVs.
            n_bootstraps (int, optional): Number of bootstrap samples to use for statistical analysis. Defaults to 1000.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
            method (str, optional): Type of correlation to use. Options: 'spearman' or 'pearson'
        Attributes:
            _dv_df (pd.DataFrame or None): DataFrame to store observed labels, initialized as None.
            _pred_dfs (dict): Dictionary to store DataFrames of predicted labels for each classifier.
            boot_idx_cache (any): Cache for bootstrap indices.
            n_bootstraps (int): Number of bootstrap samples.
            rng (np.random.RandomState): Random number generator initialized with the given seed.
            auc_dist (dict): Dictionary to store AUC distributions for each classifier.
            labels_path (str or None): Path to the file containing the true labels.
            dv_path (list[str] or None): List of file paths containing predicted labels.
        """
        self.method      = method
        self._dv_df      = None
        self._iv_dfs     = {}
        self.corr_dist    = {}
        self.n_bootstraps = n_bootstraps
        self.rng = np.random.RandomState(seed)
        self.dv_path = dv_path
        self.iv_paths = iv_paths
        self.boot_idx_cache = self._paired_boot_indices()
        self._get_corr_dist()
        
    ### Setter/Getter Logic ###
    @property
    def dv_path(self) -> str:
        return self._dv_path

    @dv_path.setter
    def dv_path(self, path: str):
        if not isinstance(path, str):
            raise ValueError("labels_path must be a string CSV filepath")
        self._dv_path = path
        df = pd.read_csv(path)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        self._dv_df = df.astype(float)

    @property
    def iv_paths(self) -> list[str]:
        return list(self._iv_dfs.keys())

    @iv_paths.setter
    def iv_paths(self, paths: list[str]):
        if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
            raise ValueError("iv_paths must be a list of CSV filepaths")
        if len(paths) != 2:
            raise ValueError(f"iv_paths should only contain two paths to compare. Detected {len(paths)}.")
        self._iv_dfs = {}
        for idx, p in enumerate(paths):
            df = pd.read_csv(p)
            df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
            if self.dv_path is not None:
                self._validate(p, df)
            name = os.path.splitext(os.path.basename(p))[0]
            self._iv_dfs[name] = df.astype(float)

    @property
    def observations_df(self) -> pd.DataFrame:
        return self._dv_df

    @property
    def predictions_dfs(self) -> dict[str, pd.DataFrame]:
        """Returns a dict mapping filepath → its predictions DataFrame."""
        return self._iv_dfs

    def _validate(self, path: str, df: pd.DataFrame):
        """Ensure one predictions-DF lines up exactly with obs_df cols & shape."""
        if df.shape != self._dv_df.shape:
            raise ValueError(f"[{path}] shape mismatch: obs {self._dv_df.shape} vs pred {df.shape}")
        
    def _paired_boot_indices(self):
        """Returns an (n_bootstraps x n) integer array of row-indices. Generated once, and reused for each model"""
        n = self._dv_df.shape[0]
        return self.rng.randint(0, n, size=(self.n_bootstraps, n))
    
    ### Statistical Methods ###
    def _get_correlation(self, iv, dv):
        if self.method=='spearman':
            return spearmanr(iv, dv)[0]
        else:
            return pearsonr(iv, dv)[0]
        
    def _get_ttest(self, arr1, arr2):
        print("t-test results of bootstrap: \n\t", ttest_ind(arr1, arr2))

    ### Bootstrapping Logic ###
    def _bootstrap_correlations(self, iv_df: pd.DataFrame):
        dv = self._dv_df.to_numpy().ravel()
        iv_df = iv_df.to_numpy().ravel()
        correlations = [self._get_correlation(iv_df[idx], dv[idx]) for idx in self.boot_idx_cache]
        return np.asarray(correlations)
    
    def _get_corr_dist(self):
        for k in self._iv_dfs.keys():
            self.corr_dist[k] = self._bootstrap_correlations(self._iv_dfs[k])
    
    ### Plotting Methods ###
    def superiority_plot(self, out_dir=None):
        for model_a, model_b in combinations(self.corr_dist.keys(), 2):
            corr_a = self.corr_dist[model_a]
            corr_b = self.corr_dist[model_b]
            self._get_ttest(corr_a,corr_b)
            resample_viz = PairSuperiorityPlot(stat_array_1=corr_a,stat_array_2=corr_b, model1_name=model_a, model2_name=model_b, stat='Correlation', out_dir=out_dir)
            resample_viz.draw()
            
    def run(self, out_dir=None):
        """
        Run the comparison of classifiers and generate plots.
        Args:
            out_dir (str, optional): Directory to save the output plots. Defaults to None.
        """
        self.superiority_plot(out_dir)
        print("Plots saved to:", out_dir)


class CompareCorrelationPairs:
    """
    Compare R^2 across multiple correlation pairs using permutation testing.

    Args:
        dataframe (pd.DataFrame): DataFrame containing all columns referenced in pairs.
        pairs (dict[str, tuple[str, str]]): Mapping of pair name -> (x_col, y_col).
        n_permutations (int, optional): Number of permutations for null distribution. Defaults to 1000.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        method (str, optional): Correlation method: 'spearman' or 'pearson'. Defaults to 'spearman'.
        two_tailed (bool, optional): If True, use two-tailed p-values. Defaults to True.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        pairs: dict[str, tuple[str, str]],
        n_permutations: int = 1000,
        seed: int = 42,
        method: str = "spearman",
        two_tailed: bool = True,
    ):
        self.data_df = dataframe
        self.pairs = pairs
        self.n_permutations = n_permutations
        self.rng = np.random.RandomState(seed)
        self.method = method
        self.two_tailed = two_tailed
        self._validate_pairs()

        self.observed_corrs = self._compute_pair_correlations(self.data_df)
        self.observed_r2 = {k: v ** 2 if not np.isnan(v) else np.nan for k, v in self.observed_corrs.items()}
        self.delta_r2_df = self._build_delta_matrix(self.observed_r2)
        self.permuted_delta_r2 = self._compute_permuted_delta_r2()
        self.delta_p_df = self._compute_p_values()

    def _validate_pairs(self):
        if not isinstance(self.pairs, dict) or not self.pairs:
            raise ValueError("pairs must be a non-empty dict mapping name -> (x_col, y_col)")
        for name, pair in self.pairs.items():
            if not isinstance(name, str):
                raise ValueError(f"Pair name must be a string. Found: {name}")
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"Pair '{name}' must be a 2-item list/tuple of column names.")
            x_col, y_col = pair
            if x_col not in self.data_df.columns or y_col not in self.data_df.columns:
                raise ValueError(f"Pair '{name}' references missing columns: {x_col}, {y_col}")

    def _get_correlation(self, x_vals, y_vals):
        if self.method == "spearman":
            return spearmanr(x_vals, y_vals)[0]
        return pearsonr(x_vals, y_vals)[0]

    def _get_pair_values(self, df: pd.DataFrame, x_col: str, y_col: str):
        pair_df = df[[x_col, y_col]].dropna()
        if pair_df.shape[0] < 2:
            return None, None
        return pair_df[x_col].to_numpy(), pair_df[y_col].to_numpy()

    def _compute_pair_correlations(self, df: pd.DataFrame) -> dict[str, float]:
        correlations = {}
        for name, (x_col, y_col) in self.pairs.items():
            x_vals, y_vals = self._get_pair_values(df, x_col, y_col)
            if x_vals is None:
                correlations[name] = np.nan
                continue
            correlations[name] = self._get_correlation(x_vals, y_vals)
        return correlations

    def _build_delta_matrix(self, values: dict[str, float]) -> pd.DataFrame:
        names = list(values.keys())
        matrix = np.full((len(names), len(names)), np.nan)
        for i, name_i in enumerate(names):
            for j, name_j in enumerate(names):
                val_i = values[name_i]
                val_j = values[name_j]
                if np.isnan(val_i) or np.isnan(val_j):
                    continue
                matrix[i, j] = val_i - val_j
        return pd.DataFrame(matrix, index=names, columns=names)

    def _permute_df(self) -> pd.DataFrame:
        permuted = {}
        for col in self.data_df.columns:
            values = self.data_df[col].to_numpy()
            permuted[col] = self.rng.permutation(values)
        return pd.DataFrame(permuted)

    def _compute_permuted_delta_r2(self) -> np.ndarray:
        names = list(self.pairs.keys())
        permuted = np.full((self.n_permutations, len(names), len(names)), np.nan)
        for idx in range(self.n_permutations):
            perm_df = self._permute_df()
            perm_corrs = self._compute_pair_correlations(perm_df)
            perm_r2 = {k: v ** 2 if not np.isnan(v) else np.nan for k, v in perm_corrs.items()}
            permuted[idx] = self._build_delta_matrix(perm_r2).to_numpy()
        return permuted

    def _compute_p_values(self) -> pd.DataFrame:
        obs = self.delta_r2_df.to_numpy()
        perm = self.permuted_delta_r2
        pvals = np.full(obs.shape, np.nan)
        valid_mask = ~np.isnan(obs)
        if self.two_tailed:
            counts = np.sum(np.abs(perm) >= np.abs(obs), axis=0)
        else:
            counts = np.where(
                obs >= 0,
                np.sum(perm >= obs, axis=0),
                np.sum(perm <= obs, axis=0),
            )
        pvals[valid_mask] = (counts[valid_mask] + 1) / (self.n_permutations + 1)
        np.fill_diagonal(pvals, 1.0)
        return pd.DataFrame(pvals, index=self.delta_r2_df.index, columns=self.delta_r2_df.columns)

    def plot_heatmaps(self, out_dir: str = None, show: bool = True):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        self._plot_heatmap(self.delta_r2_df, "Delta R^2", "coolwarm", axes[0])
        self._plot_heatmap(self.delta_p_df, "Delta p-values", "viridis_r", axes[1])
        if out_dir:
            fig.savefig(os.path.join(out_dir, "delta_r2_heatmap.svg"), bbox_inches="tight")
            fig.savefig(os.path.join(out_dir, "delta_pvalue_heatmap.svg"), bbox_inches="tight")
        if show:
            plt.show()
        return fig, axes

    def _plot_heatmap(self, data_df: pd.DataFrame, title: str, cmap: str, ax):
        simple_heatmap(
            data_df,
            dataset_name=title,
            ax=ax,
            palette=cmap,
            annot=True,
            fmt=".2f",
            labels=list(data_df.columns),
        )

    def run(self, out_dir: str = None, show: bool = True):
        """
        Returns delta R^2 and p-value heatmaps with their underlying DataFrames.
        """
        fig, axes = self.plot_heatmaps(out_dir=out_dir, show=show)
        return self.delta_r2_df, self.delta_p_df, fig, axes
