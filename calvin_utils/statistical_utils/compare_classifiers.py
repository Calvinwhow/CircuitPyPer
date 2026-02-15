import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import norm
from calvin_utils.plotting_utils.pair_superiority_plot import PairSuperiorityPlot

from sklearn.metrics import roc_auc_score, roc_curve

class CompareClassifiers:
    def __init__(self,  labels_path: str = None,  pred_paths: list[str] = None, n_bootstraps: int = 1000, seed: int = 42):
        """
        Initializes the CompareClassifiers object.
        Args:
            labels_path (str, optional): Path to the file containing the true labels. Defaults to None.
            pred_paths (list[str], optional): List of file paths containing predicted labels from different classifiers. Defaults to None.
            n_bootstraps (int, optional): Number of bootstrap samples to use for statistical analysis. Defaults to 1000.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
        Attributes:
            _obs_df (pd.DataFrame or None): DataFrame to store observed labels, initialized as None.
            _pred_dfs (dict): Dictionary to store DataFrames of predicted labels for each classifier.
            _boot_idx_cache (any): Cache for bootstrap indices, initialized as None.
            n_bootstraps (int): Number of bootstrap samples.
            rng (np.random.RandomState): Random number generator initialized with the given seed.
            auc_dist (dict): Dictionary to store AUC distributions for each classifier.
            labels_path (str or None): Path to the file containing the true labels.
            pred_paths (list[str] or None): List of file paths containing predicted labels.
        """
        
        self._obs_df      = None
        self._pred_dfs    = {}
        self._boot_idx_cache = None
        self.n_bootstraps = n_bootstraps
        self.rng = np.random.RandomState(seed)
        self.auc_dist = {}
        self.labels_path = labels_path
        self.pred_paths = pred_paths

    ### Setter/Getter Logic ###
    
    @property
    def labels_path(self) -> str:
        return self._labels_path

    @labels_path.setter
    def labels_path(self, path: str):
        if not isinstance(path, str):
            raise ValueError("labels_path must be a string CSV filepath")
        self._labels_path = path
        df = pd.read_csv(path)
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        if not np.all(df.sum(axis=1) == 1):
            raise ValueError("labels CSV must be one-hot encoded (each row sums to 1)")
        self._obs_df = df.astype(int)

    @property
    def pred_paths(self) -> list[str]:
        return list(self._pred_dfs.keys())

    @pred_paths.setter
    def pred_paths(self, paths: list[str]):
        if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
            raise ValueError("pred_paths must be a list of CSV filepaths")

        self._pred_dfs = {}
        for idx, p in enumerate(paths):
            df = pd.read_csv(p)
            df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
            if self._obs_df is not None:
                self._validate_single_prediction(p, df)
            self._pred_dfs[f"Model {idx}"] = df.astype(float)

    @property
    def observations_df(self) -> pd.DataFrame:
        return self._obs_df

    @property
    def predictions_dfs(self) -> dict[str, pd.DataFrame]:
        """Returns a dict mapping filepath → its predictions DataFrame."""
        return self._pred_dfs

    def _validate_single_prediction(self, path: str, pdf: pd.DataFrame):
        """Ensure one predictions-DF lines up exactly with obs_df cols & shape."""
        missing = set(self._obs_df.columns) - set(pdf.columns)
        extra   = set(pdf.columns)      - set(self._obs_df.columns)
        if missing or extra:
            raise ValueError(f"[{path}] column mismatch: missing={missing}, extra={extra}")
        if pdf.shape != self._obs_df.shape:
            raise ValueError(f"[{path}] shape mismatch: obs {self._obs_df.shape} vs pred {pdf.shape}")
    
    ### Bootstrapping Logic ###
    
    def _paired_boot_indices(self, n):
        """Returns an (n_bootstraps x n) integer array of row-indices. Generated once, and reused for each model"""
        if self._boot_idx_cache is None:
            self._boot_idx_cache = self.rng.randint(0, n, size=(self.n_bootstraps, n))
        return self._boot_idx_cache
    
    def _bootstrap_aucs(self, probs: pd.DataFrame):
        y_true = self._obs_df.to_numpy().ravel()
        y_prob = probs[self._obs_df.columns].to_numpy().ravel()
        boot_idx = self._paired_boot_indices(len(y_true))
        aucs = [roc_auc_score(y_true[idx], y_prob[idx]) for idx in boot_idx]
        return np.asarray(aucs)
    
    ### Statistical Logic ###
    
    def _micro_roc(self, probs: pd.DataFrame):
        """Return (fpr, tpr, auc) for micro-average."""
        y_true = self._obs_df.to_numpy().ravel()
        y_prob = probs[self._obs_df.columns].to_numpy().ravel()
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        return fpr, tpr, auc
    
    def _compute_midrank(self, x):
        J = np.argsort(x)
        Z = x[J]
        n = len(x)
        T = np.zeros(n)
        i = 0
        while i < n:
            j = i
            while j < n and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1) + 1
            i = j
        out = np.empty(n)
        out[J] = T
        return out

    def _fast_delong(self, pred_pos, pred_neg):
        m = len(pred_pos)
        n = len(pred_neg)
        pos_ranks = self._compute_midrank(np.concatenate([pred_pos, pred_neg]))[:m]
        auc = (pos_ranks - 1).sum() / (m * n) - (m + 1) / (2 * n)

        v01 = (pos_ranks - pos_ranks.mean()) / n
        v10 = ((self._compute_midrank(np.concatenate([pred_neg, pred_pos]))[:n] - 1)
            - (pos_ranks.mean())) / m
        s01 = np.cov(v01, bias=True)
        s10 = np.cov(v10, bias=True)
        se  = np.sqrt((s01 / m) + (s10 / n))
        return auc, se
    
    
    ### P Value Methods ###
    def pairwise_delong(self):
        """
        Returns a DataFrame of two–sided DeLong p-values
        comparing every pair of models stored in self._pred_dfs.
        Uses the micro-average predictions for each model.
        """

        names  = list(self._pred_dfs.keys())
        y_true = self._obs_df.to_numpy().ravel()
        results = []

        # pre-compute positive / negative labels indices once
        pos_idx = y_true == 1
        neg_idx = ~pos_idx

        for a, b in combinations(names, 2):
            pa = self._pred_dfs[a][self._obs_df.columns].to_numpy().ravel()
            pb = self._pred_dfs[b][self._obs_df.columns].to_numpy().ravel()

            auc_a, se_a = self._fast_delong(pa[pos_idx], pa[neg_idx])
            auc_b, se_b = self._fast_delong(pb[pos_idx], pb[neg_idx])

            var  = se_a**2 + se_b**2          # covariance ≈ 0 (indep. models)
            z    = (auc_a - auc_b) / np.sqrt(var)
            pval = 2 * (1 - norm.cdf(abs(z)))

            results.append((a, b, auc_a, auc_b, z, pval))
        df = pd.DataFrame(results,
                          columns=["Model_A", "Model_B",
                                   "AUC_A", "AUC_B",
                                   "DeLong Test Z", "DeLong Test p"])
        return df

    ### Plotting Methods ###
    def _palette(self):
        return sns.color_palette("tab10", len(self._pred_dfs))

    def plot_micro_roc_curves(self, ax=None):
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = "Helvetica"
        plt.rcParams["font.size"]  = 16

        ax  = ax or plt.gca()
        pal = self._palette()

        for i, (model_name, df) in enumerate(self._pred_dfs.items()):
            fpr, tpr, auc = self._micro_roc(df)
            ax.plot(fpr, tpr, lw=2, color=pal[i],
                    label=f"{model_name} (AUC {auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=2)
        ax.set_xlabel("False-Positive Rate")
        ax.set_ylabel("True-Positive Rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_title("Micro-Average ROC Curves")
        ax.tick_params(axis="both", labelsize=16)
        ax.legend(fontsize=16)
        return ax

    def plot_micro_auc_ci(self, ax=None, alpha=.05):
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = "Helvetica"
        plt.rcParams["font.size"]  = 16

        ax  = ax or plt.gca()
        pal = self._palette()

        means, lows, highs, labels, pvals = [], [], [], [], []
        for i, (model_name, df) in enumerate(self._pred_dfs.items()):
            aucs = self._bootstrap_aucs(df)
            means.append(np.mean(aucs))
            lo, hi = np.percentile(aucs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            lows.append(lo); highs.append(hi)
            labels.append(model_name)
            self.auc_dist[model_name] = aucs
            pvals.append((np.sum(aucs < 0.5))/self.n_bootstraps)

        for i, (mu, lo, hi, p) in enumerate(zip(means, lows, highs, pvals)):
            ax.hlines(i, lo, hi, color=pal[i], lw=2)
            ax.scatter(mu, i, color=pal[i], edgecolor="k", zorder=3, s=45)
            ax.text(lo - 0.02, i, f"p={p:.3f}", va="center", ha="right", fontsize=16)   # annotate p-value

        ax.axvline(0.5, ls="--", color="grey", lw=2)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=16)
        ax.set_xlim(0.4, 1.0)
        ax.set_xlabel("AUROC")
        ax.set_title("Micro-AUC 95 % CI")
        ax.tick_params(axis="both", labelsize=16)
        ax.grid(axis="x", linestyle=":", alpha=.6)
        return ax
    
    def plot(self, out_dir=None):
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = "Helvetica"
        plt.rcParams["font.size"]  = 16

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"wspace": .35})
        self.plot_micro_roc_curves(ax1)
        self.plot_micro_auc_ci(ax2)

        plt.tight_layout()
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            fig.savefig(os.path.join(out_dir, "auc_comparison.svg"), bbox_inches="tight")
    
    def superiority_plot(self, out_dir=None):
        for model_a, model_b in combinations(self.auc_dist.keys(), 2):
            auc_a = self.auc_dist[model_a]
            auc_b = self.auc_dist[model_b]
            resample_viz = PairSuperiorityPlot(stat_array_1=auc_a,stat_array_2=auc_b, model1_name=model_a, model2_name=model_b, stat='AUC', out_dir=out_dir)
            resample_viz.draw()
            
    def run(self, out_dir=None):
        """
        Run the comparison of classifiers and generate plots.
        Args:
            out_dir (str, optional): Directory to save the output plots. Defaults to None.
        """
        self.plot(out_dir)
        self.superiority_plot(out_dir)
        print("Plots saved to:", out_dir)
