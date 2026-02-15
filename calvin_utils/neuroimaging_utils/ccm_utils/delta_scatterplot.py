import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, t

class DeltaCorrelationScatter:
    def __init__(self, x_array_1, x_array_2, y_array, label_1="X₁", label_2="X₂", y_label="Y", stat_label="r", out_dir=None, permutation_result=None, method='pearson'):
        """
        Parameters
        ----------
        x_array_1 : array-like
            X values for condition / model 1 (black)
        x_array_2 : array-like
            X values for condition / model 2 (grey)
        y_array : array-like
            Shared Y values
        permutation_result : dict or None
            Optional dict with keys:
                {'delta_r', 'p'}
            If None, analytic Δr is computed and shown without p-value.
        """

        self.x1 = np.asarray(x_array_1)
        self.x2 = np.asarray(x_array_2)
        self.y = np.asarray(y_array)

        if not (len(self.x1) == len(self.x2) == len(self.y)):
            raise ValueError("x_array_1, x_array_2, and y_array must be same length")

        self.label_1 = label_1
        self.label_2 = label_2
        self.y_label = y_label
        self.stat_label = stat_label
        self.out_dir = out_dir
        self.permutation_result = permutation_result
        self.method = method

        # ---- style (mirrors ResampleVisualizer) ----
        self.BLACK = "#211D1E"
        self.GREY = "#8E8E8E"
        self.WHITE = "#FFFFFF"

        self.prepare_out_dir()
    
    ### Helper Utils
    def prepare_out_dir(self):
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)
            
    ### Statistical Utils
    def compute_correlations(self):
        if self.method=='pearson':
            r1, _ = pearsonr(self.x1, self.y)
            r2, _ = pearsonr(self.x2, self.y)
        elif self.method=='spearman':
            r1, _ = spearmanr(self.x1, self.y)
            r2, _ = spearmanr(self.x2, self.y)
        return r1, r2, r1 - r2
    
    @staticmethod
    def analytic_ci(x, y, x_grid, alpha=0.05):
        slope, intercept = np.polyfit(x, y, 1)
        y_hat = slope * x_grid + intercept
        resid = y - (slope * x + intercept)
        s_err = np.sqrt(np.sum(resid**2) / (len(x) - 2))
        t_val = t.ppf(1 - alpha / 2, df=len(x) - 2)
        ci = t_val * s_err * np.sqrt(1 / len(x) + (x_grid - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
        return y_hat, y_hat - ci, y_hat + ci

    ### Plotting Utils
    def setup_axis(self, ax):
        ax.set_xlabel("X", fontsize=18)
        ax.set_ylabel(self.y_label, fontsize=18)
        ax.tick_params(labelsize=14)
        sns.despine(ax=ax)

    def plot_scatter(self, ax, x, y, color, label, alpha):
        ax.scatter(x, y, s=60, color=color, alpha=alpha, edgecolors=self.WHITE, linewidth=1.2, label=label, zorder=3)

    def plot_regression_with_ci(self, ax, x, y, color):
        x_grid = np.linspace(min(x), max(x), 300)
        y_hat, lo, hi = self.analytic_ci(x, y, x_grid)

        ax.plot(x_grid, y_hat, color=color, linewidth=2.5, zorder=4)
        ax.fill_between(x_grid, lo, hi, color=color, alpha=0.4, zorder=2)

    def annotate_stats(self, ax, r1, r2, delta_r):
        if self.permutation_result is not None:
            p = self.permutation_result["p"]
            text = (
                f"{self.label_1}: r = {r1:.3f}\n"
                f"{self.label_2}: r = {r2:.3f}\n"
                f"Δr = {delta_r:.3f}\n"
                f"p = {p:.4f}"
            )
        else:
            text = (
                f"{self.label_1}: r = {r1:.3f}\n"
                f"{self.label_2}: r = {r2:.3f}\n"
                f"Δr = {delta_r:.3f}"
            )

        ax.text(0.05, 0.95, text, transform=ax.transAxes, ha="left", va="top", fontsize=14, color=self.BLACK)

    ### Orchestration Methods
    def draw(self, show=True):
        r1, r2, delta_r = self.compute_correlations()
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        self.plot_scatter(ax, self.x1, self.y, self.BLACK, self.label_1, alpha=0.9)
        self.plot_scatter(ax, self.x2, self.y, self.GREY,  self.label_2, alpha=0.9)
        self.plot_regression_with_ci(ax, self.x1, self.y, self.BLACK)
        self.plot_regression_with_ci(ax, self.x2, self.y, self.GREY)
        self.setup_axis(ax)
        self.annotate_stats(ax, r1, r2, delta_r)
        ax.legend(frameon=False, fontsize=14)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        plt.tight_layout()
        if self.out_dir is not None:
            fname = f"delta_scatter_{self.label_1}_vs_{self.label_2}"
            fig.savefig(os.path.join(self.out_dir, fname + ".svg"))
        plt.show()
