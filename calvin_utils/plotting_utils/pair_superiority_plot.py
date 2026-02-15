import os 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

class PairSuperiorityPlot:
    def __init__(self, stat_array_1, stat_array_2, model1_name="X", model2_name="Y", stat="R²", out_dir=None, observed_stat_array=None, method='bootstrap'):
        """
        Initializes the resampling plot object for visualizing paired delta statistics.
        This class is used to plot the paired delta between a statistic observed for 
        each resampling, and can visualize either bootstraps or permutations.
        Args:
            stat_array_1 (float): The statistical values 
            stat_array_2 (float): The R-squared value for the second region of interest (ROI).
            model1_name (str, optional): Name of the first model. Defaults to "X".
            model2_name (str, optional): Name of the second model. Defaults to "Y".
            stat (str, optional): The statistic name to be plotted. Defaults to "R²".
            out_dir (str, optional): Where to save. 
            observed_stat_array (float): The observed statistical values, where index 0 is the first observation and index 1 is the other.
            method (str): bootstrap | permtuation. If permutation, draws the distribution vertical line at the Delta's value. IF bootstrap, plot at 0.
        """
        self.stat_array_1 = stat_array_1
        self.stat_array_2 = stat_array_2 
        self.delta_array = self._stat_array_1 - self._stat_array_2
        self.model1_name = model1_name
        self.model2_name = model2_name
        self.out_dir = out_dir
        self.observed_stat_array = observed_stat_array
        self.stat = stat
        self.method = method
        self.BLACK = '#211D1E'
        self.GREY = '#8E8E8E'
        self.WHITE = "#FFFFFF"
        self.colors = self.compute_line_colors_abrupt()
        self.prepare_out_dir()

    @property
    def stat_array_1(self):
        return self._stat_array_1

    @stat_array_1.setter
    def stat_array_1(self, value):
        value = np.array(value)  # Convert to numpy array
        if hasattr(self, '_stat_array_2') and len(value) != len(self._stat_array_2):
            raise ValueError("stat_array_1 and stat_array_2 must have the same length.")
        self._stat_array_1 = value

    @property
    def stat_array_2(self):
        return self._stat_array_2

    @stat_array_2.setter
    def stat_array_2(self, value):
        value = np.array(value)
        if hasattr(self, '_stat_array_1') and len(value) != len(self._stat_array_1):
            raise ValueError("stat_array_1 and stat_array_2 must have the same length.")
        self._stat_array_2 = value

    def prepare_out_dir(self):
        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)
            
    def get_black_to_grey_colormap_abrupt(self):
        return LinearSegmentedColormap.from_list("black_grey_abrupt", [
            (0.0, self.GREY),
            (0.5, self.GREY),
            (0.50001, self.BLACK),
            (1.0, self.BLACK)
        ])

    def compute_line_colors_abrupt(self):
        slope_diff = self.stat_array_2 - self.stat_array_1 
        max_abs = np.max(np.abs(slope_diff))
        normed = (slope_diff + max_abs) / (2 * max_abs)  # scale to [0, 1] with zero at 0.5
        cmap = self.get_black_to_grey_colormap_abrupt()
        return cmap(normed)

    def compute_stats(self, alpha=0.05, analytic=False):
        differences = np.array(self.delta_array)
        if (self.method=='permutation') and (self.observed_stat_array):
            print("Statistical method: permutation, one-tailed")
            delta = float(self.observed_stat_array[0] - self.observed_stat_array[1])
            p_val = np.mean(differences > delta) 
            stat1 = float(self.observed_stat_array[0])      # r2 1
            stat2 = float(self.observed_stat_array[1])      # r2 2
        elif self.method=='bootstrap':
            if analytic:
                print("Confidence Interval Method: Analytic Formula")
                delta = np.mean(differences)
                std_error = np.std(differences, ddof=1) / np.sqrt(len(differences))
                z_score = abs(np.percentile(np.random.normal(0, 1, 100000), 100 * (1 - alpha / 2)))
                stat1 = delta - z_score * std_error     # ci lower
                stat2 = delta + z_score * std_error     # ci upper
            else:
                print("Confidence Interval Method: Empiric Percentile")
                p_val = np.mean(differences > 0)
                delta = np.mean(differences)
                stat1 = np.percentile(differences, 100 * (alpha / 2))       # ci lower
                stat2 = np.percentile(differences, 100 * (1 - alpha / 2))   # ci upper
        else:
            raise RuntimeError("Provide appropriate method input with corresponding observed_stat_array argument")
        return {"p": p_val, "stat1": stat1, "stat2": stat2, "delta": delta}

    def plot_paired_slopes(self, ax):
        self.plot_resample_dots(ax, self.stat_array_1, 1, self.GREY)
        self.plot_resample_dots(ax, self.stat_array_2, 0, self.BLACK)
        for i in range(len(self.stat_array_1)):
            ax.plot([0, 1], [self.stat_array_2[i], self.stat_array_1[i]], color=self.colors[i], linewidth=2, alpha=0.8)
        if self.observed_stat_array is not None:
            ax.plot([0, 1], [self.observed_stat_array[1], self.observed_stat_array[0]], color=self.BLACK, linewidth=2, alpha=1.0, zorder=9)
            self.plot_resample_dots(ax, self.observed_stat_array[0], 1, self.WHITE, edgecolor='black', alpha=1, linewidth=2, zorder=10)     # H_1 (model 1) goes on right
            self.plot_resample_dots(ax, self.observed_stat_array[1], 0, self.WHITE, edgecolor='black', alpha=1, linewidth=2, zorder=10)     # H_0 goes on left
        self.setup_slope_subplot(ax)

    def plot_resample_dots(self, ax, y_vals, x_coord, color, size=150, edgecolor='white', alpha=0.9, linewidth=1.2, zorder=3):
        ax.scatter(np.full(len(y_vals), x_coord), y_vals, color=color,
                   edgecolors=edgecolor, linewidth=linewidth, alpha=alpha, s=size, marker='o', zorder=zorder)

    def setup_slope_subplot(self, ax):
        ax.set_xticks([0, 1])
        ax.set_xticklabels([self.model2_name, self.model1_name])
        ax.set_title(f"Paired {self.stat}", fontsize=20)
        ax.set_ylabel(f"{self.stat}", fontsize=20)
        ax.set_xlim(-0.5, 1.5)
        y_max = max(np.max(self.stat_array_1), np.max(self.stat_array_2))
        ax.set_ylim(top=y_max*1.2)
        ax.tick_params(labelsize=16)
        sns.despine(ax=ax)

    def setup_histogram(self, ax, abs_limit, ymax_padded):
        sns.histplot(self.delta_array, bins=40, kde=True, color=self.BLACK, ax=ax, edgecolor='white')
        if self.method == 'bootstrap':
            line_value = 0
        elif (self.observed_stat_array is not None) and (self.method=='permutation'):
            line_value = float(self.observed_stat_array[0] - self.observed_stat_array[1])
            print(line_value)
        elif (self.observed_stat_array is None) and (self.method=='permutation'):
            print("Error: permutation plotting requested, but self.observed_stat_array=None. Pass observed_stat_array as an argument.")
            line_value = 0
        else:
            line_value = 0
        ax.axvline(line_value, color=self.GREY, linestyle='--')         # 0 if 
        ax.set_xlim([-abs_limit, abs_limit])
        ax.set_ylim([0, ymax_padded])
        ax.set_title(f"Δ{self.stat} Distribution", fontsize=20)
        ax.set_xlabel(f"Additional {self.stat} (Δ{self.stat})", fontsize=20)
        ax.set_ylabel("Number of Resamples", fontsize=20)
        ax.tick_params(labelsize=16)
        if self.method=='bootstrap':
            ax.text(abs_limit * -0.95, ymax_padded * 0.95, f"Favours {self.model2_name}", ha='left', va='top', fontsize=16, color=self.BLACK)
            ax.text(abs_limit * 0.95, ymax_padded * 0.95, f"Favours {self.model1_name}", ha='right', va='top', fontsize=16, color=self.BLACK)
        else:
            ax.text(abs_limit * -0.95, ymax_padded * 0.95, f"Favours H₁", ha='left', va='top', fontsize=16, color=self.BLACK)
            ax.text(abs_limit * 0.95, ymax_padded * 0.95, f"Favours H₀", ha='right', va='top', fontsize=16, color=self.BLACK)
            
        sns.despine(ax=ax)
        return line_value

    def annotate_histogram_ci(self, ax, delta, stat1, stat2, p, line_value):
        left_density = np.sum(self.delta_array < line_value)
        right_density = np.sum(self.delta_array >= line_value)
        if self.method=='bootstrap':
            x_text = 0.05 if left_density < right_density else 0.95
            ha_text = 'left' if left_density < right_density else 'right'
        else:
            x_text = 0.05 if left_density > right_density else 0.95
            ha_text = 'left' if left_density > right_density else 'right'
        
        p_val = 1 - p if p > 0.5 else p
        if self.method=='bootstrap':
            ci_text = f"Δ{self.stat} = {delta:.4f}\nCI: [{stat1:.4f}, {stat2:.4f}]\np = {p_val:.4f}"
        elif self.method=='permutation':
            ci_text = f"Δ{self.stat} = {delta:.4f}\n{self.stat} {self.model1_name}: {stat1:.4f}\n{self.stat} {self.model2_name}: {stat2:.4f}\np = {p_val:.4f}"
        else:
            raise RuntimeError("self.method must be bootstrap or permutation.")
        ax.text(x_text, 0.9, ci_text, ha=ha_text, va='top', fontsize=14, color=self.BLACK, transform=ax.transAxes)

    def annotate_paired_slopes(self, ax):
        t,p = ttest_rel(np.array(self.stat_array_1), np.array(self.stat_array_2))
        x_text = 0.05
        ha_text = 'left' 
        stat_text = f"t = {t:.4f}\np = {p:.4f}"
        ax.text(x_text, 0.95, stat_text, ha=ha_text, va='top', fontsize=14, color=self.BLACK, transform=ax.transAxes)

    def draw(self, verbose=True, ax_pair=None, save=True):
        hist_counts, _ = np.histogram(self.delta_array, bins=40)
        ymax_padded = np.max(hist_counts) * 1.10
        abs_limit = np.max(np.abs(self.delta_array))

        stats = self.compute_stats()
        if ax_pair is None:
            fig, axes = plt.subplots(1, 2, figsize=(10, 6), gridspec_kw={'width_ratios': [0.7, 1.3]})
        else:
            axes = ax_pair
            fig = axes[0].figure

        self.plot_paired_slopes(axes[0])
        # Increase the width of the axis lines
        for spine in axes[0].spines.values():
            spine.set_linewidth(2)
        for spine in axes[1].spines.values():
            spine.set_linewidth(2)
        self.annotate_paired_slopes(axes[0])

        line_value = self.setup_histogram(axes[1], abs_limit, ymax_padded)
        self.annotate_histogram_ci(
            axes[1],
            stats['delta'],
            stats['stat1'],
            stats['stat2'],
            stats['p'],
            line_value
        )
        # Save the figure
        if save and self.out_dir is not None and ax_pair is None:
            name = f'superiority_plot-{self.model1_name}-vs-{self.model2_name}_{len(self.stat_array_1)}{self.method}s.svg'
            save_path = os.path.join(self.out_dir, name)
            fig.savefig(save_path, format='svg')
        if ax_pair is None:
            plt.tight_layout()
            if verbose:
                plt.show()
