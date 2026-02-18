import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr


@dataclass
class SimpleDamagePlotter:
    """
    Four-panel scatter plotter for comparing two columns in a CSV or DataFrame.

    Panels:
    1) Raw values
    2) Log10 values (positive-only)
    3) Z-scored values
    4) Rank percentiles

    Styling mirrors simple_scatter for consistency.
    """

    csv_path: str | None = None
    df: pd.DataFrame | None = None
    x_col: str = ""
    y_col: str = ""
    color_col: str | None = None
    dataset_name: str = "Damage Scatter"
    output_dir: str | None = None
    x_label: str | None = None
    y_label: str | None = None

    def _load_df(self) -> pd.DataFrame:
        if self.df is not None:
            return self.df.copy()
        if self.csv_path is None:
            raise ValueError("Provide either df or csv_path.")
        return pd.read_csv(self.csv_path)

    def _base_style(self, ax):
        ax.tick_params(axis='both', labelsize=16)
        for spine in ax.spines.values():
            spine.set_linewidth(2)

    def _add_stats_text(self, ax, x_vals, y_vals):
        rho, p = spearmanr(x_vals, y_vals, nan_policy='omit')
        r, pr = pearsonr(x_vals, y_vals)
        x_pos = 0.05
        y_pos = 0.95 if rho > 0 else 0.15
        ax.text(
            x_pos, y_pos,
            f"Rho = {rho:.2f}, p = {p:.2e}\nR = {r:.2f}, p = {pr:.2e}",
            fontsize=16,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(facecolor='white', alpha=0, edgecolor='none')
        )

    def _plot_panel(self, ax, plot_df, title, x_label, y_label):
        if plot_df.empty:
            ax.set_title(title, fontsize=20)
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=ax.transAxes)
            self._base_style(ax)
            return

        if self.color_col:
            sns.regplot(
                data=plot_df, x="x", y="y", ax=ax,
                scatter=False, line_kws={'color': '#8E8E8E', 'zorder': 2}
            )
            sns.scatterplot(
                data=plot_df, x="x", y="y", hue="color", ax=ax,
                s=150, edgecolor="white", linewidth=2, alpha=0.98, zorder=3
            )
        else:
            sns.regplot(
                data=plot_df, x="x", y="y", ax=ax,
                scatter_kws={'alpha': 0.98, 'color': '#8E8E8E', 's': 150, 'edgecolors': 'white', 'linewidth': 2, 'zorder': 3},
                line_kws={'color': '#8E8E8E', 'zorder': 2}
            )

        self._add_stats_text(ax, plot_df["x"], plot_df["y"])
        ax.set_title(title, fontsize=20)
        ax.set_xlabel(x_label, fontsize=20)
        ax.set_ylabel(y_label, fontsize=20)
        self._base_style(ax)

    def _prep_plot_df(self, df, x_vals, y_vals):
        out = pd.DataFrame({"x": x_vals, "y": y_vals})
        if self.color_col:
            out["color"] = df[self.color_col].values
        return out.dropna(subset=["x", "y"])

    def plot_four_panel(self):
        df = self._load_df()
        required = {self.x_col, self.y_col}
        if not required.issubset(df.columns):
            raise ValueError(f"Missing required columns: {sorted(required - set(df.columns))}")
        if self.color_col and self.color_col not in df.columns:
            raise ValueError(f"Missing color_col: {self.color_col}")

        x_label = self.x_label or self.x_col
        y_label = self.y_label or self.y_col

        x = df[self.x_col].astype(float)
        y = df[self.y_col].astype(float)

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # 1) Raw
        raw_df = self._prep_plot_df(df, x, y)
        self._plot_panel(axes[0, 0], raw_df, "Raw", x_label, y_label)

        # 2) Log10 (positive-only)
        pos_mask = (x > 0) & (y > 0)
        log_df = self._prep_plot_df(df[pos_mask], np.log10(x[pos_mask]), np.log10(y[pos_mask]))
        title = "Log10 (positive-only)" if not log_df.empty else "Log10 (insufficient positive values)"
        self._plot_panel(axes[0, 1], log_df, title, f"log10({x_label})", f"log10({y_label})")

        # 3) Z-scored
        x_z = (x - x.mean()) / (x.std() if x.std() != 0 else 1)
        y_z = (y - y.mean()) / (y.std() if y.std() != 0 else 1)
        z_df = self._prep_plot_df(df, x_z, y_z)
        self._plot_panel(axes[1, 0], z_df, "Z-scored", f"z({x_label})", f"z({y_label})")

        # 4) Rank percentiles
        x_r = x.rank(pct=True)
        y_r = y.rank(pct=True)
        r_df = self._prep_plot_df(df, x_r, y_r)
        self._plot_panel(axes[1, 1], r_df, "Rank percentile", f"rank({x_label})", f"rank({y_label})")

        plt.tight_layout()
        if self.output_dir is not None:
            os.makedirs(os.path.join(self.output_dir, "scatterplots"), exist_ok=True)
            out_path = os.path.join(self.output_dir, f"scatterplots/{self.dataset_name}_four_panel.svg")
            plt.savefig(out_path, bbox_inches="tight")
        plt.show()
        return fig
