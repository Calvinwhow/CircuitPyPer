"""Box-and-jitter statistical plotting helpers."""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class BoxJitterPlotter:
    """Box-and-jitter plot for any categorical grouping."""

    def __init__(
        self,
        figsize: tuple[int, int] = (6, 6),
        box_palette: str = "Set2",
        jitter_color: str = "black",
        jitter_alpha: float = 0.6,
        jitter_size: int = 4,
        title_fontsize: int = 20,
        label_fontsize: int = 20,
        tick_fontsize: int = 16,
        spine_width: int = 2,
    ):
        self.figsize = figsize
        self.box_palette = box_palette
        self.jitter_color = jitter_color
        self.jitter_alpha = jitter_alpha
        self.jitter_size = jitter_size
        self.title_fontsize = title_fontsize
        self.label_fontsize = label_fontsize
        self.tick_fontsize = tick_fontsize
        self.spine_width = spine_width

    def plot(
        self,
        df: pd.DataFrame,
        group_col: str,
        y_col: str,
        title: str,
        out_dir: str | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
    ) -> None:
        if df.empty:
            raise ValueError("`df` is empty.")
        if group_col not in df.columns or y_col not in df.columns:
            raise KeyError("`group_col` or `y_col` not found in dataframe.")

        plt.figure(figsize=self.figsize)
        ax = sns.boxplot(
            data=df,
            x=group_col,
            y=y_col,
            palette=self.box_palette,
            width=0.6,
            showcaps=True,
            fliersize=0,
            zorder=1,
        )
        sns.stripplot(
            data=df,
            x=group_col,
            y=y_col,
            color=self.jitter_color,
            size=self.jitter_size,
            alpha=self.jitter_alpha,
            jitter=0.25,
            ax=ax,
            zorder=2,
        )

        ax.set_title(title, fontsize=self.title_fontsize)
        ax.set_xlabel(xlabel if xlabel else group_col, fontsize=self.label_fontsize)
        ax.set_ylabel(ylabel if ylabel else y_col, fontsize=self.label_fontsize)
        ax.tick_params(axis="both", labelsize=self.tick_fontsize)
        for spine in ax.spines.values():
            spine.set_linewidth(self.spine_width)

        plt.tight_layout()

        if out_dir:
            os.makedirs(os.path.join(out_dir, "boxjitter_plots"), exist_ok=True)
            plt.savefig(
                os.path.join(out_dir, f"boxjitter_plots/{title}_boxjitter.svg"),
                bbox_inches="tight",
            )

        plt.show()
        
        
        
        
from scipy.stats import kruskal, wilcoxon
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb_color):
    return '#%02x%02x%02x' % rgb_color

def lighten_color(rgb, factor):
    return tuple(min(int(c + (255 - c) * factor), 255) for c in rgb)

def generate_grey_shades(primary_hex, n):
    base_rgb = hex_to_rgb(primary_hex)
    return [rgb_to_hex(lighten_color(base_rgb, i / (n * 1.8))) for i in range(n)]

def plot_horizontal_boxplot_with_stats(
    df,
    xlabel="Value",
    out_dir=None,
    primary_color="#8E8E8E",
    secondary_color="#211D1E"
):
    '''Generates a boxplot for each row in a dataframe, assuming observations (nans allowed) are stored across the columns.'''
    df = df.copy()
    label_col = df.columns[0]
    df = df.set_index(label_col)
    df.index.name = 'Category'

    # Sort categories by median
    medians = df.median(axis=1).sort_values(ascending=False)
    df = df.loc[medians.index]

    # Stats
    print("Wilcoxon Signed-Rank Test (median ≠ 0):")
    for row_name, row in df.iterrows():
        values = row.dropna().values
        if len(values) > 0:
            try:
                stat, p = wilcoxon(values)
                print(f"  {row_name}: stat={stat:.3f}, p={p:.4f}")
            except ValueError:
                print(f"  {row_name}: Invalid (likely all zeros or constant values)")
        else:
            print(f"  {row_name}: No valid data")

    groups = [row.dropna().values for _, row in df.iterrows()]
    if len(groups) > 1:
        kw_stat, kw_p = kruskal(*groups)
        print(f"\nKruskal-Wallis Test across categories: stat={kw_stat:.3f}, p={kw_p:.4f}")

    # Reshape for plotting
    df_long = df.reset_index().melt(id_vars='Category', var_name='Observation', value_name='Value')

    # Create shade palette from primary color
    n_categories = df.shape[0]
    shade_palette = generate_grey_shades(primary_color, n_categories)
    palette_map = dict(zip(df.index.tolist(), shade_palette))

    # Plot
    plt.figure(figsize=(10, 6))
    sns.set(style='white', font_scale=1.5)
    ax = sns.boxplot(
        data=df_long,
        y='Category',
        x='Value',
        orient='h',
        palette=palette_map
    )

    # Dashed line at zero
    ax.axvline(0, linestyle='--', color='#B0B0B0', linewidth=2)

    # Style
    sns.despine()
    plt.tick_params(labelsize=20)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel('', fontsize=20)
    plt.title('', fontsize=20)
    plt.tight_layout()

    # Save if requested
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "horizontal_boxplot.svg")
        plt.savefig(out_path, format="svg")
        print(f"Saved SVG to: {out_path}")

    plt.show()
