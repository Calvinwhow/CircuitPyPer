import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class PermutationVisualizer:
    def __init__(self, stat_obs: float, stat_dist, stat: str = "stat", out_dir: str | None = None, side: str = "two-sided", bins: int = 40, kde: bool = True, correction: bool = True, absval=False):
        self.stat_obs = np.abs(stat_obs) if absval else stat_obs
        self.stat_dist = np.abs(np.asarray(stat_dist, dtype=float).ravel()) if absval else np.asarray(stat_dist, dtype=float).ravel()
        self.stat = stat
        self.out_dir = out_dir
        self.side = side.lower()
        self.bins = int(bins)
        self.kde = bool(kde)
        self.correction = bool(correction)
        self.absval = absval
        self.BLACK = '#211D1E'
        self.GREY = '#8E8E8E'

        if self.out_dir is not None:
            os.makedirs(self.out_dir, exist_ok=True)

        self.stat_dist = self.stat_dist[np.isfinite(self.stat_dist)]
        if self.stat_dist.size == 0:
            raise ValueError("stat_dist is empty after removing non-finite values.")

    def _p_value(self) -> float:
        x0 = self.stat_obs
        x = self.stat_dist
        n = x.size

        if self.side == "greater":
            k = np.sum(x >= x0)
        elif self.side == "less":
            k = np.sum(x <= x0)
        elif self.side == "two-sided":
            center = 0.0
            k = np.sum(np.abs(x - center) >= np.abs(x0 - center))
        else:
            raise ValueError("side must be 'two-sided', 'greater', or 'less'.")

        if self.correction:
            return (k + 1.0) / (n + 1.0)
        return k / n if n > 0 else np.nan

    def _annotate(self, ax, p_val: float):
        txt = f"{self.stat}₀ = {self.stat_obs:.4f}\np = {p_val:.4g} ({self.side})\nN perms = {self.stat_dist.size}"
        ax.text(0.02, 0.98, txt, ha='left', va='top', fontsize=12, color=self.BLACK,
                transform=ax.transAxes)
        
    def draw(self, name: str = 'perm_hist.svg', verbose = True):
        fig, ax = plt.subplots(figsize=(6, 6))

        # histogram
        sns.histplot(self.stat_dist, bins=self.bins, kde=self.kde, color=self.BLACK,
                    edgecolor='white', ax=ax)

        # lines
        ax.axvline(self.stat_obs, color=self.BLACK, linestyle='-', linewidth=2, zorder=4)
        ax.axvline(0, color=self.GREY, linestyle='--', linewidth=1.5)

        # p-value in title
        p_val = self._p_value()
        ax.set_title(
            f"Permutation distribution of {self.stat}  |  p = {p_val:.4g} ({self.side})",
            fontsize=16
        )

        ax.set_xlabel(self.stat, fontsize=14)
        ax.set_ylabel("Count", fontsize=14)
        sns.despine(ax=ax)
        for s in ax.spines.values():
            s.set_linewidth(2)
        ax.set_xlim(np.min(self.stat_dist), np.max(self.stat_dist))
        self._annotate(ax, p_val)

        # save + show
        if self.out_dir is not None:
            fig.savefig(os.path.join(self.out_dir, name),
                        format=name.split('.')[-1], dpi=300)
        if verbose: 
            plt.show()

        return {"p": p_val, "side": self.side, "n": int(self.stat_dist.size)}

if __name__=="__main__":
    stat_obs = 1
    stat_dist = [1,2,3]
    stat = 'my_stat'
    PermutationVisualizer(stat_obs=stat_obs, stat_dist=stat_dist, stat=stat).draw('perm_histog.svg')