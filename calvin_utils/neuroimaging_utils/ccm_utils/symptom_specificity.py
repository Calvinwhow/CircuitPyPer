import os
import re
from dataclasses import dataclass
import pandas as pd
import numpy as np
from tqdm import tqdm
import os, numpy as np, matplotlib.pyplot as plt, seaborn as sns, matplotlib
from typing import List, Tuple
from itertools import combinations, product
from calvin_utils.plotting_utils.pair_superiority_plot import PairSuperiorityPlot
from calvin_utils.neuroimaging_utils.ccm_utils.permutation_plot import PermutationVisualizer
from calvin_utils.neuroimaging_utils.ccm_utils.correlations import run_pearson, run_spearman
from calvin_utils.neuroimaging_utils.nifti_utils.damage_score_utils import DamageScorer
from calvin_utils.permutation_analysis_utils.statsmodels_palm import CalvinStatsmodelsPalm
from calvin_utils.plotting_utils.simple_box_plot import SimpleBoxPlotWrapper
plt.ion()


print("backend:", matplotlib.get_backend(), "interactive:", plt.isinteractive())


class SpecificityAnalyzer:
    def __init__(self, X, Y, y_labels, correlation='spearman', method='bootstrap', vectorize=True, out_dir=None, absval=True):
        '''
        Args:
            X: Array-like. Can accept DF or np.array. Expects shape (Observations, Variables), where observations=voxels.
            Y: Array-like. Can accept DF or np.array. Expects shape (Observations, Variables), where observations=voxels.
            y_labels: Dict mapping each y_variable's name to a category (i.e. {'nihss7': 'ataxia'})
            correlation: Which correlation (spearman or pearson) to use. Defaults to spearman. 
            method: Which statistical testing method (bootstrap or permutation) to use. Defaults to bootstrap. 
            vectorize: Whether to vectorize the correlations. Defaults to true. Set to false to use gold-standard math. 
        Returns:
        
        '''
        self.correlation = correlation
        self.method = method
        self.vectorize = vectorize
        self.out_dir = out_dir
        self.absval = absval
        self.deltas = None
        self.cols = X.columns
        Y, self.y_labels, self.y_cols = self._clean_y(Y, y_labels)
        self.observations_x, self.n_independent_vars, self.x_arr = self._get_arr_info(X)
        self.observations_y, self.n_dependent_vars, self.y_arr = self._get_arr_info(Y)
        self.y_sort_idx, self.labels, self.n_labels, self.unique_labels, self.label_mapping = self._get_label_info()
        self._validate_inputs()
        
    ### Helpers ###
    def _clean_y(self, df: pd.DataFrame, labels: dict) -> pd.DataFrame:
        # Keep only columns present in y_labels and numeric
        trg_cols = [c for c in df.columns if c in labels.keys()]
        d = df[trg_cols].copy()

        # coerce to numeric and drop non-numeric columns
        for c in list(d.columns):
            if d[c].dtype == 'object':
                print(f"Dropping non-numeric column: {c}")
                d.pop(c)

        # prune y_labels to columns that survived
        labels_dict = {c: labels[c] for c in d.columns if c in labels}
        print(d.columns, labels_dict.keys())
        return d, labels_dict, list(d.columns)

        
    def _get_arr_info(self, arr) -> Tuple[int, int]:
        if isinstance(arr, pd.DataFrame):
            arr = arr.to_numpy()
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
        return arr.shape[0], arr.shape[1], arr
    
    def _get_label_info(self):
        # labels aligned to Y columns
        labels = [self.y_labels[c] for c in self.y_cols]         
        uniq_order = list(dict.fromkeys(labels))                      # preserves order
        mapping = {lab: i for i, lab in enumerate(uniq_order)}
        int_labels = np.array([mapping[lab] for lab in labels], int)  # per-column int label

        # sort index that groups columns by label (preserves intra-label order)
        idx = np.arange(len(self.y_cols))
        y_sort_idx = sorted(idx, key=lambda i: int_labels[i])

        return (np.array(y_sort_idx, int),
                labels,
                len(uniq_order),
                np.array(uniq_order, dtype=object),
                int_labels)

    
    def _validate_inputs(self):
        if self.observations_x != self.observations_y:
            raise ValueError("Row length of x_df and y_df do not match")
    
    ### Resampling ###
    def _resample(self, resample):
        """
        Returns X and Y arrays, permutating or resampling. 
        optionally permuting the outcome data if permutation is True.
        """
        if not resample: 
            return self.x_arr, self.y_arr
        idx = np.arange(self.observations_y)
        if self.method == 'permutation':
            idx = np.random.permutation(idx)
            return self.x_arr, self.y_arr[idx, :]       # permutation only shuffles observations of the Y variables
        elif self.method == 'bootstrap':
            idx = np.random.choice(idx, len(idx), replace=True)
            return self.x_arr[idx, :], self.y_arr[idx, :]      # Bootstrap shuffles (resamples) observations of X and Y.
        else:
            raise ValueError(f"Method {self.method} not supported. Set method='bootstrap' or 'permutation")
        
    
    ### P-value Tools ###
    def _get_p_values(self, arr_obs:np.ndarray, arr_resample:np.ndarray) -> np.ndarray:
        '''
        Obs is shape (n_indep, n_labels) while perm is shape (n_indep, n_labels, perms)
        Args:
            arr_obs: observed AUC array of shape (n_indep_vars, n_ubique_labels) with labels in the same order as self.labels.
            arr_resample: resampled AUC array of shape (n_indep_vars, n_ubique_labels) with labels in the same order as self.labels.
        Return:
            Nothing! Saves plots with distributions, p-values, confidence intervals, and other statistics to your output directory. 
        '''
        n_indep, n_labels = arr_obs.shape
        label_indices = np.arange(len(self.unique_labels))
        pairs = list(combinations(label_indices, r=2))          # [(a,b), ...]
        tasks = product(range(n_indep), pairs)                   # (iv, (a,b)) generator
        p = np.zeros((n_indep, n_labels))
        for iv, (a, b) in tasks: # get combinations of pairs of labels. 
            print(f"----\n{self.cols[iv]}, {self.unique_labels[a]} vs {self.unique_labels[b]}\n----")
            if self.method == 'permutation':
                delta      = float(np.abs(arr_obs[iv, a] - arr_obs[iv, b]))
                i, j = np.triu_indices(arr_resample.shape[1], k=1)
                max_delta = np.max(np.abs(arr_resample[iv, i, :] - arr_resample[iv, j, :]), axis=0).ravel()               
                PermutationVisualizer(
                    stat_obs=delta, stat_dist=max_delta, stat='Delta Correlation', out_dir=self.out_dir, absval=True
                ).draw(f'iv{iv}_deltaAvgCorr={self.unique_labels[a]}-{self.unique_labels[b]}_maxstatperm.svg', verbose=True)
            elif self.method == 'bootstrap':
                PairSuperiorityPlot(
                    arr_resample[iv, a, :].ravel(),
                    arr_resample[iv, b, :].ravel(),
                    self.unique_labels[a], self.unique_labels[b],
                    stat='Average Correlation', out_dir=self.out_dir
                ).draw(f'iv{iv}_avgCorr={self.unique_labels[a]}-{self.unique_labels[b]}_bootstrap.svg', verbose=True)

            else:
                raise ValueError(f"Method {self.method} not supported. Set method='bootstrap' or 'permutation")
        return
    
    ### Statistical Tools ###
    def _get_AUC(self, arr: np.ndarray) -> np.ndarray:
        """
        Average correlations within each label group using the aligned label vector.
        arr shape: (n_indep, n_dep)
        """
        # build mask with labels aligned to arr columns
        # self.labels is aligned to self.y_cols; arr columns must be in same order.
        lab_vec = np.array(self.labels, dtype=object)                 # (n_dep,)
        uniq = np.array(self.unique_labels, dtype=object)             # (n_labels,)
        mask = (uniq[:, None] == lab_vec[None, :]).astype(float)      # (n_labels, n_dep)

        counts = mask.sum(axis=1, keepdims=True)                      # (n_labels,1)
        counts[counts == 0] = 1.0
        # (n_indep, n_dep) @ (n_dep, n_labels) -> (n_indep, n_labels)
        return arr @ mask.T / counts.T


    def _correlate(self, X, Y) -> np.ndarray:
        '''
        Runs multiple X arrays across multiple Y arrays.
        Returns:
            np.ndarray of correlation values, shape (Indepvars, Depvars).
        '''
        if self.correlation=='pearson':
            return run_pearson(X, Y, self.vectorize)
        elif self.correlation=='spearman':
            return run_spearman(X, Y, self.vectorize)
        else:
            raise ValueError(f"correlation={self.correlation} not implemented. Please set correlation='spearman' or 'pearson'")
    
    ### Loop Orchestration and Handling ###
    def _run_correlation(self, resample=False) -> np.ndarray:
        '''Looped handling correlation function'''
        X, Y = self._resample(resample)
        if self.absval:
            return np.abs(self._correlate(X,Y))
        else:
            return self._correlate(X,Y) # returns shape (Indepvars, Depvars)

    def extract_r_values_long(self, CORR: np.ndarray = None) -> pd.DataFrame:
        """
        Return one row per observed X-to-Y correlation.

        Columns
        -------
        measurement
            Column name from X.
        outcome
            Column name from Y.
        label
            Category label for the Y/outcome column.
        r
            Correlation value. Honors ``self.absval`` when CORR is not provided.
        """
        if CORR is None:
            CORR = self._run_correlation(resample=False)

        rows = []
        for iv, measurement in enumerate(self.cols):
            for dep, outcome in enumerate(self.y_cols):
                rows.append({
                    "measurement": measurement,
                    "outcome": outcome,
                    "label": self.labels[dep],
                    "r": float(CORR[iv, dep]),
                })
        return pd.DataFrame(rows)

    def extract_group_comparison_r_values(
        self,
        target_labels,
        other_labels=None,
        *,
        target_name: str = None,
        other_name: str = "Other",
        CORR: np.ndarray = None,
        flatten_single_measurement: bool = True,
    ) -> pd.DataFrame:
        """
        Build a wide dataframe for ``SimpleBoxPlotWrapper`` pairwise plots.

        With multiple measurements, each measurement gets two tuple-named
        columns: ``(measurement, target_name)`` and
        ``(measurement, other_name)``. With one measurement, columns are
        flattened to ``target_name`` and ``other_name`` by default.

        Parameters
        ----------
        target_labels : str or list-like
            Y-label category/categories to compare, e.g. ``"motor"``.
        other_labels : str or list-like, optional
            Categories to combine into the comparator. If omitted, all labels
            not in ``target_labels`` are pooled.
        target_name : str, optional
            Display/column name for the target group. Defaults to joined
            target label names.
        other_name : str, optional
            Display/column name for the pooled comparator.
        CORR : np.ndarray, optional
            Precomputed correlation array of shape ``(n_indep, n_dep)``.
        flatten_single_measurement : bool, optional
            If True and there is only one X/measurement column, return simple
            columns instead of a single top-level measurement MultiIndex.
        """
        def _as_set(value):
            if value is None:
                return None
            if isinstance(value, str):
                return {value}
            return set(value)

        target_set = _as_set(target_labels)
        other_set = _as_set(other_labels)
        if target_name is None:
            target_name = "+".join(str(label) for label in target_set)

        r_long = self.extract_r_values_long(CORR=CORR)
        known_labels = set(r_long["label"].unique())
        missing_target = target_set - known_labels
        if missing_target:
            raise ValueError(f"target_labels not found in y_labels: {sorted(missing_target)}")
        if other_set is not None:
            missing_other = other_set - known_labels
            if missing_other:
                raise ValueError(f"other_labels not found in y_labels: {sorted(missing_other)}")
        else:
            other_set = known_labels - target_set

        if not other_set:
            raise ValueError("No labels available for the comparison group.")

        out = {}
        max_len = 0
        flatten_columns = flatten_single_measurement and len(self.cols) == 1
        for measurement in self.cols:
            m = r_long["measurement"] == measurement
            target_vals = r_long.loc[m & r_long["label"].isin(target_set), "r"].reset_index(drop=True)
            other_vals = r_long.loc[m & r_long["label"].isin(other_set), "r"].reset_index(drop=True)
            if flatten_columns:
                out[target_name] = target_vals
                out[other_name] = other_vals
            else:
                out[(measurement, target_name)] = target_vals
                out[(measurement, other_name)] = other_vals
            max_len = max(max_len, len(target_vals), len(other_vals))

        for key, values in out.items():
            out[key] = values.reindex(range(max_len))
        return pd.DataFrame(out)
        
    def _run_loop(self, n_resamples):
        if n_resamples < 1:
            print("No resamples (permutations or bootstraps) requested.")
            return
        AUC_p = np.zeros((self.n_independent_vars, self.n_labels, n_resamples)) # shape (indep_vars, n_labels, perms)
        for i in tqdm(range(n_resamples), desc=f'running {self.method}'):
            CORR = self._run_correlation(resample=True)  # shape (n_indep, n_dep)
            AUC_p[:, :, i] = self._get_AUC(CORR)         # shape (n_indep, n_label, 1)
        return AUC_p
    
    ### Sorting Utils ###
    def _sort_arr(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, float)
        a = a[np.argsort(-np.abs(a))]   # sort by |r| descending (real values preserved)
        out = []
        for i, v in enumerate(a):   # radial placement: largest near center, then alternate sides
            if i % 2 == 0:
                out.append(v)
            else:
                out.insert(0, v)
        return np.array(out)

    def _sort_corrs(self, arr: np.ndarray, sort_within_labels: bool = True) -> np.ndarray:
        """
        Sort correlations to group by label (aligned via y_sort_idx),
        then radial sort within each label group.
        """
        arr = arr[self.y_sort_idx]                # group by label using aligned index
        L = np.array(self.labels)[self.y_sort_idx]
        if not sort_within_labels:
            return arr
        for label in self.unique_labels:
            subidx = (L == label)
            arr[subidx] = self._sort_arr(arr[subidx])
        return arr

    
    ### Plotting Utils ###
    def _compute_group_meta(self):
        import numpy as np
        labels_sorted = np.array(self.labels)[self.y_sort_idx]
        group_order   = list(self.unique_labels)
        group_sizes   = [int(np.sum(labels_sorted == lab)) for lab in group_order]
        group_edges   = np.cumsum(group_sizes)
        group_starts  = np.concatenate(([0], group_edges[:-1]))
        group_centers = group_starts + np.array(group_sizes) / 2.0
        return labels_sorted, group_order, group_sizes, group_edges, group_starts, group_centers

    def _palette(self, group_order):
        import matplotlib.pyplot as plt
        base = plt.get_cmap('tab20').colors
        return {lab: base[i % len(base)] for i, lab in enumerate(group_order)}

    def _lighten(self, rgb, factor=0.65):
        import numpy as np
        from matplotlib import colors as mcolors
        r,g,b = mcolors.to_rgb(rgb)
        return (1 - factor) * np.array([r,g,b]) + factor * np.array([1,1,1])

    def _darken(self, rgb, factor=0.85):
        import numpy as np
        from matplotlib import colors as mcolors
        r,g,b = mcolors.to_rgb(rgb)
        return tuple(np.clip(factor * np.array([r,g,b]), 0, 1))

    def _draw_label_kde(
        self, ax, start, end, r_vals, line_color, x_fine_global,
        *, bw_scale=1.2, fill_alpha=0.14
    ):
        """
        KDE-like smooth curve for bars start..end-1 using kernel regression on x-index.
        Adds zero 'ghost' points at group edges so tails round to 0.
        Evaluates on the provided global grid for clean overlays.
        """
        import numpy as np
        import matplotlib.patheffects as pe

        if end <= start: 
            return

        # sample points (bar centers) for this label
        x_seg = np.arange(start, end, dtype=float)
        y_seg = np.asarray(r_vals[start:end], float).ravel()

        # ghost zeros at half-bars for rounded tails
        L, R = start - 0.5, (end - 1) + 0.5
        x_aug = np.r_[L, x_seg, R]
        y_aug = np.r_[0.0, y_seg, 0.0]

        # bandwidth (Silverman)
        def _bw(z):
            z = np.asarray(z, float).ravel()
            n = z.size
            if n <= 1: return 0.5
            iqr = np.subtract(*np.percentile(z, [75, 25]))
            s   = np.std(z, ddof=1)
            sigma = min(s, iqr/1.34) if (s>0 and iqr>0) else max(s, iqr, 1.0)
            return max(0.9 * sigma * n**(-1/5), 0.5)
        bw = _bw(x_seg) * float(bw_scale)

        # Nadaraya–Watson kernel regression (Gaussian) on x_fine_global
        X = x_aug[:, None]                                   # (n,1)
        Z = (x_fine_global[None, :] - X) / bw                # (n,m)
        W = np.exp(-0.5 * Z * Z)                             # (n,m)
        y_fit = (W * y_aug[:, None]).sum(0) / (W.sum(0) + 1e-12)

        # mask = (x_fine_global >= L) & (x_fine_global <= R)
        y_fit[x_fine_global < L] = np.nan
        y_fit[x_fine_global > R] = np.nan
        
        # draw line across the full axis; fill only within this block
        ax.plot(
            x_fine_global, y_fit, color=line_color, linewidth=3.0, alpha=0.95, zorder=4,
            path_effects=[pe.Stroke(linewidth=4.0, foreground='white', alpha=0.35), pe.Normal()]
        )

    def _draw_global_kde(
        self, ax, r_vals, line_color, x_fine_global,
        *, bw_scale=1.2
    ):
        """
        Smooth one continuous curve across every bar using Gaussian kernel
        regression on x-index.
        """
        self._draw_label_kde(
            ax=ax,
            start=0,
            end=len(r_vals),
            r_vals=r_vals,
            line_color=line_color,
            x_fine_global=x_fine_global,
            bw_scale=bw_scale,
            fill_alpha=0,
        )

    def _plot(
        self,
        CORR: np.ndarray,
        scale: float = 1,
        sort_within_labels: bool = True,
        smooth_by_label: bool = True,
        absval: bool = False
    ):
        BLACK, GREY = '#211D1E', '#8E8E8E'
        sns.set_theme(style="white", context="talk")

        # group meta / palette
        labels_sorted, group_order, group_sizes, group_edges, group_starts, group_centers = self._compute_group_meta()
        color_map = self._palette(group_order)

        _, n_dep = CORR.shape
        x_bars = np.arange(n_dep)
        # global fine grid for smooth overlays
        upsample = 120
        x_fine_global = np.linspace(-0.5, n_dep - 0.5, upsample * n_dep + 1)

        for iv in range(self.n_independent_vars):
            if absval:
                CORR = np.abs(CORR)
            r_vals = self._sort_corrs(
                CORR[iv, :].astype(float),
                sort_within_labels=sort_within_labels,
            ).ravel()

            # bars (lighter tints)
            bar_colors = [self._lighten(color_map[lab], factor=0.65) for lab in labels_sorted]
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(x_bars, r_vals, width=0.9, color=bar_colors, edgecolor="white", linewidth=1.3, zorder=2)

            if not smooth_by_label:
                self._draw_global_kde(
                    ax,
                    r_vals,
                    line_color=BLACK,
                    x_fine_global=x_fine_global,
                    bw_scale=scale,
                )

            # per-label KDE overlays (darker line)
            for lab, start, size in zip(group_order, group_starts, group_sizes):
                if size <= 0: 
                    continue
                end = start + size
                if smooth_by_label:
                    line_color = self._darken(color_map[lab], factor=0.85)
                    self._draw_label_kde(ax, start, end, r_vals, line_color, x_fine_global,
                                        bw_scale=scale, fill_alpha=0)

                # average R value for this category
                avg_r = np.nanmean(r_vals[start:end])
                ax.text(start + size/2 - 0.5, avg_r, f"{avg_r:.2f}",
                        ha='center', va='bottom' if avg_r >= 0 else 'top',
                        fontsize=12, color=BLACK, fontweight='bold')

            # separators
            for edge in group_edges[:-1]:
                ax.axvline(edge - 0.5, ls='--', color=GREY, lw=1.6, zorder=1)

            # labels centered on groups
            ax.set_xticks(group_centers-0.5)
            ax.set_xticklabels(group_order, fontsize=16, color=BLACK)

            # cosmetics
            ax.set_ylabel(f'Correlation ({self.correlation})', fontsize=20, color=BLACK)
            ax.set_title(f'{self.cols[iv]}', fontsize=22, color=BLACK)
            ax.axhline(0, color=GREY, lw=1.6)
            ax.grid(axis='y', alpha=0.15)
            for s in ax.spines.values():
                s.set_linewidth(2); s.set_color(BLACK)
            ax.set_xlim(-0.5, n_dep - 0.5)

            ymax = float(np.nanmax(CORR)); ymin = float(np.nanmin(CORR))
            pad = 0.06 * (ymax - ymin if ymax > ymin else 1.0)
            ax.set_ylim(0 if self.absval else ymin - pad, ymax + pad)

            sns.despine(ax=ax)
            fig.tight_layout()
            if self.out_dir:
                os.makedirs(self.out_dir, exist_ok=True)
                safe_col = re.sub(r"[^0-9A-Za-z._-]+", "_", str(self.cols[iv])).strip("_")
                fig.savefig(os.path.join(self.out_dir, f'specificity_{safe_col}.svg'), format='svg')
            plt.show()
            plt.close(fig)

    ### Orchestrator ### 
    def run(
        self,
        n_resamples=1000,
        scale=0.85,
        sort_within_labels: bool = True,
        smooth_by_label: bool = True,
    ):
        
        CORR  = self._run_correlation()                 # 1 - get correlation of each col of X to every col of Y. Sum R values within the columns defined by y_labels. 
        AUC   = self._get_AUC(CORR)     
        AUC_p = self._run_loop(n_resamples)             # 2 - Repeat step 1 with bootstrapped or permuted data 1000 times. 
        p     = self._get_p_values(AUC, AUC_p)          # 3 - compare observed AUC and permuted AUC
        self._plot(
            CORR,
            scale,
            sort_within_labels=sort_within_labels,
            smooth_by_label=smooth_by_label,
        )                                               # 4 - For each col of X, plot the R-values from step 1, coloured/grouped by y_label. 


@dataclass(frozen=True)
class SpecificityAnalysisResult:
    """Tables and analyzer produced by :class:`NetworkSpecificityAnalysis`."""

    prepared_data: pd.DataFrame
    damage_scores: pd.DataFrame
    observed_correlations: pd.DataFrame
    group_comparison: pd.DataFrame | None
    analyzer: SpecificityAnalyzer


class NetworkSpecificityAnalysis:
    """Run the notebook network-specificity workflow from a script config.

    Spreadsheet filtering and neuroimaging import are delegated to
    :class:`CalvinStatsmodelsPalm`; this class only orchestrates damage scoring,
    specificity resampling, tabular outputs, and the optional comparison plot.
    """

    def __init__(
        self,
        *,
        input_path,
        out_dir,
        file_column,
        target_maps_directory,
        target_map_pattern,
        mask_path,
        label_dict,
        sheet=None,
        required_columns=None,
        keep_rows=None,
        drop_rows=None,
        path_replacements=None,
        damage_metric="cosine",
        correlation="spearman",
        method="permutation",
        vectorize=False,
        absval=False,
        n_resamples=1000,
        sort_within_labels=False,
        smooth_by_label=True,
        fillna_value=0,
        target_labels=None,
        other_labels=None,
        target_name=None,
        other_name="Other",
        draw_comparison_plot=True,
        comparison_dataset_name=" ",
        comparison_xlabel=" ",
        comparison_ylabel=None,
        comparison_figsize=(4, 6),
    ):
        self.input_path = input_path
        self.out_dir = out_dir
        self.file_column = file_column
        self.target_maps_directory = target_maps_directory
        self.target_map_pattern = target_map_pattern
        self.mask_path = mask_path
        self.label_dict = dict(label_dict)
        self.sheet = sheet
        configured_required = (
            [required_columns]
            if isinstance(required_columns, str)
            else list(required_columns or [])
        )
        self.required_columns = list(
            dict.fromkeys([file_column, *configured_required])
        )
        self.keep_rows = list(keep_rows or [])
        self.drop_rows = list(drop_rows or [])
        self.path_replacements = list(path_replacements or [])
        self.damage_metric = damage_metric
        self.correlation = correlation
        self.method = method
        self.vectorize = vectorize
        self.absval = absval
        self.n_resamples = n_resamples
        self.sort_within_labels = sort_within_labels
        self.smooth_by_label = smooth_by_label
        self.fillna_value = fillna_value
        self.target_labels = target_labels
        self.other_labels = other_labels
        self.target_name = target_name
        self.other_name = other_name
        self.draw_comparison_plot = draw_comparison_plot
        self.comparison_dataset_name = comparison_dataset_name
        self.comparison_xlabel = comparison_xlabel
        self.comparison_ylabel = comparison_ylabel
        self.comparison_figsize = comparison_figsize
        self.analysis_dir = os.path.join(str(out_dir), method)

    def prepare_data(self):
        """Return the filtered spreadsheet and subject-by-map score table."""
        os.makedirs(self.analysis_dir, exist_ok=True)
        palm = CalvinStatsmodelsPalm(
            input_csv_path=self.input_path,
            output_dir=self.out_dir,
            sheet=self.sheet,
        )
        replacements = (
            {self.file_column: self.path_replacements}
            if self.path_replacements
            else None
        )
        data_df = palm.prepare_spreadsheet_data(
            required_columns=self.required_columns,
            keep_rows=self.keep_rows,
            drop_rows=self.drop_rows,
            path_replacements=replacements,
        )

        subject_df = palm.import_neuroimaging_data(
            file_column=self.file_column,
            mask_path=self.mask_path,
            process_special_values=False,
        )
        target_df = palm.import_neuroimaging_data(
            import_path=self.target_maps_directory,
            file_pattern=self.target_map_pattern,
            mask_path=self.mask_path,
            process_special_values=False,
        )

        damage_scorer = DamageScorer(self.mask_path, subject_df, target_df)
        damage_df = damage_scorer.calculate_damage_scores(self.damage_metric)
        damage_df = damage_scorer.sort_dataframes_by_index(damage_df)

        if len(damage_df) != len(data_df):
            raise ValueError(
                "Imported subject images and prepared spreadsheet rows do not match: "
                f"{len(damage_df)} images for {len(data_df)} rows."
            )

        data_df = data_df.reset_index(drop=True).fillna(self.fillna_value)
        damage_df = damage_df.reset_index(drop=True).fillna(self.fillna_value)
        data_df.to_csv(
            os.path.join(self.analysis_dir, "prepared_spreadsheet.csv"), index=False
        )
        damage_output = damage_df.copy()
        damage_output.insert(0, "subject", data_df[self.file_column].to_numpy())
        damage_output.to_csv(
            os.path.join(self.analysis_dir, "damage_scores.csv"), index=False
        )
        return data_df, damage_df

    def run(self):
        """Execute specificity resampling and return all generated tables."""
        data_df, damage_df = self.prepare_data()
        analyzer = SpecificityAnalyzer(
            X=damage_df,
            Y=data_df,
            y_labels=self.label_dict,
            correlation=self.correlation,
            method=self.method,
            vectorize=self.vectorize,
            out_dir=self.analysis_dir,
            absval=self.absval,
        )
        analyzer.run(
            n_resamples=self.n_resamples,
            sort_within_labels=self.sort_within_labels,
            smooth_by_label=self.smooth_by_label,
        )

        observed = analyzer.extract_r_values_long()
        observed.to_csv(
            os.path.join(self.analysis_dir, "observed_correlations.csv"), index=False
        )

        comparison = None
        if self.target_labels is not None:
            comparison = analyzer.extract_group_comparison_r_values(
                target_labels=self.target_labels,
                other_labels=self.other_labels,
                target_name=self.target_name,
                other_name=self.other_name,
            )
            comparison.to_csv(
                os.path.join(self.analysis_dir, "group_comparison_r_values.csv"),
                index=False,
            )
            if self.draw_comparison_plot:
                target_name = self.target_name
                if target_name is None:
                    labels = (
                        [self.target_labels]
                        if isinstance(self.target_labels, str)
                        else list(self.target_labels)
                    )
                    target_name = "+".join(str(label) for label in labels)
                if len(analyzer.cols) == 1:
                    plot_columns = [(target_name, self.other_name)]
                    group_labels = None
                else:
                    plot_columns = [
                        (
                            (measurement, target_name),
                            (measurement, self.other_name),
                        )
                        for measurement in analyzer.cols
                    ]
                    group_labels = [str(measurement) for measurement in analyzer.cols]
                SimpleBoxPlotWrapper(comparison).plot(
                    columns=plot_columns,
                    group_labels=group_labels,
                    pair_names=[target_name, self.other_name],
                    dataset_name=self.comparison_dataset_name,
                    xlabel=self.comparison_xlabel,
                    ylabel=(
                        self.comparison_ylabel
                        or f"Correlation ({self.correlation.title()})"
                    ),
                    figsize=self.comparison_figsize,
                    out_dir=self.analysis_dir,
                )

        return SpecificityAnalysisResult(
            prepared_data=data_df,
            damage_scores=damage_df,
            observed_correlations=observed,
            group_comparison=comparison,
            analyzer=analyzer,
        )


def get_column_labels(df):
    print("Please copy and paste this into the following cell to edit it: \n")
    print("label_dict = {")
    for c in df.columns:
        print(f"'{c}': '',")
    print("}")
    
if __name__=="main":
    X = [1,2,3]
    Y = [1,2,3]
    y_labels = {}
    correlation='spearman'
    out_dir = None
    SpecificityAnalyzer(X, Y, y_labels, correlation, out_dir).run()
