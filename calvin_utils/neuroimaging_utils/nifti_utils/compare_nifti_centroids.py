from glob import glob
from pathlib import Path
import pprint
import os
import pandas as pd
import nibabel as nib
import numpy as np
from scipy.stats import ttest_ind, ttest_rel
from sklearn.cluster import KMeans
from calvin_utils.plotting_utils.simple_box_plot import SimpleBoxPlotPair
from calvin_utils.plotting_utils.simple_dynamite_plot import SimpleDynamitePlotPair
from calvin_utils.plotting_utils.simple_cdf_plot import SimpleCdfPlotPair
from calvin_utils.plotting_utils.simple_line_plot import SimpleLinePlot


class NiftiCentroidStats:
    """
    Compare centroids of two sets of NIfTI files.

    Parameters
    ----------
    grp1_glob, grp2_glob : str
        Recursive glob patterns for the two groups (e.g. '/data/A/**/*.nii*').
    n_centroids : int, default 1
        How many centroids to extract **per file** (k‑means if > 1).
    mirror : bool, default True
        If n_centroids > 1 and *mirror* is True, x‑coordinates of every
        centroid‑bucket are forced to share the same sign (midline mirroring)
        and **all buckets are then concatenated**, so the test is run on a
        *single* merged centroid array per group.

    Public methods
    --------------
    run()                 → summary_xyz, ttests_xyz
    compare_norms()       → summary_norm, ttests_norm
    """

    # ------------------------------------------------------------------ #
    # constructor
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        grp1_glob: str | None = None,
        grp2_glob: str | None = None,
        *,
        csv_path: str | None = None,
        subject_col: str | None = None,
        nifti_col_a: str | None = None,
        nifti_col_b: str | None = None,
        n_centroids: int = 1,
        mirror: bool = True,
        out_dir: str | None = None,
        plot_type: str | None = None,
        group_labels: list[str] | None = None,
    ):
        if n_centroids < 1:
            raise ValueError("n_centroids must be ≥ 1")
        self.grp1_glob, self.grp2_glob = grp1_glob, grp2_glob
        self.n_centroids = n_centroids
        self.mirror = mirror and n_centroids > 1
        self.csv_path = csv_path
        self.subject_col = subject_col
        self.nifti_col_a = nifti_col_a
        self.nifti_col_b = nifti_col_b
        self.out_dir = out_dir
        if plot_type is not None and plot_type not in {"box", "dynamite", "cdf", "line"}:
            raise ValueError("plot_type must be one of: 'box', 'dynamite', 'cdf', 'line', or None")
        self.plot_type = plot_type
        if group_labels is None:
            self.group_labels = ["group1", "group2"]
        else:
            if len(group_labels) != 2:
                raise ValueError("group_labels must have exactly 2 entries.")
            self.group_labels = list(group_labels)
        self._csv_mode = csv_path is not None
        if self._csv_mode:
            self.df = pd.read_csv(self.csv_path)
            self._set_csv_columns()
            self.grp1_files = []
            self.grp2_files = []
        else:
            if not self.grp1_glob or not self.grp2_glob:
                raise ValueError("Provide grp1_glob and grp2_glob when csv_path is not set.")
            self.grp1_files = self._keep_nii(glob(self.grp1_glob, recursive=True))
            self.grp2_files = self._keep_nii(glob(self.grp2_glob, recursive=True))

    def _set_csv_columns(self):
        cols = list(self.df.columns)
        if self.nifti_col_a and self.nifti_col_b:
            if self.nifti_col_a not in cols or self.nifti_col_b not in cols:
                raise ValueError("csv_col_a/csv_col_b not found in CSV")
            return
        if len(cols) == 2:
            self.nifti_col_a, self.nifti_col_b = cols
            return
        if self.subject_col and self.subject_col in cols:
            remaining = [c for c in cols if c != self.subject_col]
            if len(remaining) == 2:
                self.nifti_col_a, self.nifti_col_b = remaining
                return
        if "subject" in cols:
            remaining = [c for c in cols if c != "subject"]
            if len(remaining) == 2:
                self.nifti_col_a, self.nifti_col_b = remaining
                return
        raise ValueError("Could not infer CSV path columns. Provide nifti_col_a and nifti_col_b.")

    # ------------------------------------------------------------------ #
    # public orchestrators
    # ------------------------------------------------------------------ #
    def run(self, name="Each coordinate (XYZ)", paired=False):
        """Welch t‑tests on X, Y, Z."""
        if self._csv_mode:
            xyz1, xyz2, ids = self._get_paired_centroids_from_csv()
            self._save_centroids_csv_paired(xyz1, xyz2, ids)
            summary = self._xyz_summary(xyz1, xyz2)
            ttests = self._xyz_ttests(xyz1, xyz2, paired=True)
            self._show_results(summary, ttests, name + "  (paired CSV)")
            self._plot_xyz_pair(xyz1, xyz2, name=name, tag="paired_csv")
            return summary, ttests
        if self.mirror:
            xyz1, xyz2 = self._get_merged_mirrored_both()
            self._save_centroids_csv_groups(xyz1, xyz2)
            summary = self._xyz_summary(xyz1, xyz2)
            ttests  = self._xyz_ttests(xyz1, xyz2)
            self._show_results(summary, ttests, name + "  (mirrored & merged)")
            self._plot_xyz_pair(xyz1, xyz2, name=name, tag="mirrored_merged")
            return summary, ttests

        # no mirroring → keep buckets separate
        grp1, grp2 = self._get_centroid_sets_both()
        self._save_centroids_csv_groups(np.vstack(grp1), np.vstack(grp2))
        summary = {f"centroid_{i}": self._xyz_summary(g1, g2)
                   for i, (g1, g2) in enumerate(zip(grp1, grp2))}
        ttests  = {f"centroid_{i}": self._xyz_ttests(g1, g2, paired)
                   for i, (g1, g2) in enumerate(zip(grp1, grp2))}
        self._show_results(summary, ttests, name)
        for i, (g1, g2) in enumerate(zip(grp1, grp2)):
            self._plot_xyz_pair(g1, g2, name=name, tag=f"centroid_{i}")
        return summary, ttests

    def compare_norms(self, name="Euclidean norm of XYZ", paired=False):
        """Welch t‑test on ||XYZ||."""
        if self._csv_mode:
            xyz1, xyz2, ids = self._get_paired_centroids_from_csv()
            self._save_centroids_csv_paired(xyz1, xyz2, ids)
            n1, n2 = np.linalg.norm(xyz1, axis=1), np.linalg.norm(xyz2, axis=1)
            summary, ttests = self._norm_stats_and_test(n1, n2, paired=True)
            self._show_results(summary, ttests, name + "  (paired CSV)")
            self._plot_norm_pair(n1, n2, name=name, tag="paired_csv")
            return summary, ttests
        if self.mirror:
            xyz1, xyz2 = self._get_merged_mirrored_both()
            self._save_centroids_csv_groups(xyz1, xyz2)
            n1, n2 = np.linalg.norm(xyz1, axis=1), np.linalg.norm(xyz2, axis=1)
            summary, ttests = self._norm_stats_and_test(n1, n2, paired)
            self._show_results(summary, ttests, name + "  (mirrored & merged)")
            self._plot_norm_pair(n1, n2, name=name, tag="mirrored_merged")
            return summary, ttests

        grp1, grp2 = self._get_centroid_sets_both()
        self._save_centroids_csv_groups(np.vstack(grp1), np.vstack(grp2))
        summary, ttests = {}, {}
        for i, (g1, g2) in enumerate(zip(grp1, grp2)):
            n1, n2 = np.linalg.norm(g1, axis=1), np.linalg.norm(g2, axis=1)
            s, t = self._norm_stats_and_test(n1, n2, paired)
            summary[f"centroid_{i}"], ttests[f"centroid_{i}"] = s['merged'], t['norm']
            self._plot_norm_pair(n1, n2, name=name, tag=f"centroid_{i}")
        self._show_results(summary, ttests, name)
        return summary, ttests

    # ------------------------------------------------------------------ #
    # helpers – file handling / centroid extraction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _keep_nii(paths):
        return [p for p in paths if Path(p).suffix in {'.nii', '.gz'}]

    @staticmethod
    def _voxel_to_world(vox_xyz, affine):
        return (affine @ np.append(vox_xyz, 1.0))[:3]

    def _centroids_of_file(self, f):
        """Return (n_centroids,3) world‑space centroids for one file."""
        img   = nib.load(f)
        idx   = np.transpose(np.nonzero(img.get_fdata()))
        if self.n_centroids == 1:
            ctrs = [idx.mean(0)]
        else:
            kmeans = KMeans(n_clusters=self.n_centroids, n_init='auto').fit(idx)
            ctrs = [idx[kmeans.labels_ == k].mean(0) for k in range(self.n_centroids)]
            ctrs.sort(key=lambda c: c[0])        # left → right order
        return np.vstack([self._voxel_to_world(c, img.affine) for c in ctrs])

    @staticmethod
    def _centroid_from_nifti_path(path):
        img = nib.load(path)
        data = img.get_fdata()
        mask = data != 0
        if not np.any(mask):
            return np.array([np.nan, np.nan, np.nan], dtype=float)
        coords = np.column_stack(np.nonzero(mask))
        hom = np.c_[coords, np.ones(len(coords))]
        world = (img.affine @ hom.T).T[:, :3]
        return world.mean(axis=0)
    
    @staticmethod
    def _centroid_from_csv(path, c1='X', c2='Y', c3='Z'):
        '''Return centroid from CSV with cols: x, y, z'''
        df = pd.read_csv(path, usecols=[c1, c2, c3])
        return [df.to_numpy()]          # shape (n_samples, 3)

    # bucket‑by‑centroid arrays
    def _centroid_sets_for_group(self, files):
        buckets = [[] for _ in range(self.n_centroids)]
        for f in files:
            ctrs = self._centroids_of_file(f)
            for i, c in enumerate(ctrs):
                buckets[i].append(c)
        return [np.vstack(b) for b in buckets]

    def _get_centroid_sets_both(self):
        '''Gets centroid directly from nifti or extracts from a CSV'''
        if (os.path.splitext(self.grp1_glob)[1] == os.path.splitext(self.grp2_glob)[1]) and (os.path.splitext(self.grp1_glob)[1] == ".csv"):
            return (self._centroid_from_csv(self.grp1_glob), 
                    self._centroid_from_csv(self.grp2_glob))
        else:
            return (self._centroid_sets_for_group(self.grp1_files),
                    self._centroid_sets_for_group(self.grp2_files))

    def _get_paired_centroids_from_csv(self):
        df = self.df.dropna(subset=[self.nifti_col_a, self.nifti_col_b])
        ids = df[self.subject_col].tolist() if self.subject_col and self.subject_col in df.columns else df.index.tolist()
        c1 = df[self.nifti_col_a].apply(self._centroid_from_nifti_path)
        c2 = df[self.nifti_col_b].apply(self._centroid_from_nifti_path)
        xyz1 = np.vstack(c1.to_numpy())
        xyz2 = np.vstack(c2.to_numpy())
        valid = (~np.isnan(xyz1).any(axis=1)) & (~np.isnan(xyz2).any(axis=1))
        ids = [i for i, ok in zip(ids, valid) if ok]
        return xyz1[valid], xyz2[valid], ids

    def _save_centroids_csv_paired(self, xyz1, xyz2, ids):
        if self.out_dir is None:
            return
        os.makedirs(self.out_dir, exist_ok=True)
        df = pd.DataFrame({
            "subject": ids,
            "x_a": xyz1[:, 0],
            "y_a": xyz1[:, 1],
            "z_a": xyz1[:, 2],
            "l2_a": np.linalg.norm(xyz1, axis=1),
            "x_b": xyz2[:, 0],
            "y_b": xyz2[:, 1],
            "z_b": xyz2[:, 2],
            "l2_b": np.linalg.norm(xyz2, axis=1),
        })
        df.to_csv(os.path.join(self.out_dir, "centroids_paired.csv"), index=False)

    def _save_centroids_csv_groups(self, xyz1, xyz2):
        if self.out_dir is None:
            return
        os.makedirs(self.out_dir, exist_ok=True)
        df1 = pd.DataFrame({
            "group": "group1",
            "x": xyz1[:, 0],
            "y": xyz1[:, 1],
            "z": xyz1[:, 2],
            "l2": np.linalg.norm(xyz1, axis=1),
        })
        df2 = pd.DataFrame({
            "group": "group2",
            "x": xyz2[:, 0],
            "y": xyz2[:, 1],
            "z": xyz2[:, 2],
            "l2": np.linalg.norm(xyz2, axis=1),
        })
        df = pd.concat([df1, df2], ignore_index=True)
        df.to_csv(os.path.join(self.out_dir, "centroids_groups.csv"), index=False)

    # ------------------------------------------------------------------ #
    # mirroring / merging
    # ------------------------------------------------------------------ #
    @staticmethod
    def _mirror_bucket(arr):
        """Flip x so the bucket's mean x becomes non‑negative."""
        if np.mean(arr[:, 0]) < 0:
            arr = arr.copy()
            arr[:, 0] *= -1
        return arr

    def _merged_mirrored(self, bucket_list):
        mirrored = [self._mirror_bucket(b) for b in bucket_list]
        return np.vstack(mirrored)

    def _get_merged_mirrored_both(self):
        g1_buckets, g2_buckets = self._get_centroid_sets_both()
        return (self._merged_mirrored(g1_buckets),
                self._merged_mirrored(g2_buckets))

    # ------------------------------------------------------------------ #
    # stats helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _xyz_summary(xyz1, xyz2):
        return {
            'group1': {'n': len(xyz1), 'mean': xyz1.mean(0), 'std': xyz1.std(0, ddof=1)},
            'group2': {'n': len(xyz2), 'mean': xyz2.mean(0), 'std': xyz2.std(0, ddof=1)},
        }

    @staticmethod
    def _xyz_ttests(xyz1, xyz2, paired=False):
        if min(len(xyz1), len(xyz2)) < 2:
            return {ax: {'t': np.nan, 'p': np.nan} for ax in 'xyz'}
        
        if paired:
            return {ax: {'t': res.statistic, 'p': res.pvalue}
                for ax, res in zip('xyz',
                                   (ttest_rel(xyz1[:, i], xyz2[:, i])
                                    for i in range(3)))}
        
        return {ax: {'t': res.statistic, 'p': res.pvalue}
                for ax, res in zip('xyz',
                                   (ttest_ind(xyz1[:, i], xyz2[:, i], equal_var=False)
                                    for i in range(3)))}

    @staticmethod
    def _norm_stats_and_test(n1, n2, paired=False):
        summary = {
            'merged': {
                'group1': {'n': len(n1), 'mean': n1.mean(), 'std': n1.std(ddof=100)},
                'group2': {'n': len(n2), 'mean': n2.mean(), 'std': n2.std(ddof=100)},
            }
        }
        if paired:
            t, p = (np.nan, np.nan) if min(len(n1), len(n2)) < 2 else ttest_rel(n1, n2)
        else:
            t, p = (np.nan, np.nan) if min(len(n1), len(n2)) < 2 else ttest_ind(n1, n2, equal_var=True)
        ttests = {'norm': {'t': t, 'p': p}}
        return summary, ttests

    # ------------------------------------------------------------------ #
    # plotting helpers
    # ------------------------------------------------------------------ #
    def _plot_xyz_pair(self, xyz1, xyz2, name: str, tag: str):
        if self.plot_type is None:
            return
        df = self._xyz_dataframe_pair(xyz1, xyz2)
        title = f"{name} ({tag})"
        if self.plot_type == "cdf":
            plotter = SimpleCdfPlotPair(
                df,
                group_col="group",
                category_col="axis",
                value_col="value",
                dataset_name=title,
                out_dir=self.out_dir,
                xlabel="Axis",
                ylabel="Cumulative Density",
            )
            plotter.run()
            return
        if self.plot_type == "line":
            plotter = SimpleLinePlot(
                df,
                x_col="axis",
                y_col="value",
                hue_col="group",
                dataset_name=title,
                out_dir=self.out_dir,
                xlabel="Axis",
                ylabel="Coordinate",
                sort_x=False,
            )
            plotter.run()
            return
        plotter_cls = SimpleDynamitePlotPair if self.plot_type == "dynamite" else SimpleBoxPlotPair
        plotter = plotter_cls(
            df,
            group_col="group",
            category_col="axis",
            value_col="value",
            dataset_name=title,
            out_dir=self.out_dir,
            xlabel="Axis",
            ylabel="Coordinate",
            order=["X", "Y", "Z"],
            hue_order=self.group_labels,
        )
        plotter.run()

    def _plot_norm_pair(self, n1, n2, name: str, tag: str):
        if self.plot_type is None:
            return
        df = self._norm_dataframe_pair(n1, n2)
        title = f"{name} ({tag})"
        if self.plot_type == "cdf":
            plotter = SimpleCdfPlotPair(
                df,
                group_col="group",
                category_col=None,
                value_col="value",
                dataset_name=title,
                out_dir=self.out_dir,
                xlabel="Group",
                ylabel="Cumulative Density",
            )
            plotter.run()
            return
        if self.plot_type == "line":
            plotter = SimpleLinePlot(
                df,
                x_col="index",
                y_col="value",
                hue_col="group",
                dataset_name=title,
                out_dir=self.out_dir,
                xlabel="Index",
                ylabel="Norm",
            )
            plotter.run()
            return
        plotter_cls = SimpleDynamitePlotPair if self.plot_type == "dynamite" else SimpleBoxPlotPair
        plotter = plotter_cls(
            df,
            group_col="group",
            category_col=None,
            value_col="value",
            dataset_name=title,
            out_dir=self.out_dir,
            xlabel="Group",
            ylabel="Norm",
            hue_order=self.group_labels,
        )
        plotter.run()

    def _xyz_dataframe_pair(self, xyz1, xyz2):
        rows = []
        for axis, idx in zip(["X", "Y", "Z"], range(3)):
            rows += [{"group": self.group_labels[0], "axis": axis, "value": v} for v in xyz1[:, idx]]
            rows += [{"group": self.group_labels[1], "axis": axis, "value": v} for v in xyz2[:, idx]]
        return pd.DataFrame(rows)

    def _norm_dataframe_pair(self, n1, n2):
        rows = []
        for i, v in enumerate(n1):
            rows.append({"group": self.group_labels[0], "value": float(v), "index": i})
        for i, v in enumerate(n2):
            rows.append({"group": self.group_labels[1], "value": float(v), "index": i})
        return pd.DataFrame(rows)

    # pretty printing
    @staticmethod
    def _show_results(summary, tests, name):
        print(f"\n----- {name} -----")
        print("Summary:")
        pprint.pprint(summary, compact=True)
        print("T‑tests:")
        pprint.pprint(tests, compact=True)


class NiftiCentroidComparisonStats:
    """
    Compare centroid differences for a target column pair vs other column pairs.

    For each pair, centroids are computed from nonzero voxels in each NIfTI,
    converted to world coordinates using the affine, and averaged. The per-row
    centroid differences (A - B) are compared between target and other pairs
    using Welch's t-test on X/Y/Z and L2 distance.

    Parameters
    ----------
    csv_path : str
        Path to CSV containing NIfTI path columns.
    target_pair : dict
        Single k:v pair defining the comparison of interest (e.g. {"colA": "colB"}).
    other_pairs : dict or None, default None
        Dict of k:v pairs for other comparisons (e.g. {"colC": "colD"}).
        If None, all unique column pairs (excluding subject_col and target_pair) are used.
    subject_col : str or None, default "subject"
        Optional subject column (ignored for computation; excluded from comparisons).
    one_sided : bool, default True
        If True, uses alternative="greater" (target > others).
    threshold : float, default 0.0
        Threshold for binarizing data (mask = data > threshold).
    """

    def __init__(
        self,
        csv_path: str,
        *,
        target_pair: dict,
        other_pairs: dict | None = None,
        subject_col: str | None = "subject",
        one_sided: bool = True,
        threshold: float = 0.0,
    ):
        if not isinstance(target_pair, dict) or len(target_pair) != 1:
            raise ValueError("target_pair must be a single k:v pair dict")
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.subject_col = subject_col
        self.one_sided = one_sided
        self.threshold = threshold

        self.target_a, self.target_b = next(iter(target_pair.items()))
        self.other_pairs = other_pairs
        self._set_other_pairs()

        self.target_deltas = None
        self.other_deltas = None
        self.tests = None

    def _set_other_pairs(self):
        cols = list(self.df.columns)
        required = [self.target_a, self.target_b]
        missing = [c for c in required if c not in cols]
        if missing:
            raise ValueError(f"target_pair columns not found in CSV: {missing}")

        if self.other_pairs is None:
            exclude = {self.target_a, self.target_b}
            if self.subject_col and self.subject_col in cols:
                exclude.add(self.subject_col)
            valid_cols = [c for c in cols if c not in exclude]
            self.other_pairs = {a: b for a, b in combinations(valid_cols, 2)}
            return

        if not isinstance(self.other_pairs, dict):
            raise ValueError("other_pairs must be a dict of k:v column pairs")
        missing_pairs = []
        for k, v in self.other_pairs.items():
            if k not in cols or v not in cols:
                missing_pairs.append((k, v))
        if missing_pairs:
            raise ValueError(f"other_pairs not found in CSV: {missing_pairs}")

    def _centroid_from_path(self, path):
        img = nib.load(path)
        data = img.get_fdata()
        mask = data > self.threshold
        if not np.any(mask):
            return np.array([np.nan, np.nan, np.nan], dtype=float)
        coords = np.column_stack(np.nonzero(mask))
        hom = np.c_[coords, np.ones(len(coords))]
        world = (img.affine @ hom.T).T[:, :3]
        return world.mean(axis=0)

    def _pair_deltas(self, col_a, col_b):
        df = self.df.dropna(subset=[col_a, col_b])
        c1 = df[col_a].apply(self._centroid_from_path)
        c2 = df[col_b].apply(self._centroid_from_path)
        a = np.vstack(c1.to_numpy())
        b = np.vstack(c2.to_numpy())
        valid = (~np.isnan(a).any(axis=1)) & (~np.isnan(b).any(axis=1))
        return a[valid] - b[valid]

    def _compute_target_deltas(self):
        self.target_deltas = self._pair_deltas(self.target_a, self.target_b)
        return self.target_deltas

    def _compute_other_deltas(self):
        if not self.other_pairs:
            self.other_deltas = np.zeros((0, 3), dtype=float)
            return self.other_deltas
        deltas = []
        for col_a, col_b in self.other_pairs.items():
            d = self._pair_deltas(col_a, col_b)
            if d.size:
                deltas.append(d)
        if deltas:
            self.other_deltas = np.vstack(deltas)
        else:
            self.other_deltas = np.zeros((0, 3), dtype=float)
        return self.other_deltas

    def run(self, name: str = "Target centroid deltas vs other comparisons"):
        self._compute_target_deltas()
        self._compute_other_deltas()

        alt = "greater" if self.one_sided else "two-sided"
        tests = {}
        for i, axis in enumerate(["x", "y", "z"]):
            tests[axis] = ttest_ind(
                self.target_deltas[:, i],
                self.other_deltas[:, i],
                equal_var=False,
                nan_policy="omit",
                alternative=alt,
            )

        t_norm = np.linalg.norm(self.target_deltas, axis=1)
        o_norm = np.linalg.norm(self.other_deltas, axis=1)
        tests["l2"] = ttest_ind(
            t_norm,
            o_norm,
            equal_var=False,
            nan_policy="omit",
            alternative=alt,
        )

        summary = {
            "target_pair": (self.target_a, self.target_b),
            "n_target": int(len(self.target_deltas)),
            "n_other": int(len(self.other_deltas)),
            "target_mean": self.target_deltas.mean(axis=0) if len(self.target_deltas) else np.array([np.nan, np.nan, np.nan]),
            "other_mean": self.other_deltas.mean(axis=0) if len(self.other_deltas) else np.array([np.nan, np.nan, np.nan]),
            "target_l2_mean": float(np.nanmean(t_norm)) if len(t_norm) else np.nan,
            "other_l2_mean": float(np.nanmean(o_norm)) if len(o_norm) else np.nan,
            "alternative": alt,
        }

        self.tests = tests
        NiftiCentroidStats._show_results(
            summary,
            {k: {"t": v.statistic, "p": v.pvalue} for k, v in tests.items()},
            name,
        )
        return summary, tests


# ------------------------------------------------------------------ #
# demo
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    stats = NiftiCentroidStats(
        grp1_glob='/data/groupA/**/*.nii*',
        grp2_glob='/data/groupB/**/*.nii*',
        n_centroids=2,
        mirror=True,
    )
    stats.run()
    stats.compare_norms()
