import os
from itertools import combinations
import pprint

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, ttest_ind
from calvin_utils.ccm_utils.bounding_box import NiftiBoundingBox


class NiftiDiceStats:
    """
    Compute Dice overlap between two NIfTI columns in a CSV, paired within subject.

    Parameters
    ----------
    csv_path : str
        Path to CSV containing a subject column and two NIfTI path columns.
    subject_col : str, default "subject"
        Column name for subject ID.
    nifti_col_a, nifti_col_b : str
        Column names for the two NIfTI paths to compare within subject.
    reference_col : str or None, default None
        Optional column name for reference NIfTI paths to build a null distribution.
    one_sample_mu : float, default 0.5
        Population mean for one-sample t-test on subject-average Dice.
    threshold : float, default 0
        Threshold for binarizing NIfTI data (mask = data > threshold).
    empty_value : float, default 1.0
        Dice value when both masks are empty.
    """

    def __init__(
        self,
        csv_path: str,
        *,
        subject_col: str = "subject",
        nifti_col_a: str = "nifti_a",
        nifti_col_b: str = "nifti_b",
        reference_col: str | None = None,
        one_sample_mu: float = 0.5,
        threshold: float = 0.0,
        empty_value: float = 1.0,
    ):
        self.csv_path = csv_path
        self.subject_col = subject_col
        self.nifti_col_a = nifti_col_a
        self.nifti_col_b = nifti_col_b
        self.reference_col = reference_col
        self.one_sample_mu = one_sample_mu
        self.threshold = threshold
        self.empty_value = empty_value

        self.df = pd.read_csv(csv_path)
        self._validate_columns()

        self.row_dice = None
        self.subject_dice = None
        self.reference_dice = None
        self.one_sample_test = None
        self.two_sample_test = None
        self._mask_cache = {}
        self._shape_cache = {}

    def _validate_columns(self):
        self.df = self.df.dropna(subset=[self.nifti_col_a, self.nifti_col_b])   # only keep when both rows present
        
        required = {self.subject_col, self.nifti_col_a, self.nifti_col_b}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        if self.reference_col and self.reference_col not in self.df.columns:
            raise ValueError(f"reference_col '{self.reference_col}' not found in CSV")

    def _load_mask(self, path: str) -> np.ndarray:
        if not isinstance(path, str) or not path:
            raise ValueError(f"Invalid NIfTI path: {path}")
        path = os.path.expanduser(path)
        data = nib.load(path).get_fdata()
        return data > self.threshold

    def _get_mask_for_path(self, path: str):
        if path in self._mask_cache:
            return self._mask_cache[path], self._shape_cache[path]
        if not isinstance(path, str) or not path:
            raise ValueError(f"Invalid NIfTI path: {path}")
        path = os.path.expanduser(path)
        img = nib.load(path)
        data = img.get_fdata()
        mask = data > self.threshold
        self._mask_cache[path] = mask
        self._shape_cache[path] = mask.shape
        return mask, mask.shape

    def _dice(self, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        if mask_a.shape != mask_b.shape:
            raise ValueError(f"Shape mismatch: {mask_a.shape} vs {mask_b.shape}")
        a = mask_a.astype(bool)
        b = mask_b.astype(bool)
        a_sum = a.sum()
        b_sum = b.sum()
        if a_sum == 0 and b_sum == 0:
            return self.empty_value
        inter = np.logical_and(a, b).sum()
        return (2.0 * inter) / (a_sum + b_sum)

    def _dice_from_bbox(self, path_a: str, path_b: str) -> float:
        bbox = NiftiBoundingBox([path_a, path_b])
        bbox.generate_bounding_box()
        bbox.add_niftis_to_bounding_box()
        stacked = bbox.stacked_nifti
        mask_a = stacked[..., 0] > self.threshold
        mask_b = stacked[..., 1] > self.threshold
        return self._dice(mask_a, mask_b)

    def _dice_from_paths(self, path_a: str, path_b: str) -> float:
        mask_a, shape_a = self._get_mask_for_path(path_a)
        mask_b, shape_b = self._get_mask_for_path(path_b)
        if shape_a != shape_b:
            return self._dice_from_bbox(path_a, path_b)
        return self._dice(mask_a, mask_b)

    def _compute_row_dice(self):
        dice_vals = []
        for row in self.df.itertuples(index=False):
            path_a = getattr(row, self.nifti_col_a)
            path_b = getattr(row, self.nifti_col_b)
            dice_vals.append(self._dice_from_paths(path_a, path_b))
        self.row_dice = np.asarray(dice_vals, dtype=float)
        self.df["dice"] = self.row_dice
        return self.row_dice

    def _compute_subject_dice(self):
        if self.row_dice is None:
            self._compute_row_dice()
        grouped = self.df.groupby(self.subject_col)["dice"].mean()
        self.subject_dice = grouped.to_numpy(dtype=float)
        return self.subject_dice

    def _compute_reference_dice(self):
        if not self.reference_col:
            return None
        paths = [p for p in self.df[self.reference_col].dropna().tolist() if isinstance(p, str) and p]
        if len(paths) < 2:
            self.reference_dice = np.array([], dtype=float)
            return self.reference_dice

        dice_vals = []
        for p1, p2 in combinations(paths, 2):
            dice_vals.append(self._dice_from_paths(p1, p2))
        self.reference_dice = np.asarray(dice_vals, dtype=float)
        return self.reference_dice

    def run(self, name: str = "Dice overlap within subject"):
        self._compute_subject_dice()
        self.one_sample_test = ttest_1samp(self.subject_dice, popmean=self.one_sample_mu, nan_policy="omit")

        if self.reference_col:
            self._compute_reference_dice()
            self.two_sample_test = ttest_ind(
                self.subject_dice,
                self.reference_dice,
                equal_var=False,
                nan_policy="omit",
            )

        summary = {
            "n_subjects": int(len(self.subject_dice)),
            "mean_dice": float(np.nanmean(self.subject_dice)) if len(self.subject_dice) else np.nan,
            "std_dice": float(np.nanstd(self.subject_dice, ddof=1)) if len(self.subject_dice) > 1 else np.nan,
            "one_sample_mu": self.one_sample_mu,
            "one_sample_t": float(self.one_sample_test.statistic) if self.one_sample_test else np.nan,
            "one_sample_p": float(self.one_sample_test.pvalue) if self.one_sample_test else np.nan,
        }

        if self.reference_col:
            summary.update({
                "n_reference_pairs": int(len(self.reference_dice)) if self.reference_dice is not None else 0,
                "reference_mean": float(np.nanmean(self.reference_dice)) if self.reference_dice is not None and len(self.reference_dice) else np.nan,
                "reference_std": float(np.nanstd(self.reference_dice, ddof=1)) if self.reference_dice is not None and len(self.reference_dice) > 1 else np.nan,
                "two_sample_t": float(self.two_sample_test.statistic) if self.two_sample_test else np.nan,
                "two_sample_p": float(self.two_sample_test.pvalue) if self.two_sample_test else np.nan,
            })

        self._show_results(summary, name)
        return summary

    @staticmethod
    def _show_results(summary, name):
        print(f"\n----- {name} -----")
        pprint.pprint(summary, compact=True)


class NiftiDiceComparisonStats(NiftiDiceStats):
    """
    Compare Dice for a target column pair vs other column-pair comparisons.

    Parameters
    ----------
    csv_path : str
        Path to CSV containing NIfTI path columns.
    target_pair : dict
        Single k:v pair defining the comparison of interest (e.g. {"colA": "colB"}).
    other_pairs : dict or None, default None
        Dict of k:v pairs for other comparisons (e.g. {"colC": "colD", "colE": "colF"}).
        If None, all unique column pairs (excluding subject_col and target_pair) are used.
    subject_col : str or None, default "subject"
        Optional subject column (ignored for computation; excluded from comparisons).
    one_sided : bool, default True
        If True, uses alternative="greater" (target > others).
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
        empty_value: float = 1.0,
    ):
        if not isinstance(target_pair, dict) or len(target_pair) != 1:
            raise ValueError("target_pair must be a single k:v pair dict")
        target_a, target_b = next(iter(target_pair.items()))
        super().__init__(
            csv_path=csv_path,
            subject_col=subject_col or "subject",
            nifti_col_a=target_a,
            nifti_col_b=target_b,
            reference_col=None,
            one_sample_mu=0.0,
            threshold=threshold,
            empty_value=empty_value,
        )
        self.target_a = target_a
        self.target_b = target_b
        self.subject_col = subject_col
        self.one_sided = one_sided
        self.other_pairs = other_pairs
        self._set_other_pairs()
        self.target_dice = None
        self.other_dice = None
        self.two_sample_test = None

    def _set_other_pairs(self):
        cols = list(self.df.columns)
        if self.other_pairs is None:
            exclude = {self.target_a, self.target_b}
            if self.subject_col and self.subject_col in cols:
                exclude.add(self.subject_col)
            valid_cols = [c for c in cols if c not in exclude]
            self.other_pairs = {a: b for a, b in combinations(valid_cols, 2)}
            return
        if not isinstance(self.other_pairs, dict):
            raise ValueError("other_pairs must be a dict of k:v column pairs")
        missing = []
        for k, v in self.other_pairs.items():
            if k not in cols or v not in cols:
                missing.append((k, v))
        if missing:
            raise ValueError(f"other_pairs not found in CSV: {missing}")

    def _compute_target_dice(self):
        df = self.df.dropna(subset=[self.target_a, self.target_b])
        vals = []
        for row in df.itertuples(index=False):
            path_a = getattr(row, self.target_a)
            path_b = getattr(row, self.target_b)
            vals.append(self._dice_from_paths(path_a, path_b))
        self.target_dice = np.asarray(vals, dtype=float)
        return self.target_dice

    def _compute_other_dice(self):
        if not self.other_pairs:
            self.other_dice = np.array([], dtype=float)
            return self.other_dice

        vals = []
        for col_a, col_b in self.other_pairs.items():
            df_pair = self.df.dropna(subset=[col_a, col_b])
            for row in df_pair.itertuples(index=False):
                path_a = getattr(row, col_a)
                path_b = getattr(row, col_b)
                vals.append(self._dice_from_paths(path_a, path_b))
        self.other_dice = np.asarray(vals, dtype=float)
        return self.other_dice

    def run(self, name: str = "Target Dice vs Other Comparisons"):
        self._compute_target_dice()
        self._compute_other_dice()

        alt = "greater" if self.one_sided else "two-sided"
        self.two_sample_test = ttest_ind(
            self.target_dice,
            self.other_dice,
            equal_var=False,
            nan_policy="omit",
            alternative=alt,
        )

        summary = {
            "target_pair": (self.target_a, self.target_b),
            "n_target": int(len(self.target_dice)),
            "target_mean": float(np.nanmean(self.target_dice)) if len(self.target_dice) else np.nan,
            "target_std": float(np.nanstd(self.target_dice, ddof=1)) if len(self.target_dice) > 1 else np.nan,
            "n_other": int(len(self.other_dice)),
            "other_mean": float(np.nanmean(self.other_dice)) if len(self.other_dice) else np.nan,
            "other_std": float(np.nanstd(self.other_dice, ddof=1)) if len(self.other_dice) > 1 else np.nan,
            "welch_t": float(self.two_sample_test.statistic) if self.two_sample_test else np.nan,
            "welch_p": float(self.two_sample_test.pvalue) if self.two_sample_test else np.nan,
            "alternative": alt,
        }

        self._show_results(summary, name)
        return summary


if __name__ == "__main__":
    stats = NiftiDiceStats(
        csv_path="/path/to/paths.csv",
        subject_col="subject",
        nifti_col_a="path_a",
        nifti_col_b="path_b",
        reference_col="reference_path",
        one_sample_mu=0.5,
        threshold=0.0,
    )
    stats.run()
