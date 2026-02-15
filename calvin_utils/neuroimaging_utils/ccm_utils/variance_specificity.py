import os
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class VarianceSpecificity:
    """
    Rank rows (maps) by how discriminative a column of interest is versus other columns.

    Discrimination score per row:
        n * (r_col ** 2) - sum(r_other ** 2)
    where n is the number of other columns.
    """

    def __init__(
        self,
        csv_path: str,
        col_of_interest: str,
        output_path: Optional[str] = None,
        map_col: Optional[str] = None,
        missing_fill: float = 0.0,
        figsize: Tuple[int, int] = (10, 8),
    ):
        # Inputs
        self.csv_path = csv_path
        self.col_of_interest = col_of_interest
        self.output_path = output_path
        self.map_col = map_col
        self.missing_fill = missing_fill
        self.figsize = figsize

        # Internal state
        self.df_raw = None
        self.df_clean = None
        self.symptom_cols = None
        self.r2_df = None
        self.discrimination_df = None
        self.rank_df = None
        self.fig = None
        self.ax = None

    # Setters / Getters
    def set_csv_path(self, csv_path: str):
        self.csv_path = csv_path

    def get_csv_path(self) -> str:
        return self.csv_path

    def set_col_of_interest(self, col_of_interest: str):
        self.col_of_interest = col_of_interest

    def get_col_of_interest(self) -> str:
        return self.col_of_interest

    def set_output_path(self, output_path: Optional[str]):
        self.output_path = output_path

    def get_output_path(self) -> Optional[str]:
        return self.output_path

    def set_map_col(self, map_col: Optional[str]):
        self.map_col = map_col

    def get_map_col(self) -> Optional[str]:
        return self.map_col

    def set_missing_fill(self, missing_fill: float):
        self.missing_fill = missing_fill

    def get_missing_fill(self) -> float:
        return self.missing_fill

    # Internal methods
    def _load_csv(self) -> pd.DataFrame:
        self.df_raw = pd.read_csv(self.csv_path)
        return self.df_raw

    def _validate_inputs(self):
        if self.df_raw is None:
            raise ValueError("CSV not loaded. Call _load_csv first.")
        if self.df_raw.shape[1] < 2:
            raise ValueError("CSV must contain a map name column plus at least one symptom column.")

        if self.map_col is None:
            self.map_col = self.df_raw.columns[0]
        if self.map_col not in self.df_raw.columns:
            raise ValueError(f"map_col '{self.map_col}' not found in CSV columns.")
        if self.col_of_interest not in self.df_raw.columns:
            raise ValueError(f"col_of_interest '{self.col_of_interest}' not found in CSV columns.")
        if self.col_of_interest == self.map_col:
            raise ValueError("col_of_interest cannot be the same as map_col.")

    def _clean_df(self) -> pd.DataFrame:
        df = self.df_raw.copy()
        # Ensure map names are strings
        df[self.map_col] = df[self.map_col].astype(str)

        self.symptom_cols = [c for c in df.columns if c != self.map_col]
        for c in self.symptom_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.fillna(self.missing_fill)
        self.df_clean = df
        return df

    def _compute_r2(self) -> pd.DataFrame:
        # R2 derived from R-values
        r2 = self.df_clean.copy()
        r2[self.symptom_cols] = r2[self.symptom_cols].astype(float) ** 2
        self.r2_df = r2
        return r2

    def _compute_discrimination(self) -> pd.DataFrame:
        if self.r2_df is None:
            raise ValueError("R2 not computed. Call _compute_r2 first.")
        df = self.r2_df.copy()
        other_cols = [c for c in self.symptom_cols if c != self.col_of_interest]
        n = len(other_cols)
        if n == 0:
            raise ValueError("Need at least one other symptom column to compute discrimination.")
        df["discrimination"] = (n * df[self.col_of_interest]) - df[other_cols].sum(axis=1)
        self.discrimination_df = df[[self.map_col, "discrimination"]].copy()
        return self.discrimination_df

    def _rank_rows(self) -> pd.DataFrame:
        if self.discrimination_df is None:
            raise ValueError("Discrimination not computed. Call _compute_discrimination first.")
        rank_df = self.discrimination_df.sort_values("discrimination", ascending=False).reset_index(drop=True)
        self.rank_df = rank_df
        return rank_df

    def _plot(self) -> Tuple[plt.Figure, plt.Axes]:
        if self.rank_df is None:
            raise ValueError("Ranked data not available. Call _rank_rows first.")
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        sns.barplot(
            data=self.rank_df,
            x="discrimination",
            y=self.map_col,
            ax=self.ax,
            color="#4C72B0"
        )
        self.ax.set_title(f"Variance Specificity: {self.col_of_interest}")
        self.ax.set_xlabel("Discrimination Score")
        self.ax.set_ylabel("Map")
        self.fig.tight_layout()
        return self.fig, self.ax

    def _save_plot(self) -> str:
        if self.fig is None:
            raise ValueError("Figure not available. Call _plot first.")
        if self.output_path is None:
            base = os.path.splitext(os.path.basename(self.csv_path))[0]
            fname = f"{base}_variance_specificity_{self.col_of_interest}.svg"
            self.output_path = os.path.join(os.path.dirname(self.csv_path), fname)

        out_dir = os.path.dirname(self.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        self.fig.savefig(self.output_path, format="svg")
        return self.output_path

    # Public orchestration
    def run(self) -> Tuple[pd.DataFrame, str, plt.Figure, plt.Axes]:
        self._load_csv()
        self._validate_inputs()
        self._clean_df()
        self._compute_r2()
        self._compute_discrimination()
        self._rank_rows()
        self._plot()
        output_path = self._save_plot()
        return self.rank_df, output_path, self.fig, self.ax
