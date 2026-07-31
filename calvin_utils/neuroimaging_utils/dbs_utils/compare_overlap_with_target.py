import glob
import os
import tempfile

import pandas as pd

from calvin_utils.neuroimaging_utils.nifti_utils.damage_score_utils import DamageScorer
from calvin_utils.plotting_utils.pair_superiority_plot import PairSuperiorityPlot


class CompareOverlapWithTarget:
    """
    Score two VTA columns against a target NIfTI using DamageScorer.

    Default CSV columns are VTA1, VTA2, and Target. If target_path is provided,
    the Target column is not required. Also saves a paired superiority plot
    comparing the two scored overlap columns.
    """

    def __init__(
        self,
        csv_path: str,
        *,
        vta_col1: str = "VTA1",
        vta_col2: str = "VTA2",
        target_path: str | None = None,
        target_col: str = "Target",
        output_dir: str | None = None,
        output_stem: str = "compare_overlap_with_target",
        output_path: str | None = None,
        figure_path: str | None = None,
        mask_path: str | None = None,
        selected_damage: str | list[str] = "avg_in_subject",
        group_labels: list[str] | None = None,
        stat_label: str = "Overlap",
        target_threshold: float = 0.0,
        score_nonzero_only: bool = False,
        resample_to_target: bool = True,
        verbose: bool = False,
        log_resample: bool = False,
        show: bool = True,
    ):
        self.csv_path = csv_path
        self.vta_col1 = vta_col1
        self.vta_col2 = vta_col2
        self.target_path = target_path
        self.target_col = target_col
        self.output_dir = output_dir
        self.output_stem = output_stem
        self.output_path = output_path
        self.figure_path = figure_path
        self.mask_path = mask_path
        self.selected_damage = selected_damage
        self.group_labels = group_labels or [vta_col1, vta_col2]
        self.stat_label = stat_label
        self.target_threshold = float(target_threshold)
        self.score_nonzero_only = bool(score_nonzero_only)
        self.resample_to_target = bool(resample_to_target)
        self.verbose = bool(verbose)
        self.log_resample = bool(log_resample)
        self.show = bool(show)

        self.df = None
        self.overlap_df = None

    def run(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.csv_path)
        self._validate_inputs()

        scoring_df = self._build_scoring_df()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            scoring_df.to_csv(tmp_path, index=False)
            scorer = DamageScorer(mask_path=self.mask_path)
            scored = scorer.score_csv_against_target(
                tmp_path,
                path_col=self.vta_col1,
                target_path=self._single_target_path(),
                selected_damage=self.selected_damage,
                target_suffix=f"{self.vta_col1}_vs_target",
                out_path=tmp_path,
                target_threshold=self.target_threshold,
                score_nonzero_only=self.score_nonzero_only,
                resample_to_target=self.resample_to_target,
                verbose=self.verbose,
                log_resample=self.log_resample,
            )
            scored.to_csv(tmp_path, index=False)
            scored = scorer.score_csv_against_target(
                tmp_path,
                path_col=self.vta_col2,
                target_path=self._single_target_path(),
                selected_damage=self.selected_damage,
                target_suffix=f"{self.vta_col2}_vs_target",
                out_path=tmp_path,
                target_threshold=self.target_threshold,
                score_nonzero_only=self.score_nonzero_only,
                resample_to_target=self.resample_to_target,
                verbose=self.verbose,
                log_resample=self.log_resample,
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        self.overlap_df = scored
        self._rename_default_metric_columns()
        self._save()
        self._plot_paired_superiority()
        return self.overlap_df

    def _validate_inputs(self):
        if self.mask_path is None:
            raise ValueError("mask_path is required because this class delegates scoring to DamageScorer.")
        if not isinstance(self.selected_damage, str):
            raise ValueError("selected_damage must be a single metric for paired superiority plotting.")
        if len(self.group_labels) != 2:
            raise ValueError("group_labels must contain exactly two labels.")

        required = {self.vta_col1, self.vta_col2}
        if self.target_path is None:
            required.add(self.target_col)

        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    def _build_scoring_df(self) -> pd.DataFrame:
        df = self.df.dropna(subset=[self.vta_col1, self.vta_col2]).copy()
        df[self.vta_col1] = df[self.vta_col1].map(self._resolve_path)
        df[self.vta_col2] = df[self.vta_col2].map(self._resolve_path)
        return df

    def _single_target_path(self) -> str:
        if self.target_path is not None:
            return self._resolve_path(self.target_path)

        targets = self.df[self.target_col].dropna().map(self._resolve_path).unique()
        if len(targets) != 1:
            raise ValueError("Target column must contain exactly one unique target path.")
        return targets[0]

    def _resolve_path(self, path: str) -> str:
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f"Invalid path: {path}")

        expanded = os.path.expanduser(path.strip())
        if os.path.isfile(expanded):
            return expanded

        matches = sorted(
            match for match in glob.glob(expanded)
            if os.path.isfile(match) and not os.path.basename(match).startswith("._")
        )
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise FileNotFoundError(f"No file found for path: {path}")
        raise ValueError(f"Path matched multiple files: {path}")

    def _rename_default_metric_columns(self):
        if self.selected_damage != "avg_in_subject":
            return

        self.overlap_df = self.overlap_df.rename(
            columns={
                f"avg_in_subject_{self.vta_col1}_vs_target": f"overlap_{self.vta_col1}_vs_target",
                f"avg_in_subject_{self.vta_col2}_vs_target": f"overlap_{self.vta_col2}_vs_target",
            }
        )

    def _score_column(self, vta_col: str) -> str:
        if self.selected_damage == "avg_in_subject":
            return f"overlap_{vta_col}_vs_target"
        if isinstance(self.selected_damage, str):
            metric = DamageScorer._output_metric_name(DamageScorer._normalize_metric_name(self.selected_damage))
            return f"{metric}_{vta_col}_vs_target"
        raise ValueError("Paired superiority plot requires selected_damage to be a single metric.")

    def _plot_paired_superiority(self):
        import matplotlib.pyplot as plt

        col1 = self._score_column(self.vta_col1)
        col2 = self._score_column(self.vta_col2)
        data = self.overlap_df[[col1, col2]].dropna()
        if data.empty:
            raise ValueError("No paired overlap scores available to plot.")

        out_dir = os.path.dirname(os.path.abspath(self._resolve_figure_path()))
        os.makedirs(out_dir, exist_ok=True)

        plotter = PairSuperiorityPlot(
            stat_array_1=data[col1].to_numpy(dtype=float),
            stat_array_2=data[col2].to_numpy(dtype=float),
            model1_name=self.group_labels[0],
            model2_name=self.group_labels[1],
            stat=self.stat_label,
            out_dir=None,
            method="bootstrap",
        )
        plotter.draw(verbose=self.show, save=False)
        fig = plt.gcf()
        fig.savefig(self.figure_path, bbox_inches="tight")
        if not self.show:
            plt.close(fig)

    def _resolve_figure_path(self) -> str:
        if self.figure_path is None:
            out_dir = self._resolve_output_dir()
            self.figure_path = os.path.join(out_dir, f"{self.output_stem}.svg")
        return self.figure_path

    def _resolve_output_dir(self) -> str:
        if self.output_dir is not None:
            return os.path.abspath(os.path.expanduser(self.output_dir))
        if self.output_path is not None:
            return os.path.dirname(os.path.abspath(self.output_path))
        return os.path.dirname(os.path.abspath(self.csv_path))

    def _save(self):
        if self.output_path is None:
            out_dir = self._resolve_output_dir()
            self.output_path = os.path.join(out_dir, f"{self.output_stem}.csv")
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        self.overlap_df.to_csv(self.output_path, index=False)
