import glob
import os
import tempfile

import pandas as pd

from calvin_utils.neuroimaging_utils.ccm_utils.bounding_box import NiftiBoundingBox
from calvin_utils.neuroimaging_utils.nifti_utils.damage_score_utils import DamageScorer
from calvin_utils.plotting_utils.pair_superiority_plot import PairSuperiorityPlot


class CompareOverlapWithTarget:
    """
    Score two VTA columns against one target NIfTI and save the results.
    """

    OUTPUT_STEM = "compare_overlap_with_target"

    def __init__(
        self,
        csv_path: str,
        *,
        vta_col1: str = "VTA1",
        vta_col2: str = "VTA2",
        target_path: str,
        output_dir: str,
        mask_path: str | None = None,
        selected_damage: str = "avg_in_subject",
        resample_to_target: bool = True,
        log_resample: bool = False,
    ):
        self.csv_path = csv_path
        self.vta_col1 = vta_col1
        self.vta_col2 = vta_col2
        self.target_path = target_path
        self.output_dir = os.path.abspath(os.path.expanduser(output_dir))
        self.mask_path = mask_path
        self.selected_damage = selected_damage
        self.resample_to_target = bool(resample_to_target)
        self.log_resample = bool(log_resample)

        self.output_path = os.path.join(self.output_dir, f"{self.OUTPUT_STEM}.csv")
        self.figure_path = os.path.join(self.output_dir, f"{self.OUTPUT_STEM}.svg")

        self.df = None
        self.overlap_df = None

    def run(self):
        self.df = pd.read_csv(self.csv_path)
        self._validate_inputs()
        os.makedirs(self.output_dir, exist_ok=True)

        scoring_df = self._build_scoring_df()
        self._ensure_mask_path(scoring_df)
        scored = self._score_overlap_columns(scoring_df)

        self.overlap_df = scored
        self._rename_default_metric_columns()
        self._save()
        return self._plot_paired_superiority()

    def _score_overlap_columns(self, scoring_df: pd.DataFrame) -> pd.DataFrame:
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
                resample_to_target=self.resample_to_target,
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
                resample_to_target=self.resample_to_target,
                log_resample=self.log_resample,
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return scored

    def _validate_inputs(self):
        if not isinstance(self.selected_damage, str):
            raise ValueError("selected_damage must be a single metric for paired superiority plotting.")

        missing = {self.vta_col1, self.vta_col2} - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    ### MASKING METHODS ###

    def _ensure_mask_path(self, scoring_df: pd.DataFrame):
        if self.mask_path is not None:
            self.mask_path = self._resolve_path(self.mask_path)
            return

        mask_files = self._collect_mask_source_files(scoring_df)
        self.mask_path = self._generate_mask_from_files(mask_files, self.output_dir)

    def _collect_mask_source_files(self, scoring_df: pd.DataFrame) -> list[str]:
        files = []
        for col in (self.vta_col1, self.vta_col2):
            files.extend(scoring_df[col].dropna().tolist())
        files.append(self._single_target_path())
        return self._dedupe_paths(files)

    @staticmethod
    def _dedupe_paths(paths: list[str]) -> list[str]:
        deduped = []
        seen = set()
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            deduped.append(path)
        return deduped

    @staticmethod
    def _generate_mask_from_files(files: list[str], output_dir: str) -> str:
        if not files:
            raise ValueError("Cannot generate a mask because no NIfTI files were provided.")

        bbox = NiftiBoundingBox(files)
        bbox.gen_mask(output_dir)
        return os.path.join(output_dir, "mask.nii.gz")

    def _build_scoring_df(self) -> pd.DataFrame:
        df = self.df.dropna(subset=[self.vta_col1, self.vta_col2]).copy()
        df[self.vta_col1] = df[self.vta_col1].map(self._resolve_path)
        df[self.vta_col2] = df[self.vta_col2].map(self._resolve_path)
        return df

    def _single_target_path(self) -> str:
        return self._resolve_path(self.target_path)

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

        plotter = PairSuperiorityPlot(
            stat_array_1=data[col2].to_numpy(dtype=float),
            stat_array_2=data[col1].to_numpy(dtype=float),
            model1_name=self.vta_col2,
            model2_name=self.vta_col1,
            stat="Overlap",
            out_dir=None,
            method="bootstrap",
        )
        plotter.draw(verbose=False, save=False)
        fig = plt.gcf()
        fig.savefig(self.figure_path, bbox_inches="tight")
        plt.close(fig)
        return fig

    def _save(self):
        self.overlap_df.to_csv(self.output_path, index=False)
