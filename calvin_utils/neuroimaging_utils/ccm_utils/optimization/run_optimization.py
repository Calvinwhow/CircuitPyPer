"""Symptom-basis optimization: run() reads top to bottom as the steps of one analysis, and each step is one module."""

from functools import partial
from pathlib import Path
from . import algorithms
from .inference import basis_map
from .output import saving
from .pipeline import basis_damage, cross_validation, fitting, pre_processing, prediction


class RunOptimization:
    """Score patients against the basis library, cross-validate and fit every model, predict, rank, save, and infer."""

    def __init__(self, datasets, *, cohorts=None, basis_dir=basis_damage.BASIS_DIR, methods=algorithms.__all__, sparse_semidefinite=False, neural_network=False, fully_trainable=False, weight_mode="unweighted", dataset_weights=None, dataset_signs=None, regularization_strength=0.1, project_prediction_variance=False, learning_rate=0.01, max_epochs=2000, tolerance=1e-7, patience=100, weight_decay=0.0, gradient_clip=5.0, device="auto", seed=42):
        """
        datasets: dict[str, dict] of length D; each entry contains X shape (N, V), y shape (N,), and optional ids shape (N,).
        cohorts, basis_dir: the basis library, every shard in basis_dir when cohorts is None.
        methods: algorithms to fit and compare, without sparse_semidefinite unless that flag is set; neural_network adds the anatomical network built on them, and fully_trainable lets it retrain their coefficients.
        weight_mode, dataset_weights, regularization_strength, project_prediction_variance: algorithm settings. dataset_signs and learning_rate through device: network settings.
        """
        self.datasets = datasets
        self.cohorts = cohorts
        self.basis_dir = basis_dir
        self.dataset_weights = dataset_weights
        self.seed = seed
        self.fit_models = partial(fitting.main, methods=tuple(method for method in methods if sparse_semidefinite or method != "sparse_semidefinite"), neural_network=neural_network, fully_trainable=fully_trainable, weight_mode=weight_mode, dataset_weights=dataset_weights, dataset_signs=dataset_signs, regularization_strength=regularization_strength, project_prediction_variance=project_prediction_variance, learning_rate=learning_rate, max_epochs=max_epochs, tolerance=tolerance, patience=patience, weight_decay=weight_decay, gradient_clip=gradient_clip, device=device)

    def run(self, *, cross_validation=None, test_datasets=None, out_dir=None, output_type=None, mask_path=None, store_history=False, store_map_stack=False, render_gif=False, gif_fps=5, gif_max_frames=60, gif_view="auto", winner_only=False, n_permutations=0):
        """
        cross_validation: "lodo" | "loocv" | int | None; leave-one-dataset-out, pooled patient folds across every dataset, or one fit on every patient (None), the only fit with a map to test and regress.
        test_datasets: dict[str, dict] | None; external datasets with X, y and ids, never used for fitting.
        out_dir, output_type, mask_path, store_history, store_map_stack, render_gif, gif_fps, gif_max_frames, gif_view, winner_only: outputs. n_permutations: basis-map regression of the winner.
        """
        self.preprocess(test_datasets, cross_validation)
        self.score_bases()
        self.fit(cross_validation)
        self.predict()
        self.rank()
        self.save(out_dir, output_type, mask_path, store_history, store_map_stack, render_gif, gif_fps, gif_max_frames, gif_view, winner_only)
        self.infer(output_type, mask_path, n_permutations)
        return self.comparison

    def preprocess(self, test_datasets, scheme):
        self.prepared = pre_processing.main(self.datasets, test_datasets, scheme)

    def score_bases(self):
        self.damage = basis_damage.main(self.prepared["datasets"], self.prepared["test_datasets"], self.cohorts, self.basis_dir)

    def fit(self, scheme):
        self.results = cross_validation.main(self.damage, scheme, self.fit_models, self.seed)

    def predict(self):
        self.results = prediction.main(self.results, self.damage)

    def rank(self):
        self.comparison = prediction.rank(self.results)
        self.result = self.results[self.comparison["winner"]]

    def save(self, out_dir, output_type, mask_path, store_history, store_map_stack, render_gif, gif_fps, gif_max_frames, gif_view, winner_only):
        self.saved = saving.main(self.results, self.damage, out_dir, self.comparison["winner"], winner_only, output_type, mask_path, store_history, store_map_stack, render_gif, gif_fps, gif_max_frames, gif_view)

    def infer(self, output_type, mask_path, n_permutations):
        self.result["basis_map_regression"] = basis_map.infer(self.result, self.saved.get(self.comparison["winner"]), output_type, mask_path, n_permutations)

    def gradient_step(self, gradient, learning_rate=0.01):
        """Exposed so an external optimizer can drive the winning algorithm: one gradient-ascent step on its weights, returning its native merit."""
        self.result = prediction.main({self.result["method"]: fitting.step(self.result, self.damage, gradient, learning_rate, self.dataset_weights)}, self.damage)[self.result["method"]]
        return fitting.merit(self.result, self.damage, self.dataset_weights)

    @classmethod
    def run_targets(cls, dataset_specs, targets=None, *, test_specs=None, subject_col=None, cross_validation="lodo", out_dir=None, output_type=None, mask_path=None, store_history=False, store_map_stack=False, render_gif=False, gif_fps=5, gif_max_frames=60, gif_view="auto", winner_only=False, n_permutations=0, **settings):
        """Run the whole analysis once per exact target column, then summarize held-out performance across targets.
        dataset_specs, test_specs: {name: {"csv_path", "nifti_column", "subject_column"}}; targets: every symptom in the basis library when None; settings: RunOptimization arguments.
        """
        targets = basis_damage.symptoms(basis_damage.load_library(settings.get("cohorts"), settings.get("basis_dir", basis_damage.BASIS_DIR))) if targets is None else tuple(targets)
        fitting_cohorts = pre_processing.load_target_cohorts(dataset_specs, targets, mask_path, subject_col)
        test_cohorts = pre_processing.load_target_cohorts(test_specs, targets, mask_path, subject_col) if test_specs else None
        comparisons = {}
        for index, target in enumerate(targets, start=1):
            print(f"Target {index}/{len(targets)}: {target}", flush=True)
            comparisons[target] = cls(pre_processing.target_datasets(fitting_cohorts, target), **settings).run(cross_validation=cross_validation, test_datasets=None if test_cohorts is None else pre_processing.target_datasets(test_cohorts, target), out_dir=None if out_dir is None else Path(out_dir) / target, output_type=output_type, mask_path=mask_path, store_history=store_history, store_map_stack=store_map_stack, render_gif=render_gif, gif_fps=gif_fps, gif_max_frames=gif_max_frames, gif_view=gif_view, winner_only=winner_only, n_permutations=n_permutations)
        saving.target_validation(comparisons, out_dir)
        return comparisons
