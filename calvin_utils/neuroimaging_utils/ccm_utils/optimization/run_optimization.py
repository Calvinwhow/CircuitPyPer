"""Single linear dispatcher for symptom-basis optimization."""

import numpy as np
from pathlib import Path
from . import algorithms
from .output import saving
from functools import partial
from .inference import basis_map
from .projections import prediction_variance
from .selection import prepare_regularization
from .solvers import adam, proximal_gradient, semidefinite
from .pipeline import basis_damage, cross_validation, pre_processing, prediction


class RunOptimization:
    """Preprocess, score patients against the basis shards once, then fit, predict, and optionally save one weighted localization."""

    def __init__(self, datasets, *, cohorts=None, basis_dir=basis_damage.BASIS_DIR, method="rms_rho", weight_mode="unweighted", dataset_weights=None, regularization_strength=0.1, project_prediction_variance=False, seed=42):
        """Configure one symptom-basis optimization.
        datasets: dict[str, dict] of length D; each entry contains X shape (N, V), y shape (N,), and optional IDs shape (N,).
        cohorts: iterable[str] | None; basis-library cohorts to load, or every available cohort when omitted.
        basis_dir: str | Path; directory containing basis shard NPY and JSON pairs.
        method: str; optimization backend named in algorithms.__all__.
        weight_mode: str; RMS-rho initialization: unweighted, gaussian, or sample-size weighted.
        dataset_weights: dict[str, float] | None; optional dataset contributions keyed like datasets.
        regularization_strength: float; elastic-net budget relative to unpenalized merit gain.
        project_prediction_variance: bool; whether compatible solvers project every update to unit prediction variance.
        seed: int; random seed for initialization, fitting, and validation.
        """
        self.datasets = datasets
        self.cohorts = cohorts
        self.basis_dir = basis_dir
        self.method = method
        self.weight_mode = weight_mode
        self.dataset_weights = dataset_weights
        self.regularization_strength = regularization_strength
        self.project_prediction_variance = project_prediction_variance
        self.seed = seed
        self.result = None
        self.results = None
        self.prepared = None
        self.damage = None

    def preprocess(self):
        return pre_processing.main(self.datasets)

    def score_bases(self, prepared, test_datasets):
        """One pass over the selected shards; returns patient-by-basis damage for the fitting and test datasets."""
        return basis_damage.main(prepared["datasets"], test_datasets, self.cohorts, self.basis_dir)

    def optimize(self, X, y, names, sample_sizes, seed, l1_strength, l2_strength):
        """Fit one optimization backend.

        X: dict[str, np.ndarray] with D dataset-name keys; each value has shape (N, B), where N is patients and B is basis maps.
        y: dict[str, np.ndarray] with D dataset-name keys; each value has shape (N,) and contains outcomes aligned with X.
        """
        dataset_weights = pre_processing.dataset_weights(X, self.dataset_weights)
        covariance = prediction_variance.gram(X, dataset_weights)
        project = partial(prediction_variance.project, covariance=covariance) if self.project_prediction_variance else None
        if self.method == "rms_rho":
            initial = pre_processing.initial_weights(len(names), self.weight_mode, sample_sizes=sample_sizes, names=names, seed=seed)
            return algorithms.rms_rho.main(X, y, solver=adam.main, random_state=seed, initial_weights=initial, l1_strength=l1_strength, l2_strength=l2_strength, dataset_weights=dataset_weights)
        if self.method == "elastic_net_ols":
            return algorithms.elastic_net_ols.main(X, y, solver=proximal_gradient.main, project=project, l1_strength=l1_strength, l2_strength=l2_strength)
        if self.method == "pairwise_proxy":
            return algorithms.pairwise_proxy.main(X, y, solver=proximal_gradient.main, project=project, l1_strength=l1_strength, l2_strength=l2_strength, dataset_weights=dataset_weights)
        if self.method == "sparse_semidefinite":
            return algorithms.sparse_semidefinite.main(X, y, solver=semidefinite.main, l1_strength=l1_strength, l2_strength=l2_strength, dataset_weights=dataset_weights)
        if self.method == "rayleigh_refinement":
            return algorithms.rayleigh_refinement.main(X, y, solver=adam.main, project=project, l1_strength=l1_strength, l2_strength=l2_strength, dataset_weights=dataset_weights)
        raise ValueError(f"Unknown optimization method {self.method!r}.")

    def prepare_regularization(self, X, y, names, sample_sizes, seed):
        reference = self.optimize(X, y, names, sample_sizes, seed, 0.0, 0.0)
        parameters = reference["theta"] if self.method == "sparse_semidefinite" else reference["weights"]
        if self.method == "sparse_semidefinite":
            l1_reference = algorithms.sparse_semidefinite.l1(parameters, 1.0, lambda values: np.linalg.norm(values, 1))
            l2_reference = algorithms.sparse_semidefinite.l2(parameters, 1.0, lambda values: np.square(values).sum())
        else:
            algorithm = getattr(algorithms, self.method)
            l1_reference = algorithm.l1(parameters, 1.0)
            l2_reference = algorithm.l2(parameters, 1.0)
        return prepare_regularization.main(reference["merit"], reference["null_merit"], l1_reference, l2_reference, self.regularization_strength)

    def fit_model(self, X, y, names, sample_sizes, seed):
        regularization = self.prepare_regularization(X, y, names, sample_sizes, seed)
        result = self.optimize(X, y, names, sample_sizes, seed, regularization["l1_strength"], regularization["l2_strength"])
        result["weights"] = prediction_variance.normalize(result["weights"], X, pre_processing.dataset_weights(X, self.dataset_weights))
        result.update(regularization)
        return result

    def fit(self, damage, within_dataset_folds, cross_dataset_validation):
        return cross_validation.main(damage, self.fit_model, within_dataset_folds=within_dataset_folds, cross_dataset=cross_dataset_validation, seed=self.seed)

    def predict(self, damage, weights):
        predictions, performance = prediction.main(damage["datasets"], weights)
        test = None
        if damage["test_datasets"] is not None:
            test_predictions, test_summary = prediction.test(damage["test_datasets"], weights, prediction.fitting_ids(damage["datasets"]))
            test = {"predictions": test_predictions, "summary": test_summary}
        return predictions, performance, test

    def _run_method(self, within_dataset_folds, cross_dataset_validation):
        """Fit the current method on the stored damage and assemble its result; the shards are read again only to build the final map."""
        fitted = self.fit(self.damage, within_dataset_folds, cross_dataset_validation)
        fit = fitted["fit"]
        if self.method == "sparse_semidefinite":
            print(f"Sparse SDP leading-eigenvalue spectral mass: {100 * fit['leading_eigenvalue_fraction']:.4f}%")
            print(fit["recovery_recommendation"])
        weights = np.asarray(fit["weights"])
        predictions, performance, test = self.predict(self.damage, weights)
        symptom_map = basis_damage.combine(weights, self.damage["library"], self.damage["scales"])
        self.result = {"method": self.method, "map_names": self.damage["map_names"], "library": self.damage["library"], "scales": self.damage["scales"], "weights": weights, "symptom_map": symptom_map, "fit": fit, "predictions": predictions, "performance": performance, "within_dataset_validation": fitted["within_dataset"], "cross_dataset_validation": fitted["cross_dataset"], "test": test}
        self.result["weight_table"] = saving.weights(self.result)
        return self.result

    def _update_weights(self, weights):
        X = {name: data["X"] for name, data in self.damage["datasets"].items()}
        self.result["weights"] = prediction_variance.normalize(weights, X, pre_processing.dataset_weights(X, self.dataset_weights))
        self.result["symptom_map"] = basis_damage.combine(self.result["weights"], self.damage["library"], self.damage["scales"])
        self.result["predictions"], self.result["performance"], self.result["test"] = self.predict(self.damage, self.result["weights"])
        self.result["weight_table"] = saving.weights(self.result)

    def _get_merit(self):
        X = {name: data["X"] for name, data in self.damage["datasets"].items()}
        y = {name: data["y"] for name, data in self.damage["datasets"].items()}
        return getattr(algorithms, self.method).evaluate(self.result["weights"], X, y, pre_processing.dataset_weights(X, self.dataset_weights))

    def save(self, out_dir, output_type, mask_path, store_history, render_gif, gif_fps, gif_max_frames, gif_view, winner_only=False):
        results = {self.method: self.result} if winner_only or self.results is None else self.results
        separate_directories = len(results) > 1
        return {method: saving.main(result, Path(out_dir) / method if separate_directories else out_dir, output_type=output_type, mask_path=mask_path, store_history=store_history, render_gif=render_gif, gif_fps=gif_fps, gif_max_frames=gif_max_frames, gif_view=gif_view) for method, result in results.items()}

    def infer(self, result, out_dir, output_type, mask_path, n_permutations):
        return basis_map.main(basis_damage.stack(result["library"], result["scales"]), result["fit"]["weights"], Path(out_dir) / "final_model" / "basis_map_regression", output_type=output_type, mask_path=mask_path, n_permutations=n_permutations)

    ### Public API ###
    def gradient_step(self, gradient, learning_rate=0.01):
        """Apply one external gradient-ascent step and return native merit.
        gradient: np.ndarray with shape (B,); external derivative with respect to fitted coefficients.
        learning_rate: float; external update step size.
        """
        self._update_weights(self.result["weights"] + learning_rate * np.asarray(gradient, dtype=float))
        return self._get_merit()

    def run(self, *, within_dataset_folds=None, cross_dataset_validation=False, test_datasets=None, out_dir=None, output_type=None, mask_path=None, store_history=False, render_gif=False, gif_fps=5, gif_max_frames=60, gif_view="auto", n_permutations=0):
        """Run independent validation branches and the final joint fit.
        within_dataset_folds: int | "loocv" | None; isolated patient folds per dataset or no patient CV.
        cross_dataset_validation: bool; whether to fit D-1 datasets and test the omitted dataset.
        test_datasets: dict[str, dict] | None; external datasets containing X shape (N, V), y shape (N,), and IDs shape (N,).
        out_dir: str | Path | None; optional root output directory.
        output_type: str | None; neuroimage output type, or no map export when omitted.
        mask_path: str | Path | None; mask or ordered feature reference used for output.
        store_history: bool; whether to save coefficient and objective history.
        render_gif: bool; whether to render optimization history as a GIF.
        gif_fps: int; GIF frames per second.
        gif_max_frames: int; maximum rendered history frames.
        gif_view: str; visualization view passed to the history renderer.
        n_permutations: int; basis-map regression permutations, or zero to skip inference.
        """
        self.results = None
        self.prepared = self.preprocess()
        self.damage = self.score_bases(self.prepared, test_datasets)
        self._run_method(within_dataset_folds, cross_dataset_validation)
        if out_dir is not None:
            self.save(out_dir, output_type, mask_path, store_history, render_gif, gif_fps, gif_max_frames, gif_view)
        if n_permutations:
            if out_dir is None or output_type is None:
                raise ValueError("Basis-map regression requires out_dir and output_type.")
            self.result["basis_map_regression"] = self.infer(self.result, out_dir, output_type, mask_path, n_permutations)
        return self.result

    def orchestrate_run(self, methods=algorithms.__all__, within_dataset_folds=None, cross_dataset_validation=False, test_datasets=None):
        """Sweep the basis library once and compare selected algorithms.
        methods: iterable[str]; optimization backends named in algorithms.__all__.
        within_dataset_folds: int | "loocv" | None; isolated patient folds per dataset or no patient CV.
        cross_dataset_validation: bool; whether to fit D-1 datasets and test the omitted dataset.
        test_datasets: dict[str, dict] | None; external datasets containing X shape (N, V), y shape (N,), and IDs shape (N,).
        """
        self.prepared = self.preprocess()
        self.damage = self.score_bases(self.prepared, test_datasets)
        results = {}
        for method in methods:
            self.method = method
            results[method] = self._run_method(within_dataset_folds, cross_dataset_validation)

        # get performances 
        if test_datasets is not None:
            summaries = {method: result["test"]["summary"] for method, result in results.items()}
        elif cross_dataset_validation:
            summaries = {method: result["cross_dataset_validation"]["summary"] for method, result in results.items()}
        elif within_dataset_folds is not None:
            summaries = {method: result["within_dataset_validation"]["summary"] for method, result in results.items()}
        else:
            summaries = {method: result["performance"] for method, result in results.items()}
        performance = {method: float(np.sqrt(np.mean([values["spearman_rho"] ** 2 for values in summary.values()]))) for method, summary in summaries.items()}
        ranking = sorted(performance, key=performance.get, reverse=True)
        for method in ranking:
            print(f"{method}: {performance[method]:.6f}")
        winner = ranking[0]
        print(f"Winner: {winner}")
        
        # Store and return readouts
        self.method = winner
        self.result = results[winner]
        self.results = results
        return {"winner": winner, "winner_map": self.result["symptom_map"], "ranking": ranking, "performance": performance, "summaries": summaries, "results": results}

    @classmethod
    def run_targets(cls, dataset_specs, targets, *, test_specs=None, subject_col=None, basis_cohorts=None, basis_dir=basis_damage.BASIS_DIR, methods=algorithms.__all__, weight_mode="unweighted", dataset_weights=None, regularization_strength=0.1, project_prediction_variance=False, seed=42, within_dataset_folds=None, cross_dataset_validation=True, out_dir=None, output_type=None, mask_path=None, store_history=False, render_gif=False, gif_fps=5, gif_max_frames=60, gif_view="auto", winner_only=False, n_permutations=0):
        """Optimize every exact target column through the same validation pipeline.
        dataset_specs: dict[str, dict] of length D; CSV cohort specifications containing csv_path, nifti_column, and optional subject_column.
        targets: iterable[str]; exact clinical outcome columns fitted independently.
        test_specs: dict[str, dict] | None; optional external CSV cohort specifications.
        subject_col: str | None; default patient identifier column.
        basis_cohorts: iterable[str] | None; basis-library cohorts to load.
        basis_dir: str | Path; directory containing basis shards.
        methods: iterable[str]; optimization backends named in algorithms.__all__.
        weight_mode: str; RMS-rho initialization mode.
        dataset_weights: dict[str, float] | None; optional dataset contributions.
        regularization_strength: float; elastic-net budget relative to unpenalized merit gain.
        project_prediction_variance: bool; whether compatible solvers project every update.
        seed: int; random seed for fitting and validation.
        within_dataset_folds: int | "loocv" | None; isolated patient folds per dataset or no patient CV.
        cross_dataset_validation: bool; whether to fit D-1 datasets and test the omitted dataset.
        out_dir: str | Path | None; optional output root containing one directory per target.
        output_type: str | None; neuroimage output type, or no map export when omitted.
        mask_path: str | Path | None; mask or ordered feature reference used for input and output.
        store_history: bool; whether to save coefficient and objective history.
        render_gif: bool; whether to render optimization history as a GIF.
        gif_fps: int; GIF frames per second.
        gif_max_frames: int; maximum rendered history frames.
        gif_view: str; visualization view passed to the history renderer.
        winner_only: bool; whether to save only the highest-performing algorithm.
        n_permutations: int; basis-map regression permutations, or zero to skip inference.
        """
        targets = tuple(targets)
        fitting_cohorts = pre_processing.load_target_cohorts(dataset_specs, targets, mask_path, subject_col)
        test_cohorts = pre_processing.load_target_cohorts(test_specs, targets, mask_path, subject_col) if test_specs else None
        comparisons = {}
        for target in targets:
            fitting = pre_processing.target_datasets(fitting_cohorts, target)
            test = None if test_cohorts is None else pre_processing.target_datasets(test_cohorts, target)
            optimizer = cls(fitting, cohorts=basis_cohorts, basis_dir=basis_dir, weight_mode=weight_mode, dataset_weights=dataset_weights, regularization_strength=regularization_strength, project_prediction_variance=project_prediction_variance, seed=seed)
            comparisons[target] = optimizer.orchestrate_run(methods, within_dataset_folds, cross_dataset_validation, test)
            if out_dir is not None:
                target_dir = Path(out_dir) / target
                optimizer.save(target_dir, output_type, mask_path, store_history, render_gif, gif_fps, gif_max_frames, gif_view, winner_only=winner_only)
                if n_permutations:
                    inference_dir = target_dir if winner_only else target_dir / optimizer.method
                    optimizer.result["basis_map_regression"] = optimizer.infer(optimizer.result, inference_dir, output_type, mask_path, n_permutations)
        if out_dir is not None:
            saving.target_validation(comparisons, out_dir)
        return comparisons
