"""Fit and update one basis-weight optimization algorithm."""

from dataclasses import replace
from functools import partial

import numpy as np

from .model_fitter_schema import AlgorithmFitDetails, AlgorithmResult, Regularization, RmsStart, SparseSemidefiniteDiagnostics


class AlgorithmFitter:
    """Calibrate, fit, normalize, and update basis-weight algorithms.

    damage: DamageMatrix | scalar object | canonical basis damage and reconstruction state.
    config: ModelFitterConfig | scalar object | shared fitting configuration.
    dependencies: ModelFitterDependencies | scalar object | numerical operations supplied by the composition root.
    """

    def __init__(self, damage, config, dependencies):
        self.damage = damage
        self.config = config
        self.dependencies = dependencies

    def fit(self, method, datasets, seed, final):
        """result: AlgorithmResult | scalar object | one calibrated and normalized algorithm solution."""
        damage_scores, outcomes = self._algorithm_inputs(datasets)
        regularization = self._calibrate(method, damage_scores, outcomes, seed)
        fitted = self._optimize(method, damage_scores, outcomes, seed, regularization.l1_strength, regularization.l2_strength)
        dataset_weights = self.dependencies.dataset_weights(damage_scores, self.config.dataset_weights)
        weights = self.dependencies.prediction_variance.normalize(fitted["weights"], damage_scores, dataset_weights)
        symptom_map = self._symptom_map(weights, final)
        return AlgorithmResult(method=method, damage=self.damage, symptom_map=symptom_map, weights=weights, regularization=regularization, details=self._details(fitted))

    def step(self, result, gradient, learning_rate):
        """result: AlgorithmResult | scalar object | supplied result after one normalized gradient-ascent update."""
        damage_scores, _ = self._algorithm_inputs(self.damage.fitting_datasets)
        dataset_weights = self.dependencies.dataset_weights(damage_scores, self.config.dataset_weights)
        updated = result.weights + learning_rate * np.asarray(gradient, dtype=float)
        weights = self.dependencies.prediction_variance.normalize(updated, damage_scores, dataset_weights)
        return replace(result, weights=weights, symptom_map=self.damage.reconstruct_map(weights), apparent=None, test=None, basis_map_regression=None)

    def merit(self, result):
        """merit: float | scalar | algorithm-specific unpenalized objective at the current coefficients."""
        damage_scores, outcomes = self._algorithm_inputs(self.damage.fitting_datasets)
        dataset_weights = self.dependencies.dataset_weights(damage_scores, self.config.dataset_weights)
        algorithm = self._algorithm(result.method)
        return algorithm.evaluate(result.weights, damage_scores, outcomes, dataset_weights)

    @staticmethod
    def _algorithm_inputs(datasets):
        """Resolve typed datasets to the numerical arrays required by existing algorithms."""
        damage_scores = {data.dataset_name: data.basis_damage_scores for data in datasets}
        outcomes = {data.dataset_name: data.outcome_values for data in datasets}
        return damage_scores, outcomes

    def _calibrate(self, method, damage_scores, outcomes, seed):
        """Return calibrated penalties from one unpenalized reference fit."""
        reference = self._optimize(method, damage_scores, outcomes, seed, 0.0, 0.0)
        l1_reference, l2_reference = self._reference_penalties(method, reference)
        values = self.dependencies.calibrate_regularization(reference["merit"], reference["null_merit"], l1_reference, l2_reference, self.config.regularization_strength)
        return Regularization(**values)

    def _reference_penalties(self, method, reference):
        """Resolve vector or lifted-matrix penalty measurements explicitly."""
        if method == "sparse_semidefinite":
            l1_reference = self.dependencies.algorithms.sparse_semidefinite.l1(reference["theta"], 1.0, lambda values: np.linalg.norm(values, 1))
            l2_reference = self.dependencies.algorithms.sparse_semidefinite.l2(reference["theta"], 1.0, lambda values: np.square(values).sum())
            return l1_reference, l2_reference
        algorithm = self._algorithm(method)
        return algorithm.l1(reference["weights"], 1.0), algorithm.l2(reference["weights"], 1.0)

    def _optimize(self, method, damage_scores, outcomes, seed, l1_strength, l2_strength):
        """Dispatch one algorithm through its required solver and arguments."""
        dataset_weights = self.dependencies.dataset_weights(damage_scores, self.config.dataset_weights)
        projector = self._prediction_projector(damage_scores, dataset_weights)
        if method == "rms_rho":
            initial = self.dependencies.initial_weights(len(self.damage.map_names), self.config.weight_mode, sample_sizes=self.damage.sample_sizes, names=self.damage.map_names, seed=seed)
            return self.dependencies.algorithms.rms_rho.main(damage_scores, outcomes, solver=self.dependencies.adam_solver, random_state=seed, initial_weights=initial, l1_strength=l1_strength, l2_strength=l2_strength, dataset_weights=dataset_weights)
        if method == "elastic_net_ols":
            return self.dependencies.algorithms.elastic_net_ols.main(damage_scores, outcomes, solver=self.dependencies.proximal_solver, project=projector, l1_strength=l1_strength, l2_strength=l2_strength)
        if method == "pairwise_proxy":
            return self.dependencies.algorithms.pairwise_proxy.main(damage_scores, outcomes, solver=self.dependencies.proximal_solver, project=projector, l1_strength=l1_strength, l2_strength=l2_strength, dataset_weights=dataset_weights)
        if method == "sparse_semidefinite":
            return self.dependencies.algorithms.sparse_semidefinite.main(damage_scores, outcomes, solver=self.dependencies.semidefinite_solver, l1_strength=l1_strength, l2_strength=l2_strength, dataset_weights=dataset_weights)
        if method == "rayleigh_refinement":
            return self.dependencies.algorithms.rayleigh_refinement.main(damage_scores, outcomes, solver=self.dependencies.adam_solver, project=projector, l1_strength=l1_strength, l2_strength=l2_strength, dataset_weights=dataset_weights)
        raise ValueError(f"Unknown optimization method {method!r}.")

    def _prediction_projector(self, damage_scores, dataset_weights):
        """Resolve the optional unit-prediction-variance projection."""
        if not self.config.project_prediction_variance:
            return None
        covariance = self.dependencies.prediction_variance.gram(damage_scores, dataset_weights)
        return partial(self.dependencies.prediction_variance.project, covariance=covariance)

    def _symptom_map(self, weights, final):
        """Resolve map reconstruction only for a fit on every patient."""
        if not final:
            return None
        return self.damage.reconstruct_map(weights)

    def _algorithm(self, method):
        """Resolve a method name to its numerical algorithm module."""
        if method == "rms_rho":
            return self.dependencies.algorithms.rms_rho
        if method == "elastic_net_ols":
            return self.dependencies.algorithms.elastic_net_ols
        if method == "pairwise_proxy":
            return self.dependencies.algorithms.pairwise_proxy
        if method == "sparse_semidefinite":
            return self.dependencies.algorithms.sparse_semidefinite
        if method == "rayleigh_refinement":
            return self.dependencies.algorithms.rayleigh_refinement
        raise ValueError(f"Unknown optimization method {method!r}.")

    @staticmethod
    def _details(fitted):
        """Convert one numerical algorithm dictionary into the canonical typed schema."""
        starts = tuple(RmsStart(**record) for record in fitted.get("starts", ()))
        sparse = AlgorithmFitter._sparse_diagnostics(fitted)
        history = tuple((np.asarray(weights), float(value)) for weights, value in fitted.get("history", ()))
        return AlgorithmFitDetails(
            objective=float(fitted["objective"]), merit=float(fitted["merit"]), null_merit=float(fitted["null_merit"]), history=history,
            converged=fitted.get("converged"), n_iterations=fitted.get("n_iterations"), dataset_signs=dict(fitted.get("dataset_signs", {})),
            r2=fitted.get("r2"), dataset_r2=dict(fitted.get("dataset_r2", {})), orientation_iterations=fitted.get("orientation_iterations"),
            initial_weights=fitted.get("initial_weights"), leading_generalized_eigenvalue=fitted.get("leading_generalized_eigenvalue"), pooled_g_rank=fitted.get("pooled_g_rank"),
            association_matrix=fitted.get("B"), covariance_matrix=fitted.get("G"), best_start=fitted.get("best_start"), starts=starts,
            start_weights=fitted.get("start_weights"), sparse_semidefinite=sparse,
        )

    @staticmethod
    def _sparse_diagnostics(fitted):
        """Resolve optional sparse-semidefinite diagnostics."""
        if "theta" not in fitted:
            return None
        return SparseSemidefiniteDiagnostics(
            theta=fitted["theta"], eigenvalues=fitted["eigenvalues"], status=fitted["status"], solver=fitted["solver"], solve_time=float(fitted["solve_time"]),
            leading_eigenvalue=float(fitted["leading_eigenvalue"]), positive_eigenvalue_sum=float(fitted["positive_eigenvalue_sum"]),
            leading_eigenvalue_fraction=float(fitted["leading_eigenvalue_fraction"]), numerical_rank=int(fitted["numerical_rank"]),
            spectral_mass_threshold=float(fitted["spectral_mass_threshold"]), leading_vector_recovery_acceptable=bool(fitted["leading_vector_recovery_acceptable"]),
            recovery_recommendation=fitted["recovery_recommendation"],
        )
