"""Typed data exchanged by the model-fitting hierarchy."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ModelFitterConfig:
    """Configuration shared by every fit in one optimization run.

    methods: tuple[str] | length algorithm_count | ordered optimization algorithms to fit.
    neural_network: bool | scalar | whether to fit anatomical neural integration after the algorithms.
    fully_trainable: bool | scalar | whether neural integration may update basis coefficients.
    weight_mode: str | scalar | initialization rule used by algorithms requiring initial coefficients.
    dataset_weights: dict[str, float] or None | length dataset_count | optional contribution of each clinical dataset.
    dataset_signs: dict[str, float] or None | length dataset_count | optional neural-network outcome orientations.
    regularization_strength: float | scalar | penalty budget as a fraction of unpenalized merit gain.
    project_prediction_variance: bool | scalar | whether supported algorithms project coefficients during fitting.
    learning_rate: float | scalar | neural-network optimizer step size.
    max_epochs: int | scalar | maximum neural-network training epochs.
    tolerance: float | scalar | neural-network improvement threshold.
    patience: int | scalar | neural-network epochs allowed without sufficient improvement.
    weight_decay: float | scalar | neural-network Adam weight decay.
    gradient_clip: float or None | scalar | optional neural-network gradient magnitude limit.
    device: str | scalar | requested neural-network compute device.
    """

    methods: tuple[str, ...]
    neural_network: bool
    fully_trainable: bool
    weight_mode: str
    dataset_weights: dict[str, float] | None
    dataset_signs: dict[str, float] | None
    regularization_strength: float
    project_prediction_variance: bool
    learning_rate: float
    max_epochs: int
    tolerance: float
    patience: int
    weight_decay: float
    gradient_clip: float | None
    device: str


@dataclass(frozen=True)
class ModelFitterDependencies:
    """Numerical functions supplied by the optimization composition root.

    algorithms: object | scalar module | available basis-weight algorithm modules.
    adam_solver: callable | scalar callable | ADAM numerical solver.
    proximal_solver: callable | scalar callable | proximal-gradient numerical solver.
    semidefinite_solver: callable | scalar callable | semidefinite numerical solver.
    prediction_variance: object | scalar module | prediction-variance projection and normalization operations.
    calibrate_regularization: callable | scalar callable | elastic-net penalty calibration function.
    dataset_weights: callable | scalar callable | dataset contribution normalization function.
    initial_weights: callable | scalar callable | basis-coefficient initialization function.
    network_fit: callable | scalar callable | unchanged anatomical-network fitting boundary.
    """

    algorithms: Any
    adam_solver: Any
    proximal_solver: Any
    semidefinite_solver: Any
    prediction_variance: Any
    calibrate_regularization: Any
    dataset_weights: Any
    initial_weights: Any
    network_fit: Any


@dataclass(frozen=True)
class Regularization:
    """Calibrated elastic-net penalty values for one algorithm.

    regularization_strength: float | scalar | requested fraction of unpenalized merit gain assigned to penalties.
    elastic_net_alpha: float | scalar | fraction of combined penalty strength assigned to L1.
    reference_merit: float | scalar | unpenalized fitted merit.
    reference_null_merit: float | scalar | merit of the null parameters.
    reference_merit_gain: float | scalar | improvement of the unpenalized fit over null.
    l1_reference: float | scalar | unit-strength L1 cost of the unpenalized parameters.
    l2_reference: float | scalar | unit-strength L2 cost of the unpenalized parameters.
    l1_strength: float | scalar | calibrated L1 penalty coefficient.
    l2_strength: float | scalar | calibrated L2 penalty coefficient.
    """

    regularization_strength: float
    elastic_net_alpha: float
    reference_merit: float
    reference_null_merit: float
    reference_merit_gain: float
    l1_reference: float
    l2_reference: float
    l1_strength: float
    l2_strength: float


@dataclass(frozen=True)
class RmsStart:
    """One RMS-rho multistart attempt.

    start: int | scalar | one-based start number.
    initialization: str | scalar | initialization rule used for this start.
    seed: int or None | scalar | random seed associated with this start.
    training_rms_rho: float | scalar | fitted RMS Spearman correlation.
    iterations: int | scalar | optimizer iterations completed.
    selected: bool | scalar | whether this start supplied the final coefficients.
    """

    start: int
    initialization: str
    seed: int | None
    training_rms_rho: float
    iterations: int
    selected: bool


@dataclass(frozen=True)
class SparseSemidefiniteDiagnostics:
    """Recovery and solver diagnostics for a sparse semidefinite fit.

    theta: np.ndarray | shape basis_count by basis_count | fitted lifted positive-semidefinite parameter matrix.
    eigenvalues: np.ndarray | shape basis_count | eigenvalues of the symmetrized lifted solution.
    status: str | scalar | conic-solver termination status.
    solver: str | scalar | conic solver used.
    solve_time: float | scalar | elapsed solve time in seconds.
    leading_eigenvalue: float | scalar | largest nonnegative lifted eigenvalue.
    positive_eigenvalue_sum: float | scalar | sum of nonnegative lifted eigenvalues.
    leading_eigenvalue_fraction: float | scalar | spectral mass captured by the leading eigenvector.
    numerical_rank: int | scalar | numerical rank of the lifted solution.
    spectral_mass_threshold: float | scalar | required leading-eigenvalue fraction.
    leading_vector_recovery_acceptable: bool | scalar | whether recovery passed the spectral threshold.
    recovery_recommendation: str | scalar | human-readable recovery assessment.
    """

    theta: NDArray[np.float64]
    eigenvalues: NDArray[np.float64]
    status: str
    solver: str
    solve_time: float
    leading_eigenvalue: float
    positive_eigenvalue_sum: float
    leading_eigenvalue_fraction: float
    numerical_rank: int
    spectral_mass_threshold: float
    leading_vector_recovery_acceptable: bool
    recovery_recommendation: str


@dataclass(frozen=True)
class AlgorithmFitDetails:
    """Typed numerical details retained from one algorithm fit.

    objective: float | scalar | penalized objective at the returned coefficients.
    merit: float | scalar | unpenalized algorithm merit at the returned coefficients.
    null_merit: float | scalar | merit of null parameters.
    history: tuple[tuple[np.ndarray, float]] | length optimizer_iteration_count | coefficient and objective trajectory.
    converged: bool or None | scalar | iterative-solver convergence flag when applicable.
    n_iterations: int or None | scalar | numerical iterations completed when applicable.
    dataset_signs: dict[str, float] | length dataset_count | fitted dataset orientations when applicable.
    r2: float or None | scalar | pooled OLS coefficient of determination when applicable.
    dataset_r2: dict[str, float] | length dataset_count | dataset-specific OLS coefficients of determination.
    orientation_iterations: int or None | scalar | pairwise orientation updates when applicable.
    initial_weights: np.ndarray or None | shape basis_count | analytical starting coefficients when applicable.
    leading_generalized_eigenvalue: float or None | scalar | leading Rayleigh initialization eigenvalue.
    pooled_g_rank: int or None | scalar | supported pooled covariance rank.
    association_matrix: np.ndarray or None | shape basis_count by basis_count | fitted algorithm association matrix.
    covariance_matrix: np.ndarray or None | shape basis_count by basis_count | pooled basis-damage covariance matrix.
    best_start: int or None | scalar | selected one-based RMS-rho start.
    starts: tuple[RmsStart] | length RMS start count | RMS-rho multistart summaries.
    start_weights: np.ndarray or None | shape start_count by basis_count | final coefficients from every RMS-rho start.
    sparse_semidefinite: SparseSemidefiniteDiagnostics or None | scalar object | lifted-solution diagnostics when applicable.
    """

    objective: float
    merit: float
    null_merit: float
    history: tuple[tuple[NDArray[np.float64], float], ...] = ()
    converged: bool | None = None
    n_iterations: int | None = None
    dataset_signs: dict[str, float] = field(default_factory=dict)
    r2: float | None = None
    dataset_r2: dict[str, float] = field(default_factory=dict)
    orientation_iterations: int | None = None
    initial_weights: NDArray[np.float64] | None = None
    leading_generalized_eigenvalue: float | None = None
    pooled_g_rank: int | None = None
    association_matrix: NDArray[np.float64] | None = None
    covariance_matrix: NDArray[np.float64] | None = None
    best_start: int | None = None
    starts: tuple[RmsStart, ...] = ()
    start_weights: NDArray[np.float64] | None = None
    sparse_semidefinite: SparseSemidefiniteDiagnostics | None = None


@dataclass(frozen=True)
class PredictionRecord:
    """One observed outcome and model prediction.

    dataset: str | scalar | dataset containing the patient.
    patient_id: str | scalar | stable patient identifier.
    observed: float | scalar | observed clinical outcome.
    prediction: float | scalar | model prediction.
    fold: int or None | scalar | one-based validation fold when cross-validated.
    """

    dataset: str
    patient_id: str
    observed: float
    prediction: float
    fold: int | None = None


@dataclass(frozen=True)
class WeightRecord:
    """One basis coefficient retained from one validation fold.

    fold: int | scalar | one-based validation fold.
    map_name: str | scalar | basis-map name.
    weight: float | scalar | fitted basis coefficient.
    """

    fold: int
    map_name: str
    weight: float


@dataclass(frozen=True)
class DatasetPerformance:
    """Prediction performance for one dataset.

    n_patients: int | scalar | number of evaluated patients.
    spearman_rho: float | scalar | Spearman correlation between predictions and outcomes.
    p_value: float | scalar | two-sided Spearman p-value.
    n_missing_outcomes: int | scalar | excluded patients lacking outcomes.
    rho_method: str or None | scalar | fold aggregation rule when cross-validated.
    n_folds: int or None | scalar | number of contributing validation folds.
    fold_rhos: tuple[float] | length usable fold count | correlations computed within sufficiently large folds.
    pooled_spearman_rho: float or None | scalar | correlation of pooled out-of-fold predictions.
    """

    n_patients: int
    spearman_rho: float
    p_value: float
    n_missing_outcomes: int = 0
    rho_method: str | None = None
    n_folds: int | None = None
    fold_rhos: tuple[float, ...] = ()
    pooled_spearman_rho: float | None = None


@dataclass(frozen=True)
class ModelEvaluation:
    """Predictions and summary statistics from one evaluation stage.

    predictions: tuple[PredictionRecord] | length evaluated_patient_count | patient-level predictions.
    summary: dict[str, DatasetPerformance] | length dataset_count | performance indexed by dataset name.
    weights: tuple[WeightRecord] | length recorded_fold_weight_count | algorithm coefficients retained across folds.
    scheme: str or None | scalar | cross-validation scheme, absent for apparent and external evaluation.
    """

    predictions: tuple[PredictionRecord, ...]
    summary: dict[str, DatasetPerformance]
    weights: tuple[WeightRecord, ...] = ()
    scheme: str | None = None


@dataclass(kw_only=True)
class ModelResult:
    """Common state carried by every fitted or cross-validated model.

    method: str | scalar | unique model name.
    damage: object | scalar | canonical damage matrix used by the model.
    symptom_map: np.ndarray or None | shape voxel_count | final fitted map, absent for cross-validation.
    apparent: ModelEvaluation or None | scalar object | performance on fitting patients.
    cross_validation: ModelEvaluation or None | scalar object | held-out fitting performance.
    test: ModelEvaluation or None | scalar object | performance on an external cohort.
    basis_map_regression: object or None | scalar object | optional post-fit basis-map regression output.
    """

    method: str
    damage: Any
    symptom_map: NDArray[np.float32] | None
    apparent: ModelEvaluation | None = None
    cross_validation: ModelEvaluation | None = None
    test: ModelEvaluation | None = None
    basis_map_regression: Any | None = None


@dataclass(kw_only=True)
class AlgorithmResult(ModelResult):
    """Final state produced by one basis-weight optimization algorithm.

    weights: np.ndarray | shape basis_count | normalized coefficients on the basis maps.
    regularization: Regularization | scalar object | calibrated penalty values.
    details: AlgorithmFitDetails | scalar object | typed numerical diagnostics and history.
    """

    weights: NDArray[np.float64]
    regularization: Regularization
    details: AlgorithmFitDetails


@dataclass(frozen=True)
class NetworkParameters:
    """Exported parameters of the anatomical integration network.

    algorithm_weights: np.ndarray | shape basis_count by algorithm_count | basis coefficients entering the network.
    algorithm_bias: np.ndarray | shape representation_count by algorithm_count | algorithm-layer biases.
    algorithm_mix: np.ndarray | shape representation_count by algorithm_count | algorithm-layer mixing weights.
    representation_bias: np.ndarray | shape representation_count | representation-layer biases.
    representation_mix: np.ndarray | shape representation_count | final representation mixing weights.
    representation_baseline: np.ndarray | shape representation_count | representation baseline contributions.
    """

    algorithm_weights: NDArray[np.float32]
    algorithm_bias: NDArray[np.float32]
    algorithm_mix: NDArray[np.float32]
    representation_bias: NDArray[np.float32]
    representation_mix: NDArray[np.float32]
    representation_baseline: NDArray[np.float32]


@dataclass(frozen=True)
class NetworkTraining:
    """Typed output of anatomical-network training.

    predictions: dict[str, np.ndarray] | length dataset_count | normalized predictions indexed by dataset.
    prediction_scale: float | scalar | pooled prediction standard deviation removed from the fitted map.
    loss: float | scalar | final pairwise-softplus loss.
    history: tuple[dict[str, float]] | length epoch_count | epoch and loss records.
    parameter_trajectory: tuple[dict[str, object]] or None | length recorded_epoch_count | parameter snapshots used for replay.
    epochs: int | scalar | completed training epochs.
    device: str | scalar | compute device used.
    optimizer: str | scalar | neural optimizer name.
    batch_mode: str | scalar | neural batching strategy.
    """

    predictions: dict[str, NDArray[np.float32]]
    prediction_scale: float
    loss: float
    history: tuple[dict[str, float], ...]
    parameter_trajectory: tuple[dict[str, object], ...] | None
    epochs: int
    device: str
    optimizer: str
    batch_mode: str


@dataclass(kw_only=True)
class NetworkResult(ModelResult):
    """Final state produced by anatomical neural integration.

    weights: np.ndarray | shape basis_count by algorithm_count | oriented algorithm coefficients entering the network.
    domain_maps: np.ndarray | shape representation_count by voxel_count | additive anatomical-domain maps.
    model: object | scalar | fitted anatomical network.
    mode: str | scalar | frozen or fine-tuned basis-coefficient mode.
    methods: tuple[str] | length algorithm_count | integrated algorithm names.
    algorithm_fits: tuple[AlgorithmResult] | length algorithm_count | source algorithm solutions.
    parameters: NetworkParameters | scalar object | exported network parameters.
    representation_maps: np.ndarray | shape representation_count by voxel_count | post-activation representation maps.
    basis_representations: tuple[str] | length basis_count | anatomical domain assigned to every basis.
    algorithm_orientations: np.ndarray | shape algorithm_count | fitted algorithm directions.
    algorithm_training_correlations: np.ndarray | shape dataset_count by algorithm_count | training correlations used for orientation.
    dataset_signs: dict[str, float] | length dataset_count | fitted clinical outcome directions.
    dataset_weights: dict[str, float] | length dataset_count | normalized dataset contributions.
    training: NetworkTraining | scalar object | typed neural training output.
    """

    weights: NDArray[np.float64]
    domain_maps: NDArray[np.float32]
    model: Any
    mode: str
    methods: tuple[str, ...]
    algorithm_fits: tuple[AlgorithmResult, ...]
    parameters: NetworkParameters
    representation_maps: NDArray[np.float32]
    basis_representations: tuple[str, ...]
    algorithm_orientations: NDArray[np.float64]
    algorithm_training_correlations: NDArray[np.float64]
    dataset_signs: dict[str, float]
    dataset_weights: dict[str, float]
    training: NetworkTraining


@dataclass(frozen=True)
class ModelResults:
    """Ordered model results returned by one fitting or validation pass.

    models: tuple[ModelResult] | length model_count | one typed result per configured model.
    """

    models: tuple[ModelResult, ...]

    @property
    def methods(self):
        """methods: tuple[str] | length model_count | model names in fitting order."""
        return tuple(model.method for model in self.models)

    def get(self, method):
        """Return the uniquely named model result."""
        matches = tuple(model for model in self.models if model.method == method)
        if len(matches) != 1:
            raise KeyError(f"Expected one result named {method!r}; found {len(matches)}.")
        return matches[0]

    def replace(self, updated):
        """Return the collection with one same-named result replaced."""
        if updated.method not in self.methods:
            raise KeyError(f"Unknown model result {updated.method!r}.")
        models = []
        for model in self.models:
            if model.method == updated.method:
                models.append(updated)
            else:
                models.append(model)
        return ModelResults(tuple(models))


@dataclass(frozen=True)
class ModelComparison:
    """Ranked model performance for one optimization target.

    winner: str | scalar | highest-ranked model name.
    ranking: tuple[str] | length model_count | model names ordered from best to worst.
    performance: dict[str, float] | length model_count | RMS Spearman score for each model.
    summaries: dict[str, dict[str, DatasetPerformance]] | length model_count | selected evaluation summaries by model.
    """

    winner: str
    ranking: tuple[str, ...]
    performance: dict[str, float]
    summaries: dict[str, dict[str, DatasetPerformance]]
