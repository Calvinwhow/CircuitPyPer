"""Fit anatomical neural integration from typed algorithm results."""

from .model_fitter_schema import NetworkParameters, NetworkResult, NetworkTraining


class NetworkFitter:
    """Translate typed fitting state across the unchanged neural-network boundary.

    damage: DamageMatrix | scalar object | canonical basis damage and reconstruction state.
    config: ModelFitterConfig | scalar object | shared fitting configuration.
    dependencies: ModelFitterDependencies | scalar object | numerical operations supplied by the composition root.
    """

    METHOD = "anatomical_neural_network"

    def __init__(self, damage, config, dependencies):
        self.damage = damage
        self.config = config
        self.dependencies = dependencies

    def fit(self, algorithm_results, datasets, seed, final):
        """result: NetworkResult | scalar object | anatomical integration of all fitted algorithms."""
        algorithm_results = tuple(algorithm_results)
        fits = {result.method: {"weights": result.weights} for result in algorithm_results}
        network_damage = self._network_damage(datasets)
        fitted = self.dependencies.network_fit(
            fits, network_damage, seed, final, self.config.dataset_weights, self.config.dataset_signs,
            self.config.fully_trainable, self.config.learning_rate, self.config.max_epochs, self.config.tolerance,
            self.config.patience, self.config.weight_decay, self.config.gradient_clip, self.config.device,
        )
        training_values = fitted["fit"]
        parameters = self._parameters(fitted["network_parameters"])
        parameter_trajectory = self._parameter_trajectory(training_values["parameter_trajectory"])
        training = NetworkTraining(
            predictions=dict(training_values["predictions"]), prediction_scale=float(training_values["prediction_scale"]), loss=float(training_values["loss"]),
            history=tuple(training_values["history"]), parameter_trajectory=parameter_trajectory,
            epochs=int(training_values["epochs"]), device=training_values["device"], optimizer=training_values["optimizer"], batch_mode=training_values["batch_mode"],
        )
        return NetworkResult(
            method=self.METHOD, damage=self.damage, symptom_map=fitted["symptom_map"], weights=fitted["weights"], domain_maps=fitted["domain_maps"],
            model=fitted["model"], mode=fitted["mode"], methods=tuple(fitted["methods"]), algorithm_fits=algorithm_results, parameters=parameters,
            representation_maps=fitted["representation_maps"], basis_representations=tuple(fitted["basis_representations"]),
            algorithm_orientations=fitted["algorithm_orientations"], algorithm_training_correlations=fitted["algorithm_training_correlations"],
            dataset_signs=dict(fitted["dataset_signs"]), dataset_weights=dict(fitted["dataset_weights"]), training=training,
        )

    def _network_damage(self, datasets):
        """Build the single private dictionary required by the unchanged network API."""
        translated = {data.dataset_name: {"X": data.basis_damage_scores, "patients": data.patient_maps, "y": data.outcome_values, "ids": data.patient_ids} for data in datasets}
        return {"datasets": translated, "library": self.damage.basis_library, "scales": self.damage.basis_l2_norms}

    @staticmethod
    def _parameters(values):
        """Convert exported network parameters to their typed schema."""
        return NetworkParameters(
            algorithm_weights=values["algorithm_weights"], algorithm_bias=values["algorithm_bias"], algorithm_mix=values["algorithm_mix"],
            representation_bias=values["representation_bias"], representation_mix=values["representation_mix"], representation_baseline=values["representation_baseline"],
        )

    @staticmethod
    def _parameter_trajectory(values):
        """Resolve optional neural parameter recording outside schema construction."""
        if values is None:
            return None
        return tuple(values)
