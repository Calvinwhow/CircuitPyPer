"""Single entry and exit for fitting every optimization model."""

from .model_fitter_utils.algorithm_fitter import AlgorithmFitter
from .model_fitter_utils.network_fitter import NetworkFitter
from .model_fitter_utils.model_fitter_schema import ModelResults


class ModelFitter:
    """Fit configured algorithms and optionally integrate them anatomically.

    damage: DamageMatrix | scalar object | canonical basis damage and reconstruction state.
    config: ModelFitterConfig | scalar object | immutable configuration shared by every fit.
    dependencies: ModelFitterDependencies | scalar object | numerical operations supplied by the composition root.
    algorithm_fitter: AlgorithmFitter | scalar object | child responsible for basis-weight algorithms.
    network_fitter: NetworkFitter | scalar object | child responsible for anatomical neural integration.
    """

    def __init__(self, damage, config, dependencies):
        self.damage = damage
        self.config = config
        self.dependencies = dependencies
        self.algorithm_fitter = AlgorithmFitter(damage, config, dependencies)
        self.network_fitter = NetworkFitter(damage, config, dependencies)

    def fit(self, seed, final, datasets=None):
        """results: ModelResults | scalar object | ordered typed results from every configured model."""
        datasets = self._datasets(datasets)
        results = []
        for index, method in enumerate(self.config.methods):
            print(f"  fitting {method}", flush=True)
            results.append(self.algorithm_fitter.fit(method, datasets, seed + index, final))
        if self.config.neural_network:
            print(f"  fitting {self.network_fitter.METHOD}", flush=True)
            results.append(self.network_fitter.fit(results, datasets, seed, final))
        return ModelResults(tuple(results))

    def step(self, result, gradient, learning_rate):
        """result: AlgorithmResult | scalar object | updated algorithm result."""
        return self.algorithm_fitter.step(result, gradient, learning_rate)

    def merit(self, result):
        """merit: float | scalar | current native merit of an algorithm result."""
        return self.algorithm_fitter.merit(result)

    def _datasets(self, datasets):
        """Resolve omitted fold data to all fitting datasets."""
        if datasets is None:
            return self.damage.fitting_datasets
        return tuple(datasets)
