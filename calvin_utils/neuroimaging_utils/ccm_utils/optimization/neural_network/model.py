"""Readable composition of the anatomical neural-network layers."""

from .dependencies import require_torch
from .layers import AlgorithmMapLayer, AnatomicalRepresentationLayer, FinalLocalizationLayer


torch = require_torch()
nn = torch.nn


class AnatomicalNetwork(nn.Module):
    """Compose coefficient reconstruction, anatomical integration, and localization."""

    def __init__(self, algorithm_weights, representation_masks, *, fixed_algorithm_maps=None, basis_maps=None, mode="frozen", scratch_scale=0.01):
        super().__init__()
        self.algorithm_layer = AlgorithmMapLayer(algorithm_weights, representation_masks, fixed_maps=fixed_algorithm_maps, basis_maps=basis_maps, mode=mode, scratch_scale=scratch_scale)
        # The integration layers start where their output equals the linear combination of the initial algorithm maps.
        with torch.no_grad(): initial_algorithm_maps = self.algorithm_layer()
        self.anatomy_layer = AnatomicalRepresentationLayer(initial_algorithm_maps)
        self.localization_layer = FinalLocalizationLayer(self.anatomy_layer.linear_offsets())

    @property
    def mode(self):
        return self.algorithm_layer.mode

    @property
    def algorithm_weights(self):
        return self.algorithm_layer.algorithm_weights

    @property
    def algorithm_bias(self):
        return self.anatomy_layer.algorithm_bias

    @property
    def algorithm_mix(self):
        return self.anatomy_layer.algorithm_mix

    @property
    def representation_bias(self):
        return self.anatomy_layer.representation_bias

    @property
    def representation_mix(self):
        return self.localization_layer.representation_mix

    @property
    def representation_baseline(self):
        return self.localization_layer.representation_baseline

    def masked_algorithm_maps(self):
        return self.algorithm_layer()

    def forward_map(self):
        algorithm_maps = self.algorithm_layer()
        representation_maps = self.anatomy_layer(algorithm_maps)
        localization_map = self.localization_layer(representation_maps)
        return localization_map, representation_maps, algorithm_maps

    def forward(self, patient_maps):
        localization_map, representation_maps, algorithm_maps = self.forward_map()
        predictions = {name: patients @ localization_map for name, patients in patient_maps.items()}
        return {"map": localization_map, "representations": representation_maps, "algorithm_maps": algorithm_maps, "predictions": predictions}

    def unfreeze_filters(self, basis_maps):
        self.algorithm_layer.unfreeze(basis_maps)
        return self

    def freeze_filters(self):
        self.algorithm_layer.freeze()
        return self

    def export(self):
        parameters = {}
        parameters["mode"] = self.mode
        parameters["algorithm_weights"] = self.algorithm_weights.detach().cpu().numpy()
        parameters["algorithm_bias"] = self.algorithm_bias.detach().cpu().numpy()
        parameters["algorithm_mix"] = self.algorithm_mix.detach().cpu().numpy()
        parameters["representation_bias"] = self.representation_bias.detach().cpu().numpy()
        parameters["representation_mix"] = self.representation_mix.detach().cpu().numpy()
        parameters["representation_baseline"] = self.representation_baseline.detach().cpu().numpy()
        return parameters
