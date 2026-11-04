"""Integrate the algorithm solutions with the anatomical network."""

import numpy as np
from . import anatomy, training
from ..pipeline import pre_processing, prediction
from .dependencies import require_torch


def initial_network(coefficients, masks, damage, fully_trainable):
    """The network at its linear starting point: frozen algorithm maps, or the coefficient matrix itself as a trainable first layer."""
    from .model import AnatomicalNetwork
    if fully_trainable:
        return AnatomicalNetwork(coefficients.copy(), masks, basis_maps=anatomy.normalized_basis(damage["library"], damage["scales"]), mode="fine_tune")  # torch shares numpy memory; training must not overwrite the algorithm solutions
    return AnatomicalNetwork(coefficients, masks, fixed_algorithm_maps=anatomy.reconstruct_algorithm_maps(coefficients, damage["library"], damage["scales"], masks), mode="frozen")


def main(fits, damage, seed, record, dataset_weights, dataset_signs, fully_trainable, learning_rate, max_epochs, tolerance, patience, weight_decay, gradient_clip, device):
    """Orient the algorithms' weight columns, then train the network on the fitting patients' voxel maps."""
    datasets = damage["datasets"]
    weights = pre_processing.dataset_weights(datasets, dataset_weights)
    supplied = None if dataset_signs is None else {name: dataset_signs[name] for name in datasets}
    coefficients, orientations, signs, correlations = anatomy.orient_algorithm_weights(np.column_stack([fit["weights"] for fit in fits.values()]), datasets, weights, supplied)
    labels = anatomy.basis_representations(damage["library"])
    require_torch().manual_seed(int(seed))
    network = initial_network(coefficients, anatomy.representation_masks(labels), damage, fully_trainable)
    trained = training.fit(network, {name: prediction.normalize(data["patients"]) for name, data in datasets.items()}, {name: data["y"] for name, data in datasets.items()}, weights, signs, learning_rate=learning_rate, max_epochs=max_epochs, tolerance=tolerance, patience=patience, weight_decay=weight_decay, gradient_clip=gradient_clip, device=device, seed=seed, record_parameters=record)
    model = trained.pop("model")
    return {"weights": coefficients, "symptom_map": trained["map"], "fit": trained, "model": model, "mode": model.mode, "methods": tuple(fits), "algorithm_fits": fits, "network_parameters": model.export(), "representation_maps": trained["representations"], "basis_representations": labels, "algorithm_orientations": orientations, "algorithm_training_correlations": correlations, "dataset_signs": signs, "dataset_weights": weights}
