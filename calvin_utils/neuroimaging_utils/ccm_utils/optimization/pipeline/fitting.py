"""Fit every model to one split of patient-by-basis damage: each algorithm directly, then the anatomical network on their solutions."""

import numpy as np
from functools import partial
from . import basis_damage, pre_processing
from .. import algorithms
from ..neural_network import fitting as network_fitting
from ..projections import prediction_variance
from ..selection import prepare_regularization
from ..solvers import adam, proximal_gradient, semidefinite

NETWORK = "anatomical_neural_network"


def optimize(method, X, y, names, sample_sizes, seed, l1_strength, l2_strength, weight_mode, dataset_weights, project_prediction_variance):
    """Run one algorithm with fixed penalties. X: dict[str, (N, B)] damage; y: dict[str, (N,)] outcomes."""
    weights = pre_processing.dataset_weights(X, dataset_weights)
    project = partial(prediction_variance.project, covariance=prediction_variance.gram(X, weights)) if project_prediction_variance else None
    if method == "rms_rho":
        initial = pre_processing.initial_weights(len(names), weight_mode, sample_sizes=sample_sizes, names=names, seed=seed)
        return algorithms.rms_rho.main(X, y, solver=adam.main, random_state=seed, initial_weights=initial, l1_strength=l1_strength, l2_strength=l2_strength, dataset_weights=weights)
    if method == "elastic_net_ols":
        return algorithms.elastic_net_ols.main(X, y, solver=proximal_gradient.main, project=project, l1_strength=l1_strength, l2_strength=l2_strength)
    if method == "pairwise_proxy":
        return algorithms.pairwise_proxy.main(X, y, solver=proximal_gradient.main, project=project, l1_strength=l1_strength, l2_strength=l2_strength, dataset_weights=weights)
    if method == "sparse_semidefinite":
        return algorithms.sparse_semidefinite.main(X, y, solver=semidefinite.main, l1_strength=l1_strength, l2_strength=l2_strength, dataset_weights=weights)
    if method == "rayleigh_refinement":
        return algorithms.rayleigh_refinement.main(X, y, solver=adam.main, project=project, l1_strength=l1_strength, l2_strength=l2_strength, dataset_weights=weights)
    raise ValueError(f"Unknown optimization method {method!r}.")


def calibrate(method, X, y, names, sample_sizes, seed, weight_mode, dataset_weights, regularization_strength, project_prediction_variance):
    """Penalty strengths from one unpenalized fit: their combined cost is regularization_strength times its gain over the null merit."""
    reference = optimize(method, X, y, names, sample_sizes, seed, 0.0, 0.0, weight_mode, dataset_weights, project_prediction_variance)
    if method == "sparse_semidefinite":
        l1_reference = algorithms.sparse_semidefinite.l1(reference["theta"], 1.0, lambda values: np.linalg.norm(values, 1))
        l2_reference = algorithms.sparse_semidefinite.l2(reference["theta"], 1.0, lambda values: np.square(values).sum())
    else:
        l1_reference = getattr(algorithms, method).l1(reference["weights"], 1.0)
        l2_reference = getattr(algorithms, method).l2(reference["weights"], 1.0)
    return prepare_regularization.main(reference["merit"], reference["null_merit"], l1_reference, l2_reference, regularization_strength)


def fit_algorithm(method, X, y, names, sample_sizes, seed, weight_mode, dataset_weights, regularization_strength, project_prediction_variance):
    """Calibrate the penalties, fit, and scale the weights to unit pooled prediction variance."""
    regularization = calibrate(method, X, y, names, sample_sizes, seed, weight_mode, dataset_weights, regularization_strength, project_prediction_variance)
    fit = optimize(method, X, y, names, sample_sizes, seed, regularization["l1_strength"], regularization["l2_strength"], weight_mode, dataset_weights, project_prediction_variance)
    fit["weights"] = prediction_variance.normalize(fit["weights"], X, pre_processing.dataset_weights(X, dataset_weights))
    return {**fit, **regularization}


def main(damage, seed, final, *, methods, neural_network, fully_trainable, weight_mode, dataset_weights, dataset_signs, regularization_strength, project_prediction_variance, learning_rate, max_epochs, tolerance, patience, weight_decay, gradient_clip, device):
    """Fit every algorithm in methods to damage["datasets"], then the network on those same solutions when neural_network is set.
    final: the fit on every fitting patient, which builds each algorithm's voxel map and records the network's per-epoch parameters; folds need neither.
    """
    X = {name: data["X"] for name, data in damage["datasets"].items()}
    y = {name: data["y"] for name, data in damage["datasets"].items()}
    library = {"map_names": damage["map_names"], "library": damage["library"], "scales": damage["scales"]}
    results = {}
    for index, method in enumerate(methods):
        print(f"  fitting {method}", flush=True)
        fit = fit_algorithm(method, X, y, damage["map_names"], damage["sample_sizes"], seed + index, weight_mode, dataset_weights, regularization_strength, project_prediction_variance)
        results[method] = {"method": method, "weights": fit["weights"], "symptom_map": basis_damage.combine(fit["weights"], damage["library"], damage["scales"]) if final else None, "fit": fit, **library}
    if neural_network:
        print(f"  fitting {NETWORK}", flush=True)
        results[NETWORK] = {"method": NETWORK, **network_fitting.main(results, damage, seed, final, dataset_weights, dataset_signs, fully_trainable, learning_rate, max_epochs, tolerance, patience, weight_decay, gradient_clip, device), **library}
    return results

def step(result, damage, gradient, learning_rate, dataset_weights):
    """One external gradient-ascent step on an algorithm's weights, rescaled to unit prediction variance, with its map rebuilt."""
    X = {name: data["X"] for name, data in damage["datasets"].items()}
    weights = prediction_variance.normalize(result["weights"] + learning_rate * np.asarray(gradient, dtype=float), X, pre_processing.dataset_weights(X, dataset_weights))
    return {**result, "weights": weights, "symptom_map": basis_damage.combine(weights, damage["library"], damage["scales"])}


def merit(result, damage, dataset_weights):
    """The algorithm's own objective at its current weights."""
    X = {name: data["X"] for name, data in damage["datasets"].items()}
    y = {name: data["y"] for name, data in damage["datasets"].items()}
    return getattr(algorithms, result["method"]).evaluate(result["weights"], X, y, pre_processing.dataset_weights(X, dataset_weights))
