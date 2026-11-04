"""Differentiable shared clinical losses."""

import numpy as np
from scipy.stats import rankdata
from .dependencies import require_torch


def prepare_pairwise(outcomes, device):
    """Precompute non-tied clinical rank pairs for every dataset."""
    torch = require_torch()
    prepared = {}
    for name, values in outcomes.items():
        ranks = np.asarray(rankdata(np.asarray(values, dtype=np.float32), method="average"), dtype=np.float32)
        left, right = np.triu_indices(len(ranks), k=1)
        direction = np.sign(ranks[left] - ranks[right])
        keep = direction != 0
        if not np.any(keep):
            raise ValueError(f"Dataset {name!r} has no ordered outcome pairs.")
        prepared[name] = (torch.as_tensor(left[keep], dtype=torch.long, device=device), torch.as_tensor(right[keep], dtype=torch.long, device=device), torch.as_tensor(direction[keep], dtype=torch.float32, device=device))
    return prepared


def dataset_pairwise_softplus(predicted, pairs, sign):
    """Compute mean softplus(-s d_ij (p_i - p_j)) for one dataset."""
    torch = require_torch()
    left, right, direction = pairs
    prediction_difference = predicted[left] - predicted[right]
    signed_margin = np.float32(sign) * direction * prediction_difference
    return torch.nn.functional.softplus(-signed_margin).mean()


def pairwise_softplus(predictions, pairs, dataset_weights, dataset_signs):
    """Return one dataset-weighted pairwise softplus loss."""
    torch = require_torch()
    loss = torch.zeros((), dtype=torch.float32, device=next(iter(predictions.values())).device)
    for name, predicted in predictions.items():
        dataset_loss = dataset_pairwise_softplus(predicted, pairs[name], dataset_signs[name])
        loss = loss + np.float32(dataset_weights[name]) * dataset_loss
    return loss
