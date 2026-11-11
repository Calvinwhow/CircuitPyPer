"""The three explicit spatial layers of the anatomical neural network."""

import numpy as np
from .anatomy import REPRESENTATIONS
from .dependencies import require_torch


torch = require_torch()
nn = torch.nn
FLOAT_DTYPE = np.float32


class AlgorithmMapLayer(nn.Module):
    """Reconstruct Z[g, v, a] = sum_k B[v, k] D[g, k] W[k, a]."""

    def __init__(self, algorithm_weights, representation_masks, *, fixed_maps=None, basis_maps=None, mode="frozen", scratch_scale=0.01):
        super().__init__()
        weights = torch.as_tensor(np.asarray(algorithm_weights, dtype=FLOAT_DTYPE))
        masks = torch.as_tensor(np.asarray(representation_masks, dtype=FLOAT_DTYPE))
        if weights.ndim != 2 or masks.shape != (len(REPRESENTATIONS), weights.shape[0]):
            raise ValueError("algorithm_weights and representation_masks do not share the basis axis.")
        self.mode = mode
        self.register_buffer("representation_masks", masks)
        if mode == "frozen":
            self._initialize_frozen(weights, fixed_maps)
        elif mode in {"fine_tune", "scratch"}:
            self._initialize_trainable(weights, basis_maps, mode, scratch_scale)
        else:
            raise ValueError("mode must be 'frozen', 'fine_tune', or 'scratch'.")

    def _initialize_frozen(self, weights, fixed_maps):
        if fixed_maps is None:
            raise ValueError("Frozen mode requires fixed_maps.")
        maps = torch.as_tensor(np.asarray(fixed_maps, dtype=FLOAT_DTYPE))
        if maps.ndim != 3 or maps.shape[0] != len(REPRESENTATIONS) or maps.shape[2] != weights.shape[1]:
            raise ValueError(f"fixed_maps must have shape ({len(REPRESENTATIONS)}, voxels, algorithms).")
        self.register_buffer("algorithm_weights", weights)
        self.register_buffer("fixed_maps", maps)
        self.basis_maps = None

    def _initialize_trainable(self, weights, basis_maps, mode, scratch_scale):
        if basis_maps is None:
            raise ValueError(f"{mode} mode requires basis_maps.")
        basis = torch.as_tensor(np.asarray(basis_maps, dtype=FLOAT_DTYPE))
        if basis.ndim != 2 or basis.shape[1] != weights.shape[0]:
            raise ValueError("basis_maps must have shape (voxels, bases).")
        initial = torch.randn_like(weights) * np.float32(scratch_scale) if mode == "scratch" else weights
        self.register_buffer("basis_maps", basis)
        self.algorithm_weights = nn.Parameter(initial)
        self.fixed_maps = None

    def forward(self):
        if self.mode == "frozen":
            return self.fixed_maps
        masked_weights = self.representation_masks[:, :, None] * self.algorithm_weights[None, :, :]
        return torch.einsum("vk,gka->gva", self.basis_maps, masked_weights)

    def unfreeze(self, basis_maps):
        if isinstance(self.algorithm_weights, nn.Parameter):
            self.algorithm_weights.requires_grad_(True)
            self.mode = "fine_tune"
            return
        weights = self.algorithm_weights.detach().clone()
        basis = torch.as_tensor(np.asarray(basis_maps, dtype=FLOAT_DTYPE), device=weights.device)
        del self._buffers["algorithm_weights"]
        del self._buffers["fixed_maps"]
        delattr(self, "basis_maps")
        self.register_buffer("basis_maps", basis)
        self.algorithm_weights = nn.Parameter(weights)
        self.fixed_maps = None
        self.mode = "fine_tune"

    def freeze(self):
        if isinstance(self.algorithm_weights, nn.Parameter):
            self.algorithm_weights.requires_grad_(False)


class AnatomicalRepresentationLayer(nn.Module):
    """Compute H[g, v] = ReLU(sum_a(alpha[g, a] ReLU(Z[g, v, a] + b[g, a])) + c[g]).

    b starts at -min_v Z[g, v, a] and c at 0, so both ReLUs pass the initial maps through unchanged and training learns
    thresholds from the linear solution. linear_offsets() is the voxel-uniform constant this start adds to each H[g].
    """

    def __init__(self, initial_algorithm_maps):
        super().__init__()
        n_algorithms = initial_algorithm_maps.shape[2]
        self.algorithm_bias = nn.Parameter(-torch.amin(initial_algorithm_maps.detach().to(torch.float32), dim=1))
        self.algorithm_mix = nn.Parameter(torch.full((len(REPRESENTATIONS), n_algorithms), 1.0 / n_algorithms, dtype=torch.float32))
        self.representation_bias = nn.Parameter(torch.zeros(len(REPRESENTATIONS), dtype=torch.float32))

    def linear_offsets(self):
        """Return k[g] = sum_a alpha[g, a] b[g, a] + c[g], the constant H[g] carries while both ReLUs are inactive."""
        return ((self.algorithm_mix * self.algorithm_bias).sum(dim=1) + self.representation_bias).detach()

    def forward(self, algorithm_maps):
        biased_algorithm_maps = algorithm_maps + self.algorithm_bias[:, None, :]
        activated_algorithm_maps = torch.relu(biased_algorithm_maps)
        mixed_algorithm_maps = torch.einsum("gva,ga->gv", activated_algorithm_maps, self.algorithm_mix)
        biased_representation_maps = mixed_algorithm_maps + self.representation_bias[:, None]
        return torch.relu(biased_representation_maps)


class FinalLocalizationLayer(nn.Module):
    """Compute M[v] = sum_g beta[g] (H[g, v] - h0[g]).

    h0 starts at the anatomy layer's linear offsets, so the initial map is exactly sum_g sum_a beta[g] alpha[g, a] Z[g, v, a].
    A voxel-uniform map term adds h0's contribution times sum_v P[v] to each patient score, so h0 is an explicit extent term.
    """

    def __init__(self, initial_baseline):
        super().__init__()
        self.representation_mix = nn.Parameter(torch.ones(len(REPRESENTATIONS), dtype=torch.float32))
        self.representation_baseline = nn.Parameter(torch.as_tensor(initial_baseline, dtype=torch.float32).clone())

    def contributions(self, representation_maps):
        """Return domain maps whose linear sum is the final localization map."""
        return (representation_maps - self.representation_baseline[:, None]) * self.representation_mix[:, None]

    def forward(self, representation_maps):
        return self.contributions(representation_maps).sum(dim=0)
