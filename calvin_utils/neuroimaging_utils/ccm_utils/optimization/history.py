"""Compact, modality-independent records of convergent-map optimization."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class OptimizationHistory:
    """Scored weights and the fixed component maps needed to replay them."""

    weights: np.ndarray       # (iterations, component maps)
    losses: np.ndarray        # (iterations,)
    maps: np.ndarray          # (component maps, features)
    map_names: tuple[str, ...]
    output_type: str = ""
    mask_path: str = ""
    objective_label: str = "RMS Spearman rho"

    def __post_init__(self):
        self.weights = np.asarray(self.weights, dtype=float)
        self.losses = np.asarray(self.losses, dtype=float)
        self.maps = np.asarray(self.maps)
        self.map_names = tuple(self.map_names)
        if self.weights.ndim != 2 or self.maps.ndim != 2:
            raise ValueError("weights and maps must be two-dimensional arrays.")
        if self.weights.shape[1] != self.maps.shape[0]:
            raise ValueError("Weights must have one column per component map.")
        if self.losses.shape != (self.weights.shape[0],):
            raise ValueError("Losses must have one value per weight snapshot.")
        if len(self.map_names) != self.maps.shape[0]:
            raise ValueError("Map names must match the component maps.")

    def map_at(self, iteration):
        """Rebuild one scored map without retaining a map history in memory."""
        return self.weights[iteration] @ self.maps

    def save(self, path):
        """Write one portable NumPy archive after optimization finishes."""
        path = Path(path)
        if path.suffix.lower() != ".npz":
            raise ValueError("Optimization history path must end in .npz.")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            version=np.array(1),
            weights=self.weights,
            losses=self.losses,
            maps=self.maps,
            map_names=np.asarray(self.map_names, dtype=str),
            output_type=np.array(self.output_type),
            mask_path=np.array(self.mask_path),
            objective_label=np.array(self.objective_label),
        )
        return path

    @classmethod
    def load(cls, path):
        """Load a history archive without permitting pickled objects."""
        with np.load(path, allow_pickle=False) as data:
            if int(data["version"]) != 1:
                raise ValueError("Unsupported optimization history version.")
            return cls(
                weights=data["weights"],
                losses=data["losses"],
                maps=data["maps"],
                map_names=tuple(data["map_names"].tolist()),
                output_type=str(data["output_type"]),
                mask_path=str(data["mask_path"]),
                objective_label=(str(data["objective_label"])
                                 if "objective_label" in data else "RMS Spearman rho"),
            )
