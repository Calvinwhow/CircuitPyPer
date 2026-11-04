"""Compact, modality-independent records of convergent-map optimization."""

import numpy as np
from pathlib import Path
from dataclasses import dataclass


@dataclass
class OptimizationHistory:
    """Scored weights and the fixed component maps needed to replay them, or, for a nonlinear model, the scored maps themselves.

    frames: (snapshots, features) maps stored directly when no weights @ maps reproduces them (the anatomical network); weights then only label the bar panel, one column per map name.
    steps: (snapshots,) iteration or epoch number of each snapshot; 1..N when omitted.
    """

    weights: np.ndarray  # (iterations, component maps)
    losses: np.ndarray  # (iterations,)
    maps: np.ndarray  # (component maps, features)
    map_names: tuple[str, ...]
    output_type: str = ""
    mask_path: str = ""
    objective_label: str = "RMS Spearman rho"
    frames: np.ndarray = None
    steps: np.ndarray = None

    def __post_init__(self):
        self.weights = np.asarray(self.weights, dtype=float)
        self.losses = np.asarray(self.losses, dtype=float)
        self.map_names = tuple(self.map_names)
        if self.weights.ndim != 2 or self.weights.shape[1] != len(self.map_names):
            raise ValueError("Weights must be two-dimensional with one column per map name.")
        if self.frames is None:
            self.maps = np.asarray(self.maps)
            if self.maps.ndim != 2 or self.maps.shape[0] != self.weights.shape[1]:
                raise ValueError("Maps must be two-dimensional with one row per weight column.")
        else:
            self.frames = np.asarray(self.frames)
            if self.frames.ndim != 2 or self.frames.shape[0] != self.weights.shape[0]:
                raise ValueError("Frames must be two-dimensional with one row per snapshot.")
        if self.losses.shape != (self.weights.shape[0],):
            raise ValueError("Losses must have one value per weight snapshot.")
        self.steps = np.arange(1, self.weights.shape[0] + 1) if self.steps is None else np.asarray(self.steps, dtype=int)
        if self.steps.shape != self.losses.shape:
            raise ValueError("Steps must have one value per snapshot.")

    @property
    def n_features(self):
        return (self.maps if self.frames is None else self.frames).shape[1]

    def map_at(self, iteration):
        """Rebuild one scored map without retaining a map history in memory, or return the stored frame."""
        if self.frames is not None:
            return self.frames[iteration]
        return self.weights[iteration].astype(self.maps.dtype) @ self.maps

    def save(self, path):
        """Write one portable NumPy archive after optimization finishes."""
        path = Path(path)
        if path.suffix.lower() != ".npz":
            raise ValueError("Optimization history path must end in .npz.")
        path.parent.mkdir(parents=True, exist_ok=True)
        stored = {"maps": self.maps} if self.frames is None else {"frames": self.frames}
        np.savez(path, version=np.array(1), weights=self.weights, losses=self.losses, steps=self.steps, map_names=np.asarray(self.map_names, dtype=str), output_type=np.array(self.output_type), mask_path=np.array(self.mask_path), objective_label=np.array(self.objective_label), **stored)
        return path

    @classmethod
    def load(cls, path):
        """Load a history archive without permitting pickled objects."""
        with np.load(path, allow_pickle=False) as data:
            if int(data["version"]) != 1:
                raise ValueError("Unsupported optimization history version.")
            return cls(weights=data["weights"], losses=data["losses"], maps=data["maps"] if "maps" in data else None, frames=data["frames"] if "frames" in data else None, steps=data["steps"] if "steps" in data else None, map_names=tuple(data["map_names"].tolist()), output_type=str(data["output_type"]), mask_path=str(data["mask_path"]), objective_label=(str(data["objective_label"]) if "objective_label" in data else "RMS Spearman rho"))
