"""Data-only schemas for the damage-matrix hierarchy."""

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CohortData:
    """Patient imaging and clinical outcomes loaded from one cohort.

    name: str | scalar | unique cohort name used in reports and validation.
    patient_maps: np.ndarray | shape patient_count by voxel_count | one spatial map per patient.
    patient_ids: np.ndarray | shape patient_count | stable unique identifier for each patient row.
    target_names: tuple[str] | length target_count | ordered names of the outcome columns.
    outcomes: np.ndarray | shape patient_count by target_count | clinical values aligned to patients and target_names.
    """

    name: str
    patient_maps: NDArray[np.float32]
    patient_ids: NDArray[np.str_]
    target_names: tuple[str, ...]
    outcomes: NDArray[np.float64]


@dataclass(frozen=True)
class BasisShard:
    """One memory-mapped basis array and its JSON metadata.

    path: Path | scalar path | location of the memory-mapped basis array.
    values: np.ndarray | shape voxel_count by shard_basis_count | basis maps stored as columns.
    metadata: dict[str, object] | scalar object | validated JSON sidecar describing the shard and its bases.
    map_names: tuple[str] | length shard_basis_count | globally qualified basis names in column order.
    """

    path: Path
    values: NDArray[np.float32]
    metadata: dict[str, object]
    map_names: tuple[str, ...]


@dataclass(frozen=True)
class CohortDamage:
    """One cohort's normalized similarity to every basis map.

    cohort: CohortData | scalar object | source imaging, IDs, and clinical outcomes for these rows.
    basis_damage_scores: np.ndarray | shape patient_count by basis_count | patient-map and basis-map cosine similarities.
    """

    cohort: CohortData
    basis_damage_scores: NDArray[np.float64]


@dataclass(frozen=True)
class TargetDataset:
    """Aligned patient rows for one cohort and one clinical target.

    dataset_name: str | scalar | readable unique label used in reports and dataset weighting.
    cohort_name: str | scalar | source cohort containing these patients.
    target_name: str | scalar | clinical outcome predicted for these patients.
    basis_damage_scores: np.ndarray | shape selected_patient_count by basis_count | normalized basis damage used by optimizers.
    patient_maps: np.ndarray | shape selected_patient_count by voxel_count | original spatial maps used for map scoring and the neural network.
    outcome_values: np.ndarray | shape selected_patient_count | observed clinical target values.
    patient_ids: np.ndarray | shape selected_patient_count | patient IDs aligned with every array row.
    """

    dataset_name: str
    cohort_name: str
    target_name: str
    basis_damage_scores: NDArray[np.float64]
    patient_maps: NDArray[np.float32]
    outcome_values: NDArray[np.float64]
    patient_ids: NDArray[np.str_]

    def subset(self, rows: NDArray[np.bool_]):
        """Return the same typed view restricted by a patient_count boolean mask."""
        return TargetDataset(self.dataset_name, self.cohort_name, self.target_name, self.basis_damage_scores[rows], self.patient_maps[rows], self.outcome_values[rows], self.patient_ids[rows])
