"""Typed views over patient-by-basis damage matrices."""

from collections.abc import Sequence
from functools import cached_property

import numpy as np

from .damage_matrix_schema import CohortDamage, TargetDataset


class DamageMatrix:
    """All basis-damage information required by one optimization analysis.

    basis_library: object | scalar | basis storage providing ordered map operations and metadata.
    basis_l2_norms: np.ndarray | shape basis_count | Euclidean norm of each basis map.
    fitting_cohorts: tuple[CohortDamage] | length fitting_cohort_count | basis damage for fitting cohorts.
    testing_cohorts: tuple[CohortDamage] or None | length testing_cohort_count | basis damage for external test cohorts.
    active_targets: tuple[str] | length active_target_count | clinical outcomes exposed to this optimization run.
    """

    def __init__(self, basis_library, basis_l2_norms, fitting_cohorts, testing_cohorts, active_targets):
        self.basis_library = basis_library
        self.basis_l2_norms = basis_l2_norms
        self.fitting_cohorts = tuple(fitting_cohorts)
        self.testing_cohorts = None if testing_cohorts is None else tuple(testing_cohorts)
        self.active_targets = tuple(active_targets)

    @property
    def map_names(self) -> tuple[str, ...]:
        """map_names: tuple[str] | length basis_count | ordered basis-map names."""
        return self.basis_library.map_names

    @property
    def sample_sizes(self) -> dict[str, int]:
        """sample_sizes: dict[str, int] | length basis_count | patient count used to build each basis map."""
        return self.basis_library.sample_sizes

    def reconstruct_map(self, weights):
        """reconstructed_map: np.ndarray | shape voxel_count | weighted combination of normalized basis maps."""
        return self.basis_library.reconstruct_map(weights, self.basis_l2_norms)

    def normalized_basis_maps(self):
        """normalized_basis_maps: np.ndarray | shape basis_count by voxel_count | unit-L2 basis maps in optimization order."""
        return self.basis_library.normalized_maps(self.basis_l2_norms)

    def select(self, targets: Sequence[str]) -> "DamageMatrix":
        """Return the same matrices with a different set of active outcomes."""
        targets = tuple(targets)
        missing = set(targets) - set(self.active_targets)
        if missing:
            raise KeyError(f"Unknown damage targets: {sorted(missing)}")
        return DamageMatrix(self.basis_library, self.basis_l2_norms, self.fitting_cohorts, self.testing_cohorts, targets)

    @cached_property
    def fitting_datasets(self) -> tuple[TargetDataset, ...]:
        """fitting_datasets: tuple[TargetDataset] | length fitting dataset count | finite fitting views for active targets."""
        return self._target_datasets(self.fitting_cohorts, keep_missing=False)

    @cached_property
    def testing_datasets(self) -> tuple[TargetDataset, ...] | None:
        """testing_datasets: tuple[TargetDataset] or None | length test dataset count | external-test views for active targets."""
        if self.testing_cohorts is None:
            return None
        return self._target_datasets(self.testing_cohorts, keep_missing=True)

    def _target_datasets(self, cohorts, keep_missing):
        datasets = []
        for item in cohorts:
            for column, target in enumerate(item.cohort.target_names):
                if target not in self.active_targets:
                    continue
                outcome = item.cohort.outcomes[:, column]
                valid = self._valid_outcome_rows(outcome, keep_missing)
                if valid.all():
                    scores = item.basis_damage_scores
                    patient_maps = item.cohort.patient_maps
                    outcome_values = outcome
                    patient_ids = item.cohort.patient_ids
                else:
                    scores = item.basis_damage_scores[valid]
                    patient_maps = item.cohort.patient_maps[valid]
                    outcome_values = outcome[valid]
                    patient_ids = item.cohort.patient_ids[valid]
                dataset_name = self._dataset_name(item.cohort.name, target)
                datasets.append(TargetDataset(dataset_name, item.cohort.name, target, scores, patient_maps, outcome_values, patient_ids))
        return tuple(datasets)

    def _dataset_name(self, cohort_name, target_name):
        """Resolve a readable unique dataset label without encoding identity in punctuation."""
        if len(self.active_targets) == 1:
            return cohort_name
        return f"{cohort_name} ({target_name})"

    @staticmethod
    def _valid_outcome_rows(outcome_values, keep_missing):
        """Resolve whether missing outcomes are retained for external prediction."""
        if keep_missing:
            return np.ones(len(outcome_values), dtype=bool)
        return np.isfinite(outcome_values)
