"""Canonical construction of patient-by-basis damage matrices."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .damage_matrix_utils.basis_library import BASIS_DIR, FEATURE_DTYPE, BasisLibrary
from .damage_matrix_utils.damage_matrix import DamageMatrix
from .damage_matrix_utils.damage_matrix_schema import CohortData, CohortDamage


class DamageMatrixFactory:
    """Project typed cohorts onto a basis library exactly once.

    basis_library: BasisLibrary | scalar object | selected memory-mapped basis shards and metadata.
    voxel_chunk_size: int | scalar | maximum spatial rows processed during one projection step.
    """

    def __init__(self, basis_cohorts: Sequence[str] | None = None, basis_dir: str | Path = BASIS_DIR, voxel_chunk_size: int = 16384):
        """Initialize basis projection.

        basis_cohorts: sequence[str] or None | length selected basis cohort count | shard names to load, or every shard when omitted.
        basis_dir: str or Path | scalar path | directory containing basis arrays and JSON metadata.
        voxel_chunk_size: int | scalar | spatial rows processed per matrix-multiplication block.
        """
        self.basis_library = BasisLibrary.load(basis_cohorts, basis_dir)
        self.voxel_chunk_size = int(voxel_chunk_size)

    def __call__(self, fitting: Sequence[CohortData], testing: Sequence[CohortData] | None = None) -> DamageMatrix:
        """Build one DamageMatrix from typed cohorts.

        fitting: sequence[CohortData] | length fitting_cohort_count | cohorts used to optimize basis weights.
        testing: sequence[CohortData] or None | length testing_cohort_count | optional external cohorts used only for final evaluation.
        returns: DamageMatrix | scalar object | normalized basis damage and metadata for the complete analysis.
        """
        fitting = self._validate_fitting_cohorts(fitting)
        testing, testing_was_supplied = self._resolve_testing_cohorts(testing)
        test_identifiers = self._test_identifiers(testing)
        self.basis_library.check_test_leakage(test_identifiers)
        cohorts = fitting + testing
        patient_maps = [cohort.patient_maps for cohort in cohorts]
        raw_dot_products, basis_l2_norms = self.basis_library.project(patient_maps, self.voxel_chunk_size)
        empty = [name for name, norm in zip(self.basis_library.map_names, basis_l2_norms) if norm == 0]
        if empty:
            raise ValueError(f"Basis maps {empty} contain no nonzero value.")
        basis_damage_scores = tuple(dot_products / basis_l2_norms / np.linalg.norm(cohort.patient_maps, axis=1)[:, None] for cohort, dot_products in zip(cohorts, raw_dot_products))
        fitting_damage = self._fitting_damage(fitting, basis_damage_scores)
        testing_damage = self._testing_damage(testing, basis_damage_scores, len(fitting), testing_was_supplied)
        active_targets = self._active_targets(fitting)
        return DamageMatrix(self.basis_library, basis_l2_norms, fitting_damage, testing_damage, active_targets)

    def _validate_fitting_cohorts(self, cohorts):
        """Validate cohort arrays and require usable variation in every fitting target."""
        validated = self._validate(cohorts)
        for cohort in validated:
            for outcome_values in cohort.outcomes.T:
                finite = np.isfinite(outcome_values)
                if finite.sum() < 3 or np.unique(outcome_values[finite]).size < 2:
                    raise ValueError(f"Cohort {cohort.name!r} contains an insufficient or constant fitting outcome.")
        if not validated:
            raise ValueError("At least one fitting cohort is required.")
        return validated

    def _resolve_testing_cohorts(self, cohorts):
        """Resolve an omitted test collection to an empty internal sequence."""
        if cohorts is None:
            return (), False
        return self._validate(cohorts), True

    @staticmethod
    def _test_identifiers(testing_cohorts):
        """Return cohort names and patient IDs for basis-leakage validation."""
        return tuple((cohort.name, cohort.patient_ids) for cohort in testing_cohorts)

    @staticmethod
    def _fitting_damage(fitting_cohorts, basis_damage_scores):
        """Attach each fitting cohort to its already-computed basis scores."""
        return tuple(CohortDamage(cohort, basis_damage_scores[index]) for index, cohort in enumerate(fitting_cohorts))

    @staticmethod
    def _testing_damage(testing_cohorts, basis_damage_scores, fitting_cohort_count, testing_was_supplied):
        """Preserve the distinction between omitted testing and an empty supplied collection."""
        if not testing_was_supplied:
            return None
        return tuple(CohortDamage(cohort, basis_damage_scores[fitting_cohort_count + index]) for index, cohort in enumerate(testing_cohorts))

    @staticmethod
    def _active_targets(fitting_cohorts):
        """Return clinical targets in first-seen column order."""
        return tuple(dict.fromkeys(target for cohort in fitting_cohorts for target in cohort.target_names))

    def _validate(self, cohorts):
        validated = []
        for cohort in cohorts:
            if not isinstance(cohort, CohortData):
                raise TypeError("DamageMatrixFactory accepts only CohortData objects.")
            maps = np.asarray(cohort.patient_maps, dtype=FEATURE_DTYPE)
            ids = np.asarray(cohort.patient_ids).astype(str).reshape(-1)
            outcomes = np.asarray(cohort.outcomes, dtype=float)
            valid_outcome_shape = (len(maps), len(cohort.target_names))
            if maps.ndim != 2 or maps.shape[1] != self.basis_library.voxel_count or outcomes.shape != valid_outcome_shape or len(ids) != len(maps):
                raise ValueError(f"Cohort {cohort.name!r} does not have one patient-map row, patient ID, and clinical-outcome row per patient.")
            if len(maps) < 3 or not np.isfinite(maps).all() or np.any(np.linalg.norm(maps, axis=1) == 0):
                raise ValueError(f"Cohort {cohort.name!r} contains insufficient or invalid patient maps.")
            validated.append(CohortData(cohort.name, maps, ids, tuple(cohort.target_names), outcomes))
        return tuple(validated)
