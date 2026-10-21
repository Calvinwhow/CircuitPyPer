"""Fast outcome-wise regression for permutation clustering analyses.

This adapter builds one symptom map per scalar outcome from a shared imaging
column. Each outcome retains its own complete-case subject cohort. Outcomes
with identical cohorts are fitted together, which avoids either discarding
partially observed subjects or repeatedly loading the same imaging files.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


class OutcomeWiseImagingRegression:
    """Generate outcome-specific imaging t-maps from a source CSV.

    Parameters
    ----------
    csv_path : path-like
        Subject-level table containing the imaging paths and scalar outcomes.
    path_column : str
        Column containing one neuroimaging or fiber file per subject.
    outcome_names : sequence of str
        Ordered scalar outcomes. This order is retained in the returned maps.
    mask_path : path-like, optional
        Mask or fiber-atlas argument forwarded to ``GiiNiiFileImport``.
    transform : {"rank", "rank_ordinal", "standardize", None}, default "rank"
        Transformation applied across subjects separately within every
        outcome's complete-case cohort. ``rank`` averages ties;
        ``rank_ordinal`` reproduces the historical stable-order ranking.
    observed_maps_path : path-like, optional
        If supplied, save the observed location-by-outcome t-statistic matrix.
    missing_policy : {"complete_case", "zero"}, default "complete_case"
        Fit every outcome on its nonmissing rows, or reproduce legacy workflows
        that retained all rows and replaced missing outcomes with zero.
    outcome_signs : scalar, sequence, or mapping, optional
        Multipliers applied to outcomes before transformation. A scalar applies
        to every outcome; mappings are keyed by outcome name.
    dataframe, imaging_data : optional
        In-memory inputs for testing or advanced use. ``imaging_data`` must be
        observation-by-location and aligned to ``dataframe``.

    Notes
    -----
    The fitted model is the no-intercept voxelwise regression used by the
    original symptom-map workflow: ``outcome ~ imaging_value``. Missing values
    are handled per outcome, rather than replacing them or forcing one common
    complete-case cohort.
    """

    def __init__(
        self,
        *,
        csv_path=None,
        path_column,
        outcome_names,
        mask_path=None,
        transform="rank",
        observed_maps_path=None,
        missing_policy="complete_case",
        outcome_signs=None,
        dataframe=None,
        imaging_data=None,
    ):
        if transform not in {"rank", "rank_ordinal", "standardize", None}:
            raise ValueError(
                "transform must be 'rank', 'rank_ordinal', 'standardize', or None."
            )
        if missing_policy not in {"complete_case", "zero"}:
            raise ValueError("missing_policy must be 'complete_case' or 'zero'.")
        if dataframe is None:
            if csv_path is None:
                raise ValueError("csv_path is required when dataframe is not supplied.")
            dataframe = pd.read_csv(csv_path)
        self.csv_path = None if csv_path is None else str(Path(csv_path))
        self.path_column = str(path_column)
        self.outcome_names = [str(name) for name in outcome_names]
        self.mask_path = None if mask_path is None else str(mask_path)
        self.transform = transform
        self.missing_policy = missing_policy
        self.observed_maps_path = (
            None if observed_maps_path is None else Path(observed_maps_path)
        )
        self.exchangeability_blocks = None

        required = [self.path_column, *self.outcome_names]
        missing = [column for column in required if column not in dataframe.columns]
        if missing:
            raise ValueError(f"Source data are missing required columns: {missing}")
        if len(set(self.outcome_names)) != len(self.outcome_names):
            raise ValueError("outcome_names must be unique.")

        paths = dataframe[self.path_column]
        valid_paths = paths.notna() & paths.astype(str).str.strip().ne("")
        if imaging_data is None:
            nonexistent = [
                path
                for path in paths.loc[valid_paths].astype(str)
                if not os.path.exists(path)
            ]
            if nonexistent:
                raise FileNotFoundError(
                    f"{len(nonexistent)} imaging paths do not exist; first: "
                    f"{nonexistent[0]}"
                )
            imaging_data = self._load_imaging(
                paths.loc[valid_paths].astype(str), self.mask_path
            )
        else:
            imaging_data = np.asarray(imaging_data)
            if imaging_data.ndim != 2:
                raise ValueError("imaging_data must be observation-by-location.")
            if imaging_data.shape[0] == len(dataframe):
                imaging_data = imaging_data[valid_paths.to_numpy()]
            elif imaging_data.shape[0] != int(valid_paths.sum()):
                raise ValueError(
                    "imaging_data rows must match either all dataframe rows or "
                    "rows with valid imaging paths."
                )

        source = dataframe.loc[valid_paths, self.outcome_names].apply(
            pd.to_numeric, errors="coerce"
        )
        self.n_outputs = len(self.outcome_names)
        self.n_locations = int(imaging_data.shape[1])
        self.outcome_signs = self._resolve_outcome_signs(outcome_signs)
        source = source * self.outcome_signs[None, :]
        if self.missing_policy == "zero":
            source = source.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        self._prepare_groups(np.asarray(imaging_data, dtype=np.float32), source)
        self.n_obs = int(self.observation_counts.max())

    def _resolve_outcome_signs(self, outcome_signs):
        if outcome_signs is None:
            signs = np.ones(self.n_outputs, dtype=np.float32)
        elif np.isscalar(outcome_signs):
            signs = np.full(self.n_outputs, float(outcome_signs), dtype=np.float32)
        elif isinstance(outcome_signs, dict):
            signs = np.asarray(
                [outcome_signs.get(name, 1.0) for name in self.outcome_names],
                dtype=np.float32,
            )
        else:
            signs = np.asarray(outcome_signs, dtype=np.float32)
        if signs.shape != (self.n_outputs,) or not np.isfinite(signs).all():
            raise ValueError("outcome_signs must provide one finite value per outcome.")
        if np.any(signs == 0):
            raise ValueError("outcome_signs cannot contain zero.")
        return signs

    @staticmethod
    def _load_imaging(paths, mask_path):
        from calvin_utils.file_utils.import_functions import GiiNiiFileImport

        importer = GiiNiiFileImport(
            import_path=paths,
            mask_path=mask_path if mask_path is not None else "default",
            transpose=False,
        )
        loaded = importer.run()
        if not isinstance(loaded, pd.DataFrame):
            raise TypeError(
                "GiiNiiFileImport.run() must return a locations-by-files DataFrame."
            )
        result = loaded.to_numpy(dtype=np.float32).T
        if result.shape[0] != len(paths):
            raise ValueError("Imported imaging files are not aligned to source rows.")
        return result

    @staticmethod
    def _transform(values, method):
        values = np.asarray(values, dtype=np.float32)
        if method == "rank":
            transformed = rankdata(values, axis=0, method="average").astype(
                np.float32
            )
            transformed -= transformed.mean(axis=0, keepdims=True)
            return transformed
        if method == "rank_ordinal":
            flat = values.reshape(values.shape[0], -1)
            order = np.argsort(flat, axis=0, kind="mergesort")
            ranks = np.empty_like(flat, dtype=np.float32)
            rows = np.arange(1, values.shape[0] + 1, dtype=np.float32)[:, None]
            ranks[order, np.arange(flat.shape[1])] = rows
            ranks -= ranks.mean(axis=0, keepdims=True)
            return ranks.reshape(values.shape)
        if method == "standardize":
            mean = values.mean(axis=0, keepdims=True)
            standard_deviation = values.std(axis=0, keepdims=True)
            return np.divide(
                values - mean,
                standard_deviation,
                out=np.zeros_like(values),
                where=standard_deviation > 1e-12,
            )
        return values.copy()

    def _prepare_groups(self, imaging_data, outcomes):
        grouped_outputs = defaultdict(list)
        valid_rows_by_output = []
        observation_counts = np.empty(self.n_outputs, dtype=int)
        for output_index, outcome in enumerate(self.outcome_names):
            values = outcomes[outcome].to_numpy(float)
            valid_rows = (
                np.arange(len(values))
                if self.missing_policy == "zero"
                else np.flatnonzero(np.isfinite(values))
            )
            if len(valid_rows) < 3:
                raise ValueError(
                    f"Outcome {outcome!r} has fewer than three usable observations."
                )
            key = tuple(valid_rows.tolist())
            grouped_outputs[key].append(output_index)
            valid_rows_by_output.append(valid_rows)
            observation_counts[output_index] = len(valid_rows)

        self.observation_counts = observation_counts
        self._groups = []
        for row_key, output_indices in grouped_outputs.items():
            rows = np.asarray(row_key, dtype=int)
            names = [self.outcome_names[index] for index in output_indices]
            x = self._transform(imaging_data[rows], self.transform)
            y = self._transform(
                outcomes.iloc[rows][names].to_numpy(np.float32), self.transform
            )
            self._groups.append(
                {
                    "rows": rows,
                    "output_indices": np.asarray(output_indices, dtype=int),
                    "x": np.ascontiguousarray(x, dtype=np.float32),
                    "y": np.ascontiguousarray(y, dtype=np.float32),
                }
            )
        self._valid_rows_by_output = valid_rows_by_output

    def draw_permutation_indices(self, n_permutations, rng):
        """Return padded, independent within-outcome permutations."""
        indices = np.full(
            (n_permutations, self.n_outputs, self.n_obs), -1, dtype=np.int32
        )
        for iteration in range(n_permutations):
            for output_index, count in enumerate(self.observation_counts):
                indices[iteration, output_index, :count] = rng.permutation(count)
        return indices

    @staticmethod
    def _fit_group(x, y):
        """Fit many no-intercept scalar outcomes against voxelwise predictors."""
        n_observations = x.shape[0]
        x_sum_squares = np.einsum("ij,ij->j", x, x, dtype=np.float64)
        xty = x.T @ y
        beta = np.divide(
            xty,
            x_sum_squares[:, None],
            out=np.zeros_like(xty, dtype=np.float64),
            where=x_sum_squares[:, None] > 0,
        )
        y_sum_squares = np.einsum("ij,ij->j", y, y, dtype=np.float64)
        residual_sum_squares = np.maximum(
            y_sum_squares[None, :] - beta**2 * x_sum_squares[:, None], 0.0
        )
        mse = residual_sum_squares / (n_observations - 1)
        variance = np.divide(
            mse,
            x_sum_squares[:, None],
            out=np.zeros_like(mse),
            where=x_sum_squares[:, None] > 0,
        )
        t_statistics = np.divide(
            beta,
            np.sqrt(variance) + 1e-6,
            out=np.zeros_like(beta),
            where=x_sum_squares[:, None] > 0,
        )
        return t_statistics.astype(np.float32, copy=False)

    def _validate_permutation_indices(self, permutation_index):
        indices = np.asarray(permutation_index)
        if indices.shape != (self.n_outputs, self.n_obs):
            raise ValueError(
                "permutation_index must have shape "
                f"({self.n_outputs}, {self.n_obs})."
            )
        for output_index, count in enumerate(self.observation_counts):
            current = indices[output_index, :count]
            if not np.array_equal(np.sort(current), np.arange(count)):
                raise ValueError(
                    f"Invalid permutation for outcome {self.outcome_names[output_index]!r}."
                )
        return indices.astype(int, copy=False)

    def run_single_multiout_regression(
        self, *, permutation=False, permutation_index=None
    ):
        """Return a location-by-outcome t-statistic tensor for one realization."""
        if permutation and permutation_index is None:
            raise ValueError("An explicit permutation_index is required.")
        indices = (
            self._validate_permutation_indices(permutation_index)
            if permutation
            else None
        )
        t_statistics = np.empty(
            (1, self.n_locations, self.n_outputs), dtype=np.float32
        )
        for group in self._groups:
            output_indices = group["output_indices"]
            y = group["y"]
            if indices is not None:
                y = np.column_stack(
                    [
                        y[indices[output_index, : len(group["rows"])], local_index]
                        for local_index, output_index in enumerate(output_indices)
                    ]
                ).astype(np.float32, copy=False)
            t_statistics[0][:, output_indices] = self._fit_group(group["x"], y)

        if not permutation and self.observed_maps_path is not None:
            self.observed_maps_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.observed_maps_path.with_suffix(
                self.observed_maps_path.suffix + ".tmp"
            )
            with open(temporary, "wb") as stream:
                np.save(stream, t_statistics[0])
            os.replace(temporary, self.observed_maps_path)
        return None, t_statistics, None

    def analysis_metadata(self):
        """Return reproducibility metadata consumed by the permutation runner."""
        return json.loads(
            json.dumps(
                {
                    "regression_adapter": self.__class__.__name__,
                    "source_csv": self.csv_path,
                    "path_column": self.path_column,
                    "mask_path": self.mask_path,
                    "transform": self.transform,
                    "missing_policy": self.missing_policy,
                    "outcome_signs": self.outcome_signs.tolist(),
                    "n_locations": self.n_locations,
                    "observation_counts": self.observation_counts.tolist(),
                    "complete_case_cohorts": len(self._groups),
                }
            )
        )


__all__ = ["OutcomeWiseImagingRegression"]
