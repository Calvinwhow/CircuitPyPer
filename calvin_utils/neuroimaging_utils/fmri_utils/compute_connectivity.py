"""Compute lesion-network maps from chunked functional-connectome rows."""

import os
import tempfile
import warnings
from pathlib import Path

import numpy as np
from tqdm import tqdm

from calvin_utils.neuroimaging_utils.fmri_utils.fp8 import COMBO_FP8_SCALE, decode_e4m3
from calvin_utils.neuroimaging_utils.nifti_utils.volume_io import NiftiIO
from calvin_utils.resource_paths import lazy_resource_path


DEFAULT_CONNECTOME_DIR = lazy_resource_path("fmri_connectome", "GSP1000", "AvgR")
DEFAULT_MASK = lazy_resource_path("MNI152_T1_2mm_brain_mask_dil.nii.gz")
DEFAULT_CHUNK_INDEX = lazy_resource_path(
    "MNI152_2mm_dil_3209v_91c_chunk_idx.nii.gz"
)


class FunctionalConnectivity:
    """Generate connectivity maps by streaming the rows needed for ``W @ C``."""

    def __init__(
        self,
        connectome_dir=DEFAULT_CONNECTOME_DIR,
        mask_path=DEFAULT_MASK,
        chunk_index_path=DEFAULT_CHUNK_INDEX,
        chunk_pattern="{chunk}_AvgR.npy",
        row_batch_mb=256,
        similarity_stats_path=None,
        combo_dir=None,
        combo_pattern="{chunk}_Combo.npy",
        std_mask_path=None,
        norm_mask_path=None,
    ):
        self.connectome_dir = connectome_dir
        self.mask_path = mask_path
        self.chunk_index_path = chunk_index_path
        self.chunk_pattern = chunk_pattern
        if row_batch_mb <= 0:
            raise ValueError("row_batch_mb must be greater than zero.")
        self.row_batch_bytes = int(row_batch_mb * 1024**2)
        self.similarity_stats_path = similarity_stats_path
        self._chunk_labels = None
        self._chunk_rows = None
        self._row_stats = None
        # Seed-normalization resources default to the precomputed-connectome layout beside connectome_dir:
        # <root>/Combo/{chunk}_Combo.npy and <root>/<root name>_{std,norm}_weighted_mask.nii.gz.
        self.combo_dir = combo_dir
        self.combo_pattern = combo_pattern
        self.std_mask_path = std_mask_path
        self.norm_mask_path = norm_mask_path
        self._voxel_std = None
        self._voxel_norm = None

    @property
    def chunk_labels(self):
        """Return one positive chunk label per masked connectome voxel."""
        if self._chunk_labels is None:
            nifti_io = NiftiIO(mask_path=str(self.mask_path))
            labels = nifti_io.import_nifti_to_numpy_array(
                [str(self.chunk_index_path)]
            )[:, 0]
            integer_labels = labels.astype(np.int32)
            if not np.array_equal(labels, integer_labels) or np.any(
                integer_labels < 1
            ):
                raise ValueError(
                    "The chunk-index image must assign a positive integer label "
                    "to every voxel in the connectome mask."
                )
            self._chunk_labels = integer_labels
        return self._chunk_labels

    @property
    def chunk_rows(self):
        """Return global masked-voxel indices for each connectome chunk."""
        if self._chunk_rows is None:
            labels = self.chunk_labels
            self._chunk_rows = tuple(
                (int(chunk), np.flatnonzero(labels == chunk))
                for chunk in np.unique(labels)
            )
        return self._chunk_rows

    @property
    def connectome_label(self):
        """Return <connectome>-<stat>[-fp8] for output filenames, e.g. yeo1000_dil-T-fp8, read from the chunks actually used."""
        connectome_dir = Path(self.connectome_dir)
        chunk, global_rows = self.chunk_rows[0]
        storage = "-fp8" if self._load_chunk(chunk, global_rows).dtype == np.uint8 else ""
        return f"{connectome_dir.parent.name}-{connectome_dir.name.removesuffix('_fp8')}{storage}"

    @staticmethod
    def _binarize_data(data, threshold):
        if threshold is None:
            raise ValueError("threshold must be provided when binarize=True.")
        return (data > threshold).astype(np.float32)

    def _load_lesion_weights(self, nifti_path):
        weights = NiftiIO(mask_path=str(self.mask_path)).import_nifti_to_numpy_array([nifti_path])[:, 0]
        return weights.reshape(self.chunk_labels.size, -1).T.astype(np.float32, copy=False)

    @staticmethod
    def _flatten_frames(values):
        return np.asarray(values).T.reshape(-1)

    @classmethod
    def _flatten_profile_batches(cls, values, frame_counts):
        offsets = np.cumsum([0, *frame_counts])
        return np.vstack([cls._flatten_frames(values[start:stop]) for start, stop in zip(offsets[:-1], offsets[1:])])

    def spatial_correlation(self, first_nifti_path, second_nifti_path):
        """Return masked voxelwise Pearson correlation between two NIfTIs."""
        first = self._load_lesion_weights(first_nifti_path).reshape(-1)
        second = self._load_lesion_weights(second_nifti_path).reshape(-1)
        finite = np.isfinite(first) & np.isfinite(second)
        if np.count_nonzero(finite) < 2:
            raise ValueError("Spatial correlation requires two finite voxels.")

        first = first[finite].astype(np.float64, copy=False)
        second = second[finite].astype(np.float64, copy=False)
        first -= np.mean(first)
        second -= np.mean(second)
        denominator = np.linalg.norm(first) * np.linalg.norm(second)
        if denominator == 0:
            raise ValueError("Spatial correlation is undefined for a constant map.")
        return float(np.dot(first, second) / denominator)

    def _chunk_path(self, chunk):
        file_name = self.chunk_pattern.format(chunk=int(chunk))
        return Path(self.connectome_dir) / file_name

    def _load_chunk(self, chunk, row_indices):
        chunk_path = self._chunk_path(chunk)
        if not chunk_path.is_file():
            raise FileNotFoundError(f"Connectome chunk not found: {chunk_path}")

        chunk_data = np.load(chunk_path, mmap_mode="r", allow_pickle=False)
        expected_shape = (row_indices.size, self.chunk_labels.size)
        if chunk_data.shape != expected_shape:
            raise ValueError(
                f"Connectome chunk {chunk_path} has shape {chunk_data.shape}; "
                f"expected {expected_shape}."
            )
        return chunk_data

    @staticmethod
    def _rows_float32(chunk_data, rows):
        """Return chunk rows as float32, decoding float8 e4m3fn bytes when the connectome is stored as uint8.

        NaN entries in float32 chunks (sporadic in yeo1000_dil) are read as 0, matching how the float8 converter stores them.
        """
        if chunk_data.dtype == np.uint8: return decode_e4m3(chunk_data[rows, :])
        return np.nan_to_num(np.asarray(chunk_data[rows, :], dtype=np.float32, order="C"), copy=True, nan=0.0)

    def _rows_per_batch(self):
        bytes_per_row = self.chunk_labels.size * np.dtype(np.float32).itemsize
        return max(1, self.row_batch_bytes // bytes_per_row)

    def _seed_weights(self, weights):
        """Return seed weights as (n_seeds, n_masked_voxels) float32, treating NaN background as zero weight."""
        weights = np.asarray(weights, dtype=np.float32)
        if weights.ndim == 1: weights = weights[None, :]
        if weights.ndim != 2 or weights.shape[1] != self.chunk_labels.size: raise ValueError(f"weights must have shape (n_seeds, n_masked_voxels); got {weights.shape}.")
        if np.isinf(weights).any(): raise ValueError("Seed weights contain infinite values.")
        missing = np.isnan(weights)
        if missing.any():
            warnings.warn(f"{int(missing.sum())} NaN seed weights in {int(missing.any(axis=1).sum())} of {weights.shape[0]} map(s) were treated as 0.", RuntimeWarning, stacklevel=3)
            weights = np.where(missing, np.float32(0), weights)
        return weights

    def _connectome_root(self):
        return Path(self.connectome_dir).parent

    def _mask_values(self, nifti_path):
        values = NiftiIO(mask_path=str(self.mask_path)).import_nifti_to_numpy_array([str(nifti_path)])[:, 0]
        if not np.all(np.isfinite(values)) or np.any(values <= 0): raise ValueError(f"{nifti_path} must be finite and positive at every connectome voxel.")
        return values.astype(np.float64, copy=False)

    @property
    def voxel_std(self):
        """Return each masked voxel's BOLD standard deviation, pooled over connectome subjects and timepoints."""
        if self._voxel_std is None:
            root = self._connectome_root()
            self._voxel_std = self._mask_values(self.std_mask_path if self.std_mask_path is not None else root / f"{root.name}_std_weighted_mask.nii.gz")
        return self._voxel_std

    @property
    def voxel_norm(self):
        """Return each masked voxel's BOLD norm, pooled over connectome subjects and timepoints."""
        if self._voxel_norm is None:
            root = self._connectome_root()
            self._voxel_norm = self._mask_values(self.norm_mask_path if self.norm_mask_path is not None else root / f"{root.name}_norm_weighted_mask.nii.gz")
        return self._voxel_norm

    def _combo_path(self, chunk):
        combo_dir = Path(self.combo_dir) if self.combo_dir is not None else self._connectome_root() / "Combo"
        return combo_dir / self.combo_pattern.format(chunk=int(chunk))

    def _combo_chunk(self, chunk, global_rows):
        combo_path = self._combo_path(chunk)
        if not combo_path.is_file(): raise FileNotFoundError(f"Combo chunk not found: {combo_path}")
        try:
            combo = np.load(combo_path, mmap_mode="r", allow_pickle=False)
        except ValueError as error:
            raise ValueError(f"{combo_path} is not a valid .npy file (damaged or incomplete); rerun 00_convert_connectome_fp8.py to rebuild it.") from error
        if combo.shape != (global_rows.size, self.chunk_labels.size) or not combo.flags.c_contiguous: raise ValueError(f"Combo chunk {combo_path} must be a row-major array of shape {(global_rows.size, self.chunk_labels.size)}.")
        return combo

    def _combo_rows(self, combo, row_voxels, local_rows, first, last):
        """Return Combo rows (columns first:last) as float32, undoing the float8 pooled-correlation encoding for uint8 chunks."""
        if combo.dtype != np.uint8: return np.asarray(combo[local_rows, first:last], dtype=np.float32)
        norms = self.voxel_norm.astype(np.float32)
        return decode_e4m3(combo[local_rows, first:last]) * (norms[row_voxels][:, None] / np.float32(COMBO_FP8_SCALE)) * norms[first:last]

    def _seed_signal_norms(self, weights):
        """Return sqrt(w^T Combo w) per seed: the norm of the seed's weighted-sum BOLD signal over all subjects."""
        active_columns = np.flatnonzero(np.any(weights != 0, axis=0))
        first, last = int(active_columns[0]), int(active_columns[-1]) + 1
        column_weights = np.ascontiguousarray(weights[:, first:last].T)
        rows_per_batch = max(1, self.row_batch_bytes // ((last - first) * np.dtype(np.float32).itemsize))
        quadratic = np.zeros(weights.shape[0], dtype=np.float64)
        for chunk, global_rows in tqdm(self.chunk_rows, desc="Streaming combo chunks"):
            chunk_weights = weights[:, global_rows]
            active_rows = np.flatnonzero(np.any(chunk_weights != 0, axis=0))
            if active_rows.size == 0: continue
            combo = self._combo_chunk(chunk, global_rows)
            for start in range(0, active_rows.size, rows_per_batch):
                local_rows = active_rows[start : start + rows_per_batch]
                # Only the column span covering the seeds is read from each row.
                projected = self._combo_rows(combo, global_rows[local_rows], local_rows, first, last) @ column_weights
                quadratic += np.einsum("sr,rs->s", chunk_weights[:, local_rows].astype(np.float64), projected.astype(np.float64))
        if np.any(quadratic <= 0): raise ValueError("A seed's summed BOLD signal has zero energy, so its map cannot be normalized.")
        return np.sqrt(quadratic)

    def _seed_maps(self, weights):
        """Return seed maps normalized like a seed whose BOLD signal is the weighted voxel sum.

        w_i: seed weight at voxel i; s_i, n_i: voxel BOLD standard deviation and norm (voxel_std, voxel_norm);
        C: the connectome stat matrix (T, AvgR, ...); Combo: summed BOLD cross-products. For each seed,
        map_j = [sum_i w_i s_i C_ij / sum_i w_i s_i] * [sum_i w_i n_i / sqrt(w^T Combo w)],
        the same normalization as pfc-toolkit's precomputed-connectome lesion mapping.
        """
        weights = self._seed_weights(weights)
        if not np.all(np.any(weights != 0, axis=1)): raise ValueError("Every seed needs at least one nonzero weight.")
        # Fail before streaming the connectome if any Combo chunk the seeds need is missing.
        needed = [(chunk, global_rows) for chunk, global_rows in self.chunk_rows if np.any(weights[:, global_rows] != 0)]
        missing = [int(chunk) for chunk, _ in needed if not self._combo_path(chunk).is_file()]
        if missing: raise FileNotFoundError(f"{len(missing)} Combo chunk(s) needed by these seeds are missing from {self._combo_path(missing[0]).parent}: {missing}")
        for chunk, global_rows in needed: self._combo_chunk(chunk, global_rows)
        std_weights = weights * self.voxel_std.astype(np.float32)
        std_totals = std_weights.sum(axis=1, dtype=np.float64)
        if np.any(std_totals == 0): raise ValueError("A seed's BOLD-standard-deviation weights sum to zero, so its map cannot be normalized.")
        maps = self._stream_connectivity(std_weights).astype(np.float64)
        scale = (weights.astype(np.float64) @ self.voxel_norm) / (std_totals * self._seed_signal_norms(weights))
        return (maps * scale[:, None]).astype(np.float32)

    def _stream_connectivity(self, weights):
        """Multiply one or more lesion vectors by only their active chunk rows."""
        weights = self._seed_weights(weights)

        connectivity = np.zeros(weights.shape, dtype=np.float32)
        scratch = np.empty_like(connectivity)
        rows_per_batch = self._rows_per_batch()

        for chunk, global_rows in tqdm(
            self.chunk_rows,
            desc="Streaming connectome chunks",
        ):
            chunk_weights = weights[:, global_rows]
            active_rows = np.flatnonzero(np.any(chunk_weights != 0, axis=0))
            if active_rows.size == 0:
                continue

            chunk_data = self._load_chunk(chunk, global_rows)

            # Lesion cohorts usually activate different rows. Multiplying the
            # union of those rows for every lesion wastes most FLOPs on known
            # zeros, so process each lesion independently when the weight
            # matrix is sparse enough to offset any repeated row reads.
            nonzero_weights = np.count_nonzero(chunk_weights[:, active_rows])
            dense_weight_slots = weights.shape[0] * active_rows.size
            if (
                weights.shape[0] > 1
                and nonzero_weights * 2 < dense_weight_slots
            ):
                for start in range(0, active_rows.size, rows_per_batch):
                    local_rows = active_rows[start : start + rows_per_batch]
                    data_block = self._rows_float32(chunk_data, local_rows)
                    block_weights = chunk_weights[:, local_rows]
                    for profile_index in range(weights.shape[0]):
                        profile_rows = np.flatnonzero(
                            block_weights[profile_index] != 0
                        )
                        if profile_rows.size == 0:
                            continue
                        np.matmul(
                            block_weights[profile_index, profile_rows],
                            data_block[profile_rows, :],
                            out=scratch[profile_index],
                        )
                        np.add(
                            connectivity[profile_index],
                            scratch[profile_index],
                            out=connectivity[profile_index],
                        )
                continue

            # Sparse lesions read only the required rows instead of scanning the
            # entire multi-gigabyte chunk. Batching bounds the temporary buffer.
            for start in range(0, active_rows.size, rows_per_batch):
                local_rows = active_rows[start : start + rows_per_batch]
                data_block = self._rows_float32(chunk_data, local_rows)
                np.matmul(chunk_weights[:, local_rows], data_block, out=scratch)
                np.add(connectivity, scratch, out=connectivity)

        return connectivity

    def _resolved_similarity_stats_path(self):
        if self.similarity_stats_path is not None:
            return Path(self.similarity_stats_path)
        return Path(self.connectome_dir) / "row_similarity_stats.npz"

    def _read_similarity_stats(self, cache_path):
        with np.load(cache_path, allow_pickle=False) as cached:
            row_sum = np.asarray(cached["row_sum"], dtype=np.float64)
            row_square_sum = np.asarray(
                cached["row_square_sum"], dtype=np.float64
            )

        expected_shape = (self.chunk_labels.size,)
        if row_sum.shape != expected_shape or row_square_sum.shape != expected_shape:
            raise ValueError(
                f"Similarity cache {cache_path} does not match the connectome: "
                f"expected vectors of shape {expected_shape}."
            )
        return row_sum, row_square_sum

    def prepare_similarity_cache(self, force=False):
        """Compute and persist per-row sums needed by cosine and Pearson maps."""
        cache_path = self._resolved_similarity_stats_path()
        if cache_path.is_file() and not force:
            self._row_stats = self._read_similarity_stats(cache_path)
            return cache_path

        row_sum = np.empty(self.chunk_labels.size, dtype=np.float64)
        row_square_sum = np.empty(self.chunk_labels.size, dtype=np.float64)
        rows_per_batch = self._rows_per_batch()

        for chunk, global_rows in tqdm(
            self.chunk_rows, desc="Preparing connectome similarity cache"
        ):
            chunk_data = self._load_chunk(chunk, global_rows)
            for start in range(0, global_rows.size, rows_per_batch):
                stop = min(start + rows_per_batch, global_rows.size)
                data_block = self._rows_float32(chunk_data, slice(start, stop))
                destination = global_rows[start:stop]
                row_sum[destination] = np.sum(
                    data_block, axis=1, dtype=np.float64
                )
                row_square_sum[destination] = np.einsum(
                    "ij,ij->i",
                    data_block,
                    data_block,
                    dtype=np.float64,
                    optimize=True,
                )

        self._row_stats = (row_sum, row_square_sum)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=cache_path.parent,
                prefix=f".{cache_path.name}.",
                suffix=".tmp.npz",
                delete=False,
            ) as temporary:
                temp_path = Path(temporary.name)
                np.savez(
                    temporary,
                    row_sum=row_sum,
                    row_square_sum=row_square_sum,
                )
            os.replace(temp_path, cache_path)
        except OSError as error:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            warnings.warn(
                f"Could not persist similarity cache {cache_path}: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
        return cache_path

    def _get_similarity_stats(self):
        if self._row_stats is None:
            cache_path = self._resolved_similarity_stats_path()
            if cache_path.is_file():
                self._row_stats = self._read_similarity_stats(cache_path)
            else:
                self.prepare_similarity_cache()
        return self._row_stats

    def _compute_inverted_profiles(self, weights, similarity):
        """Compare each lesion vector with every row of symmetric ``C``."""
        similarity = similarity.lower()
        if similarity not in {"cosine", "pearson"}:
            raise ValueError("similarity must be 'cosine' or 'pearson'.")

        weights = self._seed_weights(weights)

        row_sum, row_square_sum = self._get_similarity_stats()

        # The functional connectome is symmetric, so W @ C is also C @ W.T
        # transposed. This preserves the sparse active-row optimization.
        numerators = self._stream_connectivity(weights)
        similarities = np.zeros_like(numerators)
        n_voxels = weights.shape[1]

        if similarity == "cosine":
            row_norm = np.sqrt(np.maximum(row_square_sum, 0))
            weight_norms = np.sqrt(
                np.einsum(
                    "ij,ij->i", weights, weights, dtype=np.float64, optimize=True
                )
            )
            if np.any(weight_norms == 0):
                raise ValueError("Cosine similarity is undefined for an empty map.")

            for index, weight_norm in enumerate(weight_norms):
                denominator = row_norm * weight_norm
                np.divide(
                    numerators[index],
                    denominator,
                    out=similarities[index],
                    where=denominator > 0,
                )
            return similarities

        weight_sum = np.sum(weights, axis=1, dtype=np.float64)
        weight_square_sum = np.einsum(
            "ij,ij->i", weights, weights, dtype=np.float64, optimize=True
        )
        weight_centered_square_sum = np.maximum(
            weight_square_sum - np.square(weight_sum) / n_voxels,
            0,
        )
        if np.any(weight_centered_square_sum == 0):
            raise ValueError(
                "Pearson correlation is undefined for a constant map."
            )

        row_centered_square_sum = np.maximum(
            row_square_sum - np.square(row_sum) / n_voxels,
            0,
        )
        row_centered_norm = np.sqrt(row_centered_square_sum)

        for index in range(weights.shape[0]):
            centered_numerator = (
                numerators[index].astype(np.float64)
                - (weight_sum[index] / n_voxels) * row_sum
            )
            denominator = (
                row_centered_norm * np.sqrt(weight_centered_square_sum[index])
            )
            np.divide(
                centered_numerator,
                denominator,
                out=similarities[index],
                where=denominator > 0,
            )
        return similarities

    def generate_connectivity_profile(
        self, nifti_path, binarize=False, threshold=None
    ):
        """Return one lesion's masked connectivity vector without building ``C``."""
        weights = self._load_lesion_weights(nifti_path)
        if binarize:
            weights = self._binarize_data(weights, threshold)

        return self._flatten_frames(self._seed_maps(weights))

    def generate_profiles_from_niftis(
        self, nifti_paths, binarize=False, threshold=None
    ):
        """Process all seeds chunk-first and return flattened voxel-frame rows."""
        weights = [self._load_lesion_weights(path) for path in tqdm(nifti_paths, desc="Loading lesion weights")]
        frame_counts = [value.shape[0] for value in weights]
        weights = np.vstack(weights).astype(np.float32, copy=False)
        if binarize:
            weights = self._binarize_data(weights, threshold)
        return self._flatten_profile_batches(self._seed_maps(weights), frame_counts)

    def generate_inverted_connectivity_profile(
        self,
        nifti_path,
        similarity="cosine",
        binarize=False,
        threshold=None,
    ):
        """Return similarity between one NIfTI and every connectome row."""
        weights = self._load_lesion_weights(nifti_path)
        if binarize:
            weights = self._binarize_data(weights, threshold)
        return self._flatten_frames(self._compute_inverted_profiles(weights, similarity))

    def generate_inverted_profiles_from_niftis(
        self,
        nifti_paths,
        similarity="cosine",
        binarize=False,
        threshold=None,
    ):
        """Return flattened voxel-frame similarity rows for several NIfTIs."""
        weights = [self._load_lesion_weights(path) for path in tqdm(nifti_paths, desc="Loading similarity maps")]
        frame_counts = [value.shape[0] for value in weights]
        weights = np.vstack(weights).astype(np.float32, copy=False)
        if binarize:
            weights = self._binarize_data(weights, threshold)
        return self._flatten_profile_batches(self._compute_inverted_profiles(weights, similarity), frame_counts)

    @staticmethod
    def _safe_stem(path):
        path = Path(path)
        return path.name[:-7] if path.name.endswith(".nii.gz") else path.stem

    @staticmethod
    def _profile_output_path(nifti_path, suffix, out_dir=None):
        """Return a profile path, defaulting to a sibling ``conn`` directory."""
        nifti_path = Path(nifti_path).expanduser()
        destination = (
            Path(out_dir).expanduser()
            if out_dir is not None
            else nifti_path.parent.parent / "conn"
        )
        return destination / (
            f"{FunctionalConnectivity._safe_stem(nifti_path)}{suffix}.nii.gz"
        )

    def save_profile(self, connectivity, out_path):
        """Restore and save one flattened voxel-frame connectivity profile."""
        out_path = Path(out_path)
        if out_path.name.endswith(".nii.gz"):
            output_base = out_path.with_name(out_path.name[:-7])
        elif out_path.suffix == ".nii":
            output_base = out_path.with_suffix("")
        else:
            output_base = out_path

        output_base.parent.mkdir(parents=True, exist_ok=True)
        NiftiIO(mask_path=str(self.mask_path)).save_files(
            arr=np.asarray(connectivity, dtype=np.float32),
            file_paths=[str(output_base)],
            dry_run=False,
        )
        return output_base.with_name(f"{output_base.name}.nii.gz")

    def save_profiles_from_niftis(
        self,
        nifti_paths,
        out_dir=None,
        binarize=False,
        threshold=None,
        suffix="",
    ):
        """Save each map to a sibling ``conn`` or an explicit directory, named <stem>_<connectome_label><suffix>."""
        nifti_paths = list(nifti_paths)
        saved_paths = []
        profiles = self.generate_profiles_from_niftis(
            nifti_paths, binarize, threshold
        )

        for nifti_path, connectivity in tqdm(
            zip(nifti_paths, profiles),
            total=len(nifti_paths),
            desc="Saving functional connectivity profiles",
        ):
            out_path = self._profile_output_path(nifti_path, f"_{self.connectome_label}{suffix}", out_dir=out_dir)
            saved_paths.append(self.save_profile(connectivity, out_path))

        return saved_paths

    def save_inverted_profiles_from_niftis(
        self,
        nifti_paths,
        out_dir=None,
        similarity="cosine",
        binarize=False,
        threshold=None,
        suffix=None,
    ):
        """Save each map to a sibling ``conn`` or an explicit directory."""
        nifti_paths = list(nifti_paths)
        profiles = self.generate_inverted_profiles_from_niftis(
            nifti_paths,
            similarity=similarity,
            binarize=binarize,
            threshold=threshold,
        )
        suffix = suffix or f"_{similarity.lower()}_inverted_connectivity"
        saved_paths = []

        for nifti_path, profile in tqdm(
            zip(nifti_paths, profiles),
            total=len(nifti_paths),
            desc="Saving inverted connectivity profiles",
        ):
            out_path = self._profile_output_path(nifti_path, f"_{self.connectome_label}{suffix}", out_dir=out_dir)
            saved_paths.append(self.save_profile(profile, out_path))

        return saved_paths


FMRIConnectivity = FunctionalConnectivity
