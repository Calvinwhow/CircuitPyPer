"""Compute lesion-network maps from chunked functional-connectome rows."""

import os
import tempfile
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

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

    @staticmethod
    def _binarize_data(data, threshold):
        if threshold is None:
            raise ValueError("threshold must be provided when binarize=True.")
        return (data > threshold).astype(np.float32)

    def _load_lesion_weights(self, nifti_path):
        lesion_img = nib.load(nifti_path)
        mask_img = nib.load(str(self.mask_path))

        if lesion_img.shape[:3] != mask_img.shape or not np.allclose(
            lesion_img.affine, mask_img.affine
        ):
            from nilearn import image

            lesion_img = image.resample_to_img(
                source_img=lesion_img,
                target_img=mask_img,
                interpolation="continuous",
            )

        weights = NiftiIO(
            mask_path=str(self.mask_path)
        ).import_nifti_to_numpy_array([lesion_img])[:, 0]
        return weights.astype(np.float32, copy=False)

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

    def _rows_per_batch(self):
        bytes_per_row = self.chunk_labels.size * np.dtype(np.float32).itemsize
        return max(1, self.row_batch_bytes // bytes_per_row)

    def _stream_connectivity(self, weights):
        """Multiply one or more lesion vectors by only their active chunk rows."""
        weights = np.asarray(weights, dtype=np.float32)
        if weights.ndim == 1:
            weights = weights[None, :]
        if weights.ndim != 2 or weights.shape[1] != self.chunk_labels.size:
            raise ValueError(
                "weights must have shape (n_lesions, n_masked_voxels); "
                f"got {weights.shape}."
            )

        connectivity = np.zeros(weights.shape, dtype=np.float32)
        scratch = np.empty_like(connectivity)
        rows_per_batch = self._rows_per_batch()

        for chunk, global_rows in self.chunk_rows:
            chunk_weights = weights[:, global_rows]
            active_rows = np.flatnonzero(np.any(chunk_weights != 0, axis=0))
            if active_rows.size == 0:
                continue

            chunk_data = self._load_chunk(chunk, global_rows)

            # Dense weights are fastest as one BLAS operation directly from mmap.
            if (
                active_rows.size == global_rows.size
                and chunk_data.dtype == np.float32
            ):
                np.matmul(chunk_weights, chunk_data, out=scratch)
                np.add(connectivity, scratch, out=connectivity)
                continue

            # Sparse lesions read only the required rows instead of scanning the
            # entire multi-gigabyte chunk. Batching bounds the temporary buffer.
            for start in range(0, active_rows.size, rows_per_batch):
                local_rows = active_rows[start : start + rows_per_batch]
                data_block = np.asarray(
                    chunk_data[local_rows, :], dtype=np.float32, order="C"
                )
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
                data_block = np.asarray(
                    chunk_data[start:stop, :], dtype=np.float32, order="C"
                )
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

        weights = np.asarray(weights, dtype=np.float32)
        if weights.ndim == 1:
            weights = weights[None, :]
        if weights.ndim != 2 or weights.shape[1] != self.chunk_labels.size:
            raise ValueError(
                "weights must have shape (n_lesions, n_masked_voxels); "
                f"got {weights.shape}."
            )

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

        labels = self.chunk_labels
        if weights.size != labels.size:
            raise ValueError(
                f"Lesion has {weights.size} masked voxels, but the chunk index "
                f"contains {labels.size}."
            )

        return self._stream_connectivity(weights)[0]

    def generate_profiles_from_niftis(
        self, nifti_paths, binarize=False, threshold=None
    ):
        """Process all lesions chunk-first and return ``(lesions, voxels)``."""
        weights = np.vstack(
            [
                self._load_lesion_weights(path)
                for path in tqdm(nifti_paths, desc="Loading lesion weights")
            ]
        ).astype(np.float32, copy=False)
        if binarize:
            weights = self._binarize_data(weights, threshold)
        return self._stream_connectivity(weights)

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
        return self._compute_inverted_profiles(weights, similarity)[0]

    def generate_inverted_profiles_from_niftis(
        self,
        nifti_paths,
        similarity="cosine",
        binarize=False,
        threshold=None,
    ):
        """Return row-wise similarity maps for several NIfTIs at once."""
        weights = np.vstack(
            [
                self._load_lesion_weights(path)
                for path in tqdm(nifti_paths, desc="Loading similarity maps")
            ]
        ).astype(np.float32, copy=False)
        if binarize:
            weights = self._binarize_data(weights, threshold)
        return self._compute_inverted_profiles(weights, similarity)

    @staticmethod
    def _safe_stem(path):
        path = Path(path)
        return path.name[:-7] if path.name.endswith(".nii.gz") else path.stem

    def save_profile(self, connectivity, out_path):
        """Save a masked connectivity vector in the connectome's NIfTI space."""
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
        out_dir,
        binarize=False,
        threshold=None,
        suffix="_functional_connectivity",
    ):
        """Compute and save one NIfTI connectivity map per lesion."""
        nifti_paths = list(nifti_paths)
        out_dir = Path(out_dir)
        saved_paths = []
        profiles = self.generate_profiles_from_niftis(
            nifti_paths, binarize, threshold
        )

        for nifti_path, connectivity in tqdm(
            zip(nifti_paths, profiles),
            total=len(nifti_paths),
            desc="Saving functional connectivity profiles",
        ):
            out_path = out_dir / f"{self._safe_stem(nifti_path)}{suffix}.nii.gz"
            saved_paths.append(self.save_profile(connectivity, out_path))

        return saved_paths

    def save_inverted_profiles_from_niftis(
        self,
        nifti_paths,
        out_dir,
        similarity="cosine",
        binarize=False,
        threshold=None,
        suffix=None,
    ):
        """Compute and save one row-similarity NIfTI per input map."""
        nifti_paths = list(nifti_paths)
        profiles = self.generate_inverted_profiles_from_niftis(
            nifti_paths,
            similarity=similarity,
            binarize=binarize,
            threshold=threshold,
        )
        out_dir = Path(out_dir)
        suffix = suffix or f"_{similarity.lower()}_inverted_connectivity"
        saved_paths = []

        for nifti_path, profile in tqdm(
            zip(nifti_paths, profiles),
            total=len(nifti_paths),
            desc="Saving inverted connectivity profiles",
        ):
            out_path = out_dir / f"{self._safe_stem(nifti_path)}{suffix}.nii.gz"
            saved_paths.append(self.save_profile(profile, out_path))

        return saved_paths


FMRIConnectivity = FunctionalConnectivity
