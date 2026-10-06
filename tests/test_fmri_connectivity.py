import nibabel as nib
import numpy as np

from calvin_utils.neuroimaging_utils.fmri_utils.compute_connectivity import (
    FunctionalConnectivity,
)


def test_functional_connectivity_computes_and_saves_w_times_c(tmp_path):
    affine = np.array(
        [
            [2.0, 0.0, 0.0, -4.0],
            [0.0, 2.0, 0.0, -6.0],
            [0.0, 0.0, 2.0, -8.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    weights = np.array([1.0, 2.0, 0.0, 3.0], dtype=np.float32)
    connectome = np.arange(16, dtype=np.float32).reshape(4, 4)
    lesion_path = tmp_path / "lesion.nii.gz"
    mask_path = tmp_path / "mask.nii.gz"
    chunk_index_path = tmp_path / "chunk_idx.nii.gz"
    connectome_dir = tmp_path / "AvgR"
    connectome_dir.mkdir()

    nib.save(
        nib.Nifti1Image(weights.reshape(2, 2, 1), affine),
        lesion_path,
    )
    nib.save(
        nib.Nifti1Image(np.ones((2, 2, 1), dtype=np.uint8), affine),
        mask_path,
    )

    # Chunk rows are not contiguous in the global masked-voxel ordering.
    chunk_labels = np.array([1, 2, 1, 2], dtype=np.int16)
    nib.save(
        nib.Nifti1Image(chunk_labels.reshape(2, 2, 1), affine),
        chunk_index_path,
    )
    np.save(connectome_dir / "1_AvgR.npy", connectome[[0, 2], :])
    np.save(connectome_dir / "2_AvgR.npy", connectome[[1, 3], :])

    mapper = FunctionalConnectivity(
        connectome_dir=connectome_dir,
        mask_path=mask_path,
        chunk_index_path=chunk_index_path,
    )
    result = mapper.generate_connectivity_profile(lesion_path)
    np.testing.assert_allclose(result, weights @ connectome)

    saved_path = mapper.save_profile(
        result,
        tmp_path / "connectivity.nii.gz",
    )
    saved = nib.load(saved_path)
    np.testing.assert_allclose(saved.get_fdata().reshape(-1), result)
    np.testing.assert_allclose(saved.affine, affine)


def test_functional_connectivity_processes_cohort_chunk_first(tmp_path):
    affine = np.eye(4)
    connectome = np.arange(16, dtype=np.float32).reshape(4, 4)
    mask_path = tmp_path / "mask.nii.gz"
    chunk_index_path = tmp_path / "chunk_idx.nii.gz"
    connectome_dir = tmp_path / "AvgR"
    connectome_dir.mkdir()

    nib.save(
        nib.Nifti1Image(np.ones((2, 2, 1), dtype=np.uint8), affine),
        mask_path,
    )
    labels = np.array([1, 2, 1, 2], dtype=np.int16)
    nib.save(
        nib.Nifti1Image(labels.reshape(2, 2, 1), affine),
        chunk_index_path,
    )
    np.save(connectome_dir / "1_AvgR.npy", connectome[[0, 2], :])
    np.save(connectome_dir / "2_AvgR.npy", connectome[[1, 3], :])

    weights = np.array(
        [[1.0, 0.0, 2.0, 0.0], [0.0, 3.0, 0.0, 4.0]],
        dtype=np.float32,
    )
    lesion_paths = []
    for index, lesion_weights in enumerate(weights):
        path = tmp_path / f"lesion_{index}.nii.gz"
        nib.save(
            nib.Nifti1Image(lesion_weights.reshape(2, 2, 1), affine),
            path,
        )
        lesion_paths.append(path)

    mapper = FunctionalConnectivity(
        connectome_dir=connectome_dir,
        mask_path=mask_path,
        chunk_index_path=chunk_index_path,
        row_batch_mb=0.00002,
    )
    profiles = mapper.generate_profiles_from_niftis(lesion_paths)

    np.testing.assert_allclose(profiles, weights @ connectome)


def test_functional_connectivity_does_not_open_inactive_chunks(tmp_path):
    affine = np.eye(4)
    mask_path = tmp_path / "mask.nii.gz"
    chunk_index_path = tmp_path / "chunk_idx.nii.gz"
    lesion_path = tmp_path / "lesion.nii.gz"
    connectome_dir = tmp_path / "AvgR"
    connectome_dir.mkdir()

    nib.save(
        nib.Nifti1Image(np.ones((2, 2, 1), dtype=np.uint8), affine),
        mask_path,
    )
    labels = np.array([1, 2, 1, 2], dtype=np.int16)
    nib.save(
        nib.Nifti1Image(labels.reshape(2, 2, 1), affine),
        chunk_index_path,
    )
    nib.save(
        nib.Nifti1Image(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(2, 2, 1),
            affine,
        ),
        lesion_path,
    )
    chunk_one = np.arange(8, dtype=np.float32).reshape(2, 4)
    np.save(connectome_dir / "1_AvgR.npy", chunk_one)
    # No chunk 2 file: the lesion has no weight there, so it must not be opened.

    mapper = FunctionalConnectivity(
        connectome_dir=connectome_dir,
        mask_path=mask_path,
        chunk_index_path=chunk_index_path,
    )
    result = mapper.generate_connectivity_profile(lesion_path)

    np.testing.assert_allclose(result, chunk_one[0])


def test_inverted_connectivity_matches_rowwise_cosine_and_pearson(tmp_path):
    affine = np.eye(4)
    mask_path = tmp_path / "mask.nii.gz"
    chunk_index_path = tmp_path / "chunk_idx.nii.gz"
    lesion_path = tmp_path / "lesion.nii.gz"
    connectome_dir = tmp_path / "AvgR"
    stats_path = tmp_path / "row_stats.npz"
    connectome_dir.mkdir()

    connectome = np.array(
        [
            [1.0, 0.2, -0.3, 0.4],
            [0.2, 1.0, 0.5, -0.1],
            [-0.3, 0.5, 1.0, 0.7],
            [0.4, -0.1, 0.7, 1.0],
        ],
        dtype=np.float32,
    )
    weights = np.array([1.0, 2.0, 0.0, 3.0], dtype=np.float32)
    labels = np.array([1, 2, 1, 2], dtype=np.int16)

    nib.save(
        nib.Nifti1Image(np.ones((2, 2, 1), dtype=np.uint8), affine),
        mask_path,
    )
    nib.save(
        nib.Nifti1Image(labels.reshape(2, 2, 1), affine),
        chunk_index_path,
    )
    nib.save(
        nib.Nifti1Image(weights.reshape(2, 2, 1), affine),
        lesion_path,
    )
    np.save(connectome_dir / "1_AvgR.npy", connectome[[0, 2], :])
    np.save(connectome_dir / "2_AvgR.npy", connectome[[1, 3], :])

    mapper = FunctionalConnectivity(
        connectome_dir=connectome_dir,
        mask_path=mask_path,
        chunk_index_path=chunk_index_path,
        row_batch_mb=0.00002,
        similarity_stats_path=stats_path,
    )
    cosine = mapper.generate_inverted_connectivity_profile(
        lesion_path, similarity="cosine"
    )

    expected_cosine = (connectome @ weights) / (
        np.linalg.norm(connectome, axis=1) * np.linalg.norm(weights)
    )
    np.testing.assert_allclose(cosine, expected_cosine, rtol=1e-6, atol=1e-6)
    assert stats_path.is_file()

    # A new mapper must reuse the persistent row-statistics cache.
    mapper = FunctionalConnectivity(
        connectome_dir=connectome_dir,
        mask_path=mask_path,
        chunk_index_path=chunk_index_path,
        similarity_stats_path=stats_path,
    )
    pearson = mapper.generate_inverted_connectivity_profile(
        lesion_path, similarity="pearson"
    )
    expected_pearson = np.array(
        [np.corrcoef(row, weights)[0, 1] for row in connectome],
        dtype=np.float32,
    )
    np.testing.assert_allclose(pearson, expected_pearson, rtol=1e-6, atol=1e-6)
