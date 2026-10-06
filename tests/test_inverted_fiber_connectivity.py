import nibabel as nib
import numpy as np

from calvin_utils.neuroimaging_utils.tract_utils.inverted_connectivity import (
    InvertedFiberConnectivity,
)
from calvin_utils.neuroimaging_utils.tract_utils.fiber_io import FiberIO
from calvin_utils.neuroimaging_utils.tract_utils.voxel_seeded_connectome import (
    VoxelSeededFiberStore,
)


def _save_fibers(path, fibers):
    stored = np.empty(len(fibers), dtype=object)
    stored[:] = [np.asarray(fiber, dtype=np.float32) for fiber in fibers]
    np.savez(path, fibers=stored)


def _save_seeded_fibers(path, reference_path, fibers, seed_voxels):
    reference = nib.load(reference_path)
    VoxelSeededFiberStore.write(
        path,
        zip(seed_voxels, fibers),
        reference,
    )


def test_orientation_invariant_fixed_mm_fiber_cosine(tmp_path):
    reference_path = tmp_path / "mask.nii.gz"
    nib.save(
        nib.Nifti1Image(np.ones((5, 5, 2), dtype=np.uint8), np.eye(4)),
        reference_path,
    )

    connectome_path = tmp_path / "connectome.voxel_fibers.h5"
    fibers = [
        [[1, 1, 1], [3, 1, 1]],
        [[3, 1, 1], [1, 1, 1]],
        [[1, 3, 1], [3, 3, 1]],
    ]
    seed_one = np.ravel_multi_index((1, 1, 1), (5, 5, 2))
    seed_two = np.ravel_multi_index((1, 3, 1), (5, 5, 2))
    _save_seeded_fibers(
        connectome_path,
        reference_path,
        fibers,
        [seed_one, seed_one, seed_two],
    )

    # Same trajectory as the first two connectome fibers, opposite ordering,
    # and a different number of original vertices.
    target_path = tmp_path / "target.npz"
    _save_fibers(
        target_path,
        [[[3, 1, 1], [2.5, 1, 1], [2, 1, 1], [1, 1, 1]]],
    )

    mapper = InvertedFiberConnectivity(
        voxel_seeded_connectome_path=connectome_path,
        reference_nifti_path=reference_path,
        sample_interval_mm=1.0,
        max_length_mm=3.0,
    )
    profile = mapper.generate_inverted_connectivity_profile(target_path)
    assert profile.shape == (50,)
    np.testing.assert_allclose(profile[seed_one], 1.0, rtol=1e-6, atol=1e-6)
    assert profile[seed_two] < profile[seed_one]
    traversed_but_not_seeded = np.ravel_multi_index((2, 1, 1), (5, 5, 2))
    assert profile[traversed_but_not_seeded] == 0

    output_path = mapper.save_profile(profile, tmp_path / "cosine.nii.gz")
    saved = nib.load(output_path)
    np.testing.assert_allclose(saved.get_fdata().reshape(-1), profile)

    saved_paths = mapper.save_inverted_profiles_from_fibers(
        [target_path],
        tmp_path / "batch",
        similarity="valid_sample_cosine",
    )
    assert len(saved_paths) == 1
    assert saved_paths[0].is_file()


def test_target_values_select_binary_or_weighted_average_trajectory(tmp_path):
    reference_path = tmp_path / "mask.nii.gz"
    nib.save(
        nib.Nifti1Image(np.ones((6, 12, 2), dtype=np.uint8), np.eye(4)),
        reference_path,
    )

    connectome_path = tmp_path / "connectome.voxel_fibers.h5"
    _save_seeded_fibers(
        connectome_path,
        reference_path,
        [[[1, 2, 1], [3, 2, 1]]],
        [np.ravel_multi_index((1, 2, 1), (6, 12, 2))],
    )

    atlas_path = tmp_path / "target_atlas.npz"
    _save_fibers(
        atlas_path,
        [
            [[1, 1, 1], [3, 1, 1]],
            [[3, 3, 1], [1, 3, 1]],
            [[1, 10, 1], [3, 10, 1]],
        ],
    )
    values_path = tmp_path / "target.fib.npy"
    np.save(values_path, np.asarray([1.0, 3.0, -100.0], dtype=np.float32))
    FiberIO.write_values_description(values_path, atlas_path)

    mapper = InvertedFiberConnectivity(
        voxel_seeded_connectome_path=connectome_path,
        reference_nifti_path=reference_path,
        sample_interval_mm=1.0,
        max_length_mm=3.0,
    )

    binary = mapper.average_target_trajectory(values_path, weighting="binary")
    binary_all = mapper.average_target_trajectory(
        values_path,
        weighting="binary",
        top_percent=100,
    )
    weighted = mapper.average_target_trajectory(values_path, weighting="weighted")

    # Target inversion is positive-only by default. Binary mode then keeps the
    # upper five percent of those positive values, so only raw value 3 survives.
    # Passing 100 disables tail filtering but still excludes the negative fiber.
    np.testing.assert_allclose(binary[:, 1], 3.0)
    np.testing.assert_allclose(binary_all[:, 1], 2.0)
    np.testing.assert_allclose(weighted[:, 1], 2.5)
    # A single surviving target retains its arbitrary endpoint order; downstream
    # matching is orientation invariant.
    np.testing.assert_allclose(np.sort(binary[:, 0]), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(binary_all[:, 0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(weighted[:, 0], [1.0, 2.0, 3.0])


def test_pairwise_mean_streams_the_full_cartesian_mean(tmp_path):
    reference_path = tmp_path / "mask.nii.gz"
    nib.save(
        nib.Nifti1Image(np.ones((6, 6, 2), dtype=np.uint8), np.eye(4)),
        reference_path,
    )
    connectome_path = tmp_path / "connectome.voxel_fibers.h5"
    candidate = np.asarray([[1, 1, 1], [3, 1, 1]], dtype=np.float32)
    seed_voxel = np.ravel_multi_index((1, 1, 1), (6, 6, 2))
    _save_seeded_fibers(
        connectome_path,
        reference_path,
        [candidate],
        [seed_voxel],
    )

    target_path = tmp_path / "target_bundle.npz"
    target_fibers = [
        np.asarray([[1, 1, 1], [3, 1, 1]], dtype=np.float32),
        np.asarray([[1, 4, 1], [4, 4, 1]], dtype=np.float32),
    ]
    _save_fibers(target_path, target_fibers)

    mapper = InvertedFiberConnectivity(
        voxel_seeded_connectome_path=connectome_path,
        reference_nifti_path=reference_path,
        sample_interval_mm=1.0,
    )
    pairwise_profile = mapper.generate_inverted_connectivity_profile(
        target_path,
        target_bundle_reduction="pairwise_mean",
    )
    average_profile = mapper.generate_inverted_connectivity_profile(
        target_path,
        target_bundle_reduction="average_trajectory",
    )

    sampled_candidate = mapper.registration.sample(candidate)
    expected_scores = []
    for target in target_fibers:
        alignment = mapper.registration.align_sampled(
            candidate=sampled_candidate,
            target=mapper.registration.sample(target),
        )
        expected_scores.append(alignment.cosine_similarity)
    expected = np.mean(expected_scores)
    voxel = np.ravel_multi_index((1, 1, 1), (6, 6, 2))

    np.testing.assert_allclose(pairwise_profile[voxel], expected, rtol=1e-6)
    assert not np.isclose(pairwise_profile[voxel], average_profile[voxel])


def test_streaming_profiles_use_one_connectome_pass_and_no_disk_cache(tmp_path):
    reference_path = tmp_path / "mask.nii.gz"
    nib.save(
        nib.Nifti1Image(np.ones((5, 5, 2), dtype=np.uint8), np.eye(4)),
        reference_path,
    )
    connectome_path = tmp_path / "connectome.voxel_fibers.h5"
    _save_seeded_fibers(
        connectome_path,
        reference_path,
        [
            [[1, 1, 1], [3, 1, 1]],
            [[1, 3, 1], [3, 3, 1]],
        ],
        [
            np.ravel_multi_index((1, 1, 1), (5, 5, 2)),
            np.ravel_multi_index((1, 3, 1), (5, 5, 2)),
        ],
    )
    target_one = tmp_path / "target_one.npz"
    target_two = tmp_path / "target_two.npz"
    _save_fibers(target_one, [[[1, 1, 1], [3, 1, 1]]])
    _save_fibers(target_two, [[[1, 3, 1], [3, 3, 1]]])

    mapper = InvertedFiberConnectivity(
        voxel_seeded_connectome_path=connectome_path,
        reference_nifti_path=reference_path,
        sample_interval_mm=1.0,
    )
    original_iterator = mapper.connectome.iter_seeded_fibers
    pass_count = 0

    def tracked_iterator():
        nonlocal pass_count
        pass_count += 1
        yield from original_iterator()

    mapper.connectome.iter_seeded_fibers = tracked_iterator
    profiles = mapper.generate_inverted_profiles_streaming(
        [target_one, target_two]
    )

    assert profiles.shape == (2, 50)
    assert pass_count == 1
