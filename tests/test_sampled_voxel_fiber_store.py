import json

import nibabel as nib
import numpy as np

from calvin_utils.neuroimaging_utils.tract_utils.fiber_geometry_registration import (
    GeodesicFiberRegistration,
)
from calvin_utils.neuroimaging_utils.tract_utils.inverted_connectivity import (
    InvertedFiberConnectivity,
)
from calvin_utils.neuroimaging_utils.tract_utils.sampled_voxel_fiber_store import (
    SampledVoxelFiberStore,
)
from calvin_utils.neuroimaging_utils.tract_utils.voxel_seeded_connectome import (
    VoxelSeededFiberStore,
)


def _reference(path):
    image = nib.Nifti1Image(np.ones((8, 8, 2), dtype=np.uint8), np.eye(4))
    nib.save(image, path)
    return image


def _target(path):
    fibers = np.empty(1, dtype=object)
    fibers[0] = np.asarray([[1, 1, 1], [5, 1, 1]], dtype=np.float32)
    np.savez(path, fibers=fibers)


def test_sampled_store_writes_memory_mapped_length_buckets(tmp_path):
    reference_path = tmp_path / "reference.nii.gz"
    reference = _reference(reference_path)
    registration = GeodesicFiberRegistration(sample_interval_mm=1.0)
    seeded_fibers = [
        (8, np.asarray([[1, 1, 1], [5, 1, 1]], dtype=np.float32)),
        (3, np.asarray([[1, 2, 1], [3, 2, 1]], dtype=np.float32)),
        (3, np.asarray([[3, 3, 1], [1, 3, 1]], dtype=np.float32)),
    ]

    store = SampledVoxelFiberStore.write(
        tmp_path / "connectome.sampled_fibers",
        seeded_fibers,
        reference,
        registration,
        max_buffer_points=5,
    )

    assert store.n_fibers == 3
    assert [bucket["sample_count"] for bucket in store.buckets] == [3, 5]
    with (store.path / "manifest.json").open() as manifest_file:
        assert json.load(manifest_file)["format"] == store.FORMAT

    batches = list(store.iter_batches(max_fibers_per_batch=2))
    assert all(isinstance(batch.trajectories, np.memmap) for batch in batches)
    assert all(batch.trajectories.dtype == np.float32 for batch in batches)
    for batch in batches:
        assert np.all(np.diff(batch.seed_voxels) >= 0)
        np.testing.assert_allclose(
            batch.full_energy,
            np.einsum("bic,bic->b", batch.trajectories, batch.trajectories),
        )


def test_sampled_store_inversion_matches_raw_streaming_store(tmp_path):
    reference_path = tmp_path / "reference.nii.gz"
    reference = _reference(reference_path)
    fibers = [
        np.asarray([[1, 1, 1], [5, 1, 1]], dtype=np.float32),
        np.asarray([[5, 1, 1], [2, 1, 1]], dtype=np.float32),
        np.asarray([[1, 4, 1], [4, 4, 1]], dtype=np.float32),
        np.asarray([[1, 5, 1], [6, 5, 1]], dtype=np.float32),
    ]
    seeds = [
        np.ravel_multi_index((1, 1, 1), reference.shape),
        np.ravel_multi_index((1, 1, 1), reference.shape),
        np.ravel_multi_index((1, 4, 1), reference.shape),
        np.ravel_multi_index((1, 4, 1), reference.shape),
    ]
    raw_path = tmp_path / "connectome.voxel_fibers.h5"
    VoxelSeededFiberStore.write(raw_path, zip(seeds, fibers), reference)
    sampled_path = tmp_path / "connectome.sampled_fibers"
    SampledVoxelFiberStore.write(
        sampled_path,
        zip(seeds, fibers),
        reference,
        GeodesicFiberRegistration(sample_interval_mm=1.0),
        max_buffer_points=8,
    )
    target_path = tmp_path / "target.npz"
    _target(target_path)
    pairwise_target_path = tmp_path / "pairwise_target.npz"
    pairwise_fibers = np.empty(2, dtype=object)
    pairwise_fibers[0] = np.asarray(
        [[1, 1, 1], [5, 1, 1]], dtype=np.float32
    )
    pairwise_fibers[1] = np.asarray(
        [[1, 4, 1], [4, 4, 1]], dtype=np.float32
    )
    np.savez(pairwise_target_path, fibers=pairwise_fibers)

    raw_mapper = InvertedFiberConnectivity(raw_path, reference_path)
    sampled_mapper = InvertedFiberConnectivity(
        sampled_path,
        reference_path,
        memory_budget_mb=0.01,
        max_fiber_batch_size=2,
    )
    raw_profile = raw_mapper.generate_inverted_connectivity_profile(target_path)
    sampled_profile = sampled_mapper.generate_inverted_connectivity_profile(target_path)

    np.testing.assert_allclose(sampled_profile, raw_profile, rtol=1e-6, atol=1e-6)
    raw_pairwise = raw_mapper.generate_inverted_profiles_streaming(
        [target_path, pairwise_target_path],
        target_bundle_reduction="pairwise_mean",
    )
    sampled_pairwise = sampled_mapper.generate_inverted_profiles_streaming(
        [target_path, pairwise_target_path],
        target_bundle_reduction="pairwise_mean",
    )
    np.testing.assert_allclose(
        sampled_pairwise, raw_pairwise, rtol=1e-6, atol=1e-6
    )


def test_sampled_store_rejects_incompatible_sampling(tmp_path):
    reference_path = tmp_path / "reference.nii.gz"
    reference = _reference(reference_path)
    store = SampledVoxelFiberStore.write(
        tmp_path / "connectome.sampled_fibers",
        [(3, np.asarray([[1, 1, 1], [4, 1, 1]], dtype=np.float32))],
        reference,
        GeodesicFiberRegistration(sample_interval_mm=1.0),
    )

    with np.testing.assert_raises_regex(ValueError, "sampling must match"):
        InvertedFiberConnectivity(
            store.path,
            reference_path,
            sample_interval_mm=2.0,
        )


def test_sampled_inverter_never_resamples_connectome_fibers(tmp_path):
    reference_path = tmp_path / "reference.nii.gz"
    reference = _reference(reference_path)
    registration = GeodesicFiberRegistration(sample_interval_mm=1.0)
    store = SampledVoxelFiberStore.write(
        tmp_path / "connectome.sampled_fibers",
        [(3, np.asarray([[1, 1, 1], [4, 1, 1]], dtype=np.float32))],
        reference,
        registration,
    )
    mapper = InvertedFiberConnectivity(store.path, reference_path)
    sampled_target = registration.sample(
        np.asarray([[1, 1, 1], [4, 1, 1]], dtype=np.float32)
    )

    def forbidden_sample(_fiber, reverse=False):
        raise AssertionError("A pre-sampled connectome fiber was sampled again.")

    mapper.registration.sample = forbidden_sample
    profile = mapper.inverter.run_sampled_batches(
        store.iter_batches(),
        [sampled_target],
    )[0]

    np.testing.assert_allclose(profile[3], 1.0, atol=1e-6)
