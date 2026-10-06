import nibabel as nib
import numpy as np

from calvin_utils.neuroimaging_utils.tract_utils.inverted_connectivity import (
    InvertedFiberConnectivity,
)


def _save_fibers(path, fibers):
    stored = np.empty(len(fibers), dtype=object)
    stored[:] = [np.asarray(fiber, dtype=np.float32) for fiber in fibers]
    np.savez(path, fibers=stored)


def test_orientation_invariant_fixed_mm_fiber_cosine(tmp_path):
    reference_path = tmp_path / "mask.nii.gz"
    nib.save(
        nib.Nifti1Image(np.ones((5, 5, 2), dtype=np.uint8), np.eye(4)),
        reference_path,
    )

    connectome_path = tmp_path / "connectome.npz"
    _save_fibers(
        connectome_path,
        [
            [[1, 1, 1], [3, 1, 1]],
            [[3, 1, 1], [1, 1, 1]],
            [[1, 3, 1], [3, 3, 1]],
        ],
    )

    # Same trajectory as the first two connectome fibers, opposite ordering,
    # and a different number of original vertices.
    target_path = tmp_path / "target.npz"
    _save_fibers(
        target_path,
        [[[3, 1, 1], [2.5, 1, 1], [2, 1, 1], [1, 1, 1]]],
    )

    mapper = InvertedFiberConnectivity(
        connectome_path=connectome_path,
        reference_nifti_path=reference_path,
        sample_interval_mm=1.0,
        max_length_mm=3.0,
        fiber_batch_size=1,
        cache_dir=tmp_path / "cache",
    )
    manifest_path = mapper.prepare_geometry_cache()
    scores = mapper.match_target_fibers(target_path)

    assert manifest_path.is_file()
    assert mapper.has_geometry_cache
    np.testing.assert_allclose(scores[:2], 1.0, rtol=1e-6, atol=1e-6)
    assert scores[2] < scores[0]

    forward = np.load(mapper.forward_cache_path, mmap_mode="r")
    reverse = np.load(mapper.reverse_cache_path, mmap_mode="r")
    counts = np.load(mapper.count_cache_path, mmap_mode="r")
    assert forward.shape == reverse.shape == (3, 4, 3)
    np.testing.assert_array_equal(counts, [3, 3, 3])

    profile = mapper.generate_inverted_connectivity_profile(
        target_path, voxel_reduction="max"
    )
    assert profile.shape == (50,)
    voxel = np.ravel_multi_index((1, 1, 1), (5, 5, 2))
    np.testing.assert_allclose(profile[voxel], 1.0, rtol=1e-6, atol=1e-6)

    output_path = mapper.save_profile(profile, tmp_path / "cosine.nii.gz")
    saved = nib.load(output_path)
    np.testing.assert_allclose(saved.get_fdata().reshape(-1), profile)

    # A fresh object reuses the fixed-width memory-mapped geometry cache.
    reloaded = InvertedFiberConnectivity(
        connectome_path=connectome_path,
        reference_nifti_path=reference_path,
        sample_interval_mm=1.0,
        max_length_mm=3.0,
        cache_dir=tmp_path / "cache",
    )
    assert reloaded.has_geometry_cache
    np.testing.assert_allclose(reloaded.match_target_fibers(target_path), scores)

    saved_paths = reloaded.save_inverted_profiles_from_fibers(
        [target_path],
        tmp_path / "batch",
        similarity="padded_cosine",
        voxel_reduction="max",
    )
    assert len(saved_paths) == 1
    assert saved_paths[0].is_file()
