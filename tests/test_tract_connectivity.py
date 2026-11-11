import nibabel as nib
import numpy as np

from calvin_utils.neuroimaging_utils.tract_utils.compute_connectivity import FiberConnectivity
from calvin_utils.neuroimaging_utils.tract_utils.fiber_intersection import FiberVoxelIndexer


def test_fiber_connectivity_processes_and_saves_each_frame(tmp_path):
    reference_path = tmp_path / "reference.nii.gz"
    seed_path = tmp_path / "seed_timeseries.nii.gz"
    atlas_path = tmp_path / "atlas.npz"
    affine = np.eye(4)
    frames = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float32)
    nib.save(nib.Nifti1Image(np.ones((2, 2, 1), dtype=np.float32), affine), reference_path)
    nib.save(nib.Nifti1Image(frames.reshape(2, 2, 1, 2), affine), seed_path)

    fibers = np.empty(2, dtype=object)
    fibers[0] = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    fibers[1] = np.asarray([[0, 1, 0], [1, 1, 0]], dtype=np.float32)
    np.savez(atlas_path, fibers=fibers)

    indexer = FiberVoxelIndexer(reference_path)
    indexer.fiber_voxel_indices = [np.asarray([0, 1]), np.asarray([2, 3])]
    mapper = FiberConnectivity(indexer, mode="sum", fiber_atlas_path=atlas_path)
    profile = mapper.generate_connectivity_profile(seed_path, binarize=False, threshold=None)
    expected = np.asarray([[4, 6], [12, 14]], dtype=np.float32)
    np.testing.assert_array_equal(profile.reshape(2, 2), expected)

    output_path = tmp_path / "connectivity.fib.npy"
    descriptor_path = mapper.save_profile(profile, output_path)
    np.testing.assert_array_equal(np.load(output_path), expected)
    assert descriptor_path == tmp_path / "connectivity.fib.json"
    assert descriptor_path.is_file()

    returned = mapper.save_profiles_from_niftis(
        [seed_path], out_dir=tmp_path / "profiles", binarize=False, threshold=None
    )
    assert returned[0].name.endswith(".fib.json")
    assert returned[0].is_file()
    assert returned[0].with_name(returned[0].name[:-5] + ".npy").is_file()
