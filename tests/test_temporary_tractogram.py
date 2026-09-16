from pathlib import Path

import numpy as np
from scipy.io import savemat

from calvin_utils.neuroimaging_utils.tract_utils.temporary_tractogram import (
    TemporaryTractogram,
)


def test_temporary_tractogram_lifetime(tmp_path):
    mat_path = tmp_path / "fibers.mat"
    fibers = np.asarray([
        [1, 2, 3, 1],
        [4, 5, 6, 1],
        [7, 8, 9, 2],
        [10, 11, 12, 2],
        [13, 14, 15, 2],
    ], dtype=np.float32)
    savemat(mat_path, {"fibers": fibers, "idx": [2, 3], "vals": [2.5, -1.25]})

    with TemporaryTractogram(mat_path, temp_root=tmp_path) as tractogram:
        tck_path = tractogram.tck_path
        temp_dir = tck_path.parent
        assert tck_path.is_file()
        assert tractogram.n_fibers == 2
        np.testing.assert_array_equal(tractogram.idx, [2, 3])
        np.testing.assert_allclose(
            tractogram.point_values(), [2.5, 2.5, -1.25, -1.25, -1.25]
        )

    assert not temp_dir.exists()


def test_temporary_tractogram_cleans_up_after_error(tmp_path):
    mat_path = tmp_path / "fibers.mat"
    savemat(mat_path, {"fibers": np.ones((2, 3), dtype=np.float32)})

    try:
        with TemporaryTractogram(mat_path, temp_root=tmp_path) as tractogram:
            temp_dir = Path(tractogram.tck_path).parent
            raise RuntimeError("render failed")
    except RuntimeError:
        pass

    assert not temp_dir.exists()


def test_temporary_vector_fib_npy_uses_atlas_and_filters(tmp_path):
    values_path = tmp_path / "statistics.fib.npy"
    atlas_path = tmp_path / "atlas.npz"
    fibers = np.empty(3, dtype=object)
    fibers[0] = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    fibers[1] = np.asarray([[0, 0, 0], [0, 1, 0]], dtype=np.float32)
    fibers[2] = np.asarray([[0, 0, 0], [0, 0, 1]], dtype=np.float32)
    np.save(values_path, np.asarray([3.0, -2.0, 0.5], dtype=np.float32))
    np.savez(atlas_path, fibers=fibers)

    with TemporaryTractogram(
        values_path,
        temp_root=tmp_path,
        fiber_atlas_path=atlas_path,
        sign="positive",
        min_abs_value=1.0,
    ) as tractogram:
        temp_dir = tractogram.tck_path.parent
        assert tractogram.n_input_fibers == 3
        assert tractogram.n_fibers == 1
        np.testing.assert_allclose(tractogram.vals, [3.0])

    assert not temp_dir.exists()
