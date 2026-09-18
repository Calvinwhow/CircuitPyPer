import nibabel as nib
import numpy as np
from scipy.io import savemat

from calvin_utils.neuroimaging_utils.tract_utils.fiber_converter import (
    FiberFormatConverter,
)
from calvin_utils.neuroimaging_utils.tract_utils.fiber_io import FiberIO


FIBERS = [
    np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32),
    np.asarray([[7, 8, 9], [10, 11, 12], [13, 14, 15]], dtype=np.float32),
]
VALUES = np.asarray([2.5, -1.25], dtype=np.float32)


def _assert_conversion(result):
    loaded = nib.streamlines.load(result["tck_path"]).tractogram.streamlines
    assert len(loaded) == 2
    np.testing.assert_allclose(loaded[0], FIBERS[0])
    np.testing.assert_allclose(loaded[1], FIBERS[1])
    np.testing.assert_array_equal(result["idx"], [2, 3])
    np.testing.assert_allclose(result["vals"], VALUES)
    assert result["n_fibers"] == 2


def test_convert_ftr_mat_to_tck(tmp_path):
    mat_path = tmp_path / "fibers_ftr.mat"
    matrix = np.vstack([
        np.column_stack([FIBERS[0], np.ones(len(FIBERS[0]))]),
        np.column_stack([FIBERS[1], np.full(len(FIBERS[1]), 2)]),
    ])
    savemat(mat_path, {"fibers": matrix, "idx": [2, 3], "vals": VALUES})

    result = FiberFormatConverter.convert_leaddbs_mat_to_tck(
        mat_path, tmp_path / "fibers.tck"
    )

    _assert_conversion(result)


def test_convert_discfibers_mat_to_tck(tmp_path):
    mat_path = tmp_path / "fibers_discfibers.mat"
    fibcell = np.empty((2, 1), dtype=object)
    fibcell[:, 0] = FIBERS
    savemat(mat_path, {"fibcell": fibcell, "vals": VALUES[:, None]})

    result = FiberFormatConverter.convert_leaddbs_mat_to_tck(
        mat_path, tmp_path / "fibers.tck"
    )

    _assert_conversion(result)


def test_convert_geometry_fib_npy_to_tck(tmp_path):
    path = tmp_path / "result.fib.npy"
    fibers = np.empty(3, dtype=object)
    fibers[0] = np.column_stack([FIBERS[0], np.full(2, 10.0)])
    fibers[1] = np.column_stack([FIBERS[1], np.full(3, -8.0)])
    fibers[2] = np.column_stack([FIBERS[0], np.full(2, 2.0)])
    np.save(path, fibers, allow_pickle=True)

    result = FiberFormatConverter.convert_fib_npy_to_tck(
        path, tmp_path / "selected.tck", top_percent=50
    )

    loaded = nib.streamlines.load(result["tck_path"]).tractogram.streamlines
    assert len(loaded) == 2
    np.testing.assert_allclose(result["vals"], [10.0, -8.0])
    assert result["n_input_fibers"] == 3


def test_convert_vector_fib_npy_with_atlas(tmp_path):
    path = tmp_path / "result.fib.npy"
    atlas = tmp_path / "atlas.npz"
    np.save(path, np.asarray([2.5, -1.25], dtype=np.float32))
    np.savez(atlas, fibers=np.asarray(FIBERS, dtype=object))

    result = FiberFormatConverter.convert_fib_npy_to_tck(
        path, tmp_path / "selected.tck", fiber_atlas_path=atlas
    )

    _assert_conversion(result)
    assert result["n_input_fibers"] == 2


def test_values_npy_requires_descriptor_even_with_explicit_atlas(tmp_path):
    values_path = tmp_path / "result.fib.values.npy"
    atlas_path = tmp_path / "atlas.npz"
    np.save(values_path, VALUES)
    np.savez(atlas_path, fibers=np.asarray(FIBERS, dtype=object))

    try:
        FiberFormatConverter.load_fib_npy(
            values_path, fiber_atlas_path=atlas_path
        )
    except FileNotFoundError as error:
        assert "result.fib.desc.json" in str(error)
    else:
        raise AssertionError("A .fib.values.npy file was accepted without its descriptor")


def test_values_descriptor_accepts_relocated_atlas_override(tmp_path):
    values_path = tmp_path / "result.fib.values.npy"
    original_atlas = tmp_path / "atlas.npz"
    relocated_atlas = tmp_path / "relocated" / "atlas.npz"
    np.save(values_path, VALUES)
    np.savez(original_atlas, fibers=np.asarray(FIBERS, dtype=object))
    FiberIO.write_values_description(values_path, original_atlas)
    relocated_atlas.parent.mkdir()
    original_atlas.rename(relocated_atlas)

    streamlines, values, count = FiberFormatConverter.load_fib_npy(
        values_path, fiber_atlas_path=relocated_atlas
    )

    assert count == 2
    np.testing.assert_allclose(streamlines[0], FIBERS[0])
    np.testing.assert_array_equal(values, VALUES)


def test_convert_values_pair_directly_to_trk(tmp_path):
    values_path = tmp_path / "result.fib.values.npy"
    atlas_path = tmp_path / "atlas.npz"
    reference_path = tmp_path / "reference.tck"
    output_path = tmp_path / "result.trk"
    np.save(values_path, VALUES)
    np.savez(atlas_path, fibers=np.asarray(FIBERS, dtype=object))
    description_path = FiberIO.write_values_description(values_path, atlas_path)
    nib.streamlines.save(
        nib.streamlines.Tractogram(FIBERS, affine_to_rasmm=np.eye(4)),
        reference_path,
    )

    FiberFormatConverter(reference_path).convert_fiber_file_to_trk(
        description_path, output_path
    )

    assert (
        FiberFormatConverter.values_from_description(description_path)
        == values_path.resolve()
    )
    loaded = nib.streamlines.load(output_path).tractogram
    assert len(loaded.streamlines) == 2
    np.testing.assert_allclose(loaded.streamlines[0], FIBERS[0])
    magnitudes = np.asarray(loaded.data_per_streamline["magnitude"]).reshape(-1)
    np.testing.assert_allclose(magnitudes, VALUES)
