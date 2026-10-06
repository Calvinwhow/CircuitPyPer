import numpy as np
import pytest

from calvin_utils.neuroimaging_utils.ccm_utils.optimization.history import (
    OptimizationHistory,
)
from calvin_utils.neuroimaging_utils.ccm_utils.optimization.visualization import (
    export_optimization_map_stack,
    optimization_stack_path,
    render_gif,
)


def _history(*, output_type, mask_path, maps=None):
    return OptimizationHistory(
        weights=np.asarray([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
        losses=np.asarray([0.1, 0.2, 0.3]),
        maps=np.asarray(maps if maps is not None else [[1.0, 2.0], [3.0, 4.0]]),
        map_names=("first", "second"),
        output_type=output_type,
        mask_path=str(mask_path),
    )


def test_export_optimization_map_stack_writes_all_nifti_snapshots(tmp_path):
    nib = pytest.importorskip("nibabel")
    mask_data = np.asarray([[[1.0], [0.0]], [[1.0], [0.0]]], dtype=np.float32)
    affine = np.asarray(
        [[-2.0, 0.0, 0.0, 10.0], [0.0, 2.0, 0.0, -4.0],
         [0.0, 0.0, 2.0, 6.0], [0.0, 0.0, 0.0, 1.0]]
    )
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(nib.Nifti1Image(mask_data, affine), mask_path)
    history = _history(output_type="nii", mask_path=mask_path)

    output_path = export_optimization_map_stack(
        history, tmp_path / "optimization.nii.gz"
    )

    image = nib.load(output_path)
    data = np.asarray(image.dataobj)
    assert data.shape == (2, 2, 1, 3)
    np.testing.assert_allclose(image.affine, affine)
    np.testing.assert_allclose(data[:, :, :, 0].reshape(-1), [1.0, 0.0, 2.0, 0.0])
    np.testing.assert_allclose(data[:, :, :, 1].reshape(-1), [2.0, 0.0, 3.0, 0.0])
    np.testing.assert_allclose(data[:, :, :, 2].reshape(-1), [3.0, 0.0, 4.0, 0.0])


def test_export_optimization_map_stack_adds_iteration_axis_to_fibers(tmp_path):
    fibers = np.empty(3, dtype=object)
    fibers[0] = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    fibers[1] = np.asarray([[0, 1, 0], [1, 1, 0]], dtype=np.float32)
    fibers[2] = np.asarray([[0, 2, 0], [1, 2, 0]], dtype=np.float32)
    atlas_path = tmp_path / "atlas.npz"
    np.savez(atlas_path, fibers=fibers, fiber_mask=np.asarray([1, 0, 1]))
    history = _history(output_type="fiber", mask_path=atlas_path)

    output_path = export_optimization_map_stack(
        history, tmp_path / "optimization.fib.npy"
    )

    data = np.load(output_path)
    assert data.shape == (3, 3)
    assert data.dtype == np.float32
    np.testing.assert_allclose(
        data,
        [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [2.0, 3.0, 4.0]],
    )


def test_optimization_stack_path_matches_native_output_type(tmp_path):
    gif_path = tmp_path / "optimization.gif"
    assert optimization_stack_path(gif_path, "nifti") == tmp_path / "optimization.nii.gz"
    assert optimization_stack_path(gif_path, "fiber") == tmp_path / "optimization.fib.npy"
    assert optimization_stack_path(gif_path, "surface") is None


def test_render_gif_writes_the_same_sampled_map_snapshots(tmp_path):
    nib = pytest.importorskip("nibabel")
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(
        nib.Nifti1Image(np.ones((1, 1, 2), dtype=np.float32), np.eye(4)),
        mask_path,
    )
    history = _history(output_type="nii", mask_path=mask_path)
    gif_path = tmp_path / "optimization.gif"

    render_gif(history, gif_path, view="metrics", max_frames=2, fps=2)

    assert gif_path.is_file()
    stack = np.asarray(nib.load(tmp_path / "optimization.nii.gz").dataobj)
    assert stack.shape == (1, 1, 2, 2)
    np.testing.assert_allclose(stack[0, 0, :, 0], [1.0, 2.0])
    np.testing.assert_allclose(stack[0, 0, :, 1], [3.0, 4.0])
