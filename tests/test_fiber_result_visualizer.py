import numpy as np

from calvin_utils.neuroimaging_utils.tract_utils.fiber_result_visualizer import (
    FiberResultVisualizer,
)
from calvin_utils.neuroimaging_utils.tract_utils.fiber_io import FiberIO


def _visualizer(values, *, sign, top_percent=None):
    visualizer = FiberResultVisualizer(
        values=np.asarray(values, dtype=np.float32),
        out_dir="unused",
        sign=sign,
        top_percent=top_percent,
        save_discfibers_mat=False,
        save_ftr_mat=False,
    )
    visualizer.fibers = [np.zeros((2, 3), dtype=np.float32) for _ in values]
    return visualizer


def test_negative_selection_exports_positive_magnitudes():
    visualizer = _visualizer([-3.0, -1.0, 0.0, 2.0], sign="negative")

    visualizer.apply_selection()

    np.testing.assert_array_equal(visualizer.keep_mask, [True, True, False, False])
    np.testing.assert_array_equal(visualizer.selected_values, [3.0, 1.0])


def test_negative_top_percent_ranks_original_values_by_absolute_magnitude():
    visualizer = _visualizer(
        [-10.0, -4.0, -3.0, -1.0, 100.0],
        sign="negative",
        top_percent=50,
    )

    visualizer.apply_selection()

    np.testing.assert_array_equal(visualizer.keep_mask, [True, True, False, False, False])
    np.testing.assert_array_equal(visualizer.selected_values, [10.0, 4.0])


def test_positive_selection_preserves_values():
    visualizer = _visualizer([-3.0, 1.0, 2.0], sign="positive")

    visualizer.apply_selection()

    np.testing.assert_array_equal(visualizer.selected_values, [1.0, 2.0])


def test_both_selection_preserves_signed_bidirectional_values():
    visualizer = _visualizer([-3.0, 0.0, 2.0, np.nan], sign="Both")

    visualizer.apply_selection()

    assert visualizer.sign == "both"
    np.testing.assert_array_equal(visualizer.keep_mask, [True, False, True, False])
    np.testing.assert_array_equal(visualizer.selected_values, [-3.0, 2.0])


def test_sign_aliases_are_normalized():
    assert _visualizer([1.0], sign=" POS ").sign == "positive"
    assert _visualizer([-1.0], sign="Neg").sign == "negative"


def test_values_descriptor_supplies_atlas(tmp_path):
    atlas_path = tmp_path / "atlas.npz"
    geometry_path = tmp_path / "result.fib.npy"
    atlas_fibers = np.empty(2, dtype=object)
    atlas_fibers[0] = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    atlas_fibers[1] = np.asarray([[0, 0, 0], [0, 1, 0]], dtype=np.float32)
    result_fibers = np.empty(2, dtype=object)
    result_fibers[0] = np.column_stack([atlas_fibers[0], [2.0, 2.0]])
    result_fibers[1] = np.column_stack([atlas_fibers[1], [-1.0, -1.0]])
    np.savez(atlas_path, fibers=atlas_fibers)
    np.save(geometry_path, result_fibers, allow_pickle=True)
    values_path = FiberIO.write_values_companion(geometry_path, atlas_path)
    description_path = FiberIO.values_description_path(values_path)

    visualizer = FiberResultVisualizer(
        values_path=description_path,
        out_dir=tmp_path / "rendered",
        save_discfibers_mat=False,
        save_ftr_mat=False,
    )
    visualizer.run()

    assert visualizer.fiber_atlas_path == atlas_path.resolve()
    np.testing.assert_array_equal(visualizer.values, [2.0, -1.0])
