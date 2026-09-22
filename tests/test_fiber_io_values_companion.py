import json
from pathlib import Path

import numpy as np

from calvin_utils.neuroimaging_utils.tract_utils import fiber_io as fiber_io_module
from calvin_utils.neuroimaging_utils.tract_utils.fiber_io import FiberIO
from calvin_utils.neuroimaging_utils.output_functions import NeuroimageFileOutporter


def _atlas(path, fiber_mask=None):
    fibers = np.empty(3, dtype=object)
    fibers[0] = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    fibers[1] = np.asarray([[0, 0, 0], [0, 1, 0]], dtype=np.float32)
    fibers[2] = np.asarray([[0, 0, 0], [0, 0, 1]], dtype=np.float32)
    contents = {"fibers": fibers}
    if fiber_mask is not None:
        contents["fiber_mask"] = np.asarray(fiber_mask)
    np.savez(path, **contents)


def _approximately_symmetric_atlas(path):
    fibers = np.empty(4, dtype=object)
    fibers[0] = np.asarray(
        [[10, 0, 0], [20, 5, 0], [30, 10, 0]], dtype=np.float32
    )
    # Same mirrored trajectory with its arbitrary vertex order reversed.
    fibers[1] = np.asarray(
        [[-30, 10, 0], [-20, 5, 0], [-10, 0, 0]], dtype=np.float32
    )
    fibers[2] = np.asarray(
        [[0, -10, 0], [0, 0, 0], [0, 10, 0]], dtype=np.float32
    )
    # No contralateral neighbor within the default five-millimetre tolerance.
    fibers[3] = np.asarray(
        [[50, 40, 0], [55, 45, 0], [60, 50, 0]], dtype=np.float32
    )
    np.savez(path, fibers=fibers)


def test_save_files_writes_compact_fib_pair(tmp_path):
    atlas_path = tmp_path / "atlas.npz"
    _atlas(atlas_path, fiber_mask=[1, 0, 1])
    output = tmp_path / "contrast_tval_0"

    FiberIO(mask_path=atlas_path).save_files(
        np.asarray([2.5, 4.0], dtype=np.float32),
        [output],
        dry_run=False,
        convert_to_nifti=False,
        convert_to_leaddbs=False,
    )

    values_path = tmp_path / "contrast_tval_0.fib.npy"
    description_path = tmp_path / "contrast_tval_0.fib.json"
    np.testing.assert_array_equal(
        np.load(values_path),
        np.asarray([2.5, 0.0, 4.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        FiberIO(mask_path=atlas_path).load_map_values(description_path),
        np.asarray([2.5, 4.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        FiberIO(mask_path=atlas_path).import_fiber_to_numpy_array(
            [description_path]
        ),
        np.asarray([[2.5], [4.0]], dtype=np.float32),
    )
    description = json.loads(description_path.read_text())
    assert description["schema"] == "calvin_utils.fiber_values"
    assert description["schema_version"] == 2
    assert description["values_file"] == "contrast_tval_0.fib.npy"
    assert description["values_path"] == str(values_path.resolve())
    assert description["values"]["relative_path"] == values_path.name
    assert description["fiber_atlas"]["path"] == str(atlas_path.resolve())
    assert description["fiber_atlas"]["fiber_count"] == 3
    assert description["values"]["dtype"] == "float32"
    assert description["values"]["ordering"] == "values[i] corresponds to fiber_atlas fiber i"
    assert description["symmetry"] == {"symmetric": False}
    assert len(description["values"]["sha256"]) == 64
    assert description["mask"]["expanded_to_full_atlas"] is True
    assert description["mask"]["kept_fiber_count"] == 2
    assert not (tmp_path / "contrast_tval_0.fib.values.npy").exists()
    assert not (tmp_path / "contrast_tval_0.fib.desc.json").exists()


def test_save_files_writes_approximately_symmetric_pair_by_default(tmp_path):
    atlas_path = tmp_path / "atlas.npz"
    _approximately_symmetric_atlas(atlas_path)

    FiberIO(mask_path=atlas_path).save_files(
        np.asarray([2.0, 6.0, 9.0, 12.0], dtype=np.float32),
        [tmp_path / "contrast_tval_0"],
        dry_run=False,
        convert_to_nifti=False,
        convert_to_leaddbs=False,
    )

    symmetric_values_path = tmp_path / "contrast_tval_0_symmetric.fib.npy"
    symmetric_description_path = tmp_path / "contrast_tval_0_symmetric.fib.json"
    np.testing.assert_array_equal(
        np.load(symmetric_values_path),
        np.asarray([4.0, 4.0, 9.0, 12.0], dtype=np.float32),
    )
    symmetry = json.loads(symmetric_description_path.read_text())["symmetry"]
    assert symmetry["symmetric"] is True
    assert symmetry["reducer"] == "mean of original and nearest mirrored-neighbor value"
    assert symmetry["mapped_fiber_count"] == 3
    assert symmetry["reciprocal_fiber_count"] == 3
    assert symmetry["reciprocal_pair_count"] == 1
    assert symmetry["self_mirror_count"] == 1
    assert symmetry["unmatched_fiber_count"] == 1
    assert symmetry["mapped_fraction"] == 0.75


def test_save_files_can_disable_symmetric_pair(tmp_path):
    atlas_path = tmp_path / "atlas.npz"
    _approximately_symmetric_atlas(atlas_path)

    FiberIO(mask_path=atlas_path).save_files(
        np.asarray([2.0, 6.0, 9.0, 12.0], dtype=np.float32),
        [tmp_path / "contrast_tval_0"],
        dry_run=False,
        convert_to_nifti=False,
        convert_to_leaddbs=False,
        symmetric=False,
    )

    assert not (tmp_path / "contrast_tval_0_symmetric.fib.npy").exists()
    assert not (tmp_path / "contrast_tval_0_symmetric.fib.json").exists()


def test_named_map_wildcards_prefer_ordinary_output(tmp_path):
    atlas_path = tmp_path / "atlas.npz"
    _approximately_symmetric_atlas(atlas_path)
    values = np.asarray([2.0, 6.0, 9.0, 12.0], dtype=np.float32)
    FiberIO(mask_path=atlas_path).save_files(
        values,
        [tmp_path / "beta_predictor_0"],
        dry_run=False,
        convert_to_nifti=False,
        convert_to_leaddbs=False,
    )

    output = NeuroimageFileOutporter(output_ftype="fiber", mask_path=atlas_path)
    (ordinary,) = output.load_named_maps(tmp_path, ["beta_predictor_[0-9]*"])
    (symmetric,) = output.load_named_maps(
        tmp_path, ["beta_predictor_[0-9]*_symmetric"]
    )

    np.testing.assert_array_equal(ordinary[:, 0], values)
    np.testing.assert_array_equal(
        symmetric[:, 0], np.asarray([4.0, 4.0, 9.0, 12.0], dtype=np.float32)
    )


def test_symmetric_nifti_uses_symmetric_values_without_mirroring_again(
    tmp_path, monkeypatch
):
    atlas_path = tmp_path / "atlas.npz"
    _approximately_symmetric_atlas(atlas_path)
    calls = []

    class RecordingDensity:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def run(self):
            return {}

    monkeypatch.setattr(fiber_io_module, "TractDensity", RecordingDensity)
    FiberIO(mask_path=atlas_path).save_files(
        np.asarray([2.0, 6.0, 9.0, 12.0], dtype=np.float32),
        [tmp_path / "contrast_tval_0"],
        dry_run=False,
        convert_to_nifti=True,
        convert_to_leaddbs=False,
    )

    assert [Path(item["fiber_path"]).name for item in calls] == [
        "contrast_tval_0.fib.json",
        "contrast_tval_0_symmetric.fib.json",
    ]
    assert [Path(item["out_path"]).name for item in calls] == [
        "contrast_tval_0.nii.gz",
        "contrast_tval_0_symmetric.nii.gz",
    ]
    assert all(item["symmetric"] is False for item in calls)


def test_save_files_rejects_geometry_output(tmp_path):
    atlas_path = tmp_path / "atlas.npz"
    _atlas(atlas_path)
    output = tmp_path / "contrast_tval_0"

    try:
        FiberIO(mask_path=atlas_path).save_files(
            np.asarray([2.5, 0.0, 4.0], dtype=np.float32),
            [output],
            dry_run=False,
            convert_to_nifti=False,
            convert_to_leaddbs=False,
            save_geometry=True,
        )
    except ValueError as error:
        assert "save_geometry is no longer supported" in str(error)
    else:
        raise AssertionError("Geometry-bearing output was still accepted")


def test_native_discovery_uses_numpy_member_of_new_pair(tmp_path):
    atlas_path = tmp_path / "atlas.npz"
    _atlas(atlas_path)
    values_path = tmp_path / "statistic.fib.npy"
    np.save(values_path, np.asarray([1.0, 2.0, 3.0], dtype=np.float32))

    FiberIO.write_values_description(values_path, atlas_path)

    assert FiberIO.is_native_map_file(values_path)
    assert not FiberIO.is_native_map_file(tmp_path / "statistic.fib.json")
    assert FiberIO.native_map_stem(values_path) == "statistic"


def test_native_discovery_prefers_new_pair_over_legacy_pair(tmp_path):
    atlas_path = tmp_path / "atlas.npz"
    _atlas(atlas_path)
    new_values_path = tmp_path / "statistic.fib.npy"
    old_values_path = tmp_path / "statistic.fib.values.npy"
    values = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    np.save(new_values_path, values)
    np.save(old_values_path, values)
    FiberIO.write_values_description(new_values_path, atlas_path)
    (tmp_path / "statistic.fib.desc.json").write_text("{}")

    assert FiberIO.is_native_map_file(new_values_path)
    assert not FiberIO.is_native_map_file(old_values_path)


def test_legacy_split_pair_is_not_written(tmp_path):
    atlas_path = tmp_path / "atlas.npz"
    _atlas(atlas_path)
    old_values_path = tmp_path / "statistic.fib.values.npy"
    np.save(old_values_path, np.asarray([1.0, 2.0, 3.0], dtype=np.float32))

    try:
        FiberIO.write_values_description(old_values_path, atlas_path)
    except ValueError as error:
        assert "writing legacy" in str(error)
    else:
        raise AssertionError("The legacy split-name pair was still writable")


def test_write_values_companion_backfills_geometry_result(tmp_path):
    geometry_path = tmp_path / "statistic.fib.npy"
    fibers = np.empty(2, dtype=object)
    fibers[0] = np.asarray([[0, 0, 0, 3.0], [1, 0, 0, 3.0]], dtype=np.float32)
    fibers[1] = np.asarray([[0, 0, 0, -2.0], [0, 1, 0, -2.0]], dtype=np.float32)
    np.save(geometry_path, fibers, allow_pickle=True)

    atlas_path = tmp_path / "atlas.npz"
    fibers_xyz = np.empty(2, dtype=object)
    fibers_xyz[0] = fibers[0][:, :3]
    fibers_xyz[1] = fibers[1][:, :3]
    np.savez(atlas_path, fibers=fibers_xyz)

    values_path = FiberIO.write_values_companion(geometry_path, atlas_path)

    assert values_path.name == "statistic_values.fib.npy"
    np.testing.assert_array_equal(
        np.load(values_path), np.asarray([3.0, -2.0], dtype=np.float32)
    )
    assert (tmp_path / "statistic_values.fib.json").is_file()
