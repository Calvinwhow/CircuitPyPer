import json

import numpy as np

from calvin_utils.neuroimaging_utils.tract_utils.fiber_io import FiberIO


def _atlas(path, fiber_mask=None):
    fibers = np.empty(3, dtype=object)
    fibers[0] = np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
    fibers[1] = np.asarray([[0, 0, 0], [0, 1, 0]], dtype=np.float32)
    fibers[2] = np.asarray([[0, 0, 0], [0, 0, 1]], dtype=np.float32)
    contents = {"fibers": fibers}
    if fiber_mask is not None:
        contents["fiber_mask"] = np.asarray(fiber_mask)
    np.savez(path, **contents)


def test_save_files_writes_lightweight_values_companion(tmp_path):
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

    values_path = tmp_path / "contrast_tval_0.fib.values.npy"
    description_path = tmp_path / "contrast_tval_0.fib.desc.json"
    np.testing.assert_array_equal(
        np.load(values_path),
        np.asarray([2.5, 0.0, 4.0], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        FiberIO(mask_path=atlas_path).load_map_values(description_path),
        np.asarray([2.5, 4.0], dtype=np.float32),
    )
    description = json.loads(description_path.read_text())
    assert description["schema"] == "calvin_utils.fiber_values"
    assert description["schema_version"] == 2
    assert description["values_file"] == "contrast_tval_0.fib.values.npy"
    assert description["values_path"] == str(values_path.resolve())
    assert description["values"]["relative_path"] == values_path.name
    assert description["fiber_atlas"]["path"] == str(atlas_path.resolve())
    assert description["fiber_atlas"]["fiber_count"] == 3
    assert description["values"]["dtype"] == "float32"
    assert description["values"]["ordering"] == "values[i] corresponds to fiber_atlas fiber i"
    assert len(description["values"]["sha256"]) == 64
    assert description["mask"]["expanded_to_full_atlas"] is True
    assert description["mask"]["kept_fiber_count"] == 2
    assert not (tmp_path / "contrast_tval_0.fib.npy").exists()


def test_save_files_can_opt_in_to_legacy_geometry(tmp_path):
    atlas_path = tmp_path / "atlas.npz"
    _atlas(atlas_path)
    output = tmp_path / "contrast_tval_0"

    FiberIO(mask_path=atlas_path).save_files(
        np.asarray([2.5, 0.0, 4.0], dtype=np.float32),
        [output],
        dry_run=False,
        convert_to_nifti=False,
        convert_to_leaddbs=False,
        save_geometry=True,
    )

    values_path = tmp_path / "contrast_tval_0.fib.values.npy"
    geometry_path = tmp_path / "contrast_tval_0.fib.npy"
    assert geometry_path.is_file()
    assert values_path.stat().st_size < geometry_path.stat().st_size
    description = json.loads(
        (tmp_path / "contrast_tval_0.fib.desc.json").read_text()
    )
    assert description["source_geometry_file"] == str(geometry_path.resolve())


def test_native_discovery_prefers_complete_values_pair(tmp_path):
    atlas_path = tmp_path / "atlas.npz"
    _atlas(atlas_path)
    values_path = tmp_path / "statistic.fib.values.npy"
    geometry_path = tmp_path / "statistic.fib.npy"
    np.save(values_path, np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    np.save(geometry_path, np.empty(0, dtype=object), allow_pickle=True)

    assert FiberIO.is_native_map_file(geometry_path)
    assert not FiberIO.is_native_map_file(values_path)

    FiberIO.write_values_description(values_path, atlas_path)

    assert FiberIO.is_native_map_file(values_path)
    assert not FiberIO.is_native_map_file(geometry_path)
    assert FiberIO.native_map_stem(values_path) == "statistic"


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

    assert values_path.name == "statistic.fib.values.npy"
    np.testing.assert_array_equal(
        np.load(values_path), np.asarray([3.0, -2.0], dtype=np.float32)
    )
    assert (tmp_path / "statistic.fib.desc.json").is_file()
