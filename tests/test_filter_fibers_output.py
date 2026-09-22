import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from calvin_utils.neuroimaging_utils.tract_utils.filter_fibers import filter_fibers


def test_filter_fibers_writes_compact_fib_pair(tmp_path):
    atlas_fibers = np.empty(2, dtype=object)
    atlas_fibers[0] = np.asarray([[1, 1, 1], [2, 1, 1]], dtype=np.float32)
    atlas_fibers[1] = np.asarray([[1, 3, 1], [2, 3, 1]], dtype=np.float32)
    atlas_path = tmp_path / "atlas.npz"
    np.savez(atlas_path, fibers=atlas_fibers)

    reference_path = tmp_path / "reference.nii.gz"
    nib.save(
        nib.Nifti1Image(np.zeros((5, 5, 5), dtype=np.float32), np.eye(4)),
        reference_path,
    )

    patient_dir = tmp_path / "patients"
    patient_dir.mkdir()
    patient_path = patient_dir / "patient_01.nii.gz"
    patient_data = np.zeros((5, 5, 5), dtype=np.float32)
    patient_data[1:3, 1, 1] = 1
    nib.save(nib.Nifti1Image(patient_data, np.eye(4)), patient_path)

    result_df = filter_fibers(
        pd.DataFrame({"nifti": [str(patient_path)]}),
        nifti_col="nifti",
        fiber_atlas_path=atlas_path,
        reference_nifti_path=reference_path,
        out_dir=tmp_path / "output",
        save_matrix=False,
        show_progress=False,
    ).run()

    values_path = tmp_path / "fibers" / "patient_01_atlas.fib.npy"
    descriptor_path = tmp_path / "fibers" / "patient_01_atlas.fib.json"
    np.testing.assert_array_equal(np.load(values_path), [1.0, 0.0])
    descriptor = json.loads(descriptor_path.read_text())
    assert descriptor["values_file"] == values_path.name
    assert descriptor["fiber_atlas"]["path"] == str(atlas_path.resolve())
    assert Path(result_df.loc[0, "fiber_path"]).resolve() == values_path.resolve()
    assert Path(result_df.loc[0, "fiber_json_path"]).resolve() == descriptor_path.resolve()
    assert not (tmp_path / "fibers" / "patient_01_atlas.fib.values.npy").exists()
