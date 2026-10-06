"""Behavioral coverage for the complete sensitivity-map workflow."""

import nibabel as nib
import numpy as np
import pandas as pd

from calvin_utils.neuroimaging_utils.ccm_utils.sensitivity_map import SensitivityMap


def test_run_generates_and_saves_continuous_voxelwise_average(tmp_path):
    mask_path = tmp_path / "mask.nii.gz"
    nib.save(
        nib.Nifti1Image(np.ones((2, 2, 1), dtype=np.uint8), np.eye(4)),
        mask_path,
    )
    observations_by_voxels = pd.DataFrame(
        [[0.0, 2.0, -4.0, 0.0], [2.0, 4.0, -8.0, 0.0]]
    )

    analysis = SensitivityMap(
        df=observations_by_voxels,
        mask_path=str(mask_path),
        out_dir=str(tmp_path / "results"),
        map_type="custom",
        manual_threshold=2,
        verbose=False,
        save_absval=False,
    )
    overlap, stepwise, average = analysis.run(return_average=True)

    np.testing.assert_array_equal(overlap["df"], [1.0, 2.0, -2.0, 0.0])
    np.testing.assert_array_equal(stepwise["df"], [50.0, 100.0, -100.0, 0.0])
    np.testing.assert_array_equal(average["df"], [1.0, 3.0, -6.0, 0.0])

    average_path = (
        tmp_path
        / "results"
        / "custom_overlap_maps"
        / "threshold_2_df_average.nii.gz"
    )
    assert average_path.is_file()
    np.testing.assert_array_equal(
        nib.load(average_path).get_fdata().reshape(-1),
        average["df"],
    )


def test_mask_vectorized_average_is_unmasked_before_saving(tmp_path):
    mask = np.array([1, 0, 1, 0], dtype=np.uint8).reshape(2, 2, 1)
    mask_path = tmp_path / "sparse_mask.nii.gz"
    nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)
    data = pd.DataFrame([[1.0, 3.0], [3.0, 5.0]])

    analysis = SensitivityMap(
        df=data,
        mask_path=str(mask_path),
        out_dir=str(tmp_path / "results"),
        map_type="custom",
        manual_threshold=1,
        verbose=False,
    )
    _, _, average = analysis.run(return_average=True)

    saved = nib.load(
        tmp_path
        / "results"
        / "custom_overlap_maps"
        / "threshold_1_df_average.nii.gz"
    ).get_fdata()
    np.testing.assert_array_equal(average["df"], [2.0, 4.0])
    np.testing.assert_array_equal(saved.reshape(-1), [2.0, 0.0, 4.0, 0.0])
