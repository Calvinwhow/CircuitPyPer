import nibabel as nib
import numpy as np

from calvin_utils.neuroimaging_utils.tract_utils.fiber_io import FiberIO
from calvin_utils.neuroimaging_utils.tract_utils.tract_density import TractDensity


def test_tract_density_resolves_new_fib_pair(tmp_path):
    fibers = np.empty(2, dtype=object)
    fibers[0] = np.asarray([[1, 1, 1], [2, 1, 1]], dtype=np.float32)
    fibers[1] = np.asarray([[1, 2, 1], [2, 2, 1]], dtype=np.float32)
    atlas_path = tmp_path / "atlas.npz"
    np.savez(atlas_path, fibers=fibers)

    values_path = tmp_path / "statistic.fib.npy"
    np.save(values_path, np.asarray([2.0, -1.0], dtype=np.float32))
    description_path = FiberIO.write_values_description(values_path, atlas_path)
    assert description_path.name == "statistic.fib.json"

    reference_path = tmp_path / "reference.nii.gz"
    nib.save(
        nib.Nifti1Image(np.zeros((4, 4, 4), dtype=np.float32), np.eye(4)),
        reference_path,
    )
    output_path = tmp_path / "density.nii.gz"

    result = TractDensity(
        fiber_path=description_path,
        reference_nifti_path=reference_path,
        out_path=output_path,
        fiberset="both",
        threshold=None,
        show_progress=False,
    ).run()

    assert result["n_input_fibers"] == 2
    np.testing.assert_array_equal(result["values"], [2.0, -1.0])
    assert output_path.is_file()
    assert np.isclose(np.sum(nib.load(output_path).get_fdata()), 2.0)
