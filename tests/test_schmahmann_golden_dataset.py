"""Public pipeline contracts using a minimal deidentified Schmahmann cohort."""

import hashlib

import nibabel as nib
import numpy as np
import pytest

from calvin_utils.permutation_analysis_utils.map_damage_cv import (
    cross_validated_map_damage,
    load_native_patient_vectors,
)


MODALITY_CONTRACTS = (
    pytest.param(
        "Nifti_File_Path",
        "nii",
        (5, 902629),
        "215c9a7e4ad2b28b996f2f2dd328b5b17e1f8898a913549b7e735e2579572d9b",
        [-0.075300274352, -0.086281942568, 0.039462802576,
         -0.029833294836, 0.019685906579],
        id="nifti",
    ),
    pytest.param(
        "fiber_path_guerrera",
        "fiber",
        (5, 384466),
        "37e05c0f3bab2ae3d4c0d502a13b5dca4d9fecb9c415536fdac1cc5580e49fac",
        [-0.042766241173, 0.002465587212, 0.107128759284,
         0.032925948038, 0.170985972968],
        id="fiber",
    ),
)


def test_fixture_integrity_deidentification_and_outcomes(schmahmann_cohort):
    _, cohort = schmahmann_cohort
    assert cohort["subject_id"].tolist() == [
        f"golden-{index:03d}" for index in range(1, 6)
    ]
    np.testing.assert_array_equal(
        cohort["TotalBarsScore"].to_numpy(), [0.0, 6.0, 9.0, 12.0, 24.5]
    )

    for path_column, hash_column in (
        ("Nifti_File_Path", "nifti_sha256"),
        ("fiber_path_guerrera", "fiber_sha256"),
    ):
        for path, expected_hash in zip(cohort[path_column], cohort[hash_column]):
            assert path.is_file(), path
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash

    for path in cohort["Nifti_File_Path"]:
        image = nib.load(path)
        assert image.shape == (91, 109, 91)
        assert image.get_data_dtype() == np.dtype("float32")
        assert bytes(image.header["descrip"]).strip(b"\0") == b""
        assert bytes(image.header["aux_file"]).strip(b"\0") == b""


@pytest.mark.parametrize(
    "path_column,output_type,expected_shape,expected_hash,expected_scores",
    MODALITY_CONTRACTS,
)
def test_native_loader_and_damage_cv_public_contract(
    schmahmann_cohort,
    path_column,
    output_type,
    expected_shape,
    expected_hash,
    expected_scores,
):
    _, cohort = schmahmann_cohort
    values = load_native_patient_vectors(
        cohort[path_column], mask_path=None, output_ftype=output_type
    )
    assert values.shape == expected_shape
    assert values.dtype == np.dtype("float32")
    assert np.isfinite(values).all()

    # Exercise the production CV contract on stable features sampled across
    # the complete native space. This remains fast while detecting reordering.
    feature_indices = np.linspace(0, values.shape[1] - 1, 4096, dtype=int)
    representative_values = values[:, feature_indices]
    assert hashlib.sha256(representative_values.tobytes()).hexdigest() == expected_hash

    scores, folds = cross_validated_map_damage(
        representative_values,
        cohort["TotalBarsScore"].to_numpy(dtype=float),
        cv="loocv",
    )
    np.testing.assert_allclose(scores, expected_scores, atol=1e-7)
    np.testing.assert_array_equal(folds, np.arange(1, 6))

