"""Stable public-surface checks for the utility modules covered end to end."""

import importlib

import pytest


PUBLIC_MODULE_CONTRACTS = (
    ("calvin_utils.file_utils.import_functions", ("GiiNiiFileImport",)),
    ("calvin_utils.file_utils.csv_prep", ("CSVComposer",)),
    ("calvin_utils.neuroimaging_utils.nifti_utils.volume_io", ("NiftiIO",)),
    ("calvin_utils.neuroimaging_utils.tract_utils.fiber_io", ("FiberIO",)),
    (
        "calvin_utils.neuroimaging_utils.tract_utils.fiber_converter",
        ("FiberFormatConverter",),
    ),
    (
        "calvin_utils.permutation_analysis_utils.map_damage_cv",
        ("load_native_patient_vectors", "cross_validated_map_damage"),
    ),
    (
        "calvin_utils.permutation_analysis_utils.statsmodels_palm",
        ("SpreadsheetDataLoader", "CalvinStatsmodelsPalm"),
    ),
    (
        "calvin_utils.permutation_analysis_utils.voxelwise_regression_prep",
        ("RegressionPrep",),
    ),
    (
        "calvin_utils.permutation_analysis_utils.voxelwise_regression",
        ("VoxelwiseRegression",),
    ),
    (
        "calvin_utils.neuroimaging_utils.ccm_utils.symptom_specificity",
        ("SpecificityAnalyzer", "NetworkSpecificityAnalysis"),
    ),
    (
        "calvin_utils.neuroimaging_utils.ccm_utils.sensitivity_map",
        ("SensitivityMap",),
    ),
)


@pytest.mark.parametrize(
    "module_name,public_names",
    PUBLIC_MODULE_CONTRACTS,
    ids=[name.rsplit(".", 1)[-1] for name, _ in PUBLIC_MODULE_CONTRACTS],
)
def test_public_module_surface(module_name, public_names):
    module = importlib.import_module(module_name)
    for public_name in public_names:
        public_object = getattr(module, public_name, None)
        assert callable(public_object), f"{module_name}.{public_name} is not callable"
