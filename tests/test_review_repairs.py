"""Regression coverage for package-health defects found in the utils review."""

import importlib
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from pathlib import Path

from setuptools import find_packages

from calvin_utils.ml_utils.train_test_splitter import TrainTestSplitter
from calvin_utils.ml_utils.umap_regression import UmapRegression
from calvin_utils.neuroimaging_utils.ccm_utils.npy_regression import (
    RegressionNPYAnalysis,
)
from calvin_utils.neuroimaging_utils.ccm_utils.correlations import run_pearson
from calvin_utils.neuroimaging_utils.nifti_utils.compare_nifti_centroids import (
    NiftiCentroidComparisonStats,
)
from calvin_utils.neuroimaging_utils.nifti_utils import matrix_utilities
from calvin_utils.neuroimaging_utils.nifti_utils.volume_io import NiftiIO
from calvin_utils.neuroimaging_utils.nifti_utils.model_vta import ModelVTAWrapper
from calvin_utils.permutation_analysis_utils.palm_utils import (
    CalvinPalm,
    CalvinPalmSubmitter,
)
from calvin_utils.permutation_analysis_utils.voxelwise_regression import (
    VoxelwiseRegression,
)
from calvin_utils.resource_paths import (
    default_nifti_mask_path,
    lazy_resource_path,
    resource_path,
)
from calvin_utils.statistical_utils.voxelwise_statistical_testing import (
    voxelwise_interaction_f_stat,
)
from calvin_utils.statistical_utils.statistical_measurements import (
    FactorialPlot,
    Linear_Reg_Diagnostic,
    calculate_vif,
)


def test_all_python_subpackages_are_discovered():
    packages = set(find_packages())
    assert "calvin_utils.neuroimaging_utils" in packages
    assert "calvin_utils.neuroimaging_utils.ccm_utils" in packages
    assert "calvin_utils.neuroimaging_utils.dbs_utils" in packages
    assert "calvin_utils.neuroimaging_utils.io" in packages
    assert "calvin_utils.neuroimaging_utils.vbm_utils" in packages
    assert "calvin_utils.plotting_utils.rendering" in packages
    assert "calvin_utils.plotting_utils.statistical" in packages
    assert "calvin_utils.experimental" in packages


def test_relocated_modules_preserve_approved_legacy_imports():
    from calvin_utils.file_utils.import_functions import GiiNiiFileImport as old_importer
    from calvin_utils.neuroimaging_utils.io.importers import GiiNiiFileImport as new_importer
    from calvin_utils.neuroimaging_utils.output_functions import NeuroimageFileOutporter as old_exporter
    from calvin_utils.neuroimaging_utils.io.exporters import NeuroimageFileOutporter as new_exporter
    from calvin_utils.statistical_utils.scatterplot import simple_scatter as old_scatter
    from calvin_utils.plotting_utils.statistical.scatterplot import simple_scatter as new_scatter
    from calvin_utils.vbm_utils.postprocessing import PostProcessing as old_postprocessing
    from calvin_utils.neuroimaging_utils.vbm_utils.postprocessing import PostProcessing as new_postprocessing
    from calvin_utils.neuroimaging_utils.nifti_utils.dice_vta import DiceVTA as old_dice
    from calvin_utils.neuroimaging_utils.dbs_utils.vta_overlap import DiceVTA as new_dice
    from calvin_utils.neuroimaging_utils.dbs_utils.compare_vta_target_overlap import (
        VTADiceCorrelation,
    )
    from calvin_utils.neuroimaging_utils.ccm_utils.voxelwise_palm import RegressionNPYAnalysis as old_npy
    from calvin_utils.neuroimaging_utils.ccm_utils.npy_regression import RegressionNPYAnalysis as new_npy

    assert old_importer is new_importer
    assert old_exporter is new_exporter
    assert old_scatter is new_scatter
    assert old_postprocessing is new_postprocessing
    assert old_dice is new_dice
    assert VTADiceCorrelation is new_dice
    assert old_npy is new_npy
    assert callable(ModelVTAWrapper)


def test_permutation_classes_document_empirical_null_contract():
    from calvin_utils.permutation_analysis_utils.correlation_fwe import (
        CalvinFWEMap as CorrelationFWE,
    )
    from calvin_utils.permutation_analysis_utils.lin_reg_fwe import (
        CalvinFWEMap as RegressionFWE,
    )

    for cls in (
        CorrelationFWE,
        RegressionFWE,
        RegressionNPYAnalysis,
        VoxelwiseRegression,
        UmapRegression,
    ):
        docstring = cls.__doc__.lower()
        assert "empirical" in docstring
        assert "theoretical null" in docstring
        assert "not" in docstring


def test_npy_permutation_summary_uses_every_voxel_in_each_contrast():
    analysis = RegressionNPYAnalysis.__new__(RegressionNPYAnalysis)
    analysis.n_permutations = 2
    analysis.n_subjects = 3
    analysis.contrast_matrix = np.ones((2, 1))
    analysis.X = np.arange(3, dtype=float).reshape(-1, 1)
    analysis.Y = np.zeros((3, 3), dtype=float)
    contrast_maps = np.array([[1.0, -4.0, 2.0], [-3.0, 2.0, 6.0]])
    seen_shapes = []

    analysis.run_regression = lambda _x, _y: (None, None, None, None)
    analysis.apply_contrast = lambda *_args: (None, contrast_maps)

    def get_max_stat(values):
        values = np.asarray(values)
        seen_shapes.append(values.shape)
        return np.max(np.abs(values))

    analysis.get_max_stat = get_max_stat

    result = analysis.run_permutation()

    np.testing.assert_array_equal(result, [[4.0, 4.0], [6.0, 6.0]])
    assert seen_shapes == [(3,), (3,), (3,), (3,)]


def test_source_resources_resolve():
    expected = (
        Path(__file__).resolve().parents[1]
        / "resources"
        / "MNI152_T1_2mm_brain_mask.nii"
    ).resolve()
    assert default_nifti_mask_path().resolve() == expected
    assert resource_path("colour_luts").is_dir()


def test_resource_override_is_authoritative_and_lazy(tmp_path, monkeypatch):
    override = tmp_path / "resources"
    override.mkdir()
    sentinel = override / "sentinel.txt"
    sentinel.write_text("ok", encoding="utf-8")
    monkeypatch.setenv("CALVIN_UTILS_RESOURCES", os.fspath(override))

    assert resource_path("sentinel.txt") == sentinel
    assert os.fspath(lazy_resource_path("sentinel.txt")) == os.fspath(sentinel)


def test_nifti_defaults_use_the_definitive_mask(monkeypatch):
    expected = default_nifti_mask_path().resolve()
    loaded_paths = []

    class FakeMask:
        @staticmethod
        def get_fdata():
            return np.array([0.0, 1.0, 1.0, 0.0]).reshape(2, 2, 1)

    def fake_load(path):
        loaded_paths.append(Path(path).resolve())
        return FakeMask()

    monkeypatch.setattr(matrix_utilities.nib, "load", fake_load)
    unmasked = matrix_utilities.unmask_matrix_v2(pd.DataFrame([5.0, 6.0]))

    assert loaded_paths == [expected]
    np.testing.assert_array_equal(
        unmasked.iloc[:, 0].to_numpy(), [np.nan, 5.0, 6.0, np.nan]
    )
    assert Path(NiftiIO().resolved_mask_path).resolve() == expected


def test_default_nifti_mask_is_not_forced_onto_fiber_analysis(monkeypatch):
    from calvin_utils.neuroimaging_utils.io import importers

    received_masks = []

    class FakeFiberIO:
        output_ftype = "fiber"

        def __init__(self, mask_path=None):
            received_masks.append(mask_path)

        @staticmethod
        def import_fiber_to_numpy_array(_file_paths):
            return np.empty((0, 0))

    monkeypatch.setattr(importers, "FiberIO", FakeFiberIO)

    default_importer = importers.GiiNiiFileImport(import_path=None)
    default_importer.import_fiber_to_numpy_array([])
    analysis_importer = importers.GiiNiiFileImport(
        import_path=None, mask_path="analysis-specific-mask.nii.gz"
    )
    analysis_importer.import_fiber_to_numpy_array([])

    assert received_masks == [None, "analysis-specific-mask.nii.gz"]


def test_palm_default_mask_is_the_definitive_resource(tmp_path):
    palm = CalvinPalmSubmitter()
    assert Path(palm.prepare_mask("", tmp_path)).resolve() == (
        default_nifti_mask_path().resolve()
    )


def test_factorial_interaction_plot_has_no_debug_output(capsys, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    plotter = FactorialPlot.__new__(FactorialPlot)
    plotter.emm_df = pd.DataFrame(
        {
            "condition": ["a", "a", "b", "b"],
            "group": ["x", "y", "x", "y"],
            "predictions": [1.0, 2.0, 3.0, 4.0],
        }
    )

    plotter.create_interaction_plot(["condition", "group"])

    assert capsys.readouterr().out == ""


def test_linear_regression_diagnostics_run_on_fitted_model():
    x = np.linspace(-2.0, 2.0, 30)
    frame = pd.DataFrame({"x": x, "y": 1.5 + 2.0 * x + np.sin(x)})
    model = smf.ols("y ~ x", data=frame).fit()
    diagnostic = Linear_Reg_Diagnostic(model)
    figure, axes = plt.subplots(2, 2)

    assert diagnostic.residual_plot(axes[0, 0]) is axes[0, 0]
    assert diagnostic.qq_plot(axes[0, 1]) is axes[0, 1]
    assert diagnostic.scale_location_plot(axes[1, 0]) is axes[1, 0]
    assert diagnostic.leverage_plot(axes[1, 1]) is axes[1, 1]
    assert calculate_vif(pd.DataFrame({"intercept": 1.0, "x": x}))["features"].tolist() == [
        "intercept",
        "x",
    ]
    plt.close(figure)

    with pytest.raises(TypeError, match="RegressionResultsWrapper"):
        Linear_Reg_Diagnostic(object())


def test_script_modules_are_import_safe():
    find_and_add = importlib.import_module(
        "calvin_utils.neuroimaging_utils.nifti_utils.find_and_add_niftis"
    )
    job_submitter = importlib.import_module("calvin_utils.server_utils.job_submitter")
    assert callable(find_and_add.combine_bilateral_niftis)
    assert callable(job_submitter.build_batch_scripts)


def test_train_test_splitter_handles_existing_synthetic_rows():
    data = pd.DataFrame(
        {
            "value": np.arange(10),
            "is_synthetic": [False] * 8 + [True] * 2,
        }
    )
    train, test = TrainTestSplitter(test_size=0.25, random_state=0).split(data)
    assert len(train) == 8
    assert len(test) == 2
    assert not test["is_synthetic"].any()


def test_train_test_splitter_can_disable_oversampling(tmp_path):
    data = pd.DataFrame({"value": np.arange(10), "outcome": [0, 1] * 5})
    train, test = TrainTestSplitter(
        test_size=0.2, random_state=0, synthetic_data=False
    ).run(data, out_dir=tmp_path, stratify="outcome")
    assert len(train) == 8
    assert len(test) == 2
    assert (tmp_path / "train_data.csv").is_file()


def test_constant_correlations_are_zero_and_debug_is_safe(capsys):
    result = run_pearson(np.ones((3, 1)), np.ones((3, 1)), debug=True)
    np.testing.assert_array_equal(result, [[0.0]])
    assert "R:" in capsys.readouterr().out


def test_centroid_default_pairs_are_created(tmp_path):
    csv_path = tmp_path / "paths.csv"
    pd.DataFrame(
        {
            "subject": ["one"],
            "target_a": ["a.nii"],
            "target_b": ["b.nii"],
            "other_a": ["c.nii"],
            "other_b": ["d.nii"],
        }
    ).to_csv(csv_path, index=False)
    comparison = NiftiCentroidComparisonStats(
        str(csv_path), target_pair={"target_a": "target_b"}
    )
    assert comparison.other_pairs == {"other_a": "other_b"}


def test_palm_default_design_matrix(tmp_path):
    palm = CalvinPalm(str(tmp_path / "unused.csv"), output_dir=None)
    design = palm.create_design_matrix(data_df=pd.DataFrame({"x": [1, 2]}))
    np.testing.assert_array_equal(design["Intercept"], [1.0, 1.0])


def test_default_voxelwise_interaction_uses_partial_f():
    rng = np.random.default_rng(2)
    sample_count = 24
    clinical = rng.normal(size=sample_count)
    brain = rng.normal(size=(sample_count, 3))
    outcome = brain[:, 0] * clinical + rng.normal(scale=0.2, size=sample_count)
    index = [f"subject_{value}" for value in range(sample_count)]

    _, results, _ = voxelwise_interaction_f_stat(
        pd.DataFrame({"outcome": outcome}, index=index),
        [pd.DataFrame(brain, index=index)],
        [pd.DataFrame({"clinical": clinical}, index=index)],
    )

    assert (results["statistic_method"] == "statsmodels_f_statistic").all()
    assert np.isfinite(results["statistic"]).all()
