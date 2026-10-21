import sys
import types

import numpy as np

from calvin_utils.neuroimaging_utils.ccm_utils.optimization.inference import basis_map


def test_basis_inference_calls_existing_regression_pipeline(tmp_path, monkeypatch):
    calls = {}

    class Frame(dict):
        @property
        def columns(self):
            return list(self)

    class Prep:
        def __init__(self, **kwargs):
            calls["prep"] = kwargs

        def run(self):
            return {}, str(tmp_path / "dataset_dict.json")

    class Regression:
        def __init__(self, **kwargs):
            calls["regression"] = kwargs

        def run(self):
            calls["ran"] = True

    prep_module = types.ModuleType("voxelwise_regression_prep")
    prep_module.RegressionPrep = Prep
    regression_module = types.ModuleType("voxelwise_regression")
    regression_module.VoxelwiseRegression = Regression
    monkeypatch.setitem(sys.modules, "calvin_utils.permutation_analysis_utils.voxelwise_regression_prep", prep_module)
    monkeypatch.setitem(sys.modules, "calvin_utils.permutation_analysis_utils.voxelwise_regression", regression_module)
    monkeypatch.setattr(basis_map, "pd", types.SimpleNamespace(DataFrame=Frame))
    monkeypatch.setattr(basis_map, "save_basis_maps", lambda *args: [tmp_path / "first.nii.gz", tmp_path / "second.nii.gz"])
    weights = np.array([0.75, -0.25])
    result = basis_map.main(np.ones((2, 8)), weights, tmp_path, output_type="nii", mask_path=None, n_permutations=20)
    np.testing.assert_array_equal(calls["prep"]["design_matrix"]["optimized_weight"], weights)
    assert list(calls["prep"]["design_matrix"].columns) == ["optimized_weight"]
    assert calls["prep"]["data_transform_method"] is None
    assert calls["regression"]["n_permutations"] == 20
    assert calls["ran"] is True
    assert result["dataset"].endswith("dataset_dict.json")
