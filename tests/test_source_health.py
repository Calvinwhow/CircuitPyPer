"""Repository-wide structural and compatibility checks."""

import ast
import importlib
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SourceHealthTest(unittest.TestCase):
    def test_every_utility_and_script_is_valid_python(self):
        failures = []
        for source_root in (PROJECT_ROOT / "calvin_utils", PROJECT_ROOT / "scripts"):
            for path in sorted(source_root.rglob("*.py")):
                try:
                    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                except SyntaxError as exc:
                    failures.append(f"{path.relative_to(PROJECT_ROOT)}:{exc.lineno}: {exc.msg}")
        self.assertEqual(failures, [], "\n".join(failures))

    def test_every_legacy_ccm_shim_imports_its_canonical_module(self):
        shim_root = PROJECT_ROOT / "calvin_utils" / "ccm_utils"
        failures = []
        for path in sorted(shim_root.rglob("*.py")):
            module = ".".join(path.with_suffix("").relative_to(PROJECT_ROOT).parts)
            if module.endswith(".__init__"):
                module = module[: -len(".__init__")]
            try:
                importlib.import_module(module)
            except Exception as exc:  # Report every broken compatibility route together.
                failures.append(f"{module}: {type(exc).__name__}: {exc}")
        self.assertEqual(failures, [], "\n".join(failures))

    def test_fast_regression_utility_is_import_safe(self):
        module = importlib.import_module(
            "calvin_utils.neuroimaging_utils.ccm_utils.fast_voxelwise_regression"
        )
        self.assertTrue(callable(module.main))
        self.assertGreater(len(module.indep_var_list), 1)

    def test_optional_bayesian_dependency_is_lazy(self):
        module = importlib.import_module(
            "calvin_utils.neuroimaging_utils.ccm_utils.bayesian_optimization"
        )
        self.assertTrue(hasattr(module, "NiftiBayesianOptimizer"))

    def test_legacy_permutation_launchers_are_import_safe(self):
        modules = (
            "calvin_utils.permutation_analysis_utils.get_palm_results",
            "calvin_utils.permutation_analysis_utils.permute_dice_coefficient",
            "calvin_utils.permutation_analysis_utils.run_fsl_palm_script",
            "calvin_utils.permutation_analysis_utils.scripts_for_submission.launch_delta_r_map_palm",
        )
        for module_name in modules:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertTrue(callable(module.main))


if __name__ == "__main__":
    unittest.main()
