# Test strategy

The suite protects behavior at three levels:

1. **Repository health** parses every utility and launcher and imports legacy
   compatibility routes. This catches syntax errors, import-time execution,
   and broken module moves.
2. **Module contracts** assert stable public entry points for the core I/O,
   tract, regression, permutation, and specificity modules. Tests should call
   these public entry points rather than private class helpers whenever a
   public route exists.
3. **Behavioral pipelines** run synthetic and deidentified golden data through
   complete workflows. Assertions cover shapes, ordering, finite values,
   outputs, statistical agreement, held-out isolation, and reproducibility.

The golden cohort is accessed through the session fixture in `conftest.py`, so
tests do not depend on external volumes or machine-specific paths. Add a new
fixture only when generated data cannot represent the relevant file format or
numerical behavior.

Run everything with:

```bash
python -m pytest -q
```

Notebook outputs, execution counts, and generated widget state are removed by
the pre-commit hook. Clean the whole checkout manually with:

```bash
python scripts/strip_notebook_outputs.py
```

The `Notebook Hygiene` workflow runs the same utility in `--check` mode and
prevents generated notebook state from reaching `main`.

The installed pre-push hook calls `scripts/run_tests.py`, which finds an
activated virtual environment, `.venv` in or beside the repository, or an
explicit `CALVIN_TEST_PYTHON` interpreter.

Run the same coverage checks used by CI with:

```bash
python -m coverage run -m pytest -q
python -m coverage report
python -m coverage report --include='calvin_utils/file_utils/csv_prep.py,calvin_utils/file_utils/file_path_collector.py,calvin_utils/neuroimaging_utils/io/importers.py,calvin_utils/neuroimaging_utils/nifti_utils/volume_io.py,calvin_utils/neuroimaging_utils/tract_utils/fiber_io.py,calvin_utils/neuroimaging_utils/tract_utils/fiber_converter.py,calvin_utils/neuroimaging_utils/tract_utils/filter_fibers.py,calvin_utils/permutation_analysis_utils/map_damage_cv.py,calvin_utils/permutation_analysis_utils/voxelwise_regression.py,calvin_utils/permutation_analysis_utils/voxelwise_regression_prep.py,calvin_utils/permutation_analysis_utils/statsmodels_palm.py,calvin_utils/neuroimaging_utils/ccm_utils/symptom_specificity.py,calvin_utils/neuroimaging_utils/ccm_utils/sensitivity_map.py,calvin_utils/statistical_utils/data_transformation.py' --fail-under=55
```

When adding a core module, add its public surface to
`test_public_module_contracts.py`, exercise its behavior in the closest
pipeline suite, and add it to the core coverage command once meaningful
coverage exists.

CI currently guards a 17% repository-wide baseline (including the newly
packaged legacy neuroimaging tree) and a 55% aggregate floor for the actively
tested core modules. Raise
these floors as new module contracts are added; do not lower them to accommodate
untested changes.
