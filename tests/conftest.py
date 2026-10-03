"""Shared, public-interface fixtures for the repository test suite."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures"

# Tests must work from the repository root, its parent, and CI checkouts whose
# directory name is not necessarily ``circuit_pyper``.
for import_root in (PROJECT_ROOT, PROJECT_ROOT.parent):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture(scope="session")
def schmahmann_cohort() -> tuple[Path, pd.DataFrame]:
    """Return the golden cohort root and a table with resolved local paths."""
    root = FIXTURE_ROOT / "schmahmann_golden"
    cohort = pd.read_csv(root / "outcomes.csv")
    for column in ("Nifti_File_Path", "fiber_path_guerrera"):
        cohort[column] = cohort[column].map(lambda relative: root / relative)
    return root, cohort
