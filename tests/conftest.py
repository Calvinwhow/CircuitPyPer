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


@pytest.fixture
def basis_shards(tmp_path):
    """Write {name: voxel vector} maps as one float32 (voxels, bases) shard per cohort and return the directory."""
    import json
    import numpy as np

    def write(maps, cohort="synthetic", pathology="Synthetic", directory=tmp_path / "basis_maps"):
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / f"shard_{cohort}.npy", np.stack([np.asarray(values, dtype=np.float32) for values in maps.values()], axis=1))
        sidecar = {"cohort": cohort, "pathology": pathology, "input_path": "synthetic.csv", "subject_column": "subject", "regressand_list": list(maps), "voxelwise_vars": ["nifti"], "mask_path": "mask.nii.gz", "transform": "rank", "patient_ids": [], "bases": [{"name": name, "regressand": name, "voxelwise_var": "nifti", "n": 40} for name in maps]}
        (directory / f"shard_{cohort}.json").write_text(json.dumps(sidecar))
        return directory

    return write
