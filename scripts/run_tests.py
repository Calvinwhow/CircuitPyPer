#!/usr/bin/env python3
"""Run pytest with the project environment, even from a Git pre-push hook."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


# Keep the checkout spelling of the path. In managed workspaces ``resolve()``
# may point into a mounted session tree that does not contain the sibling venv.
PROJECT_ROOT = Path(__file__).absolute().parents[1]


def _candidate_interpreters() -> list[Path]:
    candidates = []
    override = os.environ.get("CALVIN_TEST_PYTHON")
    virtual_env = os.environ.get("VIRTUAL_ENV")
    if override:
        candidates.append(Path(override))
    if virtual_env:
        candidates.append(Path(virtual_env) / "bin" / "python")
    candidates.extend(
        (
            PROJECT_ROOT / ".venv" / "bin" / "python",
            PROJECT_ROOT.parent / ".venv" / "bin" / "python",
            Path(sys.executable),
        )
    )
    normalized = []
    for path in candidates:
        path = path.expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        # Do not resolve interpreter symlinks: a venv's Python commonly points
        # at the system binary, but it must be invoked through the venv path so
        # Python can discover the venv's installed packages.
        normalized.append(path.absolute())
    return list(dict.fromkeys(normalized))


def _has_pytest(interpreter: Path) -> bool:
    if not interpreter.is_file():
        return False
    result = subprocess.run(
        [str(interpreter), "-c", "import pytest"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def main() -> int:
    for interpreter in _candidate_interpreters():
        if _has_pytest(interpreter):
            return subprocess.call(
                [str(interpreter), "-m", "pytest", "-q", *sys.argv[1:]],
                cwd=PROJECT_ROOT,
            )
    print(
        "No Python environment with pytest was found. "
        "Install requirements-dev.txt or set CALVIN_TEST_PYTHON.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
