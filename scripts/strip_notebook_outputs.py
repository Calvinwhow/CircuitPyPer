#!/usr/bin/env python3
"""Remove generated output state from Jupyter notebooks.

With explicit paths this is suitable for pre-commit. With no paths it scans the
repository, which is useful both locally and in CI. The implementation uses only
the Python standard library so notebook hygiene never depends on Jupyter being
installed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).absolute().parents[1]
EXCLUDED_DIRECTORIES = {".git", ".mypy_cache", ".pytest_cache", ".venv", "venv"}


def clean_notebook(document: dict[str, Any]) -> bool:
    """Clear code-cell outputs/execution counts and notebook widget state."""
    changed = False
    for cell in document.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True

    # Widget state is generated output stored at notebook level and can be
    # considerably larger than the visible cell output.
    metadata = document.get("metadata")
    if isinstance(metadata, dict) and "widgets" in metadata:
        del metadata["widgets"]
        changed = True
    return changed


def discover_notebooks(root: Path = PROJECT_ROOT) -> list[Path]:
    """Return repository notebooks while ignoring VCS and environment data."""
    return sorted(
        path
        for path in root.rglob("*.ipynb")
        if not EXCLUDED_DIRECTORIES.intersection(path.relative_to(root).parts)
    )


def strip_notebook(path: Path, *, check: bool = False) -> bool:
    """Clean one notebook and return whether generated state was present."""
    raw_document = path.read_text(encoding="utf-8")
    # A few legacy paths are intentional zero-byte placeholders. They contain
    # no generated state and should not prevent cleanup of real notebooks.
    if not raw_document.strip():
        return False
    document = json.loads(raw_document)
    changed = clean_notebook(document)
    if changed and not check:
        path.write_text(
            json.dumps(document, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )
    return changed


def _notebook_paths(paths: Iterable[str]) -> list[Path]:
    explicit_paths = list(paths)
    if not explicit_paths:
        return discover_notebooks()
    return sorted(
        path
        for raw_path in explicit_paths
        if (path := Path(raw_path)).suffix == ".ipynb" and path.is_file()
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report generated notebook state without modifying files.",
    )
    parser.add_argument("paths", nargs="*", help="Notebook paths; defaults to all.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    changed_paths = []
    try:
        for path in _notebook_paths(args.paths):
            if strip_notebook(path, check=args.check):
                changed_paths.append(path)
    except (OSError, json.JSONDecodeError) as error:
        print(f"Notebook cleanup failed: {error}", file=sys.stderr)
        return 2

    if not changed_paths:
        print("Notebook outputs are clean.")
        return 0

    action = "would strip" if args.check else "stripped"
    for path in changed_paths:
        try:
            display_path = path.relative_to(PROJECT_ROOT)
        except ValueError:
            display_path = path
        print(f"{action}: {display_path}")
    if args.check:
        print("Run: python scripts/strip_notebook_outputs.py", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
