"""Resolve data files in source checkouts and installed distributions."""

from __future__ import annotations

import os
import site
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path


NIFTI_MASK_FILENAME = "MNI152_T1_2mm_brain_mask.nii"


def _resource_candidates() -> list[Path]:
    """Return resource roots for overrides, checkouts, and pip installs."""
    package_dir = Path(__file__).resolve().parent
    candidates: list[Path] = []

    if configured := os.environ.get("CALVIN_UTILS_RESOURCES"):
        # An explicit override is authoritative. A typo should not silently
        # select a different installation's resources.
        return [Path(configured).expanduser()]

    candidates.extend(
        [
            package_dir.parent / "resources",  # source/editable checkout
            package_dir / "resources",  # package-data installation
            package_dir.parent / "share" / "calvin_utils" / "resources",
        ]
    )

    data_roots = {Path(sys.prefix), Path(sysconfig.get_path("data"))}
    try:
        data_roots.add(Path(site.getuserbase()))
    except (AttributeError, TypeError):
        pass
    candidates.extend(
        root / "share" / "calvin_utils" / "resources" for root in data_roots
    )

    # ``pip --target`` places both ``calvin_utils`` and ``share`` beneath an
    # entry on sys.path, which need not be sys.prefix.
    candidates.extend(
        Path(entry) / "share" / "calvin_utils" / "resources"
        for entry in sys.path
        if entry
    )

    unique: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.expanduser().resolve(strict=False)
        if normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


def get_resource_root() -> Path:
    """Return the first available Calvin Utils resource directory.

    ``CALVIN_UTILS_RESOURCES`` can override the location. Source checkouts keep
    resources beside ``calvin_utils``; wheels install them beneath
    ``$prefix/share/calvin_utils``.
    """
    candidates = _resource_candidates()
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Calvin Utils resources were not found. Set CALVIN_UTILS_RESOURCES "
        f"to their location. Searched: {searched}"
    )


def resource_path(*parts: str) -> Path:
    """Return a path below the resolved resource root."""
    return get_resource_root().joinpath(*parts)


def default_nifti_mask_path() -> Path:
    """Return the definitive default mask for volumetric NIfTI operations."""
    path = resource_path(NIFTI_MASK_FILENAME)
    if not path.is_file():
        raise FileNotFoundError(f"Default NIfTI mask was not found: {path}")
    return path


@dataclass(frozen=True)
class LazyResourcePath(os.PathLike):
    """A path-like resource reference that resolves only when it is used."""

    parts: tuple[str, ...]

    def resolve(self) -> Path:
        return resource_path(*self.parts)

    def __fspath__(self) -> str:
        return os.fspath(self.resolve())

    def __str__(self) -> str:
        return os.fspath(self)

    def __truediv__(self, other) -> "LazyResourcePath":
        return LazyResourcePath(self.parts + (os.fspath(other),))

    def __getattr__(self, name):
        # Delegate ordinary pathlib operations (exists, is_file, suffix, etc.)
        # without resolving during module import or object construction.
        return getattr(self.resolve(), name)


def lazy_resource_path(*parts: str) -> LazyResourcePath:
    """Return a resource reference without resolving the installation yet."""
    return LazyResourcePath(tuple(parts))
