import os
from typing import Iterable, List


def normalize_paths(paths: Iterable[str]) -> List[str]:
    """Normalize and filter a list of file paths."""
    if paths is None:
        return []
    out = []
    for p in paths:
        if not isinstance(p, str):
            continue
        p = p.strip()
        if not p:
            continue
        out.append(os.path.abspath(os.path.expanduser(p)))
    return out
