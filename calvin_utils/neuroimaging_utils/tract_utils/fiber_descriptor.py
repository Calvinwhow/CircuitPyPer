"""Fibre result files: telling a fibre atlas from a result, and resolving a result.

A fibre ATLAS is geometry (.npz, .npy, .tck, .trk, .trx). A RESULT is values
attached to an atlas: a ``.fib.npy`` vector, and the ``.fib.json`` descriptor
(schema ``calvin_utils.fiber_values``) that names the atlas the values were
written against. These read and check those files; drawing them is the
viewer's business.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["ATLAS_SUFFIXES", "DESCRIPTOR_SCHEMA", "is_atlas", "is_descriptor", "read_descriptor", "streamline_values"]

# .mat is handled separately: it can carry values as well as geometry.
ATLAS_SUFFIXES = (".npz", ".npy", ".tck", ".trk", ".trx")


def is_atlas(path):
    """Whether this file is fibre GEOMETRY rather than a result.

    ``.fib.npy`` and ``.fib.values.npy`` are results -- values attached to an
    atlas -- and everything else with these suffixes is the atlas itself.
    """
    name = str(path).lower()
    if name.endswith((".fib.npy", ".fib.values.npy")):
        return False
    return name.endswith(ATLAS_SUFFIXES)


DESCRIPTOR_SCHEMA = "calvin_utils.fiber_values"


def is_descriptor(path):
    """Whether this ``.json`` is a fibre-value descriptor.

    Read rather than assumed from the extension: a ``.json`` here could equally
    be an electrode reconstruction, and the schema key is what says which.
    """
    path = Path(path).expanduser()
    if path.suffix.lower() != ".json":
        return False
    try:
        import json

        with open(path) as fh:
            return json.load(fh).get("schema") == DESCRIPTOR_SCHEMA
    except Exception:
        return False


def read_descriptor(path, verify=True):
    """Resolve a ``.fib.desc.json`` to its atlas and its values file.

    The descriptor is the entry point for a result: it names the atlas the
    values were written against, so a figure never hard-codes an atlas path and
    cannot be pointed at the wrong one by accident.

    Atlas resolution tries the absolute path first and then ``relative_path``
    against the descriptor's own folder, which is what keeps a result readable
    after the drive it was written on is mounted somewhere else.

    ``verify`` checks what the descriptor promises -- the values file's SHA-256
    and the atlas' size -- because the failure it prevents is silent: values
    joined to the wrong atlas still render, they are just wrong.
    """
    import hashlib
    import json

    path = Path(path).expanduser().resolve()
    with open(path) as fh:
        desc = json.load(fh)
    if desc.get("schema") != DESCRIPTOR_SCHEMA:
        raise ValueError(f"{path.name} is not a {DESCRIPTOR_SCHEMA} descriptor")

    atlas_info = desc.get("fiber_atlas") or {}
    candidates = [atlas_info.get("path")]
    if atlas_info.get("relative_path"):
        candidates.append(path.parent / atlas_info["relative_path"])
    atlas = next((Path(c).expanduser() for c in candidates
                  if c and Path(c).expanduser().is_file()), None)
    if atlas is None:
        raise FileNotFoundError(
            f"{path.name} names an atlas that is not on this machine: "
            f"{atlas_info.get('path')!r} (nor at its recorded relative path)")

    # The name moved from "<stem>.values.npy" to "<stem>.fib.values.npy"; both
    # spellings are read so results written either side of that still open.
    named = desc.get("values_file") or ""
    stem = named.split(".")[0] if named else path.name.split(".")[0]
    names = [n for n in (named, f"{stem}.fib.values.npy", f"{stem}.values.npy") if n]

    # Three places to look, because a descriptor does not always arrive beside
    # its values. A browser drop uploads the one file the user dragged, so the
    # descriptor lands alone in a scratch folder; the folder it was WRITTEN in
    # is then the only way back to the pair.
    folders = [path.parent]
    explicit = desc.get("values_path") or (desc.get("values") or {}).get("path")
    if explicit:
        folders.insert(0, Path(explicit).expanduser().parent)
        names.insert(0, Path(explicit).name)
    origin = desc.get("source_geometry_file")
    if origin:
        folders.append(Path(origin).expanduser().parent)
    atlas_dir = atlas_info.get("path")
    if atlas_dir:
        folders.append(Path(atlas_dir).expanduser().parent)

    values = next((folder / n for folder in folders for n in names
                   if (folder / n).is_file()), None)
    if values is None:
        raise FileNotFoundError(
            f"{path.name} names {named!r}, which is not beside it and is not at "
            f"any path the descriptor records. The values and the descriptor "
            f"are one unit and have to travel together -- drag both in, or open "
            f"the descriptor from its own folder with the file picker.")

    if verify:
        promised = (desc.get("values") or {}).get("sha256")
        if promised:
            digest = hashlib.sha256()
            with open(values, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    digest.update(block)
            if digest.hexdigest() != promised:
                raise ValueError(
                    f"{values.name} does not match the SHA-256 in "
                    f"{path.name}; it has been rewritten or replaced")
        size = atlas_info.get("size_bytes")
        if size is not None and atlas.stat().st_size != int(size):
            raise ValueError(
                f"{atlas.name} is {atlas.stat().st_size} bytes but "
                f"{path.name} recorded {size}; this is not the atlas these "
                f"values were written against")

    return {"atlas": atlas, "values": values,
            "fiber_count": atlas_info.get("fiber_count"),
            "name": path.name.split(".")[0], "descriptor": desc}


def streamline_values(path, count=None):
    """One value per atlas fibre, from a ``.fib.values.npy`` (or any 1-D array).

    ``count`` is the number of fibres the atlas holds; a mismatch means the
    values were written against a different atlas, which is worth saying plainly
    rather than colouring the wrong bundles.
    """
    path = Path(path).expanduser()
    if is_descriptor(path):
        # A descriptor stands in for the values it describes, so the validation
        # it carries travels with the overlay rather than being done once at
        # drop time and then forgotten.
        path = read_descriptor(path)["values"]
    if path.suffix.lower() == ".npz":
        store = np.load(path, allow_pickle=True)
        key = next((k for k in ("values", "vals", "arr_0") if k in store),
                   store.files[0])
        values = np.asarray(store[key])
    else:
        values = np.asarray(np.load(path, allow_pickle=True))
    values = np.asarray(values, dtype=np.float64).ravel()
    if count is not None and len(values) != count:
        raise ValueError(
            f"{path.name} has {len(values)} values but the atlas has {count} "
            f"fibers; these were written against a different atlas")
    return values
