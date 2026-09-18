"""Lead-DBS electrode models, rendered from a reconstruction.

Ported from Lead-DBS: ``ea_showelectrode.m`` and ``ea_mapelmodel2reco.m``.

A model file holds the electrode built once at a canonical position, as two
groups of surfaces -- ``insulation`` and ``contacts`` -- plus four fiducials
(``head``, ``tail``, ``x``, ``y``). A reconstruction holds the same four
fiducials where the lead actually sits in a patient. Fitting one to the other is
a single affine solve, which is the whole of the geometry:

    A = [head 1; tail 1; x 1; y 1]   in the model
    B = [head 1; tail 1; x 1; y 1]   in the patient
    X = (A \\ B)'                     and every vertex goes through X

Four non-coplanar points determine the affine exactly, so this is a solve rather
than a fit: head and tail give the trajectory, x and y fix the roll, which is
what makes a directed lead's segmented contacts land on the right side.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["load_model", "read_markers", "electrode_meshes", "model_for",
           "looks_like_electrode", "INSULATION_COLOR", "CONTACT_COLOR"]

# Lead-DBS' own defaults: a pale grey shaft with dark metal bands.
INSULATION_COLOR = "#d4d6da"
CONTACT_COLOR = "#43484f"


# -- MATLAB files -----------------------------------------------------------
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


def _is_hdf5(path):
    """Whether this is a v7.3 (HDF5) MAT-file rather than an older one.

    A v7.3 file keeps MATLAB's 128-byte text header and puts the HDF5 signature
    at offset 512, so testing byte 0 says "not HDF5" for every v7.3 file there
    is -- and every Lead-DBS electrode model is v7.3.
    """
    with open(path, "rb") as fh:
        if fh.read(8) == _HDF5_MAGIC:
            return True
        fh.seek(512)
        return fh.read(8) == _HDF5_MAGIC


def _h5_meshes(handle, group):
    """``[(vertices, faces)]`` from one struct-of-arrays group in a v7.3 file."""
    out = []
    faces, verts = group["faces"], group["vertices"]
    for i in range(faces.shape[0]):
        f = np.array(handle[faces[i, 0]])
        v = np.array(handle[verts[i, 0]])
        # MATLAB writes column-major, so HDF5 hands both back transposed.
        out.append((np.ascontiguousarray(v.T, float),
                    np.ascontiguousarray(f.T).astype(np.int64) - 1))
    return out


def _mat_meshes(struct):
    """The same, from the scipy path for pre-7.3 files."""
    out = []
    entries = np.atleast_1d(struct).ravel()
    for item in entries:
        v = np.asarray(item.vertices, float)
        f = np.asarray(item.faces).astype(np.int64) - 1
        out.append((v, f))
    return out


def _polydata(vertices, faces):
    import pyvista as pv

    quads = faces.shape[1]
    sizes = np.full((len(faces), 1), quads, np.int64)
    return pv.PolyData(vertices, np.hstack([sizes, faces]).ravel())


def load_model(path):
    """One electrode model: its surfaces and the four fiducials that place it."""
    path = Path(path).expanduser()
    if _is_hdf5(path):
        import h5py

        with h5py.File(path, "r") as handle:
            e = handle["electrode"]
            parts = {name: _h5_meshes(handle, e[name])
                     for name in ("insulation", "contacts")}
            fids = {k: np.array(e[k]).ravel()[:3]
                    for k in ("head_position", "tail_position",
                              "x_position", "y_position")}
    else:
        import scipy.io as sio

        e = sio.loadmat(str(path), struct_as_record=False,
                        squeeze_me=True)["electrode"]
        parts = {name: _mat_meshes(getattr(e, name))
                 for name in ("insulation", "contacts")}
        fids = {k: np.asarray(getattr(e, k), float).ravel()[:3]
                for k in ("head_position", "tail_position",
                          "x_position", "y_position")}

    A = np.array([[*fids["head_position"], 1.0],
                  [*fids["tail_position"], 1.0],
                  [*fids["x_position"], 1.0],
                  [*fids["y_position"], 1.0]])
    return {"name": path.stem, "A": A,
            "insulation": [_polydata(v, f) for v, f in parts["insulation"]],
            "contacts": [_polydata(v, f) for v, f in parts["contacts"]]}


def model_for(elmodel, models_dir):
    """Find a model file from Lead-DBS' human-readable electrode name.

    Lead-DBS keeps the mapping in ea_resolve_elspec.m as a long switch. Rather
    than port a table that changes with every new lead, the name is slugged the
    way the filenames already are -- "Medtronic 3389" -> medtronic_3389 -- and
    matched against what is on disk, which is self-maintaining.
    """
    models_dir = Path(models_dir).expanduser()
    slug = "".join(c if c.isalnum() else "_" for c in str(elmodel).lower())
    slug = "_".join(p for p in slug.split("_") if p)
    exact = models_dir / f"{slug}.mat"
    if exact.is_file():
        return exact
    have = {p.stem: p for p in models_dir.glob("*.mat")
            if not p.stem.endswith("_vol")}
    if slug in have:
        return have[slug]
    # Fall back to the longest stem the name starts with, so
    # "boston_scientific_vercise_directed" still finds "boston_vercise_directed".
    tokens = set(slug.split("_"))
    scored = sorted(((len(tokens & set(stem.split("_"))), stem)
                     for stem in have), reverse=True)
    if scored and scored[0][0] >= 2:
        return have[scored[0][1]]
    raise FileNotFoundError(
        f"no electrode model matching {elmodel!r} in {models_dir}")


# -- reconstructions --------------------------------------------------------
def _walk(node, handle=None):
    """MATLAB struct -> nested dict, for either file generation."""
    import h5py

    if handle is not None and isinstance(node, h5py.Group):
        return {k: _walk(node[k], handle) for k in node.keys()}
    if handle is not None and isinstance(node, h5py.Dataset):
        if node.dtype == h5py.ref_dtype:
            refs = np.asarray(node).ravel()
            return [_walk(handle[r], handle) for r in refs]
        return np.array(node)
    if hasattr(node, "_fieldnames"):
        return {k: _walk(getattr(node, k)) for k in node._fieldnames}
    if isinstance(node, np.ndarray) and node.dtype == object:
        return [_walk(x) for x in node.ravel()]
    return node


def read_markers(path):
    """``[{side, head, tail, x, y, elmodel}]`` from a reconstruction.

    Accepts an ``ea_reconstruction.mat`` of either generation, or a JSON file
    with the same shape -- which is the escape hatch for anything that is not
    MATLAB, and the reason this takes a structure rather than a file format.
    """
    path = Path(path).expanduser()
    if path.suffix.lower() == ".json":
        import json

        data = json.loads(path.read_text())
    elif _is_hdf5(path):
        import h5py

        with h5py.File(path, "r") as handle:
            data = {k: _walk(handle[k], handle) for k in handle.keys()
                    if not k.startswith("#")}
    else:
        import scipy.io as sio

        data = _walk(sio.loadmat(str(path), struct_as_record=False,
                                 squeeze_me=True))
        data.pop("__header__", None)
        data.pop("__version__", None)
        data.pop("__globals__", None)

    reco = data.get("reco", data)
    # MNI first: a figure in template space is the usual case, and a native
    # reconstruction plotted on an MNI brain would be silently in the wrong
    # place rather than obviously missing.
    space = None
    for key in ("mni", "scrf", "native"):
        if isinstance(reco, dict) and key in reco:
            space = reco[key]
            break
    if space is None:
        space = reco

    markers = space.get("markers") if isinstance(space, dict) else None
    if markers is None:
        raise ValueError(f"{path.name} has no markers; not a reconstruction?")
    if isinstance(markers, dict):
        markers = [markers]

    elmodel = data.get("elmodel") or reco.get("elmodel") if isinstance(reco, dict) else None
    if isinstance(elmodel, (list, np.ndarray)):
        elmodel = np.asarray(elmodel).ravel()[0]
    if isinstance(elmodel, bytes):
        elmodel = elmodel.decode()
    if isinstance(elmodel, np.ndarray):
        elmodel = "".join(chr(int(c)) for c in elmodel.ravel())

    out = []
    for i, entry in enumerate(markers):
        if not isinstance(entry, dict):
            continue
        try:
            fids = {k: np.asarray(entry[k], float).ravel()[:3]
                    for k in ("head", "tail", "x", "y")}
        except (KeyError, TypeError, ValueError):
            continue
        if any(len(v) != 3 or not np.isfinite(v).all() for v in fids.values()):
            continue
        out.append({"side": ["R", "L"][i] if i < 2 else str(i), **fids,
                    "elmodel": elmodel})
    if not out:
        raise ValueError(f"{path.name} has no usable markers")
    return out


# -- the transform ----------------------------------------------------------
def fit(A, markers):
    """The affine that carries the model's fiducials onto the patient's."""
    B = np.array([[*markers["head"], 1.0],
                  [*markers["tail"], 1.0],
                  [*markers["x"], 1.0],
                  [*markers["y"], 1.0]])
    return np.linalg.solve(A, B).T


def _apply(mesh, X):
    out = mesh.copy()
    pts = np.asarray(mesh.points, float)
    out.points = (X @ np.c_[pts, np.ones(len(pts))].T)[:3].T
    return out


def electrode_meshes(source, models_dir, model=None):
    """``{name: PolyData}`` for an electrode file, ready to be a group.

    ``source`` is a reconstruction; with ``model`` given, that model file is
    used instead of the one the reconstruction names. A model file passed as the
    source renders at its own canonical position, which is what makes it
    possible to look at a lead without a patient.
    """
    models_dir = Path(models_dir).expanduser()
    source = Path(source).expanduser()

    try:
        sites = read_markers(source)
    except (ValueError, KeyError, OSError):
        # Not a reconstruction: treat it as a model and place it where it was
        # built, with the identity transform.
        spec = load_model(model or source)
        sites = [{"side": "", "elmodel": spec["name"], "identity": True}]
        specs = {"": spec}
    else:
        specs = {}
        for site in sites:
            path = Path(model) if model else model_for(site["elmodel"], models_dir)
            specs[site["side"]] = load_model(path)

    group = {}
    for site in sites:
        spec = specs[site["side"]]
        X = (np.eye(4) if site.get("identity") else fit(spec["A"], site))
        tag = f"{site['side']}_" if site["side"] else ""
        for kind, meshes in (("insulation", spec["insulation"]),
                             ("contacts", spec["contacts"])):
            for i, mesh in enumerate(meshes, start=1):
                group[f"{tag}{kind}_{i:02d}"] = _apply(mesh, X)
    return group


def looks_like_electrode(path):
    """Whether this file is a lead rather than a tractogram.

    Both arrive as ``.mat``, so the extension decides nothing and the caller has
    to look inside. Only the top-level names are read -- a Lead-DBS tractogram
    is hundreds of megabytes and must not be loaded to answer this.
    """
    path = Path(path).expanduser()
    wanted = {"electrode", "reco", "markers", "elmodel"}
    try:
        if path.suffix.lower() == ".json":
            import json

            data = json.loads(path.read_text())
            return bool(wanted & set(data)) or "reco" in data
        if _is_hdf5(path):
            import h5py

            with h5py.File(path, "r") as handle:
                return bool(wanted & set(handle.keys()))
        import scipy.io as sio

        names = {n for n, _, _ in sio.whosmat(str(path))}
        return bool(wanted & names)
    except Exception:
        return False
