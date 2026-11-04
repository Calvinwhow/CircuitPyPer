"""Memory-mapped basis shards and the patient-by-basis damage they produce.

A shard is ``basis_maps/shard_{cohort}.npy``: one float32 array of shape (voxels, bases), only ever opened with
``np.load(mmap_mode="r")``. Its sidecar ``shard_{cohort}.json`` must carry every key in SIDECAR_KEYS, and each entry
of its ``bases`` list (one per column, in column order) every key in BASIS_KEYS.
"""

import json
import numpy as np
from pathlib import Path
from .pre_processing import FEATURE_DTYPE

BASIS_DIR = Path(__file__).resolve().parents[1] / "basis_maps"
SIDECAR_KEYS = ("cohort", "pathology", "input_path", "subject_column", "regressand_list", "voxelwise_vars", "mask_path", "transform", "patient_ids", "bases")
BASIS_KEYS = ("name", "regressand", "voxelwise_var", "n")


def load_shard(path):
    """Memory-map one shard and validate it against its sidecar; nothing is read into memory."""
    path = Path(path)
    sidecar_path = path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text())
    missing = [key for key in SIDECAR_KEYS if key not in sidecar]
    if missing:
        raise ValueError(f"{sidecar_path.name} is missing {missing}.")
    if path.stem != f"shard_{sidecar['cohort']}":
        raise ValueError(f"{path.name} does not match its sidecar cohort {sidecar['cohort']!r}.")
    for basis in sidecar["bases"]:
        missing = [key for key in BASIS_KEYS if key not in basis]
        if missing:
            raise ValueError(f"A basis in {sidecar_path.name} is missing {missing}.")
    values = np.load(path, mmap_mode="r")
    if values.dtype != FEATURE_DTYPE or values.ndim != 2 or values.shape[1] != len(sidecar["bases"]):
        raise ValueError(f"{path.name} must be a {np.dtype(FEATURE_DTYPE).name} (voxels, {len(sidecar['bases'])}) array; found {values.dtype} {values.shape}.")
    return {"path": path, "values": values, "sidecar": sidecar, "names": tuple(f"{sidecar['cohort']}:{basis['name']}" for basis in sidecar["bases"])}


def load_library(cohorts=None, directory=BASIS_DIR):
    """Memory-map the shards of the selected cohorts, or of every cohort when ``cohorts`` is None."""
    directory = Path(directory)
    paths = sorted(directory.glob("shard_*.npy")) if cohorts is None else [directory / f"shard_{cohort}.npy" for cohort in cohorts]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{missing} not found in {directory}.")
    if not paths:
        raise ValueError(f"No basis shards in {directory}.")
    shards = [load_shard(path) for path in paths]
    n_voxels = {shard["values"].shape[0] for shard in shards}
    if len(n_voxels) != 1:
        raise ValueError(f"Shards disagree on voxel count: {sorted(n_voxels)}.")
    mask_paths = {shard["sidecar"]["mask_path"] for shard in shards}
    if len(mask_paths) != 1:
        raise ValueError("Shards disagree on the mask defining feature order.")
    names = tuple(name for shard in shards for name in shard["names"])
    if len(set(names)) != len(names):
        raise ValueError("Basis names repeat across the selected shards.")
    sample_sizes = {name: int(basis["n"]) for shard in shards for name, basis in zip(shard["names"], shard["sidecar"]["bases"])}
    return {"shards": shards, "names": names, "n_voxels": n_voxels.pop(), "mask_path": mask_paths.pop(), "sample_sizes": sample_sizes}


def symptoms(library):
    """Every symptom the selected shards have bases for, in shard order."""
    return tuple(dict.fromkeys(basis["regressand"] for shard in library["shards"] for basis in shard["sidecar"]["bases"]))


def sweep(shards, patients, chunk_voxels):
    """Read every shard once, accumulating patient-by-basis dot products and each basis's L2 norm."""
    n_voxels = shards[0]["values"].shape[0]
    n_bases = sum(shard["values"].shape[1] for shard in shards)
    products = {key: np.zeros((len(values), n_bases)) for key, values in patients.items()}
    squares = np.zeros(n_bases)
    for start in range(0, n_voxels, chunk_voxels):
        block = np.hstack([shard["values"][start:start + chunk_voxels] for shard in shards])
        if not np.isfinite(block).all():
            raise ValueError(f"Basis shards contain NaN or infinity in voxels {start} to {start + len(block)}.")
        squares += np.square(block, dtype=np.float64).sum(axis=0)
        for key, values in patients.items():
            products[key] += values[:, start:start + chunk_voxels] @ block
    return products, np.sqrt(squares)


def bare_subjects(name, ids):
    """Subject IDs without the "{dataset}:" prefix fitting IDs may carry and sidecars do not."""
    prefix = f"{name}:"
    return np.asarray([value[len(prefix):] if value.startswith(prefix) else value for value in np.asarray(ids).astype(str)])


def check_leakage(test_datasets, library):
    """Refuse a test dataset named like a selected basis cohort whose patients built that cohort's bases."""
    sidecars = {shard["sidecar"]["cohort"]: shard["sidecar"] for shard in library["shards"]}
    leaking = sorted(name for name in set(test_datasets) & set(sidecars) if set(bare_subjects(name, test_datasets[name]["ids"])) & set(map(str, sidecars[name]["patient_ids"])))
    if leaking:
        raise ValueError(f"Test datasets {leaking} contain patients that built selected basis cohorts; leave those cohorts out of cohorts= or test on other patients.")


def main(datasets, test_datasets=None, cohorts=None, directory=BASIS_DIR, chunk_voxels=16384):
    """Reduce every patient map to patient-by-basis damage in one pass over the selected shards.
    datasets: dict[str, dict] of length D; each entry contains X with shape (N, V), y with shape (N,), and IDs with shape (N,).
    test_datasets: dict[str, dict] | None; external datasets with the same fields and shapes as datasets.
    cohorts: iterable[str] | None; basis-library cohorts to load, or every available cohort when omitted.
    directory: str | Path; directory containing basis shard NPY and JSON pairs.
    chunk_voxels: int; number of voxel rows read from each shard per sweep block.
    """
    library = load_library(cohorts, directory)
    check_leakage(test_datasets or {}, library)
    groups = {"datasets": datasets, "test_datasets": test_datasets or {}}
    patients = {}
    for group, members in groups.items():
        for name, data in members.items():
            values = np.asarray(data["X"], dtype=FEATURE_DTYPE)
            if values.ndim != 2 or values.shape[1] != library["n_voxels"]:
                raise ValueError(f"Dataset {name!r} has patient maps of shape {values.shape}; the shards have {library['n_voxels']} voxels.")
            patients[group, name] = values
    products, scales = sweep(library["shards"], patients, chunk_voxels)
    empty = [name for name, scale in zip(library["names"], scales) if scale == 0]
    if empty:
        raise ValueError(f"Basis maps {empty} contain no nonzero value.")
    damage = {key: products[key] / scales / np.linalg.norm(values, axis=1)[:, None] for key, values in patients.items()}
    scored = {group: {name: {"X": damage[group, name], "patients": patients[group, name], "y": np.asarray(data["y"], dtype=float).reshape(-1), "ids": np.asarray(data["ids"]).astype(str).reshape(-1)} for name, data in members.items()} for group, members in groups.items()}
    return {"library": library, "map_names": library["names"], "sample_sizes": library["sample_sizes"], "scales": scales, "datasets": scored["datasets"], "test_datasets": scored["test_datasets"] if test_datasets is not None else None}


def combine(weights, library, scales):
    """Build the weighted symptom map in voxel space, one memory-mapped shard at a time."""
    weights = np.asarray(weights, dtype=float).reshape(-1) / scales
    symptom_map = np.zeros(library["n_voxels"], dtype=FEATURE_DTYPE)
    offset = 0
    for shard in library["shards"]:
        width = shard["values"].shape[1]
        symptom_map += shard["values"] @ weights[offset:offset + width].astype(FEATURE_DTYPE)
        offset += width
    return symptom_map


def stack(library, scales):
    """Load every selected basis as a normalized (bases, voxels) matrix; only the optional history and basis-map inference outputs need this."""
    matrix = np.hstack([shard["values"] for shard in library["shards"]])
    matrix /= scales.astype(FEATURE_DTYPE)
    return matrix.T
