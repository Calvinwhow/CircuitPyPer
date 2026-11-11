"""Anatomical metadata, orientation, and spatial-map reconstruction."""

import numpy as np


ANATOMICAL_REPRESENTATIONS = ("fibers", "networks", "local")
REPRESENTATIONS = ANATOMICAL_REPRESENTATIONS
ALIASES = {"fiber": "fibers", "fibers": "fibers", "structural": "fibers", "tract": "fibers", "network": "networks", "networks": "networks", "functional": "networks", "connectivity": "networks", "local": "local", "voxel": "local", "voxelwise": "local", "lesion": "local"}


def canonical_representation(value):
    """Return one standardized non-composite anatomical representation."""
    key = str(value).strip().lower().replace("_", "-")
    if key not in ALIASES:
        raise ValueError(f"Unknown anatomical representation {value!r}; use fibers, networks, or local.")
    return ALIASES[key]


def basis_representations(library, supplied=None):
    """Return one explicit anatomical representation per ordered basis map."""
    names = library.map_names
    supplied_values = None if supplied is None or isinstance(supplied, dict) else tuple(supplied)
    if supplied_values is not None and len(supplied_values) != len(names):
        raise ValueError(f"Expected {len(names)} basis representations, received {len(supplied_values)}.")

    labels = []
    missing = []
    index = 0
    for shard in library.shards:
        sidecar_label = shard.metadata.get("representation")
        for name, basis in zip(shard.map_names, shard.metadata["bases"]):
            if isinstance(supplied, dict):
                value = supplied.get(name)
            elif supplied_values is not None:
                value = supplied_values[index]
            else:
                value = basis.get("representation", sidecar_label)
            if value is None:
                missing.append(name)
            else:
                labels.append(canonical_representation(value))
            index += 1
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Basis representation metadata is missing for {len(missing)} map(s): {preview}. Supply basis_representations or rebuild the shards with representation metadata.")
    return tuple(labels)


def representation_masks(labels):
    """Return masks with shape ``(3, n_bases)`` in representation order."""
    labels = tuple(canonical_representation(label) for label in labels)
    absent = [name for name in ANATOMICAL_REPRESENTATIONS if name not in labels]
    if absent:
        raise ValueError(f"The anatomical network requires at least one basis map for {absent}.")
    return np.stack([np.asarray([label == name for label in labels], dtype=np.float32) for name in ANATOMICAL_REPRESENTATIONS])


def reconstruct_algorithm_maps(weights, library, scales, masks):
    """Build the masked spatial maps once as ``(representation, voxel, algorithm)``."""
    weights = np.asarray(weights, dtype=np.float32)
    scales = np.asarray(scales, dtype=np.float32).reshape(-1)
    masks = np.asarray(masks, dtype=np.float32)
    if weights.ndim != 2 or weights.shape[0] != len(scales) or masks.shape != (len(REPRESENTATIONS), len(scales)):
        raise ValueError("Weights, scales, and representation masks do not share the basis axis.")
    maps = np.zeros((len(REPRESENTATIONS), library.voxel_count, weights.shape[1]), dtype=np.float32)
    offset = 0
    for shard in library.shards:
        width = shard.values.shape[1]
        normalized = weights[offset:offset + width] / scales[offset:offset + width, None]
        for group in range(len(REPRESENTATIONS)):
            maps[group] += shard.values @ (normalized * masks[group, offset:offset + width, None])
        offset += width
    return maps


def normalized_basis(library, scales):
    """Load the normalized voxel-by-basis matrix needed only for trainable filters."""
    scales = np.asarray(scales, dtype=np.float32).reshape(-1)
    matrix = np.empty((library.voxel_count, len(scales)), dtype=np.float32)
    offset = 0
    for shard in library.shards:
        width = shard.values.shape[1]
        matrix[:, offset:offset + width] = shard.values / scales[offset:offset + width]
        offset += width
    return matrix


def _correlation(left, right):
    left = np.asarray(left, dtype=np.float32).reshape(-1).copy()
    right = np.asarray(right, dtype=np.float32).reshape(-1).copy()
    left -= left.mean()
    right -= right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return 0.0 if denominator == 0 else float(left @ right / denominator)


def orient_algorithm_weights(weights, datasets, dataset_weights, supplied_signs=None):
    """Orient algorithm columns and establish fixed training-dataset directions."""
    weights = np.asarray(weights, dtype=np.float32).copy()
    correlations = np.asarray([[_correlation(data["X"] @ weights[:, algorithm], data["y"]) for algorithm in range(weights.shape[1])] for data in datasets.values()], dtype=np.float32)
    pooled = np.asarray([dataset_weights[name] for name in datasets], dtype=np.float32) @ correlations
    orientations = np.where(pooled < 0, -1.0, 1.0).astype(np.float32)
    weights *= orientations
    correlations *= orientations
    if supplied_signs is None:
        signs = {name: float(-1.0 if correlations[index].mean() < 0 else 1.0) for index, name in enumerate(datasets)}
    else:
        if set(supplied_signs) != set(datasets) or any(float(value) not in {-1.0, 1.0} for value in supplied_signs.values()):
            raise ValueError("dataset_signs must provide -1 or 1 for every fitting dataset.")
        signs = {name: float(supplied_signs[name]) for name in datasets}
    return weights, orientations, signs, correlations
