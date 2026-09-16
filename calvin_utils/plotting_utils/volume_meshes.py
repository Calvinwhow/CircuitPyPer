"""Turn a NIfTI into drawable geometry, so any volume can become a layer.

A lesion, a stimulation volume, a thresholded cluster, a whole atlas -- these
arrive as volumes, not meshes, and the only thing standing between them and a
figure is a surface.  This builds that surface with the SAME pipeline that made
the SUIT and AAL parcel meshes (``nifti_utils.parcel_meshes.mesh_from_mask``),
so a lesion dropped in today sits beside those meshes and looks like it belongs:
identical presmoothing, volume-preserving iso level, Taubin smoothing and
decimation.

What a volume becomes depends on what is in it, and ``split`` says which:

``single``      everything above the threshold, as one surface.  A lesion mask,
                a VTA, a thresholded cluster map.
``labels``      one surface per distinct integer value -- any atlas or
                segmentation becomes a region GROUP, stylable exactly like the
                built-in parcel sets.
``components``  one surface per connected blob above the threshold, which is how
                a multi-cluster statistical map becomes separately colourable
                pieces without anyone having to label them first.
``auto``        ``labels`` when the volume looks like a segmentation (integer
                valued, a handful of distinct values), otherwise ``single``.

Auto exists because getting this wrong is expensive in both directions: running
marching cubes 400 times over what was actually a probability map, or flooding
an entire atlas into one blob.  It guesses from the data, and the answer it
picked is returned so the caller can show it rather than leave the user
wondering.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["meshes_from_volume", "volume_summary", "SPLITS", "MAX_LABELS",
           "MIN_VOXELS", "MAX_MESHES"]

SPLITS = ("auto", "single", "labels", "components")

# Above this many distinct values a volume is not a segmentation, whatever its
# dtype says; it is a parametric map that happens to be integer-valued, and
# meshing every level of it would take minutes and produce nothing legible.
MAX_LABELS = 256

# Marching cubes plus smoothing costs roughly a second per structure, so a
# threshold that leaves 400 speckles is minutes of work for a figure nobody can
# read. Small pieces are dropped by VOXEL COUNT before any of them is meshed,
# and only the largest survivors are built; what got dropped is reported rather
# than silently discarded, because "my cluster vanished" must be answerable.
MIN_VOXELS = 27          # 3x3x3 -- below this nothing survives smoothing anyway
MAX_MESHES = 64

# Matching the parcel meshes: voxel-space presmoothing, 1 mm contouring.
DEFAULT_SMOOTHING = dict(presmooth_vox=1.2, target_mm=1.0, smooth_i=40,
                         smooth_f=0.05, decimate=0.4, min_component_frac=0.05)


def _load(path):
    import nibabel as nib

    img = nib.load(str(Path(path).expanduser()))
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim > 3:
        data = data[..., 0]
    return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0), np.asarray(img.affine, float)


def _looks_like_labels(data):
    nonzero = data[data != 0]
    if not nonzero.size:
        return False
    if not np.allclose(nonzero, np.round(nonzero)):
        return False
    return 1 < len(np.unique(nonzero)) <= MAX_LABELS


def volume_summary(path):
    """What is in a volume, for the UI to show before anything is meshed."""
    data, _ = _load(path)
    nonzero = data[data != 0]
    labels = _looks_like_labels(data)
    return {
        "shape": list(data.shape),
        "min": float(data.min()), "max": float(data.max()),
        "nonzero": int(nonzero.size),
        "distinct": int(len(np.unique(nonzero))) if nonzero.size else 0,
        "looks_like_labels": bool(labels),
        "suggested_split": "labels" if labels else "single",
    }


def meshes_from_volume(path, iso=None, split="auto", names=None,
                       min_voxels=MIN_VOXELS, max_meshes=MAX_MESHES,
                       verbose=False, **smoothing):
    """``({name: PolyData}, info)`` for one volume.

    ``iso`` is the threshold that defines "inside": voxels with ``value > iso``.
    It defaults to 0, which is what a mask or an already-thresholded map wants.
    For a raw statistical map, set it to the same number you would threshold at.

    Pieces under ``min_voxels`` are dropped and at most ``max_meshes`` of the
    largest are built; ``info["skipped"]`` and ``info["over_cap"]`` name what did
    not make it.
    """
    from calvin_utils.neuroimaging_utils.nifti_utils.parcel_meshes import (
        mesh_from_mask, sanitize_region_name,
    )

    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")

    data, affine = _load(path)
    iso = 0.0 if iso is None else float(iso)
    options = dict(DEFAULT_SMOOTHING, **{k: v for k, v in smoothing.items()
                                         if v is not None})

    if split == "auto":
        split = "labels" if _looks_like_labels(data) else "single"

    masks = {}
    if split == "labels":
        for value in np.unique(data[data != 0]):
            label = int(round(float(value)))
            name = (names or {}).get(label, f"label_{label}")
            masks[sanitize_region_name(str(name))] = data == value
    elif split == "components":
        from scipy import ndimage

        tagged, count = ndimage.label(data > iso)
        # Largest first, so "component_1" is the one someone means by "the
        # cluster" and colour choices stay stable if the threshold nudges.
        sizes = np.bincount(tagged.ravel())[1:] if count else np.zeros(0)
        for rank, index in enumerate(np.argsort(sizes)[::-1], start=1):
            masks[f"component_{rank}"] = tagged == (index + 1)
    else:
        masks[Path(path).name.split(".nii")[0]] = data > iso

    # Drop the specks before meshing anything, then keep the largest.
    sized = [(name, mask, int(mask.sum())) for name, mask in masks.items()]
    too_small = [name for name, _, n in sized if n < int(min_voxels)]
    sized = [item for item in sized if item[2] >= int(min_voxels)]
    sized.sort(key=lambda item: -item[2])
    over_cap = [name for name, _, _ in sized[int(max_meshes):]]
    masks = {name: mask for name, mask, _ in sized[:int(max_meshes)]}

    meshes, skipped = {}, list(too_small)
    for name, mask in masks.items():
        surface = mesh_from_mask(mask, affine, verbose=verbose, **options)
        if surface is None or not surface.n_points:
            skipped.append(name)      # too small or too thin to close a surface
            continue
        meshes[name] = surface

    if not meshes:
        raise ValueError(
            f"nothing above {iso} in {Path(path).name} formed a surface"
            + (f"; {len(skipped)} piece(s) were under {min_voxels} voxels"
               if skipped else "")
        )
    return meshes, {"split": split, "iso": iso, "n_meshes": len(meshes),
                    "skipped": skipped, "over_cap": over_cap}
