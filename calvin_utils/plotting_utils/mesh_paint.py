"""Turn a NIfTI into colour on a brain mesh.

There are two honest ways to put a volume onto a surface and they answer
different questions, so both are here and neither is the "real" one:

``paint_vertices`` samples the volume at every mesh vertex and gives a
continuous gradient.  It answers "what does the map look like where it touches
this surface" and it is the right choice for a cortical surface, because the
value it shows at a point genuinely came from that point.

``paint_parcels`` reduces the volume inside each regional mesh to one number
and floods the whole region with it.  It answers "how much does this map
implicate this structure", which is the question a parcellated figure is
usually asking.  The reduction is chosen, not assumed -- see ``parcel_stats``.

Both leave values that do not survive thresholding as NaN, which the renderer
draws in the opaque base colour.  An unpainted region therefore still reads as
anatomy rather than vanishing or turning into the bottom of the colour bar --
the distinction between "no signal here" and "the most negative value in the
figure" matters, and a colormap alone cannot make it.

Sampling geometry
-----------------
Meshes are in MNI world millimetres and volumes are in voxel indices, so every
sample goes through ``inv(affine)``.  Interior points for a parcel are found by
voxelising the mesh itself on a 1 mm world grid rather than by sampling the
volume's own voxels: the mesh is the thing that was drawn, so the mesh is the
thing whose inside should be measured, and a 1 mm grid weights every cubic
millimetre of a region equally whatever the volume's resolution happens to be.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyvista as pv

__all__ = [
    "sample_nifti", "paint_vertices", "parcel_stats", "parcel_interiors",
    "paint_parcels", "select_regions", "load_parcels", "coarsen", "STATS",
]

STATS = ("max", "mean_nonzero", "mean", "min", "sum")


# -- volume sampling --------------------------------------------------------
@lru_cache(maxsize=8)
def _load_volume(path, mtime):
    """Read and clean a volume once. ``mtime`` is in the key so an edited file
    on disk invalidates the entry rather than silently serving a stale map."""
    import nibabel as nib

    return _clean(nib.load(path))


def _clean(img):
    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim > 3:
        data = data[..., 0]
    # Statistical maps routinely carry NaN outside the analysis mask. Left in
    # place, a single NaN voxel poisons every trilinear sample that touches it,
    # so the whole neighbourhood silently drops out of the figure.
    return (np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0),
            np.asarray(img.affine, float))


def _volume(nifti):
    """Data and affine for a path or an in-memory image.

    Paths go through a small cache: reducing a volume inside 165 parcel meshes
    asks for the same volume 165 times, and re-reading a gzipped NIfTI each time
    costs more than every other step of the figure put together.
    """
    if hasattr(nifti, "affine"):
        return _clean(nifti)
    path = str(Path(nifti).expanduser())
    try:
        stamp = os.path.getmtime(path)
    except OSError:
        stamp = 0.0
    return _load_volume(path, stamp)


def sample_nifti(points, nifti, order=1):
    """Sample a volume at world-millimetre ``points`` (N, 3).

    ``order=1`` is trilinear, which is what a gradient wants; ``order=0`` is
    nearest neighbour, which is what you want for a label volume or any map
    whose values are categorical.  Points outside the volume return 0.
    """
    from scipy.ndimage import map_coordinates

    data, affine = _volume(nifti)
    pts = np.asarray(points, float)
    vox = np.linalg.inv(affine) @ np.c_[pts, np.ones(len(pts))].T
    return map_coordinates(data, vox[:3], order=order, mode="constant", cval=0.0)


def _select(values, sign="both", absolute=False, threshold=None):
    """Sign selection, optional |v|, and thresholding -- shared by both painters.

    ``sign`` is applied to the ORIGINAL values and ``absolute`` afterwards, so
    ``sign="negative", absolute=True`` means "the negative tail, plotted as
    magnitudes", which is how a one-tailed figure in a single colour is made.
    """
    out = np.asarray(values, float).copy()
    if sign not in ("both", "positive", "negative"):
        raise ValueError("sign must be 'both', 'positive' or 'negative'")
    if sign == "positive":
        out[out < 0] = np.nan
    elif sign == "negative":
        out[out > 0] = np.nan
    if absolute:
        out = np.abs(out)
    if threshold is not None:
        out[np.abs(out) < float(threshold)] = np.nan
    out[out == 0] = np.nan
    return out


# -- continuous painting ----------------------------------------------------
def paint_vertices(mesh, nifti, sign="both", absolute=False, threshold=None,
                   order=1, scalar="value"):
    """Copy of ``mesh`` carrying the volume's value at each vertex.

    Works on a single ``PolyData`` or on a ``{'L': ..., 'R': ...}`` hemisphere
    dict, returning the same shape it was given.
    """
    if isinstance(mesh, dict):
        return {k: paint_vertices(v, nifti, sign, absolute, threshold, order, scalar)
                for k, v in mesh.items()}

    out = mesh.copy()
    out[scalar] = _select(
        sample_nifti(out.points, nifti, order=order), sign, absolute, threshold
    )
    return out


# -- parcels ----------------------------------------------------------------
def load_parcels(source):
    """``{region_name: PolyData}`` from a registered mesh name or a directory.

    Region order follows ``atlas_LUT.txt`` when the directory ships one, so a
    figure's regions stay in atlas order rather than in whatever order the
    filesystem returns.
    """
    if isinstance(source, dict):
        return source

    name = str(source)
    directory = Path(name).expanduser()
    if not directory.is_dir():
        from calvin_utils.plotting_utils.brains import mesh_subcortex

        spec = mesh_subcortex(name)
        if not spec["custom_atlas_path"]:
            raise ValueError(
                f"{name!r} is a yabplot-packaged region set, not a directory of "
                "meshes this renderer can read. Point load_parcels at a directory, "
                "or use one of the locally built parcel meshes."
            )
        directory = Path(spec["custom_atlas_path"])

    files = {p.stem: p for p in sorted(directory.glob("*.vtk"))}
    lut = directory / "atlas_LUT.txt"
    if lut.exists():
        ordered = []
        for line in lut.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] in files:
                ordered.append(parts[1])
        files = {k: files[k] for k in ordered} | files
    return {stem: pv.read(path) for stem, path in files.items()}


def coarsen(meshes, reduction=0.7):
    """Decimate a mesh or a group of them, for the tessellated look.

    The parcel meshes are built at ~1.1 mm edges, which gives clean silhouettes
    and, at figure scale, a per-triangle wireframe so dense it reads as grey
    haze rather than as structure. Decimating to ~2-3 mm edges puts the facets
    and the wireframe back on a scale the eye can resolve: the surface shades as
    visible plates instead of as a smooth blob.

    This is a DISPLAY choice, applied per mesh before anything is merged or
    sampled, so region statistics are computed on whatever geometry is actually
    drawn and the numbers never disagree with the picture.
    """
    if not reduction:
        return meshes
    reduction = float(reduction)
    if isinstance(meshes, dict):
        return {k: coarsen(v, reduction) for k, v in meshes.items()}
    if meshes.n_cells < 80:            # already coarse; decimating destroys it
        return meshes
    return meshes.decimate_pro(reduction, preserve_topology=True).triangulate()


def _interior_points(mesh, step=1.0):
    """World coordinates of the points inside a closed mesh, on a `step` grid."""
    lo = np.floor(np.asarray(mesh.bounds).reshape(3, 2)[:, 0] / step) * step - step
    hi = np.ceil(np.asarray(mesh.bounds).reshape(3, 2)[:, 1] / step) * step + step
    dims = np.maximum(((hi - lo) / step).astype(int) + 1, 2)
    ref = pv.ImageData(dimensions=dims, spacing=(step,) * 3, origin=tuple(lo))
    mask = mesh.voxelize_binary_mask(reference_volume=ref)
    inside = np.asarray(mask.point_data["mask"]).astype(bool)
    return np.asarray(ref.points)[inside]


def parcel_interiors(parcels, step=1.0):
    """``{region: (N, 3) world points}`` inside each regional mesh.

    Voxelising 165 meshes is the slow part of a parcel figure, and it depends
    only on the MESHES -- not on the volume, the statistic or the threshold. A
    caller that is going to re-reduce the same regions (a GUI slider, a sweep
    over several contrasts) computes this once and passes it back in.
    """
    return {name: _interior_points(mesh, step=step)
            for name, mesh in load_parcels(parcels).items()}


def parcel_stats(parcels, nifti, stat="max", sign="both", absolute=False,
                 threshold=None, step=1.0, order=1, interiors=None):
    """``{region_name: value}`` reducing the volume inside each regional mesh.

    ``stat``
        ``"max"``            the most extreme voxel, sign kept -- sensitive, and
                             the right choice for a sparse thresholded map where
                             a real peak occupies a small part of a big region.
        ``"mean_nonzero"``   the average over voxels that carry a value, which
                             stops a thresholded map from being diluted by the
                             zeros around it.
        ``"mean"``           the average over the whole region, zeros included.
        ``"min"``, ``"sum"`` the obvious.

    ``sign`` keeps only one tail before reducing -- ``sign="positive",
    threshold=1.96`` is the one-tailed selection that makes a figure one-signed,
    and therefore drawn in a single-hue ramp rather than a diverging map.
    ``absolute`` takes ``|v|`` after that, so a region that is strongly negative
    ranks with the strongly positive ones instead of at the bottom.
    ``threshold`` drops voxels with ``|v| < threshold`` *before* the reduction,
    which is what makes ``"max"`` and ``"mean_nonzero"`` mean what you want on
    an unthresholded t-map.
    """
    if stat not in STATS:
        raise ValueError(f"stat must be one of {STATS}, got {stat!r}")

    parcels = load_parcels(parcels)
    out = {}
    for name, mesh in parcels.items():
        pts = (interiors or {}).get(name)
        if pts is None:
            pts = _interior_points(mesh, step=step)
        if not len(pts):
            out[name] = np.nan
            continue
        v = _select(sample_nifti(pts, nifti, order=order), sign, absolute, threshold)
        out[name] = _reduce(v[np.isfinite(v)], stat)
    return out


def _reduce(v, stat):
    if not len(v):
        return np.nan
    if stat == "max":
        return float(v[np.argmax(np.abs(v))])       # signed peak
    if stat == "min":
        return float(v[np.argmin(np.abs(v))])
    if stat == "mean":
        return float(np.mean(v))
    if stat == "sum":
        return float(np.sum(v))
    nz = v[v != 0]
    return float(np.mean(nz)) if len(nz) else np.nan


def select_regions(values, min_value=None, keep_top=None):
    """NaN out regions that do not clear a floor, or are outside the top N.

    ``threshold`` in ``parcel_stats`` drops *voxels* before the reduction; this
    drops *regions* after it. They do different jobs and a parcel figure usually
    wants both: voxel thresholding decides what counts as signal, region
    selection decides what is worth colouring. Without the second one, a "max"
    figure floods every region that contains a single suprathreshold voxel,
    which on a whole-brain map is all of them.
    """
    out = dict(values)
    if min_value is not None:
        out = {k: (v if np.isfinite(v) and abs(v) >= float(min_value) else np.nan)
               for k, v in out.items()}
    if keep_top is not None:
        ranked = sorted((k for k in out if np.isfinite(out[k])),
                        key=lambda k: -abs(out[k]))[:int(keep_top)]
        keep = set(ranked)
        out = {k: (v if k in keep else np.nan) for k, v in out.items()}
    return out


def paint_parcels(parcels, nifti=None, values=None, stat="max", sign="both",
                  absolute=False, threshold=None, min_value=None, keep_top=None,
                  step=1.0, scalar="value", interiors=None):
    """One merged ``PolyData`` of all regions, flooded with their statistic.

    Pass ``nifti`` to compute the statistic, or ``values`` to supply a
    ``{region: number}`` mapping you already have.  Regions with no value come
    back as NaN and are drawn in the base colour.

    The scalar is attached per cell, not per point: regions touch, and a point
    scalar would smear one region's value across the seam into its neighbour.
    """
    parcels = load_parcels(parcels)
    if values is None:
        if nifti is None:
            raise ValueError("paint_parcels needs either a nifti or a values mapping")
        values = parcel_stats(parcels, nifti, stat=stat, sign=sign,
                              absolute=absolute, threshold=threshold, step=step,
                              interiors=interiors)
    values = select_regions(values, min_value=min_value, keep_top=keep_top)

    meshes, scalars = [], []
    for name, mesh in parcels.items():
        v = float(values.get(name, np.nan))
        meshes.append(mesh)
        scalars.append(np.full(mesh.n_cells, v, dtype=float))

    merged = pv.merge(meshes) if len(meshes) > 1 else meshes[0].copy()
    merged[scalar] = np.concatenate(scalars)
    return merged
