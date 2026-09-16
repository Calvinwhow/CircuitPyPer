"""Render Lead-DBS / discriminative-fiber results inside the composite glass brain.

This is the standalone renderer that produced the SCA fibre figures.  It does
not go through yabplot: it draws the streamlines itself with PyVista so that
line width, tube shading, hemisphere culling and camera placement are all under
direct control, and it is one layer type in
:class:`calvin_utils.plotting_utils.brain_viewer.BrainScene`, so streamlines can
be drawn inside the same backdrop -- and alongside the same painted meshes -- as
everything else.

Typical use::

    from calvin_utils.plotting_utils.fiber_render import load_fibers, panel, hero

    fibers = load_fibers(
        ".../contrast_pval_FWE_0_positive_ftr.mat",
        max_abs_value=0.05,        # p-value map: keep the significant fibres
        max_lines=30000,
    )
    panel(fibers, "fibers_panel.png")
    hero(fibers, "fibers_left.png", view="left")

or, mixed with anything else::

    BrainScene().surface("pial_wholebrain", glass=True).fibers(fibers).render(out)

Value filtering
---------------
``sign`` / ``min_abs_value`` / ``top_percent`` mirror
``FiberFormatConverter._fiber_value_mask``.  ``max_abs_value`` is the extra one
this module adds, because a ``contrast_pval_*`` MAT stores p-values, where the
fibres you want are the *small* ones -- ``max_abs_value=0.05`` is the p<0.05 set.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv

from calvin_utils.plotting_utils.render_scene import GLASS_ALPHA, RED

__all__ = ["load_fibers", "panel", "hero", "FIBER_COLOR", "LINE_WIDTH"]

FIBER_COLOR = RED
LINE_WIDTH = 1.5


# Loading ------------------------------------------------------------------
def load_fibers(
    path,
    fiber_atlas_path=None,
    sign="both",
    min_abs_value=None,
    max_abs_value=None,
    top_percent=None,
    max_lines=None,
    step=2,
    seed=0,
):
    """Read a ``.mat`` / ``.fib.npy`` fibre result into a line ``pv.PolyData``.

    ``step`` keeps every Nth vertex of each streamline (Lead-DBS geometry is
    ~0.5 mm-spaced, so ``step=2`` is ~1 mm and invisible at figure scale while
    halving the point count).  ``max_lines`` randomly subsamples streamlines;
    past ~30k the render stops looking denser and only gets slower.

    The returned mesh carries ``"vals"`` (per point, the streamline's value) and
    ``"dir"`` (per point uint8 RGB, the |direction| DTI convention).
    """
    from calvin_utils.neuroimaging_utils.tract_utils.fiber_converter import (
        FiberFormatConverter,
    )

    path = Path(path).expanduser()
    if path.name.lower().endswith(".fib.npy"):
        streamlines, values, _ = FiberFormatConverter.load_fib_npy(
            path, fiber_atlas_path=fiber_atlas_path, sign=sign,
            min_abs_value=min_abs_value, top_percent=top_percent,
        )
    else:
        streamlines, _, values = FiberFormatConverter.load_leaddbs_mat(path)
        if values is None:
            values = np.zeros(len(streamlines), dtype=np.float32)
        keep = FiberFormatConverter._fiber_value_mask(
            values, sign, min_abs_value, top_percent
        )
        streamlines = [s for s, k in zip(streamlines, keep) if k]
        values = np.asarray(values)[keep]

    if max_abs_value is not None:
        keep = np.abs(values) <= float(max_abs_value)
        if not keep.any():
            raise ValueError(f"no fibers survive max_abs_value={max_abs_value}")
        streamlines = [s for s, k in zip(streamlines, keep) if k]
        values = values[keep]

    order = np.arange(len(streamlines))
    if max_lines is not None and len(order) > max_lines:
        order = np.sort(np.random.default_rng(seed).choice(len(order), max_lines, False))

    pts, cells, vals, cursor = [], [], [], 0
    for i in order:
        f = np.asarray(streamlines[i], dtype=np.float32)
        d = f[::max(1, int(step))]
        if len(d) < 2:
            continue
        if not np.allclose(d[-1], f[-1]):
            d = np.vstack([d, f[-1]])
        n = len(d)
        pts.append(d)
        cells.append(np.r_[n, np.arange(cursor, cursor + n)])
        vals.append(np.full(n, values[i], dtype=np.float32))
        cursor += n

    poly = pv.PolyData()
    poly.points = np.concatenate(pts)
    poly.lines = np.concatenate(cells)
    poly["vals"] = np.concatenate(vals)
    poly["dir"] = _direction_rgb(poly)
    return poly


def _direction_rgb(poly):
    pts, lines = poly.points, poly.lines
    rgb = np.zeros_like(pts)
    i = 0
    while i < len(lines):
        n = lines[i]
        ids = lines[i + 1:i + 1 + n]
        seg = np.gradient(pts[ids], axis=0)
        rgb[ids] = np.abs(seg) / np.clip(np.linalg.norm(seg, axis=1, keepdims=True), 1e-6, None)
        i += 1 + n
    return (rgb * 255).astype(np.uint8)



# Convenience wrappers -----------------------------------------------------
# These exist because "draw these fibres in a glass brain" is by far the most
# common thing to want; anything more layered goes through BrainScene directly.
def _scene(poly, coloring, color, width, mesh, glass_alpha):
    from calvin_utils.plotting_utils.brain_viewer import BrainScene

    return (BrainScene()
            .surface(mesh, glass=True, opacity=glass_alpha)
            .fibers(poly, coloring=coloring, color=color, width=width))


def panel(poly, out, views=("left", "right", "anterior", "superior"),
          coloring="solid", color=FIBER_COLOR, width=LINE_WIDTH,
          mesh="pial_wholebrain", glass_alpha=GLASS_ALPHA, **render_kw):
    return _scene(poly, coloring, color, width, mesh, glass_alpha).render(
        out, views=views, **render_kw)


def hero(poly, out, view="left", coloring="solid", color=FIBER_COLOR,
         width=1.6, mesh="pial_wholebrain", glass_alpha=GLASS_ALPHA, zoom=1.12,
         **render_kw):
    return _scene(poly, coloring, color, width, mesh, glass_alpha).render(
        out, views=view, zoom=zoom, **render_kw)
