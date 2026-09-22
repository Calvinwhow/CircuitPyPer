#!/usr/bin/env python3
"""Batch figures through Circuit Viewer.

``neuro_plotter`` is deliberately an adapter, not a second renderer.  Each
entry in :data:`FIGURES` describes mesh groups, the NIfTI overlays painted on
each mesh, and independent fiber groups.  The description is translated into a
Circuit Viewer scene, saved as ``scene.json``, and rendered through one cached
``RenderSession`` into individual SVG views and a combined SVG panel.

The previous yabplot implementation is preserved beside this file as
``neuro_plotter.py.bak``.
"""

from __future__ import annotations

import argparse
import base64
import copy
import gc
import html
import io
import json
import math
import os
import re
import sys
import webbrowser
from collections.abc import Mapping
from pathlib import Path

import nibabel as nib


# Make the source checkout usable before circuit-viewer is formally installed
# into this environment.  An installed package always wins.
CIRCUIT_PYPER_DIR = Path(__file__).resolve().parents[1]
if str(CIRCUIT_PYPER_DIR) not in sys.path:
    sys.path.insert(0, str(CIRCUIT_PYPER_DIR))

try:
    from circuit_viewer.session import RenderSession
    from circuit_viewer.spec import (
        new_fiber_layer,
        new_mesh_layer,
        new_overlay,
        new_spec,
        save_spec,
    )
except ModuleNotFoundError as exc:
    if exc.name != "circuit_viewer":
        raise
    viewer_root = Path(os.environ.get(
        "CIRCUIT_VIEWER_ROOT",
        CIRCUIT_PYPER_DIR.parent.parent / "circuit_viewer",
    )).expanduser()
    if not (viewer_root / "circuit_viewer" / "__init__.py").is_file():
        raise ModuleNotFoundError(
            "circuit-viewer is required. Install it into this environment or "
            "set CIRCUIT_VIEWER_ROOT to its source checkout."
        ) from exc
    sys.path.insert(0, str(viewer_root))
    from circuit_viewer.session import RenderSession
    from circuit_viewer.spec import (
        new_fiber_layer,
        new_mesh_layer,
        new_overlay,
        new_spec,
        save_spec,
    )


# =============================================================================
# 1. USER CONFIGURATION
# =============================================================================

NIFTI_PATH = Path(
    "/Volumes/OneTouch/01p_Schmahmann_SCA_Atrophy/results/optimization/"
    "symptom_on_lhs/network_regressions_clusters/"
    "cluster_regression_identity_standardized/"
    "Nifti_File_Path-on-cluster_motor-cluster_cognitive-cluster_emotional/"
    "regression/contrast_tval_FWE_0.nii.gz"
)
OUTPUT_DIR = NIFTI_PATH.parent / "neuro_plots"

DEFAULT_CMAP = "xrain"
XRAIN_PATH = CIRCUIT_PYPER_DIR / "resources" / "colour_luts" / "x_rain.clut"

VIEWS = [
    "left_lateral",
    "right_lateral",
    "left_medial",
    "right_medial",
    "anterior",
    "posterior",
    "superior",
    "inferior",
]
FORMATS = ["svg"]
EXPORT_SIZE = (1700, 1400)
PANEL_COLUMNS = 4
MAKE_PANEL = True
MAKE_INDIVIDUAL_VIEWS = True
MAKE_HTML = True
OPEN_HTML = True
SCALAR_BARS = True


# One figure is one Circuit Viewer scene.  A mesh group owns its NIfTI overlay
# stack; fibers are sibling layers and never belong to a mesh.  ``$source`` is
# replaced with the file passed to dispatch().
FIGURES = [
    {
        "name": "parcel_max",
        "title": "Regional maximum",
        "when": "nifti",
        "meshes": [
            {
                "mesh": "aal3_suit_parcels",
                "niftis": [
                    {
                        "path": "$source",
                        "name": "Source map",
                        "mode": "region",       # region | vertex
                        "stat": "max",          # max | nanmean | mean | sum
                        "cmap": DEFAULT_CMAP,
                        "threshold": 0,
                        "clim": None,
                    }
                ],
            }
        ],
        "fibers": [],
    },
    {
        "name": "fiber_values",
        "title": "Fiber values",
        "when": "fiber",
        "meshes": [
            {
                "mesh": "pial_wholebrain",
                "color": "#8d99ae",
                "opacity": 0.16,
                "cull_near_hemisphere": True,
                "niftis": [],
            }
        ],
        "fibers": [
            {
                "source": "$source",
                "name": "Fibers",
                "coloring": "value",           # value | direction | solid
                "cmap": DEFAULT_CMAP,
                "opacity": 1.0,
            }
        ],
    },
]


# =============================================================================
# 2. SCENE TRANSLATION
# =============================================================================

MESH_FIELDS = (
    "name", "iso", "split", "presmooth_vox", "color", "opacity",
    "faceted", "edges", "tessellate", "edge_color", "edge_width",
    "edge_opacity", "show_left", "show_right", "hidden_regions", "section",
    "section_cap", "cull_near_hemisphere",
)

FIBER_FIELDS = (
    "name", "sign", "min_abs_value", "max_abs_value", "top_percent",
    "max_lines", "step", "smooth", "tube_sides", "tube_radius", "coloring",
    "color", "opacity", "clim",
)

OVERLAY_FIELDS = (
    "sign", "absolute", "threshold", "max_value", "symmetric",
    "hide_below_min", "opacity", "min_value", "keep_top", "step",
)

STAT_ALIASES = {
    "max": "max",
    "max_in_roi": "max",
    "nanmean": "mean",
    "mean": "mean",
    "avg": "mean",
    "average": "mean",
    "mean_nonzero": "mean_nonzero",
    "avg_in_target": "mean_nonzero",
    "sum": "sum",
    "min": "min",
}

VIEW_CAMERAS = {
    "left_lateral": {"azimuth": 90.0, "elevation": 0.0},
    "right_lateral": {"azimuth": 270.0, "elevation": 0.0},
    # The intervening hemisphere is hidden by _scene_for_view().
    "left_medial": {"azimuth": 270.0, "elevation": 0.0},
    "right_medial": {"azimuth": 90.0, "elevation": 0.0},
    "anterior": {"azimuth": 0.0, "elevation": 0.0},
    "posterior": {"azimuth": 180.0, "elevation": 0.0},
    "superior": {"azimuth": 0.0, "elevation": 89.9},
    "inferior": {"azimuth": 0.0, "elevation": -89.9},
}

VIEW_ALIASES = {
    "left": "left_lateral",
    "right": "right_lateral",
    "lat_l": "left_lateral",
    "lat_r": "right_lateral",
    "med_l": "left_medial",
    "med_r": "right_medial",
    "ant": "anterior",
    "post": "posterior",
    "sup": "superior",
    "inf": "inferior",
}

VIEW_LABELS = {
    name: name.replace("_", " ").title() for name in VIEW_CAMERAS
}


def nifti_stem(path):
    """A stable stem for NIfTI and canonical fiber result names."""
    name = Path(path).name
    for suffix in (
        ".fib.values.npy", ".fib.desc.json", ".fib.npy", ".fib.json", ".values.npy",
        ".nii.gz", ".nii",
    ):
        if name.lower().endswith(suffix):
            return name[:-len(suffix)]
    return Path(name).stem


def input_kind(path):
    name = Path(path).name.lower()
    if name.endswith((".nii", ".nii.gz")):
        return "nifti"
    if name.endswith((
        ".mat", ".tck", ".trk", ".fib.npy", ".fib.json",
        ".fib.values.npy", ".fib.desc.json", ".values.npy", ".npz",
    )):
        return "fiber"
    raise ValueError(
        "Input must be a NIfTI or fiber result "
        "(.mat/.tck/.trk/.fib.npy/.fib.json/legacy .fib.values.npy/.fib.desc.json/.npz): "
        f"{path}"
    )


def _slug(value):
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return text.strip("._") or "figure"


def _resolve_path(value, source, label="file"):
    if value in (None, "", "$source"):
        path = Path(source)
    else:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path(source).parent / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return str(path)


def _mesh_reference(value, source):
    """Resolve paths while leaving registered Circuit Viewer mesh names alone."""
    if isinstance(value, (list, tuple)):
        return [_mesh_reference(item, source) for item in value]
    if value in (None, "", "$source"):
        return _resolve_path(value, source, "mesh")
    text = str(value)
    expanded = Path(text).expanduser()
    path_like = expanded.is_absolute() or "/" in text or text.startswith(".")
    if path_like:
        return _resolve_path(text, source, "mesh")
    return text


def _resolve_cmap(value):
    value = DEFAULT_CMAP if value in (None, "") else value
    normalized = re.sub(r"[^a-z0-9]", "", str(value).lower())
    if normalized == "xrain" and XRAIN_PATH.is_file():
        return str(XRAIN_PATH.resolve())
    return value


def _choice(figure, item, parent, key, default=None):
    """Figure-wide CLI/config values override item and parent defaults."""
    if figure.get(key) is not None:
        return figure[key]
    if item.get(key) is not None:
        return item[key]
    if parent.get(key) is not None:
        return parent[key]
    return default


def _normal_stat(value):
    key = str(value or "max").lower()
    if key not in STAT_ALIASES:
        raise ValueError(
            f"unknown flooding statistic {value!r}; choose max, nanmean, mean, "
            "mean_nonzero, min, or sum"
        )
    return STAT_ALIASES[key]


def _normal_mode(value):
    value = str(value or "region").lower()
    aliases = {"flood": "region", "flooding": "region", "vertices": "vertex"}
    value = aliases.get(value, value)
    if value not in {"region", "vertex"}:
        raise ValueError("mode must be 'region' (flooding) or 'vertex'")
    return value


def _overlay_entry(raw):
    if isinstance(raw, (str, os.PathLike)):
        return {"path": str(raw)}
    if not isinstance(raw, Mapping):
        raise TypeError("every NIfTI overlay must be a path or a dictionary")
    return dict(raw)


def _build_overlay(raw, parent, figure, source, fiber_values=False):
    item = _overlay_entry(raw)
    path = _resolve_path(
        item.get("path", item.get("nifti", "$source")),
        source,
        "fiber values" if fiber_values else "NIfTI overlay",
    )
    if not fiber_values and input_kind(path) != "nifti":
        raise ValueError(f"mesh overlays must be NIfTIs: {path}")

    threshold = _choice(figure, item, parent, "threshold")
    max_value = _choice(figure, item, parent, "max_value")
    if isinstance(threshold, (list, tuple)):
        if len(threshold) == 1:
            threshold = threshold[0]
        elif len(threshold) == 2:
            threshold, max_value = threshold
        else:
            raise ValueError("threshold accepts one value or [minimum, maximum]")

    clim = _choice(figure, item, parent, "clim")
    if clim is None:
        clim = _choice(figure, item, parent, "vminmax")
    if clim is not None:
        if not isinstance(clim, (list, tuple)) or len(clim) != 2:
            raise ValueError("clim must be [minimum, maximum]")
        clim = [float(clim[0]), float(clim[1])]

    kwargs = {
        "nifti": path,
        "name": item.get("name", item.get("label", nifti_stem(path))),
        "mode": _normal_mode(_choice(figure, item, parent, "mode", "region")),
        "stat": _normal_stat(_choice(figure, item, parent, "stat", "max")),
        "palette": _resolve_cmap(_choice(
            figure, item, parent, "cmap",
            _choice(figure, item, parent, "palette", DEFAULT_CMAP),
        )),
        "color": _choice(figure, item, parent, "color", "#c15656"),
        "clim": clim,
    }
    for field in OVERLAY_FIELDS:
        value = _choice(figure, item, parent, field)
        if value is not None:
            kwargs[field] = value
    if "alpha" in item and "opacity" not in kwargs:
        kwargs["opacity"] = item["alpha"]
    return new_overlay(**kwargs)


def _mesh_entries(figure):
    entries = figure.get("meshes")
    if entries is None and figure.get("mesh") is not None:
        entries = [{
            "mesh": figure["mesh"],
            "niftis": figure.get("niftis", ["$source"]),
        }]
    if entries is None:
        return []
    if isinstance(entries, Mapping):
        entries = [entries]
    return list(entries)


def _build_mesh(raw, figure, source):
    if isinstance(raw, (str, os.PathLike)):
        item = {"mesh": str(raw), "niftis": []}
    elif isinstance(raw, Mapping):
        item = dict(raw)
    else:
        raise TypeError("every mesh must be a name/path or a dictionary")
    if "mesh" not in item:
        raise ValueError("every mesh entry requires 'mesh'")

    niftis = item.get("niftis", [])
    if item.get("nifti") is not None:
        niftis = [item["nifti"], *list(niftis or [])]
    if isinstance(niftis, (str, os.PathLike, Mapping)):
        niftis = [niftis]
    overlays = [
        _build_overlay(entry, item, figure, source) for entry in (niftis or [])
    ]

    kwargs = {
        "mesh": _mesh_reference(item["mesh"], source),
        "overlays": overlays,
    }
    for field in MESH_FIELDS:
        if item.get(field) is not None:
            kwargs[field] = item[field]
    return new_mesh_layer(**kwargs)


def _fiber_entries(figure):
    entries = figure.get("fibers") or []
    if isinstance(entries, (str, os.PathLike, Mapping)):
        entries = [entries]
    return list(entries)


def _build_fiber(raw, figure, source):
    if isinstance(raw, (str, os.PathLike)):
        item = {"source": str(raw)}
    elif isinstance(raw, Mapping):
        item = dict(raw)
    else:
        raise TypeError("every fiber entry must be a path or a dictionary")

    fiber_source = item.get("source", item.get("json", "$source"))
    kwargs = {
        "source": _resolve_path(fiber_source, source, "fiber source"),
        "palette": _resolve_cmap(_choice(
            figure, item, {}, "cmap",
            _choice(figure, item, {}, "palette", DEFAULT_CMAP),
        )),
    }
    atlas = item.get("fiber_atlas_path")
    if atlas:
        kwargs["fiber_atlas_path"] = _resolve_path(atlas, source, "fiber atlas")
    for field in FIBER_FIELDS:
        if item.get(field) is not None:
            kwargs[field] = item[field]
    coloring = str(kwargs.get("coloring", "value")).lower()
    kwargs["coloring"] = {
        "orientation": "direction", "orient": "direction",
    }.get(coloring, coloring)

    overlays = item.get("overlays") or []
    if isinstance(overlays, (str, os.PathLike, Mapping)):
        overlays = [overlays]
    kwargs["overlays"] = [
        _build_overlay(entry, item, figure, source, fiber_values=True)
        for entry in overlays
    ]
    return new_fiber_layer(**kwargs)


def build_scene(source, figure):
    """Translate one FIGURES entry into a validated Circuit Viewer scene."""
    source = Path(source).expanduser().resolve()
    layers = [
        _build_mesh(entry, figure, source) for entry in _mesh_entries(figure)
    ]
    layers.extend(
        _build_fiber(entry, figure, source) for entry in _fiber_entries(figure)
    )
    if not layers:
        raise ValueError(
            f"figure {figure.get('name', '<unnamed>')!r} has no meshes or fibers"
        )

    scene = new_spec(layers)
    for field in ("background", "center", "distance", "lighting", "parallel"):
        if figure.get(field) is not None:
            scene[field] = copy.deepcopy(figure[field])
    if figure.get("camera"):
        scene["camera"].update(figure["camera"])
    if figure.get("zoom") is not None:
        scene["camera"]["zoom"] = float(figure["zoom"])
    plot_kwargs = figure.get("plot_kwargs") or {}
    if plot_kwargs.get("zoom") is not None:
        scene["camera"]["zoom"] = float(plot_kwargs["zoom"])
    return scene


# =============================================================================
# 3. SVG VIEWS AND PANELS
# =============================================================================

def _normal_views(views):
    result = []
    for raw in views or VIEWS:
        name = VIEW_ALIASES.get(str(raw).lower(), str(raw).lower())
        if name not in VIEW_CAMERAS:
            raise ValueError(
                f"unknown view {raw!r}; choose from {list(VIEW_CAMERAS)}"
            )
        if name not in result:
            result.append(name)
    return result


def _scene_for_view(scene, view):
    out = copy.deepcopy(scene)
    if view not in {"left_medial", "right_medial"}:
        return out
    keep_left = view == "left_medial"
    for layer in out["layers"]:
        if layer.get("kind") == "mesh":
            layer["show_left"] = keep_left
            layer["show_right"] = not keep_left
    return out


def _svg_bytes(png, size, title=None):
    width, height = map(int, size)
    encoded = base64.b64encode(png).decode("ascii")
    title_tag = f"<title>{html.escape(str(title))}</title>" if title else ""
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">\n'
        f'{title_tag}<image width="{width}" height="{height}" '
        f'href="data:image/png;base64,{encoded}"/>\n</svg>\n'
    ).encode("utf-8")


def _write_render(path, png, size, title=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".svg":
        path.write_bytes(_svg_bytes(png, size, title=title))
    elif suffix == ".png":
        path.write_bytes(png)
    elif suffix in {".jpg", ".jpeg"}:
        from PIL import Image

        Image.open(io.BytesIO(png)).convert("RGB").save(path, quality=95)
    else:
        raise ValueError("formats must be svg, png, jpg, or jpeg")
    return path


def _panel_bytes(tiles, size, columns=4, background="#ffffff", title=None):
    columns = max(1, min(int(columns), len(tiles)))
    rows = int(math.ceil(len(tiles) / columns))
    tile_w, tile_h = map(int, size)
    label_height, gap = 54, 18
    cell_h = label_height + tile_h
    width = columns * tile_w + (columns - 1) * gap
    height = rows * cell_h + (rows - 1) * gap
    title_tag = f"<title>{html.escape(str(title))}</title>" if title else ""
    parts = []
    for index, (name, png) in enumerate(tiles):
        row, column = divmod(index, columns)
        x = column * (tile_w + gap)
        y = row * (cell_h + gap)
        encoded = base64.b64encode(png).decode("ascii")
        label = html.escape(VIEW_LABELS[name])
        parts.append(
            f'<text x="{x + tile_w / 2:g}" y="{y + 36:g}" '
            'text-anchor="middle" font-family="Arial,Helvetica,sans-serif" '
            f'font-size="30" fill="#252a31">{label}</text>\n'
            f'<image x="{x}" y="{y + label_height}" width="{tile_w}" '
            f'height="{tile_h}" href="data:image/png;base64,{encoded}"/>\n'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">\n'
        f'{title_tag}<rect width="100%" height="100%" '
        f'fill="{html.escape(str(background))}"/>\n'
        + "".join(parts) + "</svg>\n"
    ).encode("utf-8")


def _quantitative_overlay_count(scene):
    return sum(
        len(layer.get("overlays") or [])
        for layer in scene["layers"]
        if layer.get("kind") in {"mesh", "fibers"}
    ) + sum(
        1 for layer in scene["layers"]
        if layer.get("kind") == "fibers"
        and layer.get("coloring") == "value"
        and not layer.get("overlays")
    )


def render_figure(source, output_dir, figure, session=None):
    """Render one scene, returning gallery cards and the reusable session."""
    name = _slug(figure.get("name") or "figure")
    title = figure.get("title", name.replace("_", " ").title())
    figure_dir = Path(output_dir) / name
    figure_dir.mkdir(parents=True, exist_ok=True)

    scene = build_scene(source, figure)
    project_path = Path(save_spec(scene, figure_dir / "scene.json"))
    if session is None:
        session = RenderSession(scene)
    else:
        session.set_spec(scene)

    size = tuple(figure.get("size") or EXPORT_SIZE)
    if len(size) != 2:
        raise ValueError("figure size must be [width, height]")
    views = _normal_views(figure.get("views", VIEWS))
    formats = [str(x).lower().lstrip(".") for x in figure.get("formats", FORMATS)]
    individual = bool(figure.get("individual_views", MAKE_INDIVIDUAL_VIEWS))
    make_panel = bool(figure.get("panel", MAKE_PANEL))
    scalar_bars = bool(figure.get("scalar_bars", SCALAR_BARS))

    if scalar_bars and _quantitative_overlay_count(scene) > 1:
        print(
            f"[warn] {name}: Circuit Viewer currently draws one scalar bar; "
            "multi-overlay colours remain correct, but only one scale can be shown."
        )

    tiles = []
    view_files = {}
    try:
        for view in views:
            session.set_spec(_scene_for_view(scene, view))
            png = session.render(
                size=size,
                quality="export",
                scalar_bar=scalar_bars,
                camera={**VIEW_CAMERAS[view], "zoom": scene["camera"]["zoom"]},
                antialias="ssaa",
            )
            tiles.append((view, png))
            files = {}
            if individual:
                for extension in formats:
                    path = figure_dir / f"{view}.{extension}"
                    files[extension] = _write_render(
                        path, png, size, title=f"{title} — {VIEW_LABELS[view]}"
                    )
            view_files[view] = files
    finally:
        session.set_spec(scene)

    cards = []
    if make_panel:
        panel_path = figure_dir / "all_views.svg"
        panel_path.write_bytes(_panel_bytes(
            tiles,
            size,
            columns=figure.get("panel_columns", PANEL_COLUMNS),
            background=scene.get("background", "#ffffff"),
            title=title,
        ))
        cards.append({
            "title": title,
            "files": {"svg": panel_path, "project": project_path},
        })
    if individual:
        cards.extend({
            "title": f"{title} — {VIEW_LABELS[view]}",
            "files": files,
        } for view, files in view_files.items())
    if not cards:
        cards.append({"title": title, "files": {"project": project_path}})
    return cards, session


# =============================================================================
# 4. HTML INDEXES AND PIPELINE-COMPATIBLE DISPATCH
# =============================================================================

def write_volume_viewer(nifti_paths, output_path, title="Interactive volumetric maps"):
    """Write one self-contained nilearn viewer with a map selector."""
    from nilearn import plotting
    from calvin_utils.plotting_utils.html_viewer_selector import HTMLViewerSelector

    named_paths = (
        nifti_paths.items()
        if isinstance(nifti_paths, Mapping)
        else ((None, path) for path in nifti_paths)
    )
    views = {}
    for supplied_label, nifti_path in named_paths:
        nifti_path = Path(nifti_path).expanduser().resolve()
        if not nifti_path.is_file() or input_kind(nifti_path) != "nifti":
            print(f"[warn] volumetric viewer skipped non-NIfTI input: {nifti_path}")
            continue
        try:
            image = nib.load(str(nifti_path))
            if len(image.shape) != 3:
                print(
                    f"[warn] volumetric viewer skipped non-3D NIfTI: "
                    f"{nifti_path} (shape {image.shape})"
                )
                continue
            label = str(supplied_label) if supplied_label is not None \
                else nifti_stem(nifti_path)
            unique = label
            duplicate = 2
            while unique in views:
                unique = f"{label} ({duplicate})"
                duplicate += 1
            views[unique] = plotting.view_img(image, title=unique)
        except Exception as exc:
            print(f"[warn] volumetric viewer could not load {nifti_path}: {exc}")
    if not views:
        return None

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    HTMLViewerSelector(views, title=title).save(str(output_path))
    return output_path


def _relative_href(target, output_dir):
    target = Path(target).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    return Path(os.path.relpath(target, start=output_dir)).as_posix()


def write_index(output_dir, source_nifti, cards, volume_viewer=None, title=None,
                gallery_links=None):
    """Write the local gallery consumed by the regression pipelines."""
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    page_title = title or f"{nifti_stem(source_nifti)} figures"
    blocks = []

    for card in cards:
        files = card.get("files", {})
        preview = files.get("svg") or files.get("png")
        if preview is None:
            continue
        preview_rel = _relative_href(preview, output_dir)
        links = []
        for extension, path in files.items():
            if not Path(path).is_file():
                continue
            rel = _relative_href(path, output_dir)
            label = "PROJECT" if extension == "project" else extension.upper()
            links.append(f'<a href="{html.escape(rel, quote=True)}">{label}</a>')
        blocks.append(
            '<div class="card">'
            f'<h2>{html.escape(str(card["title"]))}</h2>'
            f'<a href="{html.escape(preview_rel, quote=True)}">'
            f'<img src="{html.escape(preview_rel, quote=True)}" loading="lazy"></a>'
            f'<p>{" · ".join(links)}</p></div>'
        )

    viewer_block = ""
    if volume_viewer is not None and Path(volume_viewer).is_file():
        viewer_rel = _relative_href(volume_viewer, output_dir)
        viewer_block = (
            '<section class="viewer-card"><h2>Interactive volumetric viewer</h2>'
            f'<iframe src="{html.escape(viewer_rel, quote=True)}" '
            'title="Interactive volumetric viewer"></iframe></section>'
        )

    linked = []
    for label, target in gallery_links or []:
        if Path(target).is_file():
            rel = _relative_href(target, output_dir)
            linked.append(
                f'<li><a href="{html.escape(rel, quote=True)}">'
                f'{html.escape(str(label))}</a></li>'
            )
    links_block = (
        '<section class="card"><h2>Static figure galleries</h2><ul>'
        + "".join(linked) + "</ul></section>"
        if linked else ""
    )
    figures_block = (
        '<h2 class="section-title">Rendered figures</h2><div class="grid">'
        + "".join(blocks) + "</div>"
        if blocks else ""
    )
    page = f'''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(page_title)}</title>
<style>
body{{font-family:system-ui;margin:36px;max-width:1800px;background:#f4f5f7;color:#17212c}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}}
.card,.viewer-card{{background:white;padding:20px;border-radius:12px;margin-bottom:20px}}
.viewer-card iframe{{display:block;width:100%;height:760px;border:1px solid #d7dce2;border-radius:8px}}
.section-title{{margin-top:30px}} img{{width:100%;background:white}}
a{{color:#235c8c}} @media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>
<h1>{html.escape(page_title)}</h1>
{viewer_block}{links_block}{figures_block}
</body></html>'''
    index_path = output_dir / "index.html"
    index_path.write_text(page, encoding="utf-8")
    return index_path


def discover_rendered_cards(*roots):
    cards = {}
    for root in roots:
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            extension = path.suffix.lower().lstrip(".")
            if not path.is_file() or extension not in {"svg", "png", "jpg", "jpeg"}:
                continue
            key = path.with_suffix("")
            card = cards.setdefault(key, {
                "title": " / ".join(
                    part.replace("_", " ").title()
                    for part in path.relative_to(root).with_suffix("").parts
                ),
                "files": {},
            })
            card["files"][extension] = path
            project = path.parent / "scene.json"
            if project.is_file():
                card["files"]["project"] = project
    return list(cards.values())


def write_collection_index(output_dir, title, figure_roots, volume_niftis=None):
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cards = discover_rendered_cards(*figure_roots)
    volume_niftis = volume_niftis or []
    volume_paths = (
        list(volume_niftis.values())
        if isinstance(volume_niftis, Mapping)
        else list(volume_niftis)
    )
    volume_viewer = None
    if volume_paths:
        volume_viewer = write_volume_viewer(
            volume_niftis,
            output_dir / "volumetric_viewer.html",
            title=f"{title} volumes",
        )
    source = volume_paths[0] if volume_paths else output_dir
    path = write_index(
        output_dir, source, cards, volume_viewer=volume_viewer, title=title
    )
    print(f"Collection index: {path} ({len(cards)} rendered file sets)")
    return path


def _applies(figure, source):
    wanted = str(figure.get("when", "any")).lower()
    return wanted in {"any", "all", input_kind(source)}


def _with_overrides(figure, overrides):
    result = copy.deepcopy(figure)
    for key, value in (overrides or {}).items():
        if value is not None:
            result[key] = copy.deepcopy(value)
    plot_kwargs = result.get("plot_kwargs") or {}
    if plot_kwargs.get("color") is not None and result.get("color") is None:
        result["color"] = plot_kwargs["color"]
    if plot_kwargs.get("zoom") is not None and result.get("zoom") is None:
        result["zoom"] = plot_kwargs["zoom"]
    if result.get("mesh") is not None and result.get("meshes"):
        result["meshes"][0]["mesh"] = result["mesh"]
    if result.get("size") is None and (
        result.get("width") is not None or result.get("height") is not None
    ):
        result["size"] = (
            int(result.get("width", EXPORT_SIZE[0])),
            int(result.get("height", EXPORT_SIZE[1])),
        )
    return result


def _close_session(session):
    if session is None:
        return
    close = getattr(session, "close", None)
    if callable(close):
        close()
        return
    # Current viewer releases its plotter at process exit. close_all bounds GL
    # state for callers that deliberately run several dispatches in one process.
    try:
        import pyvista as pv

        pv.close_all()
    except Exception:
        pass
    del session
    gc.collect()


def dispatch(source_nifti, output_dir, figures=None, overrides=None,
             make_html=MAKE_HTML, open_html=OPEN_HTML, volume_viewer=None):
    """Render every applicable FIGURES entry for one source file."""
    source = Path(source_nifti).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        raise FileNotFoundError(source)

    candidates = figures if figures is not None else FIGURES
    selected = [
        _with_overrides(figure, overrides)
        for figure in candidates
        if _applies(figure, source)
    ]
    cards, session = [], None
    try:
        for figure in selected:
            print(f"Rendering: {figure.get('name', 'figure')}")
            rendered, session = render_figure(
                source, output_dir, figure, session=session
            )
            cards.extend(rendered)
    finally:
        _close_session(session)

    if not make_html:
        return cards
    if volume_viewer is None and input_kind(source) == "nifti":
        volume_viewer = write_volume_viewer(
            [source], output_dir / "volumetric_viewer.html",
            title=f"{nifti_stem(source)} volumetric view",
        )
    index_path = write_index(
        output_dir, source, cards, volume_viewer=volume_viewer
    )
    print(f"HTML viewer: {index_path}")
    if open_html:
        webbrowser.open(index_path.as_uri())
    return index_path


# =============================================================================
# 5. CLI
# =============================================================================

def parser():
    result = argparse.ArgumentParser(
        description="Render configured mesh/NIfTI/fiber scenes with Circuit Viewer."
    )
    result.add_argument("--nifti", dest="nifti_path")
    result.add_argument("--output-dir")
    result.add_argument("--figures-json", help="JSON file containing a FIGURES list")
    result.add_argument("--mesh", help="replace the first mesh in every figure")
    result.add_argument("--mode", choices=("region", "vertex", "flood", "flooding"))
    result.add_argument("--stat")
    result.add_argument("--cmap")
    result.add_argument("--threshold", type=float)
    result.add_argument("--clim", "--vminmax", nargs=2, type=float)
    result.add_argument("--views", nargs="+")
    result.add_argument("--formats", nargs="+")
    result.add_argument("--width", type=int)
    result.add_argument("--height", type=int)
    result.add_argument("--panel-columns", type=int)
    result.add_argument("--volume-viewer")
    result.add_argument("--no-panel", dest="panel", action="store_false")
    result.add_argument(
        "--no-individual-views", dest="individual_views", action="store_false"
    )
    result.add_argument(
        "--no-scalar-bars", dest="scalar_bars", action="store_false"
    )
    result.add_argument("--no-html", dest="make_html", action="store_false")
    result.add_argument("--no-open-html", dest="open_html", action="store_false")
    result.add_argument("--list", action="store_true")
    result.set_defaults(
        panel=None, individual_views=None, scalar_bars=None,
        make_html=None, open_html=None,
    )
    return result


def _load_figures(path):
    value = json.loads(Path(path).expanduser().read_text())
    if isinstance(value, Mapping):
        value = value.get("figures", [value])
    if not isinstance(value, list) or not all(isinstance(x, Mapping) for x in value):
        raise ValueError("--figures-json must contain a figure object or a list")
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    if args.list:
        print(json.dumps(FIGURES, indent=2, default=str))
        return 0

    source = Path(args.nifti_path or NIFTI_PATH).expanduser()
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else source.parent / f"{nifti_stem(source)}_figures"
    )
    figures = _load_figures(args.figures_json) if args.figures_json else FIGURES

    overrides = {}
    for key in (
        "mesh", "mode", "stat", "cmap", "threshold", "clim", "views",
        "formats", "width", "height", "panel_columns", "panel",
        "individual_views", "scalar_bars",
    ):
        value = getattr(args, key)
        if value is not None:
            overrides[key] = value

    dispatch(
        source_nifti=source,
        output_dir=output_dir,
        figures=figures,
        overrides=overrides,
        make_html=MAKE_HTML if args.make_html is None else args.make_html,
        open_html=OPEN_HTML if args.open_html is None else args.open_html,
        volume_viewer=args.volume_viewer,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
