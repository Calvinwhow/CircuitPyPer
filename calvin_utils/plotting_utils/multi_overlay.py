"""Multi-overlay renderers used by neuro_plotter."""

from pathlib import Path

import nibabel as nib
import numpy as np
import pyvista as pv
from matplotlib.colors import is_color_like, to_rgb
from scipy.ndimage import gaussian_filter, map_coordinates
from yabplot.mesh import load_bmesh
from yabplot.scene import (
    add_context_to_view,
    finalize_plot,
    get_shading_preset,
    get_view_configs,
    prepare_plotter,
    set_camera,
    setup_plotter,
)


def _select(data, sign):
    if sign == "absolute":
        return np.abs(data)
    if sign == "positive":
        return np.maximum(data, 0)
    if sign == "negative":
        return np.maximum(-data, 0)
    raise ValueError("overlay sign must be 'absolute', 'positive', or 'negative'")


def _threshold(value, data):
    value = "95%" if value is None else value
    if isinstance(value, str):
        if not value.endswith("%"):
            raise ValueError("overlay threshold must be numeric or a percentile")
        nonzero = data[np.isfinite(data) & (data > 0)]
        if not nonzero.size:
            return None
        return float(np.percentile(nonzero, float(value[:-1])))
    value = float(value)
    if value < 0:
        raise ValueError("overlay threshold must be nonnegative")
    return value


def _load(overlays, default_alpha):
    loaded = []
    for overlay in overlays:
        item = dict(overlay)
        path = Path(item.get("path", "")).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"overlay NIfTI not found: {path}")
        if not is_color_like(item.get("color")):
            raise ValueError(f"invalid overlay color: {item.get('color')!r}")
        alpha = float(item.get("alpha", default_alpha))
        if not 0 < alpha <= 1:
            raise ValueError("overlay alpha must be in (0, 1]")
        image = nib.load(str(path))
        data = image.get_fdata()
        if data.ndim > 3:
            data = data[..., 0]
        data = np.nan_to_num(data, nan=0, posinf=0, neginf=0)
        sign = item.get("sign", "absolute")
        item.update(
            path=path,
            image=image,
            data=data,
            sign=sign,
            alpha=alpha,
            threshold_value=_threshold(item.get("threshold"), _select(data, sign)),
        )
        loaded.append(item)
    return loaded


def _load_tracts(overlays, default_alpha):
    import yabplot.plotting as yp

    loaded = []
    for index, overlay in enumerate(overlays):
        item = dict(overlay)
        path = Path(item.get("_tract_path", ""))
        if not path.is_file():
            raise FileNotFoundError(f"temporary tractogram not found: {path}")
        if not is_color_like(item.get("color")):
            raise ValueError(f"invalid overlay color: {item.get('color')!r}")
        item["alpha"] = float(item.get("alpha", default_alpha))
        if not 0 < item["alpha"] <= 1:
            raise ValueError("overlay alpha must be in (0, 1]")
        mesh = yp._retrieve_static_mesh("tracts", str(path), str(index), str(path))
        if mesh is not None:
            loaded.append((item, mesh))
    return loaded


def _blend(coverage, overlays, background):
    coverage = np.clip(np.asarray(coverage, float), 0, 1)
    weights = coverage * np.array([item["alpha"] for item in overlays])
    total = weights.sum()
    if not total:
        return to_rgb(background)
    mixed = weights @ np.array([to_rgb(item["color"]) for item in overlays]) / total
    opacity = 1 - np.prod(1 - weights)
    return tuple(np.asarray(to_rgb(background)) * (1 - opacity) + mixed * opacity)


def _add_legend(plotter, overlays):
    entries = [
        [item.get("label", item["path"].name), item["color"]]
        for item in overlays
    ]
    actor = plotter.add_legend(
        entries,
        loc="upper left",
        bcolor="white",
        border=False,
        size=(0.43, min(0.06 * len(entries), 0.3)),
    )
    actor.SetPosition(0.52, 0.77)


def _surface_meshes(item, n_levels, blur_sigma):
    data = _select(item["data"], item["sign"])
    sigma = float(item.get("blur_sigma", blur_sigma))
    if sigma > 0:
        data = gaussian_filter(data, sigma=sigma)
    threshold = item["threshold_value"]
    if threshold is None or not np.any(data >= threshold):
        return []
    upper = float(item.get("vmax", np.percentile(data[data >= threshold], 99)))
    count = int(item.get("n_levels", n_levels))
    if count < 1:
        raise ValueError("n_levels must be at least 1")
    levels = [threshold] if upper <= threshold else np.linspace(threshold, upper, count)
    grid = pv.ImageData(dimensions=data.shape)
    grid["value"] = data.flatten(order="F")
    meshes = []
    for level in levels:
        mesh = grid.contour([level], scalars="value")
        if mesh.n_points:
            mesh.transform(item["image"].affine, inplace=True)
            if item.get("smooth_i", 0):
                mesh = mesh.smooth(
                    n_iter=int(item["smooth_i"]),
                    relaxation_factor=float(item.get("smooth_f", 0.01)),
                )
            meshes.append(mesh)
    return meshes


def plot_mesh_overlays(
    overlays,
    *,
    bmesh=None,
    views=None,
    layout=None,
    figsize=None,
    n_levels=1,
    blur_sigma=0,
    alpha=0.65,
    style="default",
    bmesh_alpha=0.16,
    bmesh_color="#8d99ae",
    zoom=1.2,
    tract_kwargs=None,
    display_type="matplotlib",
    export_path=None,
    show_legend=True,
):
    nifti_items = _load(
        [item for item in overlays if not item.get("_tract_path")], alpha
    )
    tract_items = _load_tracts(
        [item for item in overlays if item.get("_tract_path")], alpha
    )
    surfaces = [(item, _surface_meshes(item, n_levels, blur_sigma)) for item in nifti_items]
    surfaces = [(item, meshes) for item, meshes in surfaces if meshes]
    if not surfaces and not tract_items:
        raise ValueError("no overlay contains values or fibers after filtering")

    selected = get_view_configs(views)
    ax, display_type, figsize = prepare_plotter(None, display_type, selected, layout, figsize)
    plotter, ncols, _ = setup_plotter(selected, layout, figsize, display_type, False)
    plotter.enable_depth_peeling(number_of_peels=50)
    plotter.enable_anti_aliasing("fxaa")
    shading = get_shading_preset(style)
    line_style = {
        "render_lines_as_tubes": True,
        "line_width": 1.2,
        **(tract_kwargs or {}),
    }

    for index, config in enumerate(selected.values()):
        plotter.subplot(index // ncols, index % ncols)
        for overlay_index, (item, meshes) in enumerate(surfaces):
            shell_alpha = 1 - (1 - item["alpha"]) ** (1 / len(meshes))
            for shell_index, mesh in enumerate(meshes):
                shown = mesh
                if config["side"] == "L":
                    shown = mesh.clip(normal="x", origin=(0, 0, 0), invert=True)
                elif config["side"] == "R":
                    shown = mesh.clip(normal="x", origin=(0, 0, 0), invert=False)
                if shown.n_points:
                    plotter.add_mesh(
                        shown,
                        color=item["color"],
                        opacity=shell_alpha,
                        lighting=False,
                        smooth_shading=True,
                        name=f"overlay_{index}_{overlay_index}_{shell_index}",
                    )
        for overlay_index, (item, mesh) in enumerate(tract_items):
            plotter.add_mesh(
                mesh,
                color=item["color"],
                opacity=item["alpha"],
                name=f"tract_overlay_{index}_{overlay_index}",
                **shading,
                **line_style,
            )
        for name, mesh in (bmesh or {}).items():
            hidden = (
                (config["side"] == "L" and name == "R")
                or (config["side"] == "R" and name == "L")
            )
            if hidden:
                continue
            plotter.add_mesh(
                mesh,
                color=bmesh_color,
                opacity=bmesh_alpha,
                smooth_shading=True,
                name=f"brain_{index}_{name}",
                **shading,
            )
        if show_legend and index == 0:
            _add_legend(
                plotter,
                [item for item, _ in surfaces] + [item for item, _ in tract_items],
            )
        set_camera(plotter, config, zoom=zoom)
        plotter.hide_axes()
    return finalize_plot(plotter, export_path, display_type, ax=ax)


def _coverage(item, points, interpolation):
    voxels = nib.affines.apply_affine(np.linalg.inv(item["image"].affine), points)
    sampled = map_coordinates(
        item["data"],
        voxels.T,
        order=0 if interpolation == "nearest" else 1,
        mode="constant",
        cval=0,
    )
    threshold = item["threshold_value"]
    if threshold is None:
        return 0
    selected = _select(sampled, item["sign"])
    return float(np.mean(selected >= threshold))


def plot_parcel_overlays(
    overlays,
    *,
    atlas=None,
    custom_atlas_path=None,
    bmesh="midthickness",
    views=None,
    layout=None,
    figsize=None,
    projection_kwargs=None,
    alpha=0.85,
    nan_color="#BDBDBD",
    style="default",
    bmesh_alpha=0.15,
    bmesh_color="lightgray",
    zoom=1.2,
    tract_kwargs=None,
    custom_atlas_proc=None,
    display_type="matplotlib",
    export_path=None,
    show_legend=True,
):
    import yabplot.plotting as yp

    nifti_items = _load(
        [item for item in overlays if not item.get("_tract_path")], alpha
    )
    tract_items = _load_tracts(
        [item for item in overlays if item.get("_tract_path")], alpha
    )
    atlas = "aseg" if atlas is None and custom_atlas_path is None else atlas
    atlas_dir = yp._resolve_resource_path(
        atlas, "subcortical", custom_path=custom_atlas_path
    )
    processing = {"smooth_i": 15, "smooth_f": 0.6, **(custom_atlas_proc or {})}
    cache_key = "custom" if custom_atlas_path else atlas
    meshes = {
        name: mesh
        for name, path in yp._find_subcortical_files(atlas_dir).items()
        if (mesh := yp._retrieve_static_mesh("subcortical", cache_key, name, path, **processing))
        is not None
    }
    if not meshes:
        raise ValueError(f"no parcel meshes found in {custom_atlas_path or atlas!r}")

    interpolation = (projection_kwargs or {}).get("interpolation", "linear")
    colors = {
        name: _blend(
            [_coverage(item, mesh.points, interpolation) for item in nifti_items],
            nifti_items,
            nan_color,
        )
        for name, mesh in meshes.items()
    }
    selected = get_view_configs(views)
    ax, display_type, figsize = prepare_plotter(None, display_type, selected, layout, figsize)
    plotter, ncols, _ = setup_plotter(selected, layout, figsize, display_type, False)
    plotter.enable_anti_aliasing("msaa")
    shading = get_shading_preset(style)
    sides = {name: yp._get_side_tokens(name) for name in meshes}
    context = load_bmesh(bmesh)
    line_style = {
        "render_lines_as_tubes": True,
        "line_width": 1.2,
        **(tract_kwargs or {}),
    }

    for index, config in enumerate(selected.values()):
        plotter.subplot(index // ncols, index % ncols)
        add_context_to_view(
            plotter, context, config["side"], bmesh_alpha, bmesh_color, **shading
        )
        for name, mesh in meshes.items():
            left, right = sides[name]
            hidden = (
                (config["side"] == "L" and right and not left)
                or (config["side"] == "R" and left and not right)
            )
            if hidden:
                continue
            plotter.add_mesh(mesh, color=colors[name], **shading)
        for overlay_index, (item, mesh) in enumerate(tract_items):
            plotter.add_mesh(
                mesh,
                color=item["color"],
                opacity=item["alpha"],
                name=f"tract_overlay_{index}_{overlay_index}",
                **shading,
                **line_style,
            )
        if show_legend and index == 0:
            _add_legend(
                plotter,
                nifti_items + [item for item, _ in tract_items],
            )
        set_camera(plotter, config, zoom=zoom)
        plotter.hide_axes()
    return finalize_plot(plotter, export_path, display_type, ax=ax)


def plot_tract_overlays(
    overlays,
    *,
    bmesh="midthickness",
    views=None,
    layout=None,
    figsize=None,
    alpha=1.0,
    style="default",
    bmesh_alpha=0.15,
    bmesh_color="lightgray",
    zoom=1.2,
    tract_kwargs=None,
    display_type="matplotlib",
    export_path=None,
    show_legend=True,
):
    """Draw several prepared tractograms with one color per overlay."""
    prepared = _load_tracts(overlays, alpha)
    if not prepared:
        raise ValueError("no tract overlay contains fibers after filtering")

    selected = get_view_configs(views)
    ax, display_type, figsize = prepare_plotter(
        None, display_type, selected, layout, figsize
    )
    plotter, ncols, _ = setup_plotter(selected, layout, figsize, display_type, False)
    plotter.enable_depth_peeling(number_of_peels=10)
    plotter.enable_anti_aliasing("msaa")
    shading = get_shading_preset(style)
    context = load_bmesh(bmesh)
    line_style = {
        "render_lines_as_tubes": True,
        "line_width": 1.2,
        **(tract_kwargs or {}),
    }

    for index, config in enumerate(selected.values()):
        plotter.subplot(index // ncols, index % ncols)
        add_context_to_view(
            plotter, context, config["side"], bmesh_alpha, bmesh_color, **shading
        )
        for overlay_index, (item, mesh) in enumerate(prepared):
            plotter.add_mesh(
                mesh,
                color=item["color"],
                opacity=item["alpha"],
                name=f"tract_overlay_{index}_{overlay_index}",
                **shading,
                **line_style,
            )
        if show_legend and index == 0:
            _add_legend(plotter, [item for item, _ in prepared])
        set_camera(plotter, config, zoom=zoom, distance=150)
        plotter.hide_axes()
    return finalize_plot(plotter, export_path, display_type, ax=ax)


__all__ = ["plot_mesh_overlays", "plot_parcel_overlays", "plot_tract_overlays"]
