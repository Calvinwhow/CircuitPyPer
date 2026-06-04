import configparser
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLUT_DIR = PACKAGE_ROOT / "resources" / "colour_luts"

PathLike = Union[str, os.PathLike]
ColorNode = Tuple[float, Tuple[float, float, float, float]]
MICROGL_LUT_SUFFIXES = {".clut", ".lut"}


def _normalize_cmap_name(name: str) -> str:
    return "".join(char for char in name.lower() if char.isalnum())


def find_mricrogl_lut(cmap_name: str, clut_dir: Optional[PathLike] = None) -> Optional[Path]:
    """
    Find a bundled MRIcroGL LUT by exact or normalized name.

    Normalized matching lets ``xrain`` resolve to ``x_rain.clut`` while leaving
    ordinary Matplotlib names alone when no bundled LUT exists.
    """
    clut_dir = Path(clut_dir) if clut_dir is not None else DEFAULT_CLUT_DIR

    for suffix in MICROGL_LUT_SUFFIXES:
        exact_path = clut_dir / f"{cmap_name}{suffix}"
        if exact_path.exists():
            return exact_path

    target = _normalize_cmap_name(cmap_name)
    for lut_path in clut_dir.iterdir():
        if lut_path.suffix.lower() in MICROGL_LUT_SUFFIXES:
            if _normalize_cmap_name(lut_path.stem) == target:
                return lut_path
    return None


def _read_mricrogl_clut_nodes(clut_path: PathLike, use_alpha: bool = False) -> Iterable[ColorNode]:
    parser = configparser.ConfigParser()
    parser.optionxform = str
    read_files = parser.read(clut_path)
    if not read_files:
        raise FileNotFoundError(f"Could not read MRIcroGL CLUT file: {clut_path}")

    try:
        n_nodes = parser.getint("INT", "numnodes")
    except (configparser.NoSectionError, configparser.NoOptionError) as exc:
        raise ValueError(f"Invalid MRIcroGL CLUT file missing INT/numnodes: {clut_path}") from exc

    nodes = []
    for idx in range(n_nodes):
        intensity_key = f"nodeintensity{idx}"
        rgba_key = f"nodergba{idx}"
        try:
            intensity = parser.getint("BYT", intensity_key) / 255.0
            rgba255 = parser.get("RGBA255", rgba_key)
        except (configparser.NoSectionError, configparser.NoOptionError) as exc:
            raise ValueError(f"Invalid MRIcroGL CLUT node {idx} in: {clut_path}") from exc

        rgba_parts = [int(value) for value in rgba255.split("|")]
        if len(rgba_parts) != 4:
            raise ValueError(f"Expected RGBA255 node with four pipe-delimited values: {rgba_key}={rgba255}")

        r, g, b, a = rgba_parts
        alpha = a / 255.0 if use_alpha else 1.0
        nodes.append((intensity, (r / 255.0, g / 255.0, b / 255.0, alpha)))

    return sorted(nodes, key=lambda node: node[0])


def mricrogl_clut_to_cmap(
    clut_path: PathLike,
    name: Optional[str] = None,
    use_alpha: bool = False,
    register: bool = False,
) -> LinearSegmentedColormap:
    """
    Convert an MRIcroGL ``.clut`` node file into a Matplotlib colormap.

    MRIcroGL stores node positions as byte intensities and colors as RGBA255.
    By default this ignores the CLUT alpha channel because MRIcroGL commonly
    uses it as overlay opacity metadata, not as the visible color ramp.
    """
    clut_path = Path(clut_path)
    nodes = _read_mricrogl_clut_nodes(clut_path, use_alpha=use_alpha)
    if not nodes:
        raise ValueError(f"MRIcroGL CLUT contains no color nodes: {clut_path}")

    first_intensity = nodes[0][0]
    last_intensity = nodes[-1][0]
    if first_intensity != 0.0 or last_intensity != 1.0:
        span = last_intensity - first_intensity
        if span <= 0:
            raise ValueError(f"MRIcroGL CLUT node intensities must span a positive range: {clut_path}")
        nodes = [
            ((intensity - first_intensity) / span, rgba)
            for intensity, rgba in nodes
        ]

    cmap_name = name if name is not None else clut_path.stem
    cmap = LinearSegmentedColormap.from_list(cmap_name, nodes)

    if register:
        register_matplotlib_cmap(cmap, force=True)
    return cmap


def register_matplotlib_cmap(cmap: LinearSegmentedColormap, force: bool = False) -> LinearSegmentedColormap:
    """
    Register a colormap with Matplotlib and return it.
    """
    if cmap.name in colormaps:
        if not force:
            return colormaps[cmap.name]
        colormaps.unregister(cmap.name)
    colormaps.register(cmap, name=cmap.name)
    return cmap


def load_mricrogl_cmap(
    cmap_name: str,
    clut_dir: Optional[PathLike] = None,
    use_alpha: bool = False,
    register: bool = True,
) -> LinearSegmentedColormap:
    """
    Load a named MRIcroGL CLUT from the repository colour LUT folder.

    Examples
    --------
    ``load_mricrogl_cmap("x_rain")``
    ``load_mricrogl_cmap("xrain")``
    ``load_mricrogl_cmap("NIH")``
    """
    clut_path = find_mricrogl_lut(cmap_name, clut_dir=clut_dir)
    if clut_path is None:
        clut_dir = Path(clut_dir) if clut_dir is not None else DEFAULT_CLUT_DIR
        raise FileNotFoundError(f"Could not find MRIcroGL LUT named {cmap_name!r} in {clut_dir}")

    return mricrogl_clut_to_cmap(
        clut_path=clut_path,
        name=cmap_name,
        use_alpha=use_alpha,
        register=register,
    )


def resolve_mricrogl_or_matplotlib_cmap(cmap):
    """
    Resolve a colormap for plotting.

    Resolution order for strings:
    1) Existing ``.clut``/``.lut`` path -> parse as MRIcroGL LUT.
    2) Existing Matplotlib colormap name -> return unchanged.
    3) Bundled MRIcroGL LUT name -> parse from ``resources/colour_luts``.
    4) Anything else -> return unchanged for Matplotlib/SUITPy.
    """
    if not isinstance(cmap, str):
        return cmap

    cmap_path = Path(cmap).expanduser()
    if cmap_path.suffix.lower() in MICROGL_LUT_SUFFIXES:
        if not cmap_path.exists():
            raise FileNotFoundError(f"Colormap LUT path does not exist: {cmap_path}")
        return mricrogl_clut_to_cmap(cmap_path, register=True)

    if cmap in colormaps:
        return cmap

    clut_path = find_mricrogl_lut(cmap)
    if clut_path is not None:
        return mricrogl_clut_to_cmap(clut_path, name=cmap, register=True)
    return cmap
