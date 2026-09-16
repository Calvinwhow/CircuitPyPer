"""
brains.py
=========
What ``neuro_plotter`` can draw, and what can colour it.

Two registries, two families, one shape. A figure names a ``mesh`` (the
geometry) and an ``atlas`` (the parcellation that colours it), and the plot
family decides which registry entries are legal:

``surface``    a continuous cortical surface, optionally with whole-brain
               pieces attached. Coloured by a cortical ``atlas``, or with
               ``atlas=None`` drawn translucent with the map as isosurfaces
               inside it -- the glass-brain rendering.
``subcortex``  a set of per-region meshes, one value each. The meshes *are*
               the parcellation, so a subcortex figure needs no ``atlas``.

Compatibility is stated in every description and enforced by
:func:`require_family`, so a mesh used by the wrong family fails with a
sentence naming the family it belongs to rather than rendering something
arbitrary.
"""

from __future__ import annotations

from pathlib import Path

RESOURCE_DIR = Path(__file__).resolve().parents[2] / "resources" / "neuro_plotter_resources"

FAMILIES = ("surface", "subcortex")


# --------------------------------------------------------------------------- #
# Meshes -- the geometry a figure draws
# --------------------------------------------------------------------------- #

MESHES = {
    # --- surface: yabplot's fsLR32k cortical surfaces, cortex only ---------- #
    "pial": {
        "what": "[surface] yabplot pial cortex, cortex only",
        "family": "surface", "pieces": ["pial"],
    },
    "midthickness": {
        "what": "[surface] yabplot midthickness cortex, cortex only",
        "family": "surface", "pieces": ["midthickness"],
    },
    "white": {
        "what": "[surface] yabplot white-matter surface, cortex only",
        "family": "surface", "pieces": ["white"],
    },
    "inflated": {
        "what": "[surface] yabplot inflated cortex, cortex only",
        "family": "surface", "pieces": ["inflated"],
    },
    "very_inflated": {
        "what": "[surface] yabplot very-inflated cortex, cortex only",
        "family": "surface", "pieces": ["very_inflated"],
    },

    # --- surface: whole brains, cortex + cerebellum + brainstem ------------- #
    "pial_wholebrain": {
        "what": "[surface] pial cortex + eroded SUIT cerebellum + brainstem",
        "family": "surface", "pieces": ["pial", "pial_cerebellum", "brainstem"],
    },
    "midthickness_wholebrain": {
        "what": "[surface] midthickness cortex + eroded SUIT cerebellum + brainstem",
        "family": "surface", "pieces": ["midthickness", "pial_cerebellum", "brainstem"],
    },
    "inflated_wholebrain": {
        "what": "[surface] inflated cortex + eroded SUIT cerebellum + brainstem",
        "family": "surface", "pieces": ["inflated", "pial_cerebellum", "brainstem"],
    },
    "glass_wholebrain": {
        "what": "[surface] carved translucent hull + cerebellum; no gyral detail",
        "family": "surface", "pieces": ["glass_cerebrum", "glass_cerebellum"],
    },
    "plain_wholebrain": {
        "what": "[surface] uncarved translucent hull + cerebellum; no gyral detail",
        "family": "surface", "pieces": ["plain_cerebrum", "glass_cerebellum"],
    },

    # --- subcortex: local parcel meshes, built by parcel_meshes ------------- #
    "aal3_suit_parcels": {
        "what": "[subcortex] AAL3 fine cerebral + SUIT cerebellar, 165 region meshes",
        "family": "subcortex", "parcels": "parcellations/aal3_suit",
    },
    "suit_parcels": {
        "what": "[subcortex] SUIT cerebellar parcels, 32 region meshes",
        "family": "subcortex", "parcels": "parcellations/suit",
    },

    # --- subcortex: the sets packaged with yabplot -------------------------- #
    "aal3_subcortical": {
        "what": "[subcortex] yabplot's packaged AAL3 subcortical meshes",
        "family": "subcortex", "yabplot_atlas": "aal3",
    },
    "aal3_nocer": {
        "what": "[subcortex] yabplot's AAL3 subcortical without the cerebellum",
        "family": "subcortex", "yabplot_atlas": "aal3_nocer",
    },
    "aseg": {
        "what": "[subcortex] FreeSurfer aseg structures; subcortex only, no surface",
        "family": "subcortex", "yabplot_atlas": "aseg",
    },
    "tian2020_s1": {
        "what": "[subcortex] Melbourne Tian 2020 scale 1; subcortex only",
        "family": "subcortex", "yabplot_atlas": "tian2020_s1",
    },
    "brainnetome_sc": {
        "what": "[subcortex] yabplot Brainnetome subcortical structures",
        "family": "subcortex", "yabplot_atlas": "brainnetome_sc",
    },
    "musus100": {
        "what": "[subcortex] yabplot Melbourne subcortex atlas",
        "family": "subcortex", "yabplot_atlas": "musus100",
    },
    "musus100_dbn": {
        "what": "[subcortex] yabplot Melbourne deep-brain nuclei",
        "family": "subcortex", "yabplot_atlas": "musus100_dbn",
    },
    "musus100_tha": {
        "what": "[subcortex] yabplot Melbourne thalamic nuclei",
        "family": "subcortex", "yabplot_atlas": "musus100_tha",
    },
}


# --------------------------------------------------------------------------- #
# Atlases -- what colours a surface
# --------------------------------------------------------------------------- #

ATLASES = {
    "aal3": {
        "what": "[surface only] AAL3 cortical parcellation",
        "family": "surface", "yabplot_atlas": "aal3",
    },
    "aparc": {
        "what": "[surface only] Desikan-Killiany (aparc)",
        "family": "surface", "yabplot_atlas": "aparc",
    },
    "brainnetome": {
        "what": "[surface only] Brainnetome cortical parcellation",
        "family": "surface", "yabplot_atlas": "brainnetome",
    },
    "schaefer100": {
        "what": "[surface only] Schaefer 100 parcels",
        "family": "surface", "yabplot_atlas": "schaefer100",
    },
    "schaefer200": {
        "what": "[surface only] Schaefer 200 parcels",
        "family": "surface", "yabplot_atlas": "schaefer200",
    },
    "schaefer300": {
        "what": "[surface only] Schaefer 300 parcels",
        "family": "surface", "yabplot_atlas": "schaefer300",
    },
    "schaefer400": {
        "what": "[surface only] Schaefer 400 parcels",
        "family": "surface", "yabplot_atlas": "schaefer400",
    },
    "schaefer1000": {
        "what": "[surface only] Schaefer 1000 parcels",
        "family": "surface", "yabplot_atlas": "schaefer1000",
    },
}


# --------------------------------------------------------------------------- #
# Lookup
# --------------------------------------------------------------------------- #

def describe(registry) -> dict:
    """``{name: what it is}`` -- the catalogue for the script's config block."""
    return {name: spec["what"] for name, spec in registry.items()}


def _lookup(registry, name, label):
    if name not in registry:
        raise ValueError(
            f"{label} {name!r} is not known. Available:\n"
            + "\n".join(f"  {k:26s} {v['what']}" for k, v in registry.items())
        )
    return dict(registry[name])


def require_family(name, family, registry=None, label="mesh"):
    """Look a name up and insist it belongs to ``family``.

    A subcortex mesh in a surface figure is not a near-miss to be coerced -- it
    is a different kind of object -- so this names the family the entry does
    belong to, and the entries that would have worked.
    """
    registry = MESHES if registry is None else registry
    spec = _lookup(registry, name, label)
    if spec["family"] != family:
        ok = [k for k, v in registry.items() if v["family"] == family]
        raise ValueError(
            f"{label} {name!r} is a {spec['family']} entry ({spec['what']}), "
            f"but this is a {family} plot. {label}es that work here: {ok}"
        )
    return spec


def mesh_pieces(name) -> list:
    """The pieces of a surface mesh, for ``glass_mesh.build_bmesh``."""
    return list(require_family(name, "surface")["pieces"])


def mesh_bmesh(name, fallback="midthickness") -> str:
    """The cortical surface a surface mesh is built on.

    Parcel-on-surface plots draw on one fsLR32k surface, so a whole-brain mesh
    yields whichever surface it contains; its other pieces have no vertices the
    cortical atlas knows about.
    """
    from calvin_utils.plotting_utils.glass_mesh import BMESH_NAMES

    for piece in require_family(name, "surface")["pieces"]:
        if piece in BMESH_NAMES:
            return piece
    return fallback


def mesh_subcortex(name) -> dict:
    """Where a subcortex mesh's regions come from.

    Returns ``{"custom_atlas_path": ...}`` for meshes built locally, or
    ``{"atlas": ...}`` for a set packaged with yabplot -- the two ways yabplot
    accepts a region set.
    """
    spec = require_family(name, "subcortex")
    if "parcels" in spec:
        path = RESOURCE_DIR / spec["parcels"]
        if not path.is_dir():
            raise FileNotFoundError(
                f"mesh {name!r} needs {path}, which does not exist. Rebuild it with "
                f"calvin_utils.neuroimaging_utils.nifti_utils.parcel_meshes."
            )
        return {"custom_atlas_path": str(path), "atlas": None}
    return {"custom_atlas_path": None, "atlas": spec["yabplot_atlas"]}


def atlas_name(name):
    """The yabplot cortical atlas a surface figure is parcellated by.

    ``None`` means no parcellation: there is nothing to reduce the map into, so
    the surface is drawn translucent with the map as isosurfaces inside it.
    """
    if name is None:
        return None
    return require_family(name, "surface", ATLASES, "atlas")["yabplot_atlas"]


__all__ = [
    "MESHES", "ATLASES", "FAMILIES", "RESOURCE_DIR",
    "describe", "require_family",
    "mesh_pieces", "mesh_bmesh", "mesh_subcortex", "atlas_name",
]
