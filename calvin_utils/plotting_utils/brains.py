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

import os
from pathlib import Path

# Where the prebuilt meshes live. An application that ships its own copy points
# this at it with NEURO_PLOTTER_RESOURCES, so it does not have to reach into
# this package's install directory to find geometry it depends on.
RESOURCE_DIR = Path(os.environ.get(
    "NEURO_PLOTTER_RESOURCES",
    Path(__file__).resolve().parents[2] / "resources" / "neuro_plotter_resources",
)).expanduser()

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
    # A solid core that fills the inside of the shell, cut flat at the midline.
    # Without it a medial view looks straight through the glass at the thalamus
    # and basal ganglia; with it you get a clean interhemispheric plane. Floored
    # just under the thalamus so the medial temporal lobe stays visible.
    "subcortex": {
        "what": "[surface] solid inner core; hides the interior, planar at the midline",
        "family": "surface", "pieces": ["subcortex"],
    },
    "pial_medial": {
        "what": "[surface] pial whole brain + inner core; for medial views",
        "family": "surface",
        "pieces": ["pial", "pial_cerebellum", "brainstem", "subcortex"],
    },

    "glass_wholebrain": {
        "what": "[surface] carved translucent hull + cerebellum; no gyral detail",
        "family": "surface", "pieces": ["glass_cerebrum", "glass_cerebellum"],
    },
    "plain_wholebrain": {
        "what": "[surface] uncarved translucent hull + cerebellum; no gyral detail",
        "family": "surface", "pieces": ["plain_cerebrum", "glass_cerebellum"],
    },

    # --- surface: Lead-DBS' own surfaces, imported from CoolSurfaces -------- #
    # Kept in their own folder rather than copied into the pile above: they are
    # a third-party set with their own provenance, and the resolver searches
    # every folder under resources/meshes. The bilateral ones are directories of
    # L/R files, which is why a piece can be a directory.
    "leaddbs_ch2": {
        "what": "[surface] Lead-DBS Ch2 whole-brain surface",
        "family": "surface", "pieces": ["BrainMesh_Ch2"],
    },
    "leaddbs_icbm152": {
        "what": "[surface] Lead-DBS ICBM152 Talairach whole-brain surface",
        "family": "surface", "pieces": ["BrainMesh_ICBM152_tal"],
    },
    "leaddbs_surf": {
        "what": "[surface] Lead-DBS combined cortical surface",
        "family": "surface", "pieces": ["surf"],
    },
    "leaddbs_cortex_hires": {
        "what": "[surface] Lead-DBS bilateral high-resolution cortex, DKT labels",
        "family": "surface", "pieces": ["CortexHiRes"],
    },
    "leaddbs_cortex_lowres": {
        "what": "[surface] Lead-DBS bilateral cortex, 15,000 vertices",
        "family": "surface", "pieces": ["CortexLowRes_15000V"],
    },
    "leaddbs_surf_lh_rh": {
        "what": "[surface] Lead-DBS bilateral cortical surface pair",
        "family": "surface", "pieces": ["surf.lh-rh"],
    },
    "leaddbs_right": {
        "what": "[surface] Lead-DBS right-hemisphere brain surface",
        "family": "surface", "pieces": ["brainMeshRight"],
    },
    "leaddbs_surf_rh_ply": {
        "what": "[surface] Lead-DBS right-hemisphere PLY surface variant",
        "family": "surface", "pieces": ["surf.rh.ply"],
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


# --- supra-regions ---------------------------------------------------------
# A 170-region atlas is too many checkboxes to work through when what you want
# is "everything except the cerebellum". These are the coarse groupings, matched
# on the naming AAL3 and SUIT already use rather than stored as a 170-line
# table, so a region added later lands in the right place without an edit here.
#
# Every rule is a prefix or an exact stem, tried in order, so the specific ones
# come first: "Frontal_Med_Orb" is frontal, and "OFCmed" is too, but "Olfactory"
# has to be named because nothing in it says frontal.
SUPRA_RULES = (
    ("Cerebellum", ("Cerebellum", "Vermis")),
    ("Brainstem",  ("Brainstem", "Red_N", "SN_", "Raphe", "VTA", "LC_", "PAG")),
    ("Thalamus",   ("Thal_", "Thal-")),
    ("Basal ganglia", ("Caudate", "Putamen", "Pallidum", "N_Acc")),
    ("Cingulate",  ("ACC_", "Cingulate_")),
    ("Insula",     ("Insula",)),
    # Medial temporal sits with the temporal lobe: hippocampus and amygdala are
    # what someone means by "turn off the temporal lobe" in a lateral view.
    ("Temporal",   ("Temporal", "Heschl", "Fusiform", "Hippocampus",
                    "ParaHippocampal", "Amygdala")),
    ("Occipital",  ("Occipital", "Calcarine", "Cuneus_", "Cuneus", "Lingual")),
    ("Parietal",   ("Parietal", "Postcentral", "Precuneus", "SupraMarginal",
                    "Angular")),
    ("Frontal",    ("Frontal", "Precentral", "Supp_Motor_Area", "OFC", "Rectus",
                    "Olfactory", "Paracentral", "Rolandic_Oper")),
    ("Midline",    ("Midline",)),
)

SUPRA_NAMES = tuple(name for name, _ in SUPRA_RULES)


def supra_region(name):
    """The coarse group a region belongs to, or ``None`` if nothing matches.

    ``None`` rather than an "Other" bucket, so an atlas this does not understand
    shows no supra-region controls instead of one misleading catch-all.
    """
    stem = str(name).rsplit(".", 1)[0]
    for suffix in ("_L", "_R", "-L", "-R"):
        if stem.upper().endswith(suffix):
            stem = stem[:-2]
            break
    # Precuneus before Cuneus, Paracentral before Postcentral: longest first
    # inside each group is not enough, because the groups themselves overlap.
    for group, prefixes in SUPRA_RULES:
        for prefix in prefixes:
            if stem.lower().startswith(prefix.lower()):
                return group
    return None
