"""Build the 'subcortex' fill: a solid core that hides everything inside.

The medial-wall patch was the wrong idea -- a thin sheet cut from cortex has a
ragged border and still leaves gaps. This is the volume approach instead: take
the subcortical mask, make sure it reaches the brainstem, erode it so it sits
strictly inside the surrounding surfaces, and split it at the midline. Each half
meshes to a closed solid whose flat cut face IS the medial panel, so there is
nothing to see through and nothing poking out between the other meshes.
"""
import numpy as np, nibabel as nib, pyvista as pv
from pathlib import Path
from scipy import ndimage

import sys; sys.path.insert(0, "pkg")
from calvin_utils.neuroimaging_utils.nifti_utils.parcel_meshes import mesh_from_mask
from calvin_utils.plotting_utils import glass_mesh as gm

PIAL = Path("/mnt/user-data/uploads/yabplot/bmesh-pial")
AAL3 = Path("res/aal3_suit")
MASK = Path("/mnt/user-data/uploads/Calvin Howard--resources--atlases/"
            "MNI_structures/subcortex/subcortex_mask_2mm.nii")
OUT = Path("pkg/resources/neuro_plotter_resources")
ERODE_MM = 7.0     # inset from every surrounding surface
CLOSE_MM = 7.0     # ball that seals the ventricles before eroding
CLEAR_MM = 2.0     # gap kept between the core and the shell around it
SLAB_MM  = 5.0     # half-thickness of the solid interhemispheric wall
PANEL_GROW_MM = 4.0   # panel tucks this far under the surrounding parcels
RECESS_MM = 4.0    # body stops this far short of the midline
PANEL_ROUND_MM = 3.0  # corner radius of the panel outline
FLOOR_MM = -11.0   # a couple of mm under the thalamus (z-min -9.2)
OVERLAP_MM = 1.0   # the halves meet rather than leaving a seam at x=0

def panel_footprint():
    """The (y, z) cells where a deep structure is what you see from the midline.

    Per cell, whichever surface reaches furthest medially is the one on screen.
    Where a thalamic nucleus or a basal ganglion wins, the medial view shows that
    cluttered interior and wants a panel; where a cortical parcel wins, a panel
    would hide the parcel instead. A full-hemisphere plane gets this wrong in the
    obvious way -- it flattens the entire medial surface.
    """
    deep_names = ("Thal", "Caudate", "Putamen", "Pallidum", "N_Acc", "Amygdala",
                  "Hippocampus", "Red_N", "SN_", "VTA", "LC_", "Raphe")
    skip = ("Cerebellum", "Vermis")
    inv = np.linalg.inv(img.affine)
    shape2 = img.shape[1:]

    def depth(paths):
        out = np.full(shape2, -np.inf)
        for q in paths:
            pts = np.asarray(pv.read(q).points)
            ijk = np.rint(nib.affines.apply_affine(inv, pts)).astype(int)
            ok = np.all((ijk[:, 1:] >= 0) & (ijk[:, 1:] < np.array(shape2)), axis=1)
            np.maximum.at(out, (ijk[ok, 1], ijk[ok, 2]), pts[ok, 0])
        return out

    left = [q for q in sorted(AAL3.glob("*_L.vtk")) if not q.stem.startswith(skip)]
    exposed = depth([q for q in left if q.stem.startswith(deep_names)])
    cortex = depth([q for q in left if not q.stem.startswith(deep_names)])
    win = np.isfinite(exposed) & (exposed > cortex)
    lab, _ = ndimage.label(win)
    counts = np.bincount(lab.ravel())[1:]
    win = np.isin(lab, np.where(counts >= 8)[0] + 1)      # drop speckle
    r = int(np.ceil(PANEL_GROW_MM / vox))
    ball = np.linalg.norm(np.indices((2*r+1,)*2) - r, axis=0) <= r
    # Grown a little so the panel tucks under the parcels around it rather than
    # meeting them edge to edge, which leaves a visible hairline.
    win = ndimage.binary_fill_holes(ndimage.binary_closing(win, structure=ball))
    # Round the outline. A footprint on a 2 mm grid has staircase corners, and
    # meshing them gives a panel with a jagged rim that reads as a cut-out rather
    # than as anatomy. Smoothing a signed distance field and re-thresholding
    # rounds the corners at a real radius instead of nibbling pixels: convex
    # corners pull in, concave ones fill, and straight runs stay put.
    sdf = (ndimage.distance_transform_edt(win, sampling=[vox]*2)
           - ndimage.distance_transform_edt(~win, sampling=[vox]*2))
    return ndimage.gaussian_filter(sdf, PANEL_ROUND_MM / vox) > 0


img = nib.load(str(MASK))
mask = np.asarray(img.dataobj) > 0
_i = np.arange(mask.shape[0]); _k = np.arange(mask.shape[2])
vox = float(np.abs(np.diag(img.affine))[0])
xw = nib.affines.apply_affine(img.affine, np.c_[_i, np.zeros((len(_i), 2))])[:, 0]
zw = nib.affines.apply_affine(img.affine, np.c_[np.zeros((len(_k), 2)), _k])[:, 2]
print(f"mask      : {mask.sum()} vox · {mask.sum()*vox**3/1000:.0f} cm3")

# The mask is cerebrum-shaped, so the brainstem has to be added or the core
# stops at the midbrain and you see straight down the stem.
stem = pv.read(OUT / "brainstem.vtk")
sub, box = gm._voxelize(stem, img, nib)
if sub is not None:
    lo, _hi = box
    before = mask.sum()
    mask[lo[0]:lo[0]+sub.shape[0], lo[1]:lo[1]+sub.shape[1], lo[2]:lo[2]+sub.shape[2]] |= sub
    print(f"+brainstem: {mask.sum()-before} vox added")

# The ventricles are holes in this mask and they open onto the outside, so a
# plain fill_holes leaves them. Eroding through them is what exposed the caudate:
# the core was being eaten from the inside as well as the outside. Close first,
# with a ball wider than the ventricles are, then fill.
r = int(np.ceil(CLOSE_MM / vox))
ball = np.linalg.norm(np.indices((2*r+1,)*3) - r, axis=0) <= r
mask = ndimage.binary_closing(mask, structure=ball)
mask = ndimage.binary_fill_holes(mask)

# Bridge the interhemispheric fissure. Without this the midline is within a few
# millimetres of the mask's exterior, so eroding eats the very face the split is
# about to expose -- which is what turned a clean interhemispheric plane into a
# lumpy medial surface. Filling along x between the outermost voxels of each
# (y, z) row makes the midline deep interior; the shell clip below takes back
# anything this pushes outside the pial.
span = mask.any(axis=0)
idx = np.arange(mask.shape[0])[:, None, None]
first = np.where(span, mask.argmax(axis=0), 0)
last = np.where(span, mask.shape[0] - 1 - mask[::-1].argmax(axis=0), -1)
mask |= (idx >= first) & (idx <= last)
print(f"closed    : {mask.sum()} vox · {mask.sum()*vox**3/1000:.0f} cm3")

# Erode in millimetres, not iterations, so the inset is the same at any voxel size.
dist = ndimage.distance_transform_edt(mask, sampling=[vox]*3)
core = dist > ERODE_MM
core = ndimage.binary_fill_holes(core)
# Clip to the shell the core has to hide inside. Closing by 7 mm bridges the
# sulci and swells the brainstem, so without this the stem hangs below the real
# brainstem mesh and the core bulges through the pial in a few places. The shell
# is itself eroded, so the core stops short of it rather than touching.
shell = None
for name in ("brainstem", "pial_cerebellum"):
    q = OUT / f"{name}.vtk"
    if q.exists():
        m = pv.read(q).triangulate()
        shell = m if shell is None else shell.merge(m)
for side_ in ("L", "R"):
    v, f = nib.load(str(PIAL / f"fs_LR.32k.{side_}.pial.surf.gii")).agg_data()
    m = pv.PolyData(np.asarray(v, float),
                    np.hstack([np.full((len(f), 1), 3), f]).astype(np.int64).ravel()).triangulate()
    shell = m if shell is None else shell.merge(m)
sv, (slo, shi) = gm._voxelize(shell, img, nib)
inside = np.zeros_like(core)
inside[slo[0]:shi[0], slo[1]:shi[1], slo[2]:shi[2]] = sv
inside = ndimage.distance_transform_edt(inside, sampling=[vox]*3) > CLEAR_MM
# The brainstem is its own mesh in the composite, so the core must not grow a
# stem of its own -- bounded or not, it reads as a spike hanging below the
# cerebrum. Subtract the brainstem region and let that mesh cover from there
# down. The subtraction uses an eroded brainstem, so the two overlap by a
# couple of millimetres instead of leaving a seam you can see through.
stem_vox, (blo, bhi) = gm._voxelize(pv.read(OUT / "brainstem.vtk"), img, nib)
stem = np.zeros_like(core)
stem[blo[0]:bhi[0], blo[1]:bhi[1], blo[2]:bhi[2]] = stem_vox
stem = ndimage.distance_transform_edt(stem, sampling=[vox]*3) > CLEAR_MM

before = core.sum()
# The midline is exempt from the shell clip. The interhemispheric fissure lies
# outside BOTH pial hemispheres, so select_enclosed_points calls the midline
# "outside" and the clip punches holes through the very plane we want -- which
# is what the thalamus was showing through. Nothing in the slab can poke out
# laterally anyway; it is bounded in y and z by the mask silhouette.
midline = (np.abs(xw) <= SLAB_MM)[:, None, None]
core &= (inside | midline) & ~stem
print(f"clipped   : {before-core.sum()} vox trimmed to the shell "
      f"({core.sum()*vox**3/1000:.0f} cm3 left)")

# Stop just below the thalamus. Lower than that and the core starts hiding the
# medial temporal lobe, which is usually the thing a figure is trying to show.
core &= (zw >= FLOOR_MM)[None, None, :]
print(f"floored   : z >= {FLOOR_MM} mm · {core.sum()*vox**3/1000:.0f} cm3")

# Build the interhemispheric plane explicitly rather than hoping erosion leaves
# one. Around the third ventricle and the interthalamic adhesion the slab is
# thin in y and z, so a distance-transform erosion punches holes straight through
# the midline and the thalamus shows through them. Take the silhouette of the
# midline slab, fill it in 2D, and extrude it back across the midline: a solid
# wall by construction, still trimmed by the shell and the floor.
slab = np.abs(xw) <= SLAB_MM
yz = panel_footprint()
plane = np.zeros_like(core)
plane[slab] = yz
plane &= (zw >= FLOOR_MM)[None, None, :]
print(f"plane     : {plane.sum()} vox over |x| <= {SLAB_MM} mm, "
      f"{yz.sum()} cells of medial wall")
# Recess the body. Its cut face sits at x = 0, which is a hair in front of the
# medial cortical parcels (they reach -0.15 mm), so leaving it there flattens the
# whole medial view into a slab -- the body, not the panel, was doing that. Cut a
# slot out of it and let only the panel reach the midline, in the one region where
# there is no cortex to hide.
core &= ~((np.abs(xw) < RECESS_MM)[:, None, None])
core |= plane

lab, n = ndimage.label(core)
if n > 1:                      # eroding can shed slivers; keep the body
    core = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
print(f"eroded {ERODE_MM} mm: {core.sum()} vox · {core.sum()*vox**3/1000:.0f} cm3")

# Mesh the whole thing once, then cut. Splitting the mask first and meshing each
# half separately does not work: mesh_from_mask runs 40 Taubin passes, which
# rounds the flat cut face off into a lumpy medial surface -- the interhemispheric
# plane got smoothed away rather than eroded away. Smoothing the closed bilateral
# solid and cutting it afterwards leaves a genuinely planar face, capped so each
# half is still a closed solid.
whole = mesh_from_mask(core, img.affine, presmooth_vox=1.2, target_mm=1.0)
print(f"meshed    : {whole.n_points} pts / {whole.n_cells} faces")
# The bilateral solid is the piece the registry names. build_bmesh splits an
# unsuffixed file at the midline itself, with the same capped clip used here, so
# shipping it whole means hemisphere toggles come from the existing machinery
# rather than from two files that have to be kept in step.
whole.save(OUT / "subcortex.vtk")
for side, half in zip(("L", "R"), gm._split_at_midline(whole)):
    b = half.bounds
    flat = (np.abs(np.asarray(half.points)[:, 0]) < 1.0).sum()
    print(f"{side}: {half.n_points:6d} pts / {half.n_cells:6d} faces · vol {half.volume/1000:5.0f} cm3 · "
          f"x {b[0]:6.1f}..{b[1]:5.1f} z {b[4]:6.1f}..{b[5]:5.1f} · {flat} pts on the midline")
    half.save(OUT / f"subcortex_{side}.vtk")
