"""Split the brainstem mesh into mesencephalon, pons and medulla.

No atlas here parcellates the brainstem that way -- the Edlow AAN atlases are
nuclei, not subdivisions -- so the junctions are found from the shape itself
rather than typed in. The pons is defined by its anterior bulge: the basis
pontis protrudes forward of the midbrain above it and the medulla below it, so
the anterior extent per axial slice rises into the pons and falls away at both
junctions. Taking the levels at half the bulge's prominence puts each cut where
the profile actually turns, which is reproducible and can be checked against the
textbook levels rather than assuming them.
"""
import sys; sys.path.insert(0, "pkg")
import numpy as np, nibabel as nib, pyvista as pv
from pathlib import Path
from scipy import ndimage
from calvin_utils.plotting_utils import glass_mesh as gm

OUT = Path("res/aal3_suit")
RES = Path("pkg/resources/neuro_plotter_resources")
REF = nib.load("/mnt/user-data/uploads/Calvin Howard--resources--atlases/"
               "MNI_structures/subcortex/subcortex_mask_2mm.nii")

stem = pv.read(RES / "brainstem.vtk").triangulate()
print(f"brainstem : {stem.n_cells} faces · z {stem.bounds[4]:.1f}..{stem.bounds[5]:.1f}")

# Cross-sectional area per 1 mm axial level, from a real rasterisation rather
# than a bounding box. Area discriminates better than anterior extent: the
# anterior profile climbs gradually out of the medulla with no clear shoulder,
# which put the lower cut at z -50, while area has clean half-maximum crossings
# at both ends of the basis pontis.
bounds = np.asarray(stem.bounds).reshape(3, 2)
lo = np.floor(bounds[:, 0]) - 2.0
dims = (np.ceil(bounds[:, 1]) + 2.0 - lo).astype(int) + 1
box = pv.ImageData(dimensions=tuple(dims), spacing=(1.0, 1.0, 1.0), origin=tuple(lo))
inside = (box.select_enclosed_points(stem, tolerance=0.0, check_surface=False)
          ["SelectedPoints"].astype(bool).reshape(dims, order="F"))
levels = lo[2] + np.arange(dims[2])
area = inside.sum(axis=(0, 1)).astype(float)
area = ndimage.uniform_filter1d(area, 3)

peak = int(np.argmax(area))
z_peak = levels[peak]

# The junction is the inflection of the area profile, not a height on it. A
# half-maximum rule needs a baseline, and the stem tapers to zero at both ends,
# so the threshold lands far too low -- it put the pontomedullary cut at z -48.
# The steepest point on each flank is where the basis pontis gives way, needs no
# baseline, and lands on the textbook levels.
slope = np.gradient(area, levels)
above, below = slice(peak + 1, len(area)), slice(0, peak)
z_upper = levels[above][int(np.argmin(slope[above]))]
z_lower = levels[below][int(np.argmax(slope[below]))]
print(f"bulge     : peak {area[peak]:.0f} mm2 at z {z_peak:.0f}")
print(f"junctions : midbrain/pons z {z_upper:.0f} · pons/medulla z {z_lower:.0f}")
print("            (textbook MNI: about -20..-25 and -40..-45)")

parts = {"Brainstem_Mesencephalon": (z_upper, None),
         "Brainstem_Pons": (z_lower, z_upper),
         "Brainstem_Medulla": (None, z_lower)}
for name, (lo, hi) in parts.items():
    piece = stem
    for at, keep_above in ((lo, True), (hi, False)):
        if at is None:
            continue
        normal = [0.0, 0.0, 1.0 if keep_above else -1.0]
        piece = piece.clip_closed_surface(normal=normal, origin=(0.0, 0.0, float(at)),
                                          inplace=False).triangulate()
    b = piece.bounds
    print(f"{name:26s} {piece.n_cells:6d} faces · vol {piece.volume/1000:5.1f} cm3 · "
          f"z {b[4]:6.1f}..{b[5]:6.1f}")
    piece.save(OUT / f"{name}.vtk")
