"""
suit_cerebellum.py
==================
Build a yabplot-compatible cerebellar mesh atlas from the SUIT anatomical
parcellation, for use as the cerebellum in ``ParcelwisePlot.plot_cortical(
include_cerebellum=True)``.

Why this exists
---------------
``_plot_cortical_with_cerebellum`` draws the cerebellum from the *subcortical*
resource directory of the selected atlas.  With AAL3 that means the coarse
``Cerebelum_*`` volumes, which are low resolution, poorly delineated at the
lobular level, and lack the deep nuclei entirely.  SUIT is a dedicated
cerebellar atlas: 34 lateralised lobules and nuclei, hand-drawn at 1 mm.

What it produces
----------------
``<out_dir>/subcortical/<Region>.vtk``  one smoothed surface per region
``<out_dir>/subcortical/atlas_LUT.txt`` ``<id> <name>`` order file
``<out_dir>/subcortical/qc_mesh_properties.txt``  vertex/face/volume report

That directory is exactly the layout ``yabplot.data._find_subcortical_files``
expects, so it can be handed to any yabplot call as ``custom_atlas_path``.

Naming
------
Region names are rewritten from SUIT's ``Left_CrusI`` convention to yabplot's
trailing-side convention, and prefixed so they survive the cerebellum filter in
``_plot_cortical_with_cerebellum`` (which keeps names containing ``"cerebel"``
or ``"vermis"``) and the per-view hemisphere test (which drops names containing
``"_r"`` from left views and ``"_l"`` from right views)::

    Left_CrusI      -> Cerebellum_CrusI_L
    Right_Dentate   -> Cerebellum_Dentate_R
    Vermis_VI       -> Vermis_VI          (midline: drawn in every view)

Mesh quality notes
------------------
Three things matter for cerebellar meshes and are easy to get wrong:

1. **Boundary closure.**  SUIT's 1 mm volume is a tight cerebellar bounding
   box, so lobules touch the array edge.  Running marching cubes on an
   unpadded mask leaves those faces open, which reads as holes and black
   shading artefacts.  The mask is zero-padded by one voxel first.
2. **Smoothing in volume space, not just on the mesh.**  Marching cubes on a
   binary mask can only ever produce stair-steps at the voxel scale, and mesh
   smoothing has to fight that geometry afterwards.  Blurring the mask first
   (``presmooth_sigma``) and taking a sub-voxel iso-contour removes the steps
   before they exist, which is far more effective than any amount of mesh
   relaxation.  To keep a blur from eroding small structures, the iso-level is
   chosen so the enclosed voxel count matches the original mask
   (``preserve_volume``) rather than fixed at 0.5.

   Mesh smoothing then only has to clean up what remains.  Taubin (the
   default) alternates positive and negative passes and is roughly
   volume-preserving; ``smooth_f`` is VTK's windowed-sinc **pass band**, where
   *lower is smoother* (0.05-0.1 typical, 1.0 nearly a no-op).  Laplacian
   smoothing is available for parity with yabplot's builder, but it shrinks
   thin structures toward their medial axis.
3. **Speckle.**  Nearest-neighbour resampling leaves stray voxel islands that
   become free-floating shards.  ``min_component_frac`` drops connected
   components below a fraction of the region's largest component.

Usage
-----
::

    from calvin_utils.neuroimaging_utils.nifti_utils.suit_cerebellum import (
        SuitCerebellumAtlas,
    )

    atlas = SuitCerebellumAtlas(
        dseg_path="~/hires_backdrops/suit/atl-Anatom_space-MNI_dseg.nii",
        lut_path="~/hires_backdrops/suit/atl-Anatom.lut",
        out_dir="~/hires_backdrops/suit/yabplot_suit",
    )
    info = atlas.run()

Then plot::

    plotter.run(
        project="vol2surf", plot="cortical", atlas="aal3",
        plot_kwargs={
            "include_cerebellum": True,
            "cerebellum_atlas_path": "~/hires_backdrops/suit/yabplot_suit/subcortical",
            "views": ["left_lateral", "posterior", "inferior"],
        },
    )

CLI
---
::

    python -m calvin_utils.neuroimaging_utils.nifti_utils.suit_cerebellum \
        --dseg ~/hires_backdrops/suit/atl-Anatom_space-MNI_dseg.nii \
        --lut  ~/hires_backdrops/suit/atl-Anatom.lut \
        --out  ~/hires_backdrops/suit/yabplot_suit
"""

from __future__ import annotations

import argparse
import pathlib
import re
from typing import Iterable

import numpy as np
import nibabel as nib


__all__ = ["SuitCerebellumAtlas", "suit_name_to_yabplot"]


# --------------------------------------------------------------------------- #
# Naming
# --------------------------------------------------------------------------- #

def suit_name_to_yabplot(raw: str, prefix: str = "Cerebellum") -> str:
    """Rewrite one SUIT LUT name into a yabplot-safe region name.

    ``Left_I_IV`` -> ``Cerebellum_I_IV_L``; ``Vermis_VI`` -> ``Vermis_VI``.

    The result is checked against the two substring tests that
    ``_plot_cortical_with_cerebellum`` applies, so a name can never be silently
    dropped from the plot or drawn on the wrong side.
    """
    raw = raw.strip().replace(" ", "_")

    if raw.lower().startswith("left_"):
        name = f"{prefix}_{raw[5:]}_L"
    elif raw.lower().startswith("right_"):
        name = f"{prefix}_{raw[6:]}_R"
    elif raw.lower().startswith("vermis"):
        name = raw                      # midline, no side suffix
    else:
        name = f"{prefix}_{raw}"

    _assert_name_is_plottable(name)
    return name


def _assert_name_is_plottable(name: str) -> None:
    """Guard the substring logic in ``_plot_cortical_with_cerebellum``."""
    low = name.lower()

    if "cerebel" not in low and "vermis" not in low:
        raise ValueError(
            f"{name!r} contains neither 'cerebel' nor 'vermis' and would be "
            f"dropped by the default cerebellum_filter."
        )

    has_l, has_r = "_l" in low, "_r" in low
    if has_l and has_r:
        raise ValueError(
            f"{name!r} contains both '_l' and '_r', so the per-view hemisphere "
            f"test would drop it from every lateral view."
        )
    if low.startswith("vermis") and (has_l or has_r):
        raise ValueError(
            f"midline region {name!r} must not carry a side token, or it will "
            f"vanish from one lateral view."
        )


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #

class SuitCerebellumAtlas:
    """Turn a SUIT label volume into a directory of yabplot ``.vtk`` meshes.

    Parameters
    ----------
    dseg_path
        SUIT discrete segmentation, e.g. ``atl-Anatom_space-MNI_dseg.nii``.
        Prefer the native 1 mm volume over a 2 mm resample: marching cubes on
        2 mm cerebellar lobules produces visibly blocky meshes that no amount
        of smoothing recovers.
    lut_path
        SUIT ``.lut`` (``id r g b name`` per line) or a plain ``id name`` file.
    out_dir
        Directory to create; meshes land in ``<out_dir>/subcortical``.
    lateralized
        ``True``  keep SUIT's 34 left/right regions (recommended: lets the
                  plotter cull the far hemisphere in lateral views).
        ``False`` merge each left/right pair into one bilateral region, giving
                  21 regions that mirror the grouped SUIT outputs.
    presmooth_sigma
        Gaussian blur applied to the binary mask before marching cubes, in
        voxels. This is the setting that actually removes voxel stair-steps;
        1.2 at 1 mm is the calibrated default. Set 0 to disable.

        The value is an upper bound, not a fixed amount: a blur that is large
        relative to a structure's own radius erodes it rather than smoothing
        it (sigma 1.0 on a 27-voxel sliver is wider than the sliver). Both the
        blur and the mesh-smoothing iterations are scaled down for small
        regions by :meth:`_adaptive_params`, so the deep nuclei keep their
        size while the big lobules get the full treatment.
    preserve_volume
        Choose the iso-level so the contoured volume matches the mask's voxel
        count instead of using 0.5. Without this, blurring erodes small
        structures such as the deep nuclei.
    smooth
        ``"taubin"`` (default) or ``"laplacian"``.
    smooth_i
        Smoothing iterations.
    smooth_f
        Taubin **pass band** (lower is smoother, 0.05-0.1 typical), or the
        Laplacian relaxation factor (higher is smoother, 0-1).
    decimate
        Optional target reduction in ``[0, 1)`` passed to
        ``PolyData.decimate_pro``; 0.5 roughly halves the file size with no
        visible change at plot scale.
    min_component_frac
        Drop connected components smaller than this fraction of the region's
        largest component. Set to 0 to keep every speck.
    min_voxels
        Skip regions with fewer than this many voxels.
    """

    def __init__(
        self,
        dseg_path: str | pathlib.Path,
        lut_path: str | pathlib.Path,
        out_dir: str | pathlib.Path,
        lateralized: bool = True,
        prefix: str = "Cerebellum",
        presmooth_sigma: float = 1.2,
        preserve_volume: bool = True,
        smooth: str = "taubin",
        smooth_i: int = 40,
        smooth_f: float = 0.05,
        decimate: float | None = 0.4,
        min_component_frac: float = 0.05,
        min_voxels: int = 10,
        build_parcels: bool = True,
        build_surface: bool = True,
        midline_offset: float = 0.08,
        cap_max_edge: float = 1.5,
    ) -> None:
        self.dseg_path = pathlib.Path(dseg_path).expanduser()
        self.lut_path = pathlib.Path(lut_path).expanduser()
        self.out_dir = pathlib.Path(out_dir).expanduser()
        self.subcortical_dir = self.out_dir / "subcortical"
        self.lateralized = lateralized
        self.prefix = prefix
        self.presmooth_sigma = presmooth_sigma
        self.preserve_volume = preserve_volume
        self.smooth = smooth
        self.smooth_i = smooth_i
        self.smooth_f = smooth_f
        self.decimate = decimate
        self.min_component_frac = min_component_frac
        self.min_voxels = min_voxels
        self.build_parcels = build_parcels
        self.build_surface = build_surface
        self.midline_offset = midline_offset
        self.cap_max_edge = cap_max_edge
        self.surface_dir = self.out_dir / "surface"

        self.labels: dict[int, str] = {}
        self.targets: dict[str, list[int]] = {}
        self.written: list[tuple[int, str]] = []
        self.report: list[dict] = []

    # -- public ------------------------------------------------------------ #

    def run(self) -> dict:
        import nibabel as nib

        globals().setdefault("nib", nib)

        self.load_lut()
        self.build_targets()

        img = nib.load(str(self.dseg_path))
        result = {"out_dir": self.out_dir}

        if self.build_surface:
            result.update(self._run_surface(img))
        if not self.build_parcels:
            return result

        self.subcortical_dir.mkdir(parents=True, exist_ok=True)
        for stale in self.subcortical_dir.glob("*.vtk"):
            stale.unlink()

        data = np.asarray(img.dataobj)
        affine = img.affine

        for rid, (name, ids) in enumerate(self.targets.items(), start=1):
            mask = np.isin(data, ids)
            n_vox = int(mask.sum())

            if n_vox < self.min_voxels:
                print(f"[skip] {name}: {n_vox} voxels")
                continue

            mesh = self._mesh_from_mask(mask, affine)
            if mesh is None:
                print(f"[skip] {name}: no mesh")
                continue

            mesh.save(self.subcortical_dir / f"{name}.vtk")
            self.written.append((rid, name))
            self.report.append(
                {
                    "name": name,
                    "ids": ids,
                    "voxels": n_vox,
                    "points": mesh.n_points,
                    "faces": mesh.n_cells,
                    "volume": float(abs(mesh.volume)),
                    "bounds": tuple(round(b, 1) for b in mesh.bounds),
                }
            )
            print(
                f"[ok]   {name:26s} {n_vox:6d} vox -> "
                f"{mesh.n_points:6d} pts  {abs(mesh.volume):8.1f} mm3"
            )

        self._write_lut()
        self._write_report()

        result.update({
            "subcortical_dir": self.subcortical_dir,
            "n_regions": len(self.written),
            "names": [n for _, n in self.written],
        })
        return result

    # -- whole-hemisphere surface ------------------------------------------ #

    def _run_surface(self, img) -> dict:
        """Build one closed surface per cerebellar hemisphere.

        The surface is contoured from the union of every label -- deep nuclei
        included -- so it is a single closed object per side rather than a
        stack of lobules. Parcel colour is applied at plot time by looking each
        vertex up in ``labels.nii.gz``, so the geometry carries no parcellation
        of its own and can be coloured by any label volume in the same space.
        """
        import pyvista as pv

        self.surface_dir.mkdir(parents=True, exist_ok=True)
        data = np.asarray(img.dataobj)
        whole = data > 0

        mesh = self._mesh_from_mask(whole, img.affine)
        if mesh is None:
            raise RuntimeError("Failed to contour the whole-cerebellum mask.")
        print(
            f"[surface] whole cerebellum: {int(whole.sum())} vox -> "
            f"{mesh.n_points} pts  {abs(mesh.volume):.0f} mm3"
        )

        written = []
        for name, mesh_half in self._split_hemispheres(mesh).items():
            mesh_half.compute_normals(inplace=True, auto_orient_normals=True)
            mesh_half.save(self.surface_dir / f"{name}.vtk")
            written.append(name)
            print(
                f"[surface] {name}: {mesh_half.n_points} pts  "
                f"{abs(mesh_half.volume):.0f} mm3  centre_x={mesh_half.center[0]:.1f}  "
                f"closed={mesh_half.n_open_edges == 0}"
            )

        # Ship the parcellation alongside the geometry so the plotter can map
        # vertices to regions without being told where the atlas came from.
        nib.save(
            nib.Nifti1Image(
                np.asarray(img.dataobj).astype(np.int16), img.affine, img.header
            ),
            self.surface_dir / "labels.nii.gz",
        )
        with open(self.surface_dir / "atlas_LUT.txt", "w") as fh:
            for rid, raw in sorted(self.labels.items()):
                fh.write(f"{rid} {suit_name_to_yabplot(raw, self.prefix)}\n")

        return {"surface_dir": self.surface_dir, "surface_names": written}

    def _split_hemispheres(self, mesh) -> dict:
        """Split one cerebellar surface into left and right halves.

        The cut is a true plane, so the interior face each half shows in a
        medial view is flat -- a hemisected cerebellum, which is what the view
        is supposed to look like. Partitioning existing faces instead makes
        that face follow the mesh edges, and it reads as a shaped, sheared
        wall.

        The catch with a plane is that both halves' caps land on exactly the
        same plane. Coplanar coincident faces z-fight in a depth-buffered
        renderer, which speckles the very plane we are trying to keep clean.
        So after clipping, each half's on-plane vertices are nudged
        ``midline_offset`` mm *into* the opposite hemisphere: the two caps end
        up a fraction of a millimetre apart, each buried inside the other half
        and invisible from outside, and each still perfectly planar. The outer
        rim moves by the same sub-voxel amount, which is far below anything
        visible, and pushing outward rather than inward means the halves
        overlap slightly instead of leaving a crack at the midline.

        The cap is then subdivided. ``clip_closed_surface`` triangulates the
        cross-section as coarsely as it can -- single triangles up to 350 mm2 --
        while parcel colour is sampled per vertex, so on the hemisected face a
        few huge triangles smear whole lobules into gradients. See
        :meth:`_refine_cap`.
        """
        import numpy as np

        out = {}
        for name, normal in (("Cerebellum_L", (-1.0, 0.0, 0.0)),
                             ("Cerebellum_R", (1.0, 0.0, 0.0))):
            half = mesh.clip_closed_surface(
                normal=normal, origin=(0.0, 0.0, 0.0), inplace=False
            ).triangulate()

            if self.midline_offset:
                pts = np.asarray(half.points).copy()
                on_plane = np.abs(pts[:, 0]) < 1e-6
                # normal points away from the kept side, so -normal[0] pushes
                # the cap across the midline into the other hemisphere.
                pts[on_plane, 0] -= normal[0] * self.midline_offset
                half.points = pts

            out[name] = self._refine_cap(
                half, plane_x=-normal[0] * self.midline_offset
            )
        return out

    def _refine_cap(self, half, plane_x: float):
        """Subdivide the flat midline cap so per-vertex colour is not blocky.

        Only the cap is touched, and only by splitting its triangles: every new
        vertex lands exactly on an existing edge, so the geometry is unchanged
        (measured volume delta 0.0000 mm3, identical bounds) and nothing moves
        relative to the body.

        The split does leave T-junctions where refined cap edges meet the
        body's original rim, so ``n_open_edges`` becomes non-zero. That is a
        topological artefact rather than a visible one -- the new vertices sit
        on the rim edges, so no gap can open. It would matter only if something
        downstream did volumetric work on these meshes; the plotting path does
        not.
        """
        import numpy as np

        if not self.cap_max_edge:
            return half
        try:
            x = np.asarray(half.points)[:, 0]
            faces = half.faces.reshape(-1, 4)[:, 1:]
            on_cap = np.isclose(x, plane_x, atol=1e-5)
            cap_mask = on_cap[faces].all(axis=1)
            if cap_mask.sum() < 4:
                return half

            cap = half.extract_cells(np.flatnonzero(cap_mask)).extract_surface()
            body = half.extract_cells(np.flatnonzero(~cap_mask)).extract_surface()
            refined = cap.triangulate().subdivide_adaptive(
                max_edge_len=self.cap_max_edge, max_n_passes=4
            )
            merged = (body.triangulate() + refined).clean(tolerance=1e-6)
            merged.compute_normals(inplace=True, auto_orient_normals=True)
            return merged
        except Exception as exc:  # pragma: no cover - cosmetic only
            print(f"[surface] cap refinement skipped: {exc}")
            return half

    def load_lut(self) -> dict[int, str]:
        """Parse ``id [r g b] name``; SUIT ships floats for the colours."""
        labels: dict[int, str] = {}
        for line in self.lut_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                rid = int(parts[0])
            except ValueError:
                continue
            labels[rid] = parts[-1]
        if not labels:
            raise ValueError(f"No usable entries parsed from {self.lut_path}")
        self.labels = labels
        return labels

    def build_targets(self) -> dict[str, list[int]]:
        """Map output region name -> the label ids that compose it."""
        targets: dict[str, list[int]] = {}

        for rid, raw in sorted(self.labels.items()):
            if self.lateralized:
                name = suit_name_to_yabplot(raw, self.prefix)
            else:
                stem = re.sub(r"^(left|right)_", "", raw, flags=re.IGNORECASE)
                if raw.lower().startswith("vermis"):
                    name = raw
                else:
                    name = f"{self.prefix}_{stem}"
                    _assert_name_is_plottable(name)
            targets.setdefault(name, []).append(rid)

        self.targets = targets
        return targets

    # -- internals --------------------------------------------------------- #

    def _mesh_from_mask(self, mask: np.ndarray, affine: np.ndarray):
        """Delegate to the shared parcel-mesh pipeline.

        The blur/iso-level/smoothing logic lives in
        :mod:`calvin_utils.neuroimaging_utils.nifti_utils.parcel_meshes` so
        cerebellar and cerebral parcels are built by exactly the same code and
        can sit in one directory without looking like two atlases.

        Both sides express the blur in voxels, which is the unit that
        transfers between atlases: stair-steps are one voxel tall regardless of
        what a voxel measures.
        """
        from calvin_utils.neuroimaging_utils.nifti_utils.parcel_meshes import (
            mesh_from_mask,
        )

        return mesh_from_mask(
            mask,
            affine,
            presmooth_vox=self.presmooth_sigma,
            preserve_volume=self.preserve_volume,
            smooth=self.smooth,
            smooth_i=self.smooth_i,
            smooth_f=self.smooth_f,
            decimate=self.decimate,
            min_component_frac=self.min_component_frac,
            verbose=True,
        )

    def _write_lut(self) -> None:
        with open(self.subcortical_dir / "atlas_LUT.txt", "w") as fh:
            for rid, name in self.written:
                fh.write(f"{rid} {name}\n")

    def _write_report(self) -> None:
        path = self.subcortical_dir / "qc_mesh_properties.txt"
        with open(path, "w") as fh:
            fh.write("region\tids\tvoxels\tpoints\tfaces\tvolume_mm3\tbounds_xyz\n")
            for row in self.report:
                fh.write(
                    f"{row['name']}\t{row['ids']}\t{row['voxels']}\t"
                    f"{row['points']}\t{row['faces']}\t{row['volume']:.1f}\t"
                    f"{row['bounds']}\n"
                )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _cli(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("Usage")[0])
    ap.add_argument("--dseg", required=True)
    ap.add_argument("--lut", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-parcels", action="store_true",
                    help="skip the per-lobule meshes and build only the surface")
    ap.add_argument("--no-surface", action="store_true",
                    help="skip the whole-hemisphere surface")
    ap.add_argument("--bilateral", action="store_true",
                    help="merge left/right pairs into 21 bilateral regions")
    ap.add_argument("--smooth", default="taubin",
                    choices=["taubin", "laplacian", "none"])
    ap.add_argument("--presmooth-sigma", type=float, default=1.2,
                    help="gaussian blur of the mask in voxels before contouring")
    ap.add_argument("--no-preserve-volume", action="store_true")
    ap.add_argument("--smooth-i", type=int, default=40)
    ap.add_argument("--smooth-f", type=float, default=0.05,
                    help="taubin pass band (lower = smoother)")
    ap.add_argument("--decimate", type=float, default=0.4)
    ap.add_argument("--min-component-frac", type=float, default=0.05)
    args = ap.parse_args(argv)

    info = SuitCerebellumAtlas(
        dseg_path=args.dseg,
        lut_path=args.lut,
        out_dir=args.out,
        lateralized=not args.bilateral,
        presmooth_sigma=args.presmooth_sigma,
        preserve_volume=not args.no_preserve_volume,
        smooth=args.smooth,
        smooth_i=args.smooth_i,
        smooth_f=args.smooth_f,
        decimate=args.decimate,
        min_component_frac=args.min_component_frac,
        build_parcels=not args.no_parcels,
        build_surface=not args.no_surface,
    ).run()

    if "surface_dir" in info:
        print(f"\nsurface -> {info['surface_dir']}")
    if "n_regions" in info:
        print(f"{info['n_regions']} parcel meshes -> {info['subcortical_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
