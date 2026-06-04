import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


class ParcelwisePlot:
    """
    Overall plotting class for yabplot with separate projection and plot switchers.

    The class has individual methods for each supported yabplot projection and
    plot function. run() calls selection switchers in order:
    mesh -> projection -> atlas -> plot.

    Selections have one inflow each:
    - mesh selection comes from run(bmesh=...)
    - projection selection comes from run(project=...)
    - atlas selection comes from run(atlas=..., custom_atlas_path=...)
    - plotting selection comes from run(plot=...)

    If a selection is provided in run() and also duplicated inside kwargs, this
    class raises instead of guessing which value should win.

    Projection switcher
    -------------------
    project="vol2surf"  -> project_vol2surf()
    project="vol2tract" -> project_vol2tract()

    Projection methods are used when the input data are still volumetric
    NIfTI values but the desired plot is not voxelwise. ``vol2surf`` samples a
    NIfTI at cortical surface vertices and returns ``(lh_data, rh_data)``.
    When ``project="vol2surf"`` and ``plot="vertexwise"`` are supplied in the
    same ``run()`` call, this class automatically builds the vertexwise meshes
    and passes them to yabplot. When ``project="vol2surf"`` is paired with
    ``plot="cortical"`` or ``plot="cortical_outline"``, the projected surface
    values are reduced into the selected yabplot cortical atlas parcels and
    passed as ``data``. When a ``map_path`` is available and ``plot`` is
    ``"subcortical"``, the NIfTI is sampled at each selected yabplot
    subcortical atlas mesh and reduced to one value per structure. When
    ``project="vol2tract"`` is paired with ``plot="tracts"``, the NIfTI is
    sampled along every tract in the selected yabplot tract atlas and reduced
    to one value per tract.

    Projection string options
    -------------------------
    projection_kwargs["interpolation"] can be "nearest" or "linear".
    Use "linear" for smooth continuous maps such as t-statistics or FA/MD.
    Use "nearest" for discrete labels, atlases, masks, or p-values where
    interpolation would create invalid intermediate values.
    projection_kwargs["nan_fill"] defaults to 0.0 before projection because
    scipy/yabplot linear interpolation propagates NaNs from the source volume.
    Set it to None to preserve NaNs exactly.

    These projection switches are not needed when you already have parcel
    values, atlas data dictionaries, vertex/tract meshes with data attached, or
    when using ``plot="voxelwise"`` directly on a NIfTI volume.


    Context mesh switcher
    ---------------------
    bmesh="midthickness"      middle of the cortical ribbon
    bmesh="pial"             outer gray matter surface
    bmesh="white"            inner white matter surface
    bmesh="swm"              smoothed white matter
    bmesh="inflated"         smoothed surface exposing sulci
    bmesh="very_inflated"    spherical-like expanded surface

    Plot switcher
    -------------
    plot="vertexwise"        -> plot_vertexwise()
    plot="cortical"          -> plot_cortical()
    plot="cortical_outline" -> plot_cortical_outline()
    plot="subcortical"       -> plot_subcortical()
    plot="tracts"            -> plot_tracts()
    plot="tract"             -> plot_tracts() alias for "tracts"
    plot="voxelwise"         -> plot_voxelwise()
    plot="connectome"        -> plot_connectome()

    Common plotting string options passed through plot_kwargs
    --------------------------------------------------------
    views:
        "left_lateral", "right_lateral", "left_medial", "right_medial",
        "superior", "inferior", "anterior", "posterior"
    display_type:
        "matplotlib", "interactive", "pyvista", "object"
    style:
        "default", "matte", "glossy", "sculpted", "flat"
    cmap:
        Any colormap name accepted by matplotlib/yabplot, e.g. "coolwarm".

    Built-in atlas names are supplied through run(atlas=...). Use
    get_available_resources(category) or yabplot.get_available_resources() to
    inspect current yabplot atlas/resource names.

    Typical surface projection + plot
    ---------------------------------
        plotter = ParcelwisePlot(
            map_path="/path/to/map.nii.gz",
            out_file="/path/to/output/my_map",
        )

        plotter.run(
            project="vol2surf",
            plot="cortical",
            atlas="aal3",
            bmesh="midthickness",
            plot_kwargs={"views": ["left_lateral", "superior"]},
        )

    Direct yabplot plot
    -------------------
        plotter = ParcelwisePlot()
        print(plotter.get_atlas_regions(atlas="aal3", category="cortical")[:10])
        plotter.run(
            plot="cortical",
            atlas="aal3",
            bmesh="midthickness",
            plot_kwargs={"views": ["left_lateral", "superior"]},
        )

    Cortical atlas with parcel outlines
    -----------------------------------
        plotter = ParcelwisePlot()
        plotter.run(
            plot="cortical_outline",
            atlas="aal3",
            bmesh="inflated",
            plot_kwargs={
                "views": ["left_lateral", "superior", "left_medial"],
                "outline_color": "black",
                "outline_width": 0.8,
                "outline_radius": 0.12,
            },
        )

    ``plot="cortical_outline"`` is intentionally cortical-only. It draws
    parcel boundaries on cortical surface meshes. Subcortical, tract, and
    voxelwise plots use different geometry and will not produce reliable
    parcel outlines through this method.
    """

    ATLAS_SWITCH = {
        'cortical': [
            'aal3',
            'aparc',
            'brainnetome',
            'schaefer100',
            'schaefer1000',
            'schaefer200',
            'schaefer300',
            'schaefer400'],
        'subcortical': [
            'aal3',
            'aal3_nocer',
            'aseg',
            'brainnetome_sc',
            'musus100',
            'musus100_dbn',
            'musus100_tha',
            'tian2020_s1'],
        'tracts': [
            'hcp1065_medium',
            'hcp1065_small',
            'hcp1065_tiny',
            'xtract_large',
            'xtract_medium',
            'xtract_small',
            'xtract_tiny']
        }

    PROJECTION_SWITCH = {
        "vol2surf": "project_vol2surf",
        "vol2tract": "project_vol2tract",
    }

    BMESH_SWITCH = {
        "midthickness",
        "pial",
        "white",
        "swm",
        "inflated",
        "very_inflated",
    }

    PLOT_SWITCH = {
        "vertexwise": "plot_vertexwise",
        "cortical": "plot_cortical",
        "cortical_outline": "plot_cortical_outline",
        "subcortical": "plot_subcortical",
        "tracts": "plot_tracts",
        "voxelwise": "plot_voxelwise",
        "connectome": "plot_connectome",
    }

    def __init__(
        self,
        map_path: str | os.PathLike | None = None,
        out_file: str | os.PathLike | None = None,
    ):
        """
        Store stable file paths shared by projection and plotting calls.

        Parameters
        ----------
        map_path : str | os.PathLike | None
            Default NIfTI path for methods that need a volume input, such as
            project_vol2surf(), project_vol2tract(), plot_voxelwise(), and
            automatic NIfTI-to-atlas scoring for cortical, subcortical, and
            tract plots. A method-level ``nii_path`` can still be supplied to
            override this for direct projection methods.
        out_file : str | os.PathLike | None
            Optional output file base for rendered plots. When a plot is saved,
            the plot type is appended before the extension. For example,
            ``out_file="/tmp/my_map"`` and ``plot="cortical"`` saves to
            ``/tmp/my_map_cortical.png``. If omitted, plots render without
            exporting unless an explicit ``export_path`` is supplied in
            ``plot_kwargs``.
        """
        self.map_path = Path(map_path).expanduser() if map_path is not None else None
        self.out_file = Path(out_file).expanduser() if out_file is not None else None
        self.projection_result = None
        self.projection_kind = None
        self.projection_bmesh = None
        self.lh_data = None
        self.rh_data = None
        self.tract_data = None
        self.parcel_scores = None
        self.plot_result = None
        self.plot_output_path = None

    def run(
        self,
        project: str | None = None,
        plot: str | None = None,
        bmesh: str | None = None,
        atlas: str | None = None,
        custom_atlas_path: str | os.PathLike | None = None,
        threshold: float | tuple[float, float] | None = None,
        damage_score_metric: str = "avg_in_target",
        score_nonzero_only: bool = False,
        projection_kwargs: dict[str, Any] | None = None,
        plot_kwargs: dict[str, Any] | None = None,
    ):
        """
        Execute the yabplot pipeline using explicit switch selections.

        Standard switch order is:
        1. ``bmesh`` validates/selects the context brain mesh.
        2. ``project`` dispatches to a projection method, if requested.
        3. ``atlas`` or ``custom_atlas_path`` validates/selects atlas input.
        4. ``plot`` dispatches to a plotting method, if requested.

        For atlas-wide tract plots, ``project="vol2tract"`` is deferred until
        after atlas resolution so every tract file in the selected yabplot tract
        atlas can be sampled automatically.

        Parameters
        ----------
        project : {"vol2surf", "vol2tract"} | None
            Projection/scoring mode for volumetric input. If ``plot`` is
            omitted, the raw projection result is returned where possible.
            ``project="vol2surf"`` samples ``map_path`` to cortical surface
            vertices and can feed ``plot="vertexwise"``, ``plot="cortical"``,
            or ``plot="cortical_outline"``. ``project="vol2tract"`` samples
            ``map_path`` along each tract in a selected tract atlas when paired
            with ``plot="tracts"``/``"tract"``. Projection is not needed for
            atlas-only renders, explicit parcel dictionaries, or direct
            voxelwise rendering.
        plot : {"vertexwise", "cortical", "cortical_outline", "subcortical", "tracts", "tract", "voxelwise", "connectome"} | None
            Plot function to run. If both ``project`` and ``plot`` are supplied,
            projection/scoring runs before the final plot. ``plot="tract"`` is
            accepted as an alias for ``plot="tracts"``. If ``map_path`` is set
            and ``plot="subcortical"`` is selected with an atlas and no
            explicit ``data``, the NIfTI is sampled at each subcortical mesh and
            reduced to one value per structure.
        bmesh : {"midthickness", "pial", "white", "swm", "inflated", "very_inflated"} | None
            Context brain mesh selection. This is passed only to yabplot calls
            that accept a mesh argument.
        atlas : str | None
            Built-in yabplot atlas name. Examples include ``"aal3"``,
            ``"aparc"``, ``"schaefer100"``, ``"aseg"``, ``"musus100"``,
            and ``"xtract_tiny"`` depending on the plot category. Mutually
            exclusive with ``custom_atlas_path``. Use
            ``get_available_resources(category)`` to list valid values.
        custom_atlas_path : str | os.PathLike | None
            Path to a custom yabplot atlas. Mutually exclusive with ``atlas``.
        threshold : float | tuple[float, float] | None
            Optional NaN threshold applied before automatic parcel/tract
            scoring calls ``DamageScorer._calculate_metrics``. If scalar,
            values below the threshold are set to NaN and excluded from the ROI
            passed to ``DamageScorer``. If ``(low, high)``, values from ``low``
            through ``high`` are set to NaN and excluded from scoring. Existing
            NaNs are also excluded. This does not affect explicit ``data``
            passed in ``plot_kwargs`` and is not passed through to yabplot
            plotting functions.
            Examples:
            ``threshold=1.96`` keeps only values greater than or equal to 1.96.
            ``threshold=(-1.96, 1.96)`` removes the central band and keeps
            values below -1.96 or above 1.96. Thresholding is applied to sampled
            values for automatic ``vol2surf`` cortical scoring, automatic
            subcortical mesh scoring, and automatic ``vol2tract`` tract scoring.
        damage_score_metric : str
            Metric passed to ``DamageScorer._calculate_metrics`` for automatic
            map-to-atlas scoring. Applies to projected cortical parcels,
            automatic subcortical scoring, and automatic tract scoring. Defaults
            to ``"avg_in_target"``. Common options include
            ``"avg_in_target"``, ``"avg_in_subject"``,
            ``"spatial_correlation"``, ``"cosine"``, ``"sum"``,
            ``"num_in_roi"``, ``"max_in_roi"``, ``"min_in_roi"``, and
            ``"dice"``. Existing per-plot kwargs still override this:
            ``surface_score_metric``, ``subcortical_score_metric``, and
            ``tract_score_metric``.
        score_nonzero_only : bool
            Passed to ``DamageScorer._calculate_metrics``. If True, the scorer
            ignores exact zero values when computing the metric. This is useful
            for sparse FWE maps where significant voxels are nonzero and
            everything else is zero. Defaults to False to preserve historical
            zero-included scoring.
        projection_kwargs : dict | None
            Additional keyword arguments passed to the selected projection
            method. Do not duplicate switch inputs here; duplicates raise.
            Common string options:
            ``interpolation`` is ``"nearest"`` or ``"linear"``.
            ``nan_fill`` defaults to ``0.0``. Use ``None`` to preserve source
            NaNs during interpolation.
            These kwargs also control automatic NIfTI-to-subcortical and
            NIfTI-to-tract scoring.
        plot_kwargs : dict | None
            Additional keyword arguments passed to the selected plot method. Do
            not duplicate switch inputs here; duplicates raise.
            Common string options:
            ``views`` values are ``"left_lateral"``, ``"right_lateral"``,
            ``"left_medial"``, ``"right_medial"``, ``"superior"``,
            ``"inferior"``, ``"anterior"``, and ``"posterior"``.
            ``display_type`` is ``"matplotlib"``, ``"interactive"``,
            ``"pyvista"``, or ``"object"``.
            ``style`` is ``"default"``, ``"matte"``, ``"glossy"``,
            ``"sculpted"``, or ``"flat"``.
            For ``plot="cortical_outline"``, ``outline_color`` is any
            matplotlib/PyVista color string and ``outline_width`` controls
            parcel boundary line width. ``outline_radius`` controls the tube
            radius used for visible 3D parcel borders; set it to ``0`` or
            ``None`` to use ordinary line rendering.
            For projected cortical plots, ``surface_score_metric`` defaults to
            ``"avg_in_target"`` and controls how vertex values are reduced
            inside each cortical atlas parcel.
            For automatic NIfTI-to-subcortical plots, ``subcortical_score_metric``
            defaults to ``"avg_in_target"``. For automatic NIfTI-to-tract plots,
            ``tract_score_metric`` defaults to ``"avg_in_target"``.
            Explicit ``data`` always wins; if provided, no automatic map-to-atlas
            scoring is performed for that plot.

        Returns
        -------
        object
            The selected yabplot return value: a projection result if only
            ``project`` is supplied, or a plot object/axis if ``plot`` is
            supplied.
        """
        projection_kwargs = dict(projection_kwargs or {})
        plot_kwargs = dict(plot_kwargs or {})
        plot = self._normalize_plot_name(plot)
        bmesh_kwargs = self.bmesh_switcher(bmesh) if bmesh is not None else {}

        should_run_projection = not (
            project == "vol2tract"
            and plot == "tracts"
            and "trk_path" not in projection_kwargs
        )

        if project is not None and should_run_projection:
            projection_kwargs = self._prepare_projection_kwargs(
                project,
                projection_kwargs,
                bmesh_kwargs,
            )
            self.projection_result = self.project_switcher(project, **projection_kwargs)

        if plot is not None:
            atlas_kwargs = self.atlas_switcher(atlas, custom_atlas_path)
            plot_kwargs = self._prepare_plot_kwargs(
                plot,
                plot_kwargs,
                bmesh_kwargs,
                atlas_kwargs,
            )
            plot_kwargs = self._inject_projection_kwargs(
                project,
                plot,
                plot_kwargs,
                projection_kwargs,
                threshold,
                damage_score_metric,
                score_nonzero_only,
            )
            self.plot_result = self.plot_switcher(plot, **plot_kwargs)
            return self.plot_result

        return self.projection_result

    def project_switcher(self, project: str, **kwargs):
        try:
            method_name = self.PROJECTION_SWITCH[project]
        except KeyError as exc:
            raise ValueError(
                f"project must be one of {sorted(self.PROJECTION_SWITCH)}; got {project!r}."
            ) from exc
        return getattr(self, method_name)(**kwargs)

    def plot_switcher(self, plot: str, **kwargs):
        plot = self._normalize_plot_name(plot)
        try:
            method_name = self.PLOT_SWITCH[plot]
        except KeyError as exc:
            raise ValueError(
                f"plot must be one of {sorted(self.PLOT_SWITCH)}; got {plot!r}."
            ) from exc
        return getattr(self, method_name)(**kwargs)

    @staticmethod
    def _normalize_plot_name(plot: str | None) -> str | None:
        if plot == "tract":
            return "tracts"
        return plot

    def bmesh_switcher(self, bmesh: str) -> dict[str, str]:
        if bmesh not in self.BMESH_SWITCH:
            raise ValueError(
                f"bmesh must be one of {sorted(self.BMESH_SWITCH)}; got {bmesh!r}."
            )
        return {"bmesh": bmesh}

    def atlas_switcher(
        self,
        atlas: str | None = None,
        custom_atlas_path: str | os.PathLike | None = None,
    ) -> dict[str, str]:
        if atlas is not None and custom_atlas_path is not None:
            raise ValueError("Provide atlas or custom_atlas_path, not both.")
        if atlas is not None:
            atlas_path = Path(atlas).expanduser()
            if atlas_path.exists():
                return {"custom_atlas_path": str(atlas_path)}
            return {"atlas": atlas}
        if custom_atlas_path is not None:
            return {"custom_atlas_path": str(Path(custom_atlas_path).expanduser())}
        return {}

    def get_atlas_regions(self, category: str, atlas: str | None = None, custom_atlas_path=None):
        atlas_kwargs = self.atlas_switcher(atlas, custom_atlas_path)
        if not atlas_kwargs:
            raise ValueError("atlas or custom_atlas_path is required.")
        yab = self._import_yabplot()
        return yab.get_atlas_regions(
            atlas=atlas_kwargs.get("atlas"),
            category=category,
            custom_atlas_path=atlas_kwargs.get("custom_atlas_path"),
        )

    def get_available_resources(self, category: str | None = None):
        """
        Return yabplot's available resource names.

        Parameters
        ----------
        category : {"cortical", "subcortical", "tracts", "bmesh", "label"} | None
            Optional resource category. If omitted, yabplot returns all
            categories.
        """
        yab = self._import_yabplot()
        return yab.get_available_resources(category)

    def project_vol2surf(
        self,
        nii_path: str | os.PathLike | None = None,
        bmesh: str = "midthickness",
        mask_medial_wall: bool = True,
        interpolation: str = "linear",
        nan_fill: float | None = 0.0,
    ):
        nii_path = self._resolve_map_path(nii_path)
        project_path = self._prepare_projection_volume(nii_path, nan_fill)
        bmesh = self.bmesh_switcher(bmesh)["bmesh"]
        yab = self._import_yabplot()
        self.projection_kind = "vol2surf"
        self.projection_bmesh = bmesh
        try:
            self.lh_data, self.rh_data = yab.project_vol2surf(
                str(project_path),
                bmesh=bmesh,
                mask_medial_wall=mask_medial_wall,
                interpolation=interpolation,
            )
        finally:
            self._cleanup_projection_volume(project_path, nii_path)
        return self.lh_data, self.rh_data

    def project_vol2tract(
        self,
        trk_path: str | os.PathLike,
        nii_path: str | os.PathLike | None = None,
        interpolation: str = "linear",
        nan_fill: float | None = 0.0,
    ):
        nii_path = self._resolve_map_path(nii_path)
        project_path = self._prepare_projection_volume(nii_path, nan_fill)
        trk_path = Path(trk_path).expanduser()
        if not trk_path.exists():
            raise FileNotFoundError(f"trk_path does not exist: {trk_path}")
        yab = self._import_yabplot()
        self.projection_kind = "vol2tract"
        try:
            self.tract_data = yab.project_vol2tract(
                str(trk_path),
                str(project_path),
                interpolation=interpolation,
            )
        finally:
            self._cleanup_projection_volume(project_path, nii_path)
        return self.tract_data

    def plot_vertexwise(
        self,
        lh=None,
        rh=None,
        bmesh: str = "midthickness",
        export_path: str | os.PathLike | None = None,
        save_plot: bool = True,
        **kwargs,
    ):
        yab = self._import_yabplot()
        if lh is None and rh is None and self.lh_data is not None and self.rh_data is not None:
            bmesh = self.projection_bmesh or bmesh
            lh, rh = self._build_vertexwise_meshes(bmesh, self.lh_data, self.rh_data)
        export_path = self._resolve_export_path(export_path, save_plot, "vertexwise")
        self.plot_result = yab.plot_vertexwise(
            lh,
            rh,
            export_path=str(export_path) if export_path is not None else None,
            **kwargs,
        )
        return self.plot_result

    def plot_cortical(
        self,
        export_path: str | os.PathLike | None = None,
        save_plot: bool = True,
        **kwargs,
    ):
        yab = self._import_yabplot()
        export_path = self._resolve_export_path(export_path, save_plot, "cortical")
        self.plot_result = yab.plot_cortical(
            export_path=str(export_path) if export_path is not None else None,
            **kwargs,
        )
        return self.plot_result

    def plot_cortical_outline(
        self,
        data=None,
        atlas=None,
        custom_atlas_path=None,
        bmesh: str = "midthickness",
        views=None,
        layout=None,
        figsize=None,
        cmap: str = "coolwarm",
        vminmax: list | tuple = (None, None),
        nan_color=(1.0, 1.0, 1.0),
        style: str = "default",
        zoom: float = 1.2,
        display_type: str = "matplotlib",
        export_path: str | os.PathLike | None = None,
        save_plot: bool = True,
        ax=None,
        cbar_kwargs=None,
        outline_color: str = "black",
        outline_width: float = 1.0,
        outline_radius: float | None = 0.12,
        outline_offset: float = 0.2,
    ):
        """
        Plot a cortical yabplot atlas with explicit parcel boundary outlines.

        This is cortical-only. It uses yabplot's cortical atlas files and
        standard brain meshes, then overlays line segments wherever adjacent
        surface vertices belong to different atlas parcels.
        """
        import numpy as np
        import pyvista as pv
        from matplotlib.colors import ListedColormap
        from yabplot.data import get_surface_paths
        from yabplot.utils import load_gii
        import yabplot.plotting as yp

        if atlas is None and custom_atlas_path is None:
            atlas = "aparc"

        bmesh = self.bmesh_switcher(bmesh)["bmesh"]
        atlas_dir = yp._resolve_resource_path(atlas, "cortical", custom_path=custom_atlas_path)
        check_name = None if custom_atlas_path else atlas
        csv_path, lut_path = yp._find_cortical_files(atlas_dir, strict_name=check_name)

        target_labels = np.loadtxt(csv_path, dtype=int)
        lut_ids, lut_colors, lut_names, max_id = yp.parse_lut(lut_path)
        all_vals = yp.map_values_to_surface(data, target_labels, lut_ids, lut_names)

        lh_path, rh_path = get_surface_paths(bmesh, "bmesh")
        lh_v, lh_f = load_gii(lh_path)
        rh_v, rh_f = load_gii(rh_path)
        lh_vals = all_vals[: len(lh_v)]
        rh_vals = all_vals[len(lh_v):]
        lh_labels = target_labels[: len(lh_v)]
        rh_labels = target_labels[len(lh_v):]

        is_cat = data is None
        if is_cat:
            lut_colors = lut_colors.copy()
            lut_colors[0] = nan_color
            plot_cmap = ListedColormap(lut_colors)
            clim = (0, max_id)
            n_colors = len(lut_colors)
        else:
            finite = np.concatenate([lh_vals[np.isfinite(lh_vals)], rh_vals[np.isfinite(rh_vals)]])
            if finite.size == 0:
                raise ValueError(
                    "No finite cortical values were found for plot='cortical_outline'. "
                    "Omit data or set data=None to render atlas parcels categorically."
                )
            vmin = vminmax[0] if vminmax[0] is not None else np.nanmin(finite)
            vmax = vminmax[1] if vminmax[1] is not None else np.nanmax(finite)
            plot_cmap = cmap
            clim = (vmin, vmax)
            n_colors = 256

        sel_views = yp.get_view_configs(views)
        ax, display_type, figsize = yp.prepare_plotter(ax, display_type, sel_views, layout, figsize)
        plotter, ncols, nrows = yp.setup_plotter(
            sel_views,
            layout,
            figsize,
            display_type,
            needs_bottom_row=not is_cat,
        )
        shading = yp.get_shading_preset(style)
        scalar_bar_mapper = None

        lh_mesh = yp.make_cortical_mesh(lh_v, lh_f, lh_vals)
        rh_mesh = yp.make_cortical_mesh(rh_v, rh_f, rh_vals)
        lh_edges = self._make_boundary_edge_mesh(
            lh_v,
            lh_f,
            lh_labels,
            outline_radius=outline_radius,
            outline_offset=outline_offset,
        )
        rh_edges = self._make_boundary_edge_mesh(
            rh_v,
            rh_f,
            rh_labels,
            outline_radius=outline_radius,
            outline_offset=outline_offset,
        )

        for i, (_, cfg) in enumerate(sel_views.items()):
            plotter.subplot(i // ncols, i % ncols)
            if cfg["side"] in {"L", "both"}:
                actor = plotter.add_mesh(
                    lh_mesh,
                    scalars="Data",
                    cmap=plot_cmap,
                    clim=clim,
                    n_colors=n_colors,
                    nan_color=nan_color,
                    show_scalar_bar=False,
                    smooth_shading=True,
                    interpolate_before_map=False,
                    **shading,
                )
                if lh_edges.n_points > 0:
                    plotter.add_mesh(
                        lh_edges,
                        color=outline_color,
                        line_width=outline_width,
                        render_lines_as_tubes=not outline_radius,
                    )
                if scalar_bar_mapper is None:
                    scalar_bar_mapper = actor.mapper

            if cfg["side"] in {"R", "both"}:
                actor = plotter.add_mesh(
                    rh_mesh,
                    scalars="Data",
                    cmap=plot_cmap,
                    clim=clim,
                    n_colors=n_colors,
                    nan_color=nan_color,
                    show_scalar_bar=False,
                    smooth_shading=True,
                    interpolate_before_map=False,
                    **shading,
                )
                if rh_edges.n_points > 0:
                    plotter.add_mesh(
                        rh_edges,
                        color=outline_color,
                        line_width=outline_width,
                        render_lines_as_tubes=not outline_radius,
                    )
                if scalar_bar_mapper is None:
                    scalar_bar_mapper = actor.mapper

            yp.set_camera(plotter, cfg, zoom=zoom)
            plotter.hide_axes()

        cbar_info = []
        if not is_cat and scalar_bar_mapper:
            if display_type != "matplotlib":
                yp.add_colorbars(plotter, [scalar_bar_mapper], [""], nrows, figsize)
            else:
                cbar_info.append({"cmap": cmap, "vminmax": list(clim)})

        export_path = self._resolve_export_path(export_path, save_plot, "cortical_outline")
        self.plot_result = yp.finalize_plot(
            plotter,
            str(export_path) if export_path is not None else None,
            display_type,
            ax=ax,
            cbar_info=cbar_info,
            cbar_kwargs=cbar_kwargs,
        )
        return self.plot_result

    def plot_subcortical(
        self,
        export_path: str | os.PathLike | None = None,
        save_plot: bool = True,
        **kwargs,
    ):
        yab = self._import_yabplot()
        export_path = self._resolve_export_path(export_path, save_plot, "subcortical")
        self.plot_result = yab.plot_subcortical(
            export_path=str(export_path) if export_path is not None else None,
            **kwargs,
        )
        return self.plot_result

    def plot_tracts(
        self,
        export_path: str | os.PathLike | None = None,
        save_plot: bool = True,
        **kwargs,
    ):
        yab = self._import_yabplot()
        export_path = self._resolve_export_path(export_path, save_plot, "tracts")
        self.plot_result = yab.plot_tracts(
            export_path=str(export_path) if export_path is not None else None,
            **kwargs,
        )
        return self.plot_result

    def plot_voxelwise(
        self,
        nii_path: str | os.PathLike | None = None,
        export_path: str | os.PathLike | None = None,
        save_plot: bool = True,
        **kwargs,
    ):
        nii_path = self._resolve_map_path(nii_path)
        yab = self._import_yabplot()
        export_path = self._resolve_export_path(export_path, save_plot, "voxelwise")
        self.plot_result = yab.plot_voxelwise(
            str(nii_path),
            export_path=str(export_path) if export_path is not None else None,
            **kwargs,
        )
        return self.plot_result

    def plot_connectome(
        self,
        export_path: str | os.PathLike | None = None,
        save_plot: bool = True,
        **kwargs,
    ):
        yab = self._import_yabplot()
        export_path = self._resolve_export_path(export_path, save_plot, "connectome")
        self.plot_result = yab.plot_connectome(
            export_path=str(export_path) if export_path is not None else None,
            **kwargs,
        )
        return self.plot_result

    def _prepare_projection_kwargs(
        self,
        project: str,
        projection_kwargs: dict[str, Any],
        bmesh_kwargs: dict[str, str],
    ) -> dict[str, Any]:
        projection_kwargs = dict(projection_kwargs)
        if project == "vol2surf":
            return self._merge_selection_kwargs(
                projection_kwargs,
                bmesh_kwargs,
                selection_name="bmesh",
                exclusive_keys={"bmesh"},
            )
        return projection_kwargs

    def _prepare_plot_kwargs(
        self,
        plot: str,
        plot_kwargs: dict[str, Any],
        bmesh_kwargs: dict[str, str],
        atlas_kwargs: dict[str, str],
    ) -> dict[str, Any]:
        plot_kwargs = dict(plot_kwargs)
        self._validate_display_type_dependencies(plot_kwargs.get("display_type"))
        plot_kwargs = self._resolve_plot_colormaps(plot_kwargs)

        if plot in {"vertexwise", "cortical", "cortical_outline", "subcortical", "tracts", "voxelwise"}:
            plot_kwargs = self._merge_selection_kwargs(
                plot_kwargs,
                bmesh_kwargs,
                selection_name="bmesh",
                exclusive_keys={"bmesh", "bmesh_type"},
            )
        elif plot == "connectome" and bmesh_kwargs:
            plot_kwargs = self._merge_selection_kwargs(
                plot_kwargs,
                {"bmesh_type": bmesh_kwargs["bmesh"]},
                selection_name="bmesh",
                exclusive_keys={"bmesh", "bmesh_type"},
            )

        if plot in {"cortical", "cortical_outline", "subcortical", "tracts", "connectome"}:
            plot_kwargs = self._merge_selection_kwargs(
                plot_kwargs,
                atlas_kwargs,
                selection_name="atlas",
                exclusive_keys={"atlas", "custom_atlas_path"},
            )

        if plot in {"cortical", "cortical_outline", "subcortical", "tracts"} and plot_kwargs.get("data") == {}:
            raise ValueError(
                "Empty data={} tells yabplot to plot continuous data with no region values, "
                "so the atlas will appear blank. Omit data or set data=None to render the "
                "atlas categorically."
            )

        return plot_kwargs

    @classmethod
    def _resolve_plot_colormaps(cls, plot_kwargs: dict[str, Any]) -> dict[str, Any]:
        plot_kwargs = dict(plot_kwargs)
        for key in ("cmap", "node_cmap", "edge_cmap"):
            if key in plot_kwargs:
                plot_kwargs[key] = cls._resolve_plot_colormap(plot_kwargs[key], key)
        return plot_kwargs

    @classmethod
    def _resolve_plot_colormap(cls, cmap, parameter_name: str = "cmap"):
        if not isinstance(cmap, str):
            return cmap

        from matplotlib import colormaps
        from pyvista.plotting.colors import get_cmap_safe

        from calvin_utils.plotting_utils.mricrogl_colormaps import (
            resolve_mricrogl_or_matplotlib_cmap,
        )

        resolved = resolve_mricrogl_or_matplotlib_cmap(cmap)
        if not isinstance(resolved, str):
            return resolved

        try:
            if resolved in colormaps:
                return resolved
            get_cmap_safe(resolved)
            return resolved
        except ValueError as exc:
            raise ValueError(cls._format_invalid_cmap_error(cmap, parameter_name)) from exc

    @staticmethod
    def _format_invalid_cmap_error(cmap: str, parameter_name: str = "cmap") -> str:
        from matplotlib import colormaps

        from calvin_utils.plotting_utils.mricrogl_colormaps import (
            DEFAULT_CLUT_DIR,
            MICROGL_LUT_SUFFIXES,
        )

        matplotlib_names = sorted(colormaps)
        mricrogl_names = []
        if DEFAULT_CLUT_DIR.exists():
            mricrogl_names = sorted(
                path.stem
                for path in DEFAULT_CLUT_DIR.iterdir()
                if path.suffix.lower() in MICROGL_LUT_SUFFIXES
            )

        return (
            f"Invalid colormap for {parameter_name}: {cmap!r}.\n"
            "Available bundled MRIcroGL LUTs:\n"
            f"{', '.join(mricrogl_names) if mricrogl_names else '<none found>'}\n\n"
            "Available Matplotlib/PyVista colormaps:\n"
            f"{', '.join(matplotlib_names)}"
        )

    @staticmethod
    def _validate_display_type_dependencies(display_type: str | None) -> None:
        if display_type != "interactive":
            return

        import importlib.util

        missing = [
            package
            for package in ("trame", "nest_asyncio2")
            if importlib.util.find_spec(package) is None
        ]
        if missing:
            raise ModuleNotFoundError(
                "display_type='interactive' requires PyVista's trame notebook "
                "backend. Install missing packages in this environment with: "
                f"pip install {' '.join(missing)}"
            )

    def _inject_projection_kwargs(
        self,
        project: str | None,
        plot: str,
        plot_kwargs: dict[str, Any],
        projection_kwargs: dict[str, Any] | None = None,
        threshold: float | tuple[float, float] | None = None,
        damage_score_metric: str = "avg_in_target",
        score_nonzero_only: bool = False,
    ) -> dict[str, Any]:
        projection_kwargs = dict(projection_kwargs or {})

        if plot == "subcortical" and project is None and self.map_path is not None:
            if "data" in plot_kwargs and plot_kwargs["data"] is not None:
                return plot_kwargs
            atlas = plot_kwargs.get("atlas")
            custom_atlas_path = plot_kwargs.get("custom_atlas_path")
            if atlas is None and custom_atlas_path is None:
                return plot_kwargs
            metric = plot_kwargs.pop("subcortical_score_metric", damage_score_metric)
            interpolation = projection_kwargs.pop("interpolation", "linear")
            nan_fill = projection_kwargs.pop("nan_fill", plot_kwargs.pop("nan_fill", 0.0))
            if projection_kwargs:
                raise ValueError(
                    "Unsupported projection_kwargs for automatic subcortical scoring: "
                    f"{sorted(projection_kwargs)}."
                )
            data = self._score_subcortical_atlas(
                atlas=atlas,
                custom_atlas_path=custom_atlas_path,
                metric=metric,
                interpolation=interpolation,
                nan_fill=nan_fill,
                threshold=threshold,
                score_nonzero_only=score_nonzero_only,
            )
            return {**plot_kwargs, "data": data}

        if project is None:
            return plot_kwargs

        if project == "vol2surf" and plot == "vertexwise":
            if "lh" in plot_kwargs or "rh" in plot_kwargs:
                return plot_kwargs
            return {
                **plot_kwargs,
                "lh": None,
                "rh": None,
            }

        if project == "vol2surf" and plot in {"cortical", "cortical_outline"}:
            if "data" in plot_kwargs and plot_kwargs["data"] is not None:
                return plot_kwargs
            metric = plot_kwargs.pop("surface_score_metric", damage_score_metric)
            atlas = plot_kwargs.get("atlas")
            custom_atlas_path = plot_kwargs.get("custom_atlas_path")
            if atlas is None and custom_atlas_path is None:
                raise ValueError(
                    "atlas or custom_atlas_path is required when auto-feeding "
                    "project='vol2surf' into a cortical parcel plot."
                )
            data = self._score_projected_surface_atlas(
                atlas=atlas,
                custom_atlas_path=custom_atlas_path,
                metric=metric,
                threshold=threshold,
                score_nonzero_only=score_nonzero_only,
            )
            return {**plot_kwargs, "data": data}

        if project == "vol2surf":
            raise ValueError(
                "project='vol2surf' produces cortical vertex arrays and can only be "
                "auto-fed into plot='vertexwise', plot='cortical', or "
                "plot='cortical_outline'. Use plot='voxelwise' for direct "
                "NIfTI rendering."
            )

        if project == "vol2tract" and plot != "tracts":
            raise ValueError(
                "project='vol2tract' produces tract sample arrays and is only "
                "compatible with plot='tracts'."
            )

        if project == "vol2tract" and plot == "tracts":
            if "data" in plot_kwargs and plot_kwargs["data"] is not None:
                return plot_kwargs
            atlas = plot_kwargs.get("atlas")
            custom_atlas_path = plot_kwargs.get("custom_atlas_path")
            if atlas is None and custom_atlas_path is None:
                raise ValueError(
                    "atlas or custom_atlas_path is required when auto-feeding "
                    "project='vol2tract' into plot='tracts'."
                )
            metric = plot_kwargs.pop("tract_score_metric", damage_score_metric)
            data = self._score_tract_atlas(
                atlas=atlas,
                custom_atlas_path=custom_atlas_path,
                metric=metric,
                projection_kwargs=projection_kwargs,
                threshold=threshold,
                score_nonzero_only=score_nonzero_only,
            )
            return {**plot_kwargs, "data": data}

        return plot_kwargs

    @staticmethod
    def _build_vertexwise_meshes(bmesh: str, lh_data, rh_data):
        import yabplot as yab
        from yabplot.data import get_surface_paths

        lh_path, rh_path = get_surface_paths(bmesh, "bmesh")
        return yab.load_vertexwise_mesh(lh_path, rh_path, lh_data, rh_data)

    def _score_projected_surface_atlas(
        self,
        atlas: str | None = None,
        custom_atlas_path: str | os.PathLike | None = None,
        metric: str = "avg_in_target",
        threshold: float | tuple[float, float] | None = None,
        score_nonzero_only: bool = False,
    ) -> dict[str, float]:
        import numpy as np
        import yabplot.plotting as yp
        from calvin_utils.neuroimaging_utils.nifti_utils.damage_score_utils import (
            DamageScorer,
        )

        if self.lh_data is None or self.rh_data is None:
            raise ValueError("vol2surf projection must run before cortical parcel scoring.")

        atlas_dir = yp._resolve_resource_path(atlas, "cortical", custom_path=custom_atlas_path)
        check_name = None if custom_atlas_path else atlas
        csv_path, lut_path = yp._find_cortical_files(atlas_dir, strict_name=check_name)
        labels = np.loadtxt(csv_path, dtype=int)
        lut_ids, _, lut_names, _ = yp.parse_lut(lut_path)
        surface_values = np.concatenate([self.lh_data, self.rh_data]).astype(float)
        surface_values = self._threshold_values(surface_values, threshold)

        if surface_values.shape[0] != labels.shape[0]:
            raise ValueError(
                "Projected surface data and atlas labels are not in the same vertex space: "
                f"data length={surface_values.shape[0]}, labels length={labels.shape[0]}."
            )

        scores = {}
        for rid in lut_ids:
            roi = (labels == rid).astype(float)
            roi[~np.isfinite(surface_values)] = 0.0
            name = lut_names[rid]
            scores[name] = DamageScorer._calculate_metrics(
                surface_values,
                roi,
                [metric],
                score_nonzero_only=score_nonzero_only,
            )[metric]
        self.parcel_scores = scores
        return scores

    def _score_subcortical_atlas(
        self,
        atlas: str | None = None,
        custom_atlas_path: str | os.PathLike | None = None,
        metric: str = "avg_in_target",
        interpolation: str = "linear",
        nan_fill: float | None = 0.0,
        threshold: float | tuple[float, float] | None = None,
        score_nonzero_only: bool = False,
    ) -> dict[str, float]:
        import numpy as np
        import pyvista as pv
        import yabplot as yab
        import yabplot.plotting as yp
        from calvin_utils.neuroimaging_utils.nifti_utils.damage_score_utils import (
            DamageScorer,
        )

        nii_path = self._resolve_map_path(None)
        project_path = self._prepare_projection_volume(nii_path, nan_fill)
        atlas_dir = yp._resolve_resource_path(atlas, "subcortical", custom_path=custom_atlas_path)
        file_map = yp._find_subcortical_files(atlas_dir)
        names = yab.get_atlas_regions(
            atlas=atlas,
            category="subcortical",
            custom_atlas_path=custom_atlas_path,
        )

        try:
            scores = {}
            for name in names:
                fpath = file_map.get(name)
                if not fpath:
                    continue
                sampled = self._sample_nifti_at_points(
                    project_path,
                    pv.read(fpath).points,
                    interpolation=interpolation,
                )
                sampled = self._threshold_values(sampled, threshold)
                roi = np.isfinite(sampled).astype(float)
                scores[name] = DamageScorer._calculate_metrics(
                    sampled,
                    roi,
                    [metric],
                    score_nonzero_only=score_nonzero_only,
                )[metric]
        finally:
            self._cleanup_projection_volume(project_path, nii_path)

        self.parcel_scores = scores
        return scores

    def _score_tract_atlas(
        self,
        atlas: str | None = None,
        custom_atlas_path: str | os.PathLike | None = None,
        metric: str = "avg_in_target",
        projection_kwargs: dict[str, Any] | None = None,
        threshold: float | tuple[float, float] | None = None,
        score_nonzero_only: bool = False,
    ) -> dict[str, float]:
        import numpy as np
        import yabplot as yab
        import yabplot.plotting as yp
        from calvin_utils.neuroimaging_utils.nifti_utils.damage_score_utils import (
            DamageScorer,
        )

        projection_kwargs = dict(projection_kwargs or {})
        interpolation = projection_kwargs.pop("interpolation", "linear")
        nan_fill = projection_kwargs.pop("nan_fill", 0.0)
        if projection_kwargs:
            raise ValueError(
                "Unsupported projection_kwargs for project='vol2tract' atlas scoring: "
                f"{sorted(projection_kwargs)}."
            )

        nii_path = self._resolve_map_path(None)
        project_path = self._prepare_projection_volume(nii_path, nan_fill)
        atlas_dir = yp._resolve_resource_path(atlas, "tracts", custom_path=custom_atlas_path)
        file_map = yp._find_tract_files(atlas_dir)
        names = yab.get_atlas_regions(
            atlas=atlas,
            category="tracts",
            custom_atlas_path=custom_atlas_path,
        )

        try:
            scores = {}
            for name in names:
                fpath = file_map.get(name)
                if not fpath:
                    continue
                sampled = yab.project_vol2tract(
                    fpath,
                    str(project_path),
                    interpolation=interpolation,
                )
                sampled = self._threshold_values(sampled, threshold)
                roi = np.isfinite(sampled).astype(float)
                scores[name] = DamageScorer._calculate_metrics(
                    np.asarray(sampled, dtype=float),
                    roi,
                    [metric],
                    score_nonzero_only=score_nonzero_only,
                )[metric]
        finally:
            self._cleanup_projection_volume(project_path, nii_path)

        self.projection_kind = "vol2tract"
        self.tract_data = scores
        self.parcel_scores = scores
        return scores

    @staticmethod
    def _sample_nifti_at_points(nii_path: Path, points, interpolation: str = "linear"):
        import nibabel as nib
        import numpy as np
        from scipy.ndimage import map_coordinates

        if interpolation not in {"linear", "nearest"}:
            raise ValueError("interpolation must be 'linear' or 'nearest'.")

        img = nib.load(nii_path)
        data = img.get_fdata()
        if data.ndim > 3:
            data = data[..., 0]

        points = np.asarray(points, dtype=float)
        coords_homo = np.hstack([points, np.ones((points.shape[0], 1))])
        vox_coords = np.linalg.inv(img.affine).dot(coords_homo.T)[:3, :]
        order = 1 if interpolation == "linear" else 0
        return map_coordinates(data, vox_coords, order=order, mode="nearest")

    @staticmethod
    def _threshold_values(values, threshold: float | tuple[float, float] | None = None):
        import numpy as np

        if threshold is None:
            return np.asarray(values, dtype=float)

        out = np.asarray(values, dtype=float).copy()
        finite = np.isfinite(out)

        if isinstance(threshold, tuple):
            if len(threshold) != 2:
                raise ValueError("threshold tuple must contain exactly two values: (low, high).")
            low, high = threshold
            if low > high:
                raise ValueError(f"threshold lower bound must be <= upper bound; got {threshold}.")
            out[finite & (out >= low) & (out <= high)] = np.nan
            return out

        if isinstance(threshold, bool):
            raise TypeError("threshold must be a number, a (low, high) tuple, or None.")

        try:
            cutoff = float(threshold)
        except (TypeError, ValueError) as exc:
            raise TypeError("threshold must be a number, a (low, high) tuple, or None.") from exc

        out[finite & (out < cutoff)] = np.nan
        return out

    @staticmethod
    def _prepare_projection_volume(nii_path: Path, nan_fill: float | None) -> Path:
        if nan_fill is None:
            return nii_path

        import nibabel as nib
        import numpy as np

        img = nib.load(nii_path)
        data = img.get_fdata()
        if np.isfinite(data).all():
            return nii_path

        filled = np.nan_to_num(
            data,
            nan=nan_fill,
            posinf=nan_fill,
            neginf=nan_fill,
        )
        tmp = NamedTemporaryFile(suffix=".nii.gz", delete=False)
        tmp.close()
        nib.save(nib.Nifti1Image(filled, img.affine, img.header), tmp.name)
        return Path(tmp.name)

    @staticmethod
    def _cleanup_projection_volume(project_path: Path, source_path: Path) -> None:
        if project_path == source_path:
            return
        try:
            project_path.unlink()
        except FileNotFoundError:
            return

    @staticmethod
    def _merge_selection_kwargs(
        kwargs: dict[str, Any],
        selection_kwargs: dict[str, Any],
        selection_name: str,
        exclusive_keys: set[str] | None = None,
    ) -> dict[str, Any]:
        if not selection_kwargs:
            return kwargs
        conflict_keys = exclusive_keys if exclusive_keys is not None else set(selection_kwargs)
        duplicate_keys = sorted(set(kwargs) & conflict_keys)
        if duplicate_keys:
            raise ValueError(
                f"{selection_name} was provided through run() and also in kwargs: "
                f"{duplicate_keys}. Provide it in only one place."
            )
        return {**kwargs, **selection_kwargs}

    @staticmethod
    def _make_boundary_edge_mesh(
        vertices,
        faces,
        labels,
        outline_radius: float | None = 0.12,
        outline_offset: float = 0.2,
    ):
        import numpy as np
        import pyvista as pv

        edge_set = set()
        for face in np.asarray(faces, dtype=int):
            for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                label_a = labels[a]
                label_b = labels[b]
                if label_a == label_b or label_a == 0 or label_b == 0:
                    continue
                edge_set.add(tuple(sorted((int(a), int(b)))))

        if not edge_set:
            return pv.PolyData()

        vertices = np.asarray(vertices, dtype=float)
        if outline_offset:
            normals = vertices - vertices.mean(axis=0)
            norms = np.linalg.norm(normals, axis=1)
            normals[norms > 0] /= norms[norms > 0, None]
            vertices = vertices + normals * outline_offset

        lines = np.array(
            [[2, edge[0], edge[1]] for edge in sorted(edge_set)],
            dtype=np.int64,
        ).ravel()
        edge_mesh = pv.PolyData(vertices, lines=lines)
        if outline_radius:
            return edge_mesh.tube(radius=outline_radius, n_sides=6)
        return edge_mesh

    def _resolve_map_path(self, nii_path: str | os.PathLike | None) -> Path:
        path = Path(nii_path).expanduser() if nii_path is not None else self.map_path
        if path is None:
            raise ValueError("nii_path or map_path is required.")
        if not path.exists():
            raise FileNotFoundError(f"nii_path does not exist: {path}")
        return path

    def _resolve_export_path(
        self,
        export_path: str | os.PathLike | None,
        save_plot: bool,
        suffix: str,
    ) -> Path | None:
        if export_path is not None:
            self.plot_output_path = Path(export_path).expanduser()
        elif save_plot and self.out_file is not None:
            self.plot_output_path = self._out_file_with_suffix(suffix)
        else:
            self.plot_output_path = None

        if self.plot_output_path is not None:
            self.plot_output_path.parent.mkdir(parents=True, exist_ok=True)
        return self.plot_output_path

    def _out_file_with_suffix(self, suffix: str) -> Path:
        ext = self.out_file.suffix or ".png"
        stem_path = self.out_file.with_suffix("") if self.out_file.suffix else self.out_file
        if stem_path.name.endswith(f"_{suffix}"):
            return stem_path.with_suffix(ext)
        return stem_path.with_name(f"{stem_path.name}_{suffix}").with_suffix(ext)

    @staticmethod
    def _import_yabplot():
        try:
            import yabplot as yab
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "yabplot is not installed in this Python environment. Install it in "
                "the same environment running this code with `pip install yabplot`."
            ) from exc
        return yab


YabPlotter = ParcelwisePlot
ParcelwiseYabPlot = ParcelwisePlot
