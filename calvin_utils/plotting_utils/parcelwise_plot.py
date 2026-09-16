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

    Cortical atlas with a SUIT cerebellum
    ------------------------------------
    ``plot="cortical"`` with ``include_cerebellum=True`` draws the cerebellum
    from the *subcortical* meshes of the selected atlas. AAL3's cerebellar
    volumes are coarse; ``cerebellum_atlas_path`` points the cerebellum at a
    different mesh directory while the cortex keeps its own parcellation::

        plotter.run(
            project="vol2surf",
            plot="cortical",
            atlas="aal3",
            plot_kwargs={
                "include_cerebellum": True,
                "cerebellum_atlas_path": "~/hires_backdrops/suit/yabplot_suit/subcortical",
                "views": ["left_lateral", "posterior", "inferior"],
            },
        )

    ``cerebellum_atlas_path`` names the cerebellar *geometry* directory once,
    for both modes: the root written by ``suit_cerebellum`` (containing
    ``surface/`` and ``subcortical/``) or either leaf directly.

    ``cerebellum_parcel_atlas`` names what *parcellates* it, independently. In
    ``"surface"`` mode it is a label volume (or a directory holding one) whose
    regions colour the hemispheres; in ``"parcels"`` mode it is a mesh
    directory or a yabplot registry name such as ``"aal3"``, drawn in place of
    the surface. ``None`` uses whatever ships with the geometry.

    ``cerebellum_mode`` chooses the geometry. ``"surface"`` (the default) draws
    one closed mesh per hemisphere and colours it per vertex, so the cerebellum
    reads as a
    single object and its parcellation is independent of both its geometry and
    the cortical atlas; with no map and no ``cerebellum_data`` it renders as
    plain context geometry in ``cerebellum_base_color``. ``"parcels"`` draws one
    flat-coloured mesh per lobule from ``cerebellum_atlas_path``.

    In ``"parcels"`` mode, region names in that directory must contain one of ``cerebellum_filter``
    (default ``("cerebel", "vermis")``) to be drawn, and carry a trailing
    ``_L``/``_R`` so lateral views can cull the far hemisphere. Build such a
    directory with
    ``calvin_utils.neuroimaging_utils.nifti_utils.suit_cerebellum``.

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
        include_cerebellum: bool = False,
        **kwargs,
    ):
        if include_cerebellum:
            return self._plot_cortical_with_cerebellum(
                export_path=export_path,
                save_plot=save_plot,
                **kwargs,
            )

        yab = self._import_yabplot()
        export_path = self._resolve_export_path(
            export_path,
            save_plot,
            "cortical",
        )

        self.plot_result = yab.plot_cortical(
            export_path=str(export_path) if export_path is not None else None,
            **kwargs,
        )

        return self.plot_result
    def _plot_cortical_with_cerebellum(
        self,
        data=None,
        atlas="aal3",
        custom_atlas_path=None,
        cerebellum_atlas=None,
        cerebellum_atlas_path=None,
        cerebellum_mode="surface",
        cerebellum_surface_path=None,
        cerebellum_parcel_atlas=None,
        cerebellum_data=None,
        cerebellum_parcellate=True,
        cerebellum_base_color=(0.72, 0.72, 0.76),
        bmesh="midthickness",
        views=None,
        layout=None,
        figsize=None,
        cmap="coolwarm",
        vminmax=(None, None),
        nan_color=(1.0, 1.0, 1.0),
        style="default",
        zoom=1.2,
        display_type="matplotlib",
        export_path=None,
        save_plot=True,
        cbar_kwargs=None,
        cerebellum_score_metric="avg_in_target",
        cerebellum_filter=("cerebel", "vermis"),
    ):
        import numpy as np
        import pyvista as pv
        import yabplot as yab
        import yabplot.plotting as yp

        from yabplot.data import get_surface_paths
        from yabplot.utils import load_gii

        # ------------------------------------------------------------------
        # Cortex: same machinery as yabplot.plot_cortical()
        # ------------------------------------------------------------------

        # The cerebellum may come from a different atlas than the cortex: the
        # cortical parcellation lives on fsLR32k surfaces while the cerebellum
        # is drawn from volumetric meshes, so there is no reason the two must
        # be the same resource. Falls back to the cortical selection.
        cerebellum_atlas_resolved = (
            cerebellum_atlas if cerebellum_atlas is not None else atlas
        )
        cerebellum_path_resolved = (
            cerebellum_atlas_path
            if cerebellum_atlas_path is not None
            else custom_atlas_path
        )
        if cerebellum_path_resolved is not None:
            cerebellum_path_resolved = str(
                Path(cerebellum_path_resolved).expanduser()
            )
        if cerebellum_atlas is not None and cerebellum_atlas_path is not None:
            raise ValueError(
                "Pass cerebellum_atlas or cerebellum_atlas_path, not both."
            )

        atlas_dir = yp._resolve_resource_path(
            atlas,
            "cortical",
            custom_path=custom_atlas_path,
        )

        check_name = None if custom_atlas_path else atlas

        csv_path, lut_path = yp._find_cortical_files(
            atlas_dir,
            strict_name=check_name,
        )

        target_labels = np.loadtxt(csv_path, dtype=int)

        lut_ids, _, lut_names, _ = yp.parse_lut(lut_path)

        all_vals = yp.map_values_to_surface(
            data,
            target_labels,
            lut_ids,
            lut_names,
        )

        lh_path, rh_path = get_surface_paths(
            bmesh,
            "bmesh",
        )

        lh_v, lh_f = load_gii(lh_path)
        rh_v, rh_f = load_gii(rh_path)

        lh_vals = all_vals[:len(lh_v)]
        rh_vals = all_vals[len(lh_v):]

        lh_mesh = yp.make_cortical_mesh(
            lh_v,
            lh_f,
            lh_vals,
        )

        rh_mesh = yp.make_cortical_mesh(
            rh_v,
            rh_f,
            rh_vals,
        )

        # ------------------------------------------------------------------
        # Cerebellum: use AAL3 subcortical meshes
        # ------------------------------------------------------------------

        # custom_atlas_path was previously dropped here, so a custom atlas
        # scored against the registry copy of `atlas` while its meshes came
        # from the custom directory. Both now use the same resolved source.
        if cerebellum_mode not in {"surface", "parcels"}:
            raise ValueError(
                f'cerebellum_mode must be "surface" or "parcels", got '
                f"{cerebellum_mode!r}"
            )

        # A registry atlas (aal3, aseg, ...) has no local directory: let
        # yabplot's own resolver fetch it and skip the layout logic.
        if cerebellum_path_resolved is None and cerebellum_mode == "parcels":
            cerebellum_dir, default_label_path = None, None
        else:
            cerebellum_dir, default_label_path = self._resolve_cerebellum_source(
                cerebellum_surface_path or cerebellum_path_resolved
                if cerebellum_mode == "surface"
                else cerebellum_path_resolved,
                cerebellum_mode,
            )

        if cerebellum_mode == "surface":
            label_path, lut_path = self._resolve_cerebellum_labels(
                cerebellum_parcel_atlas, cerebellum_dir, default_label_path
            )
            cerebellum_meshes, cerebellar_finite = self._cerebellum_surface_meshes(
                surface_path=cerebellum_dir,
                label_path=label_path,
                lut_path=lut_path,
                data=cerebellum_data,
                metric=cerebellum_score_metric,
                parcellate=cerebellum_parcellate,
            )
        else:
            parcel_atlas, parcel_dir = self._resolve_cerebellum_parcels(
                cerebellum_parcel_atlas, cerebellum_atlas_resolved, cerebellum_dir
            )
            cerebellum_meshes, cerebellar_finite = self._cerebellum_parcel_meshes(
                atlas=parcel_atlas,
                atlas_path=parcel_dir,
                metric=cerebellum_score_metric,
                cerebellum_filter=cerebellum_filter,
                parcellate=cerebellum_parcellate,
            )

        # ------------------------------------------------------------------
        # Shared colour range
        # ------------------------------------------------------------------

        cortical_finite = np.concatenate([
            lh_vals[np.isfinite(lh_vals)],
            rh_vals[np.isfinite(rh_vals)],
        ])

        finite = np.concatenate([
            cortical_finite,
            cerebellar_finite,
        ])

        vmin = (
            vminmax[0]
            if vminmax[0] is not None
            else np.nanmin(finite)
        )

        vmax = (
            vminmax[1]
            if vminmax[1] is not None
            else np.nanmax(finite)
        )

        clim = (vmin, vmax)

        # ------------------------------------------------------------------
        # ONE PyVista scene
        # ------------------------------------------------------------------

        sel_views = yp.get_view_configs(views)

        ax, display_type, figsize = yp.prepare_plotter(
            None,
            display_type,
            sel_views,
            layout,
            figsize,
        )

        plotter, ncols, nrows = yp.setup_plotter(
            sel_views,
            layout,
            figsize,
            display_type,
            needs_bottom_row=True,
        )

        shading = yp.get_shading_preset(style)

        scalar_bar_mapper = None

        for i, (_, cfg) in enumerate(sel_views.items()):

            plotter.subplot(
                i // ncols,
                i % ncols,
            )

            # Cortex
            if cfg["side"] in {"L", "both"}:

                actor = plotter.add_mesh(
                    lh_mesh,
                    scalars="Data",
                    cmap=cmap,
                    clim=clim,
                    nan_color=nan_color,
                    show_scalar_bar=False,
                    smooth_shading=True,
                    interpolate_before_map=False,
                    **shading,
                )

                if scalar_bar_mapper is None:
                    scalar_bar_mapper = actor.mapper

            if cfg["side"] in {"R", "both"}:

                actor = plotter.add_mesh(
                    rh_mesh,
                    scalars="Data",
                    cmap=cmap,
                    clim=clim,
                    nan_color=nan_color,
                    show_scalar_bar=False,
                    smooth_shading=True,
                    interpolate_before_map=False,
                    **shading,
                )

                if scalar_bar_mapper is None:
                    scalar_bar_mapper = actor.mapper

            # Cerebellum
            for name, mesh in cerebellum_meshes.items():

                name_lower = name.lower()

                if cfg["side"] == "L" and (
                    "_r" in name_lower
                    or "right" in name_lower
                ):
                    continue

                if cfg["side"] == "R" and (
                    "_l" in name_lower
                    or "left" in name_lower
                ):
                    continue

                if "Data" in mesh.point_data:
                    plotter.add_mesh(
                        mesh,
                        scalars="Data",
                        cmap=cmap,
                        clim=clim,
                        nan_color=nan_color,
                        show_scalar_bar=False,
                        smooth_shading=True,
                        **shading,
                    )
                else:
                    # No parcel values available: draw the cerebellum as plain
                    # context geometry rather than a blank scalar field.
                    plotter.add_mesh(
                        mesh,
                        color=cerebellum_base_color,
                        show_scalar_bar=False,
                        smooth_shading=True,
                        **shading,
                    )

            yp.set_camera(
                plotter,
                cfg,
                zoom=zoom,
            )

            plotter.hide_axes()

        # ------------------------------------------------------------------
        # Colorbar + export
        # ------------------------------------------------------------------

        cbar_info = []

        if scalar_bar_mapper is not None:

            if display_type != "matplotlib":

                yp.add_colorbars(
                    plotter,
                    [scalar_bar_mapper],
                    [""],
                    nrows,
                    figsize,
                )

            else:

                cbar_info.append(
                    {
                        "cmap": cmap,
                        "vminmax": list(clim),
                    }
                )

        export_path = self._resolve_export_path(
            export_path,
            save_plot,
            "cortical",
        )

        self.plot_result = yp.finalize_plot(
            plotter,
            str(export_path)
            if export_path is not None
            else None,
            display_type,
            ax=ax,
            cbar_info=cbar_info,
            cbar_kwargs=cbar_kwargs,
        )

        return self.plot_result

    # ------------------------------------------------------------------ #
    # Cerebellum geometry
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_cerebellum_labels(spec, geom_dir, default_label_path):
        """Pick the label volume that colours the cerebellar surface.

        The geometry and the parcellation are separate choices: the SUIT
        surfaces are just a shape, and any integer label volume in the same
        space can colour them. ``spec`` may be

        * ``None``            -> the labels shipped beside the geometry (SUIT);
        * a ``.nii`` /``.nii.gz`` -> that volume;
        * a directory        -> ``labels.nii.gz`` in it, or in its ``surface``
                                subdirectory.

        The lookup table is taken from beside the label volume: ``atlas_LUT``,
        else ``<stem>.txt``, else the only ``.txt`` present -- matching how
        yabplot finds region orders. Names in it must match the region names
        used for scoring, which is what ties the two together.
        """
        if spec is None:
            return default_label_path, None

        spec = Path(spec).expanduser()
        if spec.is_dir():
            for cand in (spec / "labels.nii.gz", spec / "surface" / "labels.nii.gz"):
                if cand.exists():
                    label_path = cand
                    break
            else:
                raise FileNotFoundError(
                    f"No labels.nii.gz under {spec}. For colouring a surface, "
                    "cerebellum_parcel_atlas must name a label volume or a "
                    "directory containing one."
                )
        elif spec.suffix in (".nii", ".gz"):
            if not spec.exists():
                raise FileNotFoundError(f"Label volume not found: {spec}")
            label_path = spec
        else:
            raise ValueError(
                f"cerebellum_parcel_atlas={str(spec)!r} is not a label volume or "
                'directory. Registry atlas names only work with '
                'cerebellum_mode="parcels", which renders their meshes rather '
                "than colouring a surface."
            )

        stem = label_path.name.split(".")[0]
        candidates = [
            label_path.parent / "atlas_LUT.txt",
            label_path.parent / f"{stem}.txt",
            label_path.parent / f"{stem}_LUT.txt",
        ]
        lut_path = next((c for c in candidates if c.exists()), None)
        if lut_path is None:
            txts = sorted(label_path.parent.glob("*.txt"))
            if len(txts) == 1:
                lut_path = txts[0]
        if lut_path is None:
            raise FileNotFoundError(
                f"No lookup table found beside {label_path}. Expected "
                f"atlas_LUT.txt or {stem}.txt with '<id> <name>' per line."
            )
        return label_path, lut_path

    @staticmethod
    def _resolve_cerebellum_parcels(spec, default_atlas, default_dir):
        """Pick the mesh set drawn in ``cerebellum_mode="parcels"``.

        ``spec`` may be ``None`` (the geometry root's own ``subcortical``
        meshes), a directory of ``.vtk`` meshes, or a yabplot registry atlas
        name such as ``"aal3"``, which is fetched on demand.
        """
        if spec is None:
            return default_atlas, str(default_dir) if default_dir else None

        candidate = Path(spec).expanduser()
        if candidate.is_dir():
            resolved, _ = ParcelwisePlot._resolve_cerebellum_source(candidate, "parcels")
            return None, str(resolved)
        return str(spec), None

    @staticmethod
    def _resolve_cerebellum_source(base, mode: str):
        """Find the geometry directory and label volume for one atlas root.

        A cerebellar atlas is one directory tree, so callers name it once. The
        builder writes ``<root>/surface`` and ``<root>/subcortical``; either
        the root or a leaf directory is accepted, so existing configs that
        point straight at ``.../subcortical`` keep working.

        Resolving explicitly also avoids a trap: yabplot's mesh scan searches
        the given directory *and one level of subdirectories*, so handing it
        the root would silently collect ``Cerebellum_L.vtk`` and
        ``Cerebellum_R.vtk`` alongside the 32 lobules and draw both.
        """
        if base is None:
            raise ValueError(
                "No cerebellar atlas path given. Set cerebellum_atlas_path to "
                "the directory produced by "
                "calvin_utils.neuroimaging_utils.nifti_utils.suit_cerebellum."
            )
        base = Path(base).expanduser()
        if not base.is_dir():
            raise FileNotFoundError(f"Cerebellar atlas directory not found: {base}")

        if mode == "surface":
            for cand in (base / "surface", base):
                if (cand / "Cerebellum_L.vtk").exists():
                    return cand, cand / "labels.nii.gz"
            raise FileNotFoundError(
                f"No Cerebellum_L.vtk under {base} or {base / 'surface'}. "
                "Rebuild with suit_cerebellum (it writes the surface by "
                'default), or use cerebellum_mode="parcels".'
            )

        for cand in (base / "subcortical", base):
            if cand.is_dir() and any(cand.glob("*.vtk")):
                return cand, None
        raise FileNotFoundError(f"No .vtk meshes under {base} or {base / 'subcortical'}.")

    def _cerebellum_parcel_meshes(
        self, atlas, atlas_path, metric, cerebellum_filter, parcellate=True
    ):
        """One flat-coloured mesh per cerebellar region (the original mode).

        With ``parcellate=False`` the meshes come back without scalars and the
        caller draws them all in the base colour.
        """
        import pyvista as pv

        if not parcellate:
            import yabplot.plotting as yp

            file_map = yp._find_subcortical_files(
                yp._resolve_resource_path(atlas, "subcortical", custom_path=atlas_path)
            )
            meshes = {
                name: pv.read(path)
                for name, path in file_map.items()
                if any(t.lower() in name.lower() for t in cerebellum_filter)
            }
            import numpy as np

            return meshes, np.asarray([], dtype=float)

        import numpy as np
        import pyvista as pv
        import yabplot.plotting as yp

        scores = self._score_subcortical_atlas(
            atlas=atlas,
            custom_atlas_path=atlas_path,
            metric=metric,
        )
        file_map = yp._find_subcortical_files(
            yp._resolve_resource_path(atlas, "subcortical", custom_path=atlas_path)
        )
        names = [
            name
            for name in scores
            if any(token.lower() in name.lower() for token in cerebellum_filter)
        ]
        if not names:
            raise ValueError(
                f"No cerebellar regions matched cerebellum_filter={cerebellum_filter!r} "
                f"in atlas {atlas_path or atlas!r}. Available: {sorted(scores)}."
            )

        meshes = {}
        for name in names:
            fpath = file_map.get(name)
            if fpath is None:
                continue
            mesh = pv.read(fpath)
            mesh["Data"] = np.full(mesh.n_points, scores[name], dtype=float)
            meshes[name] = mesh

        finite = np.asarray(
            [scores[n] for n in names if np.isfinite(scores[n])], dtype=float
        )
        return meshes, finite

    def _cerebellum_surface_meshes(
        self, surface_path, label_path, data, metric, lut_path=None, parcellate=True
    ):
        """Whole-hemisphere surfaces coloured per vertex from a label volume.

        The geometry (``Cerebellum_L.vtk`` / ``Cerebellum_R.vtk``) carries no
        parcellation of its own. Any integer label volume in the same space can
        colour it, so the cerebellar parcellation is chosen independently of
        the geometry -- and independently of the cortical atlas.

        With ``parcellate=False``, or with no map and no explicit ``data``, the
        surfaces come back without a scalar array and the caller draws them as
        plain context geometry.
        """
        import numpy as np
        import pyvista as pv

        if surface_path is None:
            raise ValueError(
                'cerebellum_mode="surface" needs cerebellum_surface_path '
                "pointing at a directory containing Cerebellum_L.vtk and "
                "Cerebellum_R.vtk (build one with "
                "calvin_utils.neuroimaging_utils.nifti_utils.suit_cerebellum)."
            )

        surface_dir = Path(surface_path).expanduser()
        meshes = {}
        for name in ("Cerebellum_L", "Cerebellum_R"):
            fpath = surface_dir / f"{name}.vtk"
            if not fpath.exists():
                raise FileNotFoundError(f"Missing cerebellar surface: {fpath}")
            meshes[name] = pv.read(fpath)

        label_path = (
            Path(label_path).expanduser()
            if label_path is not None
            else surface_dir / "labels.nii.gz"
        )
        lut_path = (
            Path(lut_path).expanduser()
            if lut_path is not None
            else surface_dir / "atlas_LUT.txt"
        )

        # No parcellation to show: hand back bare geometry. parcellate=False is
        # the explicit request; the rest catch the cases where there is simply
        # nothing to colour with.
        if not parcellate or not label_path.exists() or (
            data is None and self.map_path is None
        ):
            return meshes, np.asarray([], dtype=float)

        lut = self._read_label_lut(lut_path)
        scores = (
            dict(data)
            if data is not None
            else self._score_label_volume(label_path, lut, metric=metric)
        )
        self.parcel_scores = scores

        # Restrict each hemisphere's lookup to its own regions plus the
        # midline. SUIT's labels do not stop cleanly at x=0 and the clipped
        # cut face sits right on it, so an unconstrained nearest-label search
        # paints a strip of the left surface with right-hemisphere values.
        allowed = {}
        for name in meshes:
            side = name[-1]
            other = f"_{'R' if side == 'L' else 'L'}"
            allowed[name] = {
                rid
                for rid, region in lut.items()
                if not region.endswith(other)
            }

        vertex_labels = self._labels_at_points(
            label_path,
            {k: m.points for k, m in meshes.items()},
            allowed_ids=allowed,
        )
        values = np.array(
            [scores.get(lut.get(i), np.nan) for i in range(max(lut) + 1)],
            dtype=float,
        )
        values[0] = np.nan

        finite = []
        for name, mesh in meshes.items():
            ids = np.clip(vertex_labels[name], 0, len(values) - 1)
            arr = values[ids]
            mesh["Data"] = arr
            finite.append(arr[np.isfinite(arr)])

        return meshes, np.concatenate(finite) if finite else np.asarray([])

    @staticmethod
    def _read_label_lut(lut_path) -> dict[int, str]:
        """``<id> <name>`` per line -> ``{id: name}``."""
        lut = {}
        for line in Path(lut_path).read_text().splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                lut[int(parts[0])] = parts[-1]
            except ValueError:
                continue
        if not lut:
            raise ValueError(f"No usable entries in {lut_path}")
        return lut

    def _score_label_volume(
        self,
        label_path,
        lut: dict[int, str],
        metric: str = "avg_in_target",
        interpolation: str = "linear",
        nan_fill: float | None = 0.0,
        threshold=None,
        score_nonzero_only: bool = False,
    ) -> dict[str, float]:
        """Score the map inside each label, volumetrically.

        The subcortical scorer samples at mesh vertices, which for a closed
        region means the map is only ever read on its shell -- fine for an
        average, wrong for ``max_in_roi`` on a small structure whose peak sits
        in the interior. Here every labelled voxel is sampled instead. The map
        is read at the label grid's world coordinates, so the two volumes need
        not share a grid.
        """
        import nibabel as nib
        import numpy as np
        from calvin_utils.neuroimaging_utils.nifti_utils.damage_score_utils import (
            DamageScorer,
        )

        nii_path = self._resolve_map_path(None)
        project_path = self._prepare_projection_volume(nii_path, nan_fill)
        try:
            lab_img = nib.load(str(label_path))
            lab = np.asarray(lab_img.dataobj)
            idx = np.argwhere(lab > 0)
            world = nib.affines.apply_affine(lab_img.affine, idx)
            sampled = self._sample_nifti_at_points(
                project_path, world, interpolation=interpolation
            )
            sampled = self._threshold_values(sampled, threshold)
            ids = lab[tuple(idx.T)]

            scores = {}
            for rid, name in lut.items():
                sel = sampled[ids == rid]
                if sel.size == 0:
                    continue
                roi = np.isfinite(sel).astype(float)
                scores[name] = DamageScorer._calculate_metrics(
                    sel, roi, [metric], score_nonzero_only=score_nonzero_only
                )[metric]
        finally:
            self._cleanup_projection_volume(project_path, nii_path)
        return scores

    @staticmethod
    def _labels_at_points(label_path, point_sets: dict, allowed_ids: dict | None = None):
        """Nearest label id for each vertex.

        Surface vertices sit exactly on the iso-contour, so rounding them to a
        voxel lands outside the labelled volume about half the time. Rather
        than nudging along normals, every background voxel is pre-mapped to its
        nearest labelled voxel, which is exact and handles the midline cut
        faces too.
        """
        import nibabel as nib
        import numpy as np
        from scipy import ndimage

        lab_img = nib.load(str(label_path))
        lab = np.asarray(lab_img.dataobj).astype(np.int32)
        inv = np.linalg.inv(lab_img.affine)

        cache = {}

        def nearest_for(key, allowed):
            if key in cache:
                return cache[key]
            field = lab if allowed is None else np.where(np.isin(lab, list(allowed)), lab, 0)
            _, inds = ndimage.distance_transform_edt(field == 0, return_indices=True)
            cache[key] = field[inds[0], inds[1], inds[2]]
            return cache[key]

        out = {}
        for name, pts in point_sets.items():
            allowed = (allowed_ids or {}).get(name)
            key = None if allowed is None else tuple(sorted(allowed))
            nearest = nearest_for(key, allowed)
            vox = np.rint(nib.affines.apply_affine(inv, np.asarray(pts))).astype(int)
            for axis in range(3):
                vox[:, axis] = np.clip(vox[:, axis], 0, lab.shape[axis] - 1)
            out[name] = nearest[vox[:, 0], vox[:, 1], vox[:, 2]]
        return out

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

        from calvin_utils.plotting_utils.colormaps import (
            resolve_cmap,
        )

        resolved = resolve_cmap(cmap)
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

        from calvin_utils.plotting_utils.colormaps import (
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
