"""Plotting utilities.

The names below are importable from here as always -- ``from
calvin_utils.plotting_utils import ParcelwisePlot`` -- but each is loaded the
first time it is asked for, not when the package is. Importing one module of
this package (``plotting_utils.simple_box_plot``, say) then costs only what that module
needs, which is what lets an application ship the renderer without the
statistics and plotting stack these names pull in.
"""

from importlib import import_module

# name -> the module it lives in
_LAZY = {
    "CustomYabplotAtlasBuilder": "calvin_utils.neuroimaging_utils.nifti_utils.generate_atlas",
    "ParcelwisePlot": "calvin_utils.plotting_utils.parcelwise_plot",
    "ParcelwiseYabPlot": "calvin_utils.plotting_utils.parcelwise_plot",
    "YabPlotter": "calvin_utils.plotting_utils.parcelwise_plot",
    "ClusterEllipsoid": "calvin_utils.plotting_utils.scatter_ellipsoids",
    "add_cluster_ellipsoids_matplotlib": "calvin_utils.plotting_utils.scatter_ellipsoids",
    "add_cluster_ellipsoids_plotly": "calvin_utils.plotting_utils.scatter_ellipsoids",
    "calculate_cluster_ellipsoids": "calvin_utils.plotting_utils.scatter_ellipsoids",
    "pairwise_cluster_separation_tests": "calvin_utils.plotting_utils.scatter_ellipsoids",
    "SimpleScatterplot3D": "calvin_utils.plotting_utils.simple_scatterplot_3d",
    "pairwise_hotelling_tests": "calvin_utils.plotting_utils.hotelling_cluster_separation",
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_LAZY[name]), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
