"""Tract utilities.

The names below are importable from here as always -- ``from
calvin_utils.neuroimaging_utils.tract_utils import InvertedFiberConnectivity``
-- but each is loaded the first time it is asked for, not when the package is.
Importing one module of this package (``tract_utils.fiber_converter``, say) then
costs only what that module needs, which is what lets an application ship the
fibre readers without nilearn and the rest of the analysis stack.
"""

from importlib import import_module

# name -> the module it lives in
_LAZY = {
    "FiberAlignment": "calvin_utils.neuroimaging_utils.tract_utils.fiber_geometry_registration",
    "FiberGeometryRegistration": "calvin_utils.neuroimaging_utils.tract_utils.fiber_geometry_registration",
    "GeodesicFiberRegistration": "calvin_utils.neuroimaging_utils.tract_utils.fiber_geometry_registration",
    "FiberTrajectorySimilarity": "calvin_utils.neuroimaging_utils.tract_utils.inverted_connectivity",
    "InvertedFiberConnectivity": "calvin_utils.neuroimaging_utils.tract_utils.inverted_connectivity",
    "TargetTrajectoryAverager": "calvin_utils.neuroimaging_utils.tract_utils.target_trajectory_averaging",
    "StreamingFiberInverter": "calvin_utils.neuroimaging_utils.tract_utils.streaming_fiber_inversion",
    "SampledFiberBatch": "calvin_utils.neuroimaging_utils.tract_utils.sampled_voxel_fiber_store",
    "SampledVoxelFiberStore": "calvin_utils.neuroimaging_utils.tract_utils.sampled_voxel_fiber_store",
    "VoxelSeededConnectomeBuilder": "calvin_utils.neuroimaging_utils.tract_utils.voxel_seeded_connectome",
    "VoxelSeededFiberStore": "calvin_utils.neuroimaging_utils.tract_utils.voxel_seeded_connectome",
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
