import numpy as np

if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [np.bool_, np.bytes_, np.str_, np.object_],
    }
if not hasattr(np, "maximum_sctype"):
    def _maximum_sctype(t):
        dtype = np.dtype(t)
        if np.issubdtype(dtype, np.complexfloating):
            return np.complex128
        if np.issubdtype(dtype, np.floating):
            return np.float64
        if np.issubdtype(dtype, np.unsignedinteger):
            return np.uint64
        if np.issubdtype(dtype, np.integer):
            return np.int64
        return dtype.type

    np.maximum_sctype = _maximum_sctype

__all__ = ["ParcelwiseDamageMap", "ParcelwiseRegressionSimilarity"]


def __getattr__(name):
    if name in __all__:
        from calvin_utils.permutation_analysis_utils.parcelwise_regression_similarity import (
            ParcelwiseDamageMap,
            ParcelwiseRegressionSimilarity,
        )

        exports = {
            "ParcelwiseDamageMap": ParcelwiseDamageMap,
            "ParcelwiseRegressionSimilarity": ParcelwiseRegressionSimilarity,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
