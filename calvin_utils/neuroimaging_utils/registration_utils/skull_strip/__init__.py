"""A scan skull-stripped with the MNI brain mask, in the frame of the registration asked for (SkullStrip).

The name is loaded the first time it is asked for, as in tract_utils.
"""

from importlib import import_module

# name -> the module it lives in
_LAZY = {"SkullStrip": "calvin_utils.neuroimaging_utils.registration_utils.skull_strip.skull_strip"}

__all__ = list(_LAZY)


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_LAZY[name]), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
