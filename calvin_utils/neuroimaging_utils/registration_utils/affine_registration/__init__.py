"""Full affine registration of a scan to a template, starting from the rigid registration (AffineRegistration).

The name is loaded the first time it is asked for, as in tract_utils.
"""

from importlib import import_module

# name -> the module it lives in
_LAZY = {"AffineRegistration": "calvin_utils.neuroimaging_utils.registration_utils.affine_registration.affine_registration"}

__all__ = list(_LAZY)


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_LAZY[name]), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
