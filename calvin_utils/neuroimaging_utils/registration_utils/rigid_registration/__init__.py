"""Rigid registration of a scan to a template by mutual information: the one implementation of rigid registration (RigidRegistration).

The name is loaded the first time it is asked for, as in tract_utils.
"""

from importlib import import_module

# name -> the module it lives in
_LAZY = {"RigidRegistration": "calvin_utils.neuroimaging_utils.registration_utils.rigid_registration.rigid_registration"}

__all__ = list(_LAZY)


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_LAZY[name]), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
