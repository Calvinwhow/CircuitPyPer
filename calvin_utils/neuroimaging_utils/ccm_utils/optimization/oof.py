"""Compatibility imports; new code should use :mod:`inner_cv`."""

from .inner_cv import prepare_inner_folds, prepare_oof_folds

__all__ = ["prepare_inner_folds", "prepare_oof_folds"]
