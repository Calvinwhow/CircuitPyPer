"""Hierarchical neural integration of deterministic symptom-basis solutions."""

from .anatomy import ANATOMICAL_REPRESENTATIONS, REPRESENTATIONS


def __getattr__(name):
    if name == "AnatomicalNetwork":
        from .model import AnatomicalNetwork
        return AnatomicalNetwork
    raise AttributeError(name)


__all__ = ["ANATOMICAL_REPRESENTATIONS", "REPRESENTATIONS", "AnatomicalNetwork"]
