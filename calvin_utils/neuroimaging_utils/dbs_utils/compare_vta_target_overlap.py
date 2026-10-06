"""Legacy shim; import from ``calvin_utils.neuroimaging_utils.dbs_utils.vta_overlap``."""

from calvin_utils.neuroimaging_utils.dbs_utils.vta_overlap import (
    DiceVTA,
    VTATargetOverlap,
)

VTADiceCorrelation = DiceVTA

__all__ = ["DiceVTA", "VTADiceCorrelation", "VTATargetOverlap"]
