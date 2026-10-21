"""Dimensionless elastic-net calibration from one unpenalized fit."""

import numpy as np


def main(merit, null_merit, l1_reference, l2_reference, strength=0.1):
    """Calibrate equal L1 and L2 contributions from an unpenalized reference fit.
    merit: float; unpenalized fitted merit.
    null_merit: float; merit of the null parameters.
    l1_reference: float; unscaled L1 penalty at the reference solution.
    l2_reference: float; unscaled L2 penalty at the reference solution.
    strength: float; total penalty budget as a fraction of merit gain over null.
    """
    gain = float(merit - null_merit)
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("The unpenalized merit must improve on the null merit.")
    if l1_reference <= 0 or l2_reference <= 0:
        raise ValueError("The unpenalized parameters must have nonzero L1 and L2 penalties.")
    budget = strength * gain
    l1_strength = budget / (2 * l1_reference)
    l2_strength = budget / (2 * l2_reference)
    alpha = l1_strength / (l1_strength + l2_strength)
    return {"regularization_strength": float(strength), "elastic_net_alpha": float(alpha), "reference_merit": float(merit), "reference_null_merit": float(null_merit), "reference_merit_gain": gain, "l1_reference": float(l1_reference), "l2_reference": float(l2_reference), "l1_strength": float(l1_strength), "l2_strength": float(l2_strength)}
