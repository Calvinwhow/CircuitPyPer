import numpy as np


class IntensityMutualInformation:
    """
    Mutual information between fixed intensities and moving intensities sampled at the same points.

    The fixed intensities (the template, sampled once) are binned at init. Each run bins the moving
    intensities and returns the mutual information of the two. Every value is shared linearly between
    its two nearest bins, so the score changes smoothly as the moving image moves, which a
    derivative-free optimiser such as Powell needs.
    """

    def __init__(self, fixed_values, moving_range, bins=32):
        """
        fixed_values (np.ndarray, shape (N,), float): fixed (template) intensities at the N sample points.
        moving_range (np.ndarray, shape (2,), float): [low, high] moving intensity mapped onto the first and last bin; values outside are clipped.
        bins (int): number of histogram bins per image, at least 2.
        """
        self.bins = bins
        self.moving_range = moving_range
        self.fixed_values = fixed_values

    ###INTERNAL METHODS###
    def _bin_positions(self, values, low, high):
        """Each value as a continuous bin position in [0, bins - 1]."""
        scaled = (np.asarray(values, float) - low) / max(high - low, 1e-12)
        return np.clip(scaled, 0.0, 1.0) * (self.bins - 1)

    def _split_bins(self, positions):
        """The lower of each value's two bins, and the weight it gives the upper one."""
        lower = np.minimum(np.floor(positions).astype(np.int64), self.bins - 2)
        return lower, positions - lower

    def _joint_histogram(self, moving_lower, moving_upper_weight):
        """The joint probability table, shape (bins, bins): fixed bins on rows, moving bins on columns."""
        joint = np.zeros(self.bins * self.bins)
        for fixed_offset, fixed_weight in ((0, 1.0 - self.fixed_upper_weight), (1, self.fixed_upper_weight)):
            for moving_offset, moving_weight in ((0, 1.0 - moving_upper_weight), (1, moving_upper_weight)):
                index = (self.fixed_lower + fixed_offset) * self.bins + (moving_lower + moving_offset)
                joint += np.bincount(index, weights=fixed_weight * moving_weight, minlength=self.bins * self.bins)
        return joint.reshape(self.bins, self.bins) / joint.sum()

    ###PROPERTY METHODS###
    @property
    def bins(self):
        return self._bins

    @bins.setter
    def bins(self, value):
        if int(value) < 2:
            raise ValueError(f"bins must be at least 2, got {value}")
        self._bins = int(value)

    @property
    def moving_range(self):
        return self._moving_range

    @moving_range.setter
    def moving_range(self, value):
        value = np.asarray(value, float).ravel()
        if value.shape != (2,) or not value[1] > value[0]:
            raise ValueError(f"moving_range must be [low, high] with high > low, got {value}")
        self._moving_range = value

    @property
    def fixed_values(self):
        return self._fixed_values

    @fixed_values.setter
    def fixed_values(self, values):
        values = np.asarray(values, float).ravel()
        if values.size < 2 or not np.isfinite(values).all():
            raise ValueError("fixed_values must hold at least two finite intensities")
        self._fixed_values = values
        low, high = np.percentile(values, [1, 99])
        self._fixed_lower, self._fixed_upper_weight = self._split_bins(self._bin_positions(values, low, max(high, low + 1e-12)))

    @property
    def fixed_lower(self):
        return self._fixed_lower

    @property
    def fixed_upper_weight(self):
        return self._fixed_upper_weight

    ###BUSINESS LOGIC###
    def run(self, moving_values):
        """
        moving_values (np.ndarray, shape (N,), float): moving (scan) intensities at the same N sample points as fixed_values.
        Returns (float): mutual information of the fixed and moving intensities, in nats.
        """
        moving_values = np.asarray(moving_values, float).ravel()
        if moving_values.shape != self.fixed_values.shape:
            raise ValueError(f"moving_values must have shape {self.fixed_values.shape}, got {moving_values.shape}")
        moving_lower, moving_upper_weight = self._split_bins(self._bin_positions(moving_values, self.moving_range[0], self.moving_range[1]))
        joint = self._joint_histogram(moving_lower, moving_upper_weight)
        independent = np.outer(joint.sum(axis=1), joint.sum(axis=0))
        nonzero = joint > 0
        return float(np.sum(joint[nonzero] * np.log(joint[nonzero] / independent[nonzero])))
