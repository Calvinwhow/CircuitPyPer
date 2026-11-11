import numpy as np


class Rotation:
    """A 3D rotation from three angles: about x, then about y, then about z."""

    def __init__(self, angles):
        """
        angles (np.ndarray, shape (3,), float): rotation angles in radians about the x, y and z axes.
        """
        self.angles = angles

    ###PROPERTY METHODS###
    @property
    def angles(self):
        return self._angles

    @angles.setter
    def angles(self, values):
        values = np.asarray(values, float).ravel()
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError(f"angles must be three finite numbers, got shape {values.shape}")
        self._angles = values

    ###BUSINESS LOGIC###
    def run(self):
        """
        Returns (np.ndarray, shape (3, 3), float): the rotation matrix, R = Rz @ Ry @ Rx.
        """
        cx, cy, cz = np.cos(self.angles)
        sx, sy, sz = np.sin(self.angles)
        about_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        about_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        about_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return about_z @ about_y @ about_x
