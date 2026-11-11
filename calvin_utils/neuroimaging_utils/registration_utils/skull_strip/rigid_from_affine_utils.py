import warnings
import numpy as np


class RigidFromAffine:
    """
    The rigid part of an affine: its rotation and translation, with its scaling and shear removed.

    The affine's linear part A (3, 3) is split by polar decomposition, A = R P, where R (3, 3) is the
    rotation closest to A and P (3, 3) is symmetric positive-definite and holds all the scaling and
    shear. With the singular value decomposition A = U S V^T (U, V orthogonal; S diagonal, the three
    stretch factors), R = U V^T. R is kept and P is dropped.

    Dropping P moves points, so the translation is chosen again: the rigid map sends the pivot c (3,)
    to the same place the affine does, R c + t_r = A c + t, so t_r = A c + t - R c. A rotation keeps
    every distance, so the result never distorts anatomy.
    """

    def __init__(self, affine_matrix, pivot_mm):
        """
        affine_matrix (np.ndarray, shape (4, 4), float): the affine, e.g. a registration from a scan's world mm to a template's world mm.
        pivot_mm (np.ndarray, shape (3,), float): point in the affine's input world mm that the rigid map sends where the affine does, e.g. the brain's centre.
        """
        self.affine_matrix = affine_matrix
        self.pivot_mm = pivot_mm

    ###INTERNAL METHODS###
    def _rotation(self, linear):
        """The rotation closest to ``linear`` (3, 3); a mirror is turned into a rotation, with a warning."""
        u, _, vt = np.linalg.svd(linear)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            warnings.warn("the affine includes a mirror flip, which a rigid map cannot hold; this usually means a left-right header problem in the scan")
            u[:, -1] *= -1
            rotation = u @ vt
        return rotation

    ###PROPERTY METHODS###
    @property
    def affine_matrix(self):
        return self._affine_matrix

    @affine_matrix.setter
    def affine_matrix(self, matrix):
        matrix = np.asarray(matrix, float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(f"affine_matrix must be a finite (4, 4) matrix, got shape {matrix.shape}")
        if abs(np.linalg.det(matrix[:3, :3])) < 1e-9:
            raise ValueError("affine_matrix is singular")
        self._affine_matrix = matrix

    @property
    def pivot_mm(self):
        return self._pivot_mm

    @pivot_mm.setter
    def pivot_mm(self, point):
        point = np.asarray(point, float).ravel()
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError(f"pivot_mm must be a finite point of shape (3,), got {point.shape}")
        self._pivot_mm = point

    ###BUSINESS LOGIC###
    def run(self):
        """
        Returns (np.ndarray, shape (4, 4), float): the rigid matrix, rotation R and translation t_r only, same input and output spaces as affine_matrix.
        """
        linear, translation = self.affine_matrix[:3, :3], self.affine_matrix[:3, 3]
        rotation = self._rotation(linear)
        rigid = np.eye(4)
        rigid[:3, :3] = rotation
        rigid[:3, 3] = linear @ self.pivot_mm + translation - rotation @ self.pivot_mm
        return rigid
