import numpy as np
from scipy.optimize import minimize

from calvin_utils.neuroimaging_utils.registration_utils.mutual_information.mutual_information import MutualInformation
from calvin_utils.neuroimaging_utils.registration_utils.rigid_registration.rotation_utils import Rotation


class RigidRegistration:
    """
    Rigid registration of a scan to a template by mutual information: the one implementation of rigid registration.

    The scan's centre of mass is placed on the template brain's to start. Powell then searches six
    parameters for the template-to-scan map x -> c_m + d + R (x - c_t) that maximises MutualInformation,
    where c_t (3,) is the template brain's centre of mass and c_m (3,) the scan's (world mm), d (3,) a
    translation in mm and R (3, 3) a rotation from three angles (Rotation). All six at zero is the
    centre-of-mass start. Rotation and translation only: the scan's size and shape are never changed.
    """

    def __init__(self, moving_path, template_path, template_mask_path, n_samples=50000, bins=32, seed=0):
        """
        moving_path (str): path to the scan to register (NIfTI file).
        template_path (str): path to the template to register to (NIfTI file), e.g. an MNI T1.
        template_mask_path (str): path to the template's brain mask (NIfTI file), on the template's grid.
        n_samples (int): number of template points inside the brain mask the mutual information is scored at, at least 1000.
        bins (int): number of histogram bins per image for the mutual information.
        seed (int): seed for drawing the sample points, so a run is repeatable.
        """
        self.mutual_information = MutualInformation(moving_path, template_path, template_mask_path, n_samples=n_samples, bins=bins, seed=seed)
        self.moving_center = self._center_of_mass(self.mutual_information.moving_data, self.mutual_information.moving_affine)
        self.template_center = self._center_of_mass(np.where(self.mutual_information.template_mask, self.mutual_information.template_data, 0.0), self.mutual_information.template_affine)
        self.parameter_units = np.array([10.0, 10.0, 10.0, 0.1, 0.1, 0.1])

    ###INTERNAL METHODS###
    def _center_of_mass(self, data, affine):
        """Intensity-weighted centre of an image, world mm, shape (3,)."""
        weights = np.clip(data, 0.0, None)
        ijk = np.argwhere(weights > 0)
        centre_ijk = np.average(ijk, axis=0, weights=weights[tuple(ijk.T)])
        return affine[:3, :3] @ centre_ijk + affine[:3, 3]

    def _template_to_moving(self, parameters):
        """(4, 4) matrix from template world mm to scan world mm, for six parameters in parameter_units."""
        real = np.asarray(parameters, float) * self.parameter_units
        rotation = Rotation(real[3:6]).run()
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = self.moving_center + real[:3] - rotation @ self.template_center
        return matrix

    def _cost(self, parameters):
        """Negative mutual information for six parameters, for the optimiser to minimise."""
        return -self.mutual_information.run(self._template_to_moving(parameters))

    ###PROPERTY METHODS###
    @property
    def mutual_information(self):
        """MutualInformation: the scorer of a template-to-scan matrix, holding the scan and template."""
        return self._mutual_information

    @mutual_information.setter
    def mutual_information(self, scorer):
        if not isinstance(scorer, MutualInformation):
            raise ValueError("mutual_information must be a MutualInformation")
        self._mutual_information = scorer

    @property
    def moving_center(self):
        """np.ndarray, shape (3,), float: the scan's intensity-weighted centre, world mm."""
        return self._moving_center

    @moving_center.setter
    def moving_center(self, point):
        self._moving_center = np.asarray(point, float)

    @property
    def template_center(self):
        """np.ndarray, shape (3,), float: the template brain's intensity-weighted centre, world mm."""
        return self._template_center

    @template_center.setter
    def template_center(self, point):
        self._template_center = np.asarray(point, float)

    @property
    def parameter_units(self):
        """np.ndarray, shape (6,), float: one optimiser unit of each parameter -- 10 mm of translation, 0.1 rad of rotation."""
        return self._parameter_units

    @parameter_units.setter
    def parameter_units(self, units):
        units = np.asarray(units, float).ravel()
        if units.shape != (6,) or not (units > 0).all():
            raise ValueError(f"parameter_units must be six positive numbers, got shape {units.shape}")
        self._parameter_units = units

    ###BUSINESS LOGIC###
    def run(self):
        """
        Returns (np.ndarray, shape (4, 4), float): the rigid matrix from the scan's world mm to the template's world mm.
        """
        parameters = minimize(self._cost, np.zeros(6), method="Powell", options={"xtol": 1e-3, "ftol": 1e-5, "maxiter": 20}).x
        return np.linalg.inv(self._template_to_moving(parameters))
