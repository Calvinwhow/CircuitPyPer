import numpy as np
from scipy.optimize import minimize

from calvin_utils.neuroimaging_utils.registration_utils.rigid_registration.rigid_registration import RigidRegistration
from calvin_utils.neuroimaging_utils.registration_utils.rigid_registration.rotation_utils import Rotation


class AffineRegistration:
    """
    Full affine registration of a scan to a template by mutual information, starting from the rigid registration.

    RigidRegistration runs first and gives the rigid template-to-scan matrix M_r (4, 4). Powell then
    searches twelve parameters for M_r P, maximising the same mutual information, where P (4, 4) maps a
    template point x -> c_t + d + R S H (x - c_t): c_t (3,) is the template brain's centre of mass (world
    mm), d (3,) a translation in mm, R (3, 3) a rotation from three angles (Rotation), S (3, 3) a diagonal
    scaling from three log-scales and H (3, 3) an upper-triangular shear from three terms. All twelve at
    zero is the rigid result. Starting from the rigid fit keeps scale and shear from trading against
    position and settling somewhere wrong.
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
        self.rigid_registration = RigidRegistration(moving_path, template_path, template_mask_path, n_samples=n_samples, bins=bins, seed=seed)
        self.parameter_units = np.array([10.0, 10.0, 10.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    ###INTERNAL METHODS###
    def _shear(self, terms):
        """(3, 3) upper-triangular shear from three terms."""
        return np.array([[1, terms[0], terms[1]], [0, 1, terms[2]], [0, 0, 1]])

    def _perturbation(self, parameters):
        """(4, 4) matrix P in template world mm, for twelve parameters in parameter_units: x -> c_t + d + R S H (x - c_t)."""
        real = np.asarray(parameters, float) * self.parameter_units
        linear = Rotation(real[3:6]).run() @ np.diag(np.exp(real[6:9])) @ self._shear(real[9:12])
        centre = self.rigid_registration.template_center
        matrix = np.eye(4)
        matrix[:3, :3] = linear
        matrix[:3, 3] = centre + real[:3] - linear @ centre
        return matrix

    def _cost(self, parameters, rigid_template_to_moving):
        """Negative mutual information of M_r P for twelve parameters, for the optimiser to minimise."""
        return -self.rigid_registration.mutual_information.run(rigid_template_to_moving @ self._perturbation(parameters))

    ###PROPERTY METHODS###
    @property
    def rigid_registration(self):
        """RigidRegistration: the rigid registration this fit starts from, holding the scan, template and scorer."""
        return self._rigid_registration

    @rigid_registration.setter
    def rigid_registration(self, registration):
        if not isinstance(registration, RigidRegistration):
            raise ValueError("rigid_registration must be a RigidRegistration")
        self._rigid_registration = registration

    @property
    def parameter_units(self):
        """np.ndarray, shape (12,), float: one optimiser unit of each parameter -- 10 mm translation, 0.1 rad rotation, 0.1 log-scale, 0.1 shear."""
        return self._parameter_units

    @parameter_units.setter
    def parameter_units(self, units):
        units = np.asarray(units, float).ravel()
        if units.shape != (12,) or not (units > 0).all():
            raise ValueError(f"parameter_units must be twelve positive numbers, got shape {units.shape}")
        self._parameter_units = units

    ###BUSINESS LOGIC###
    def run(self):
        """
        Returns (np.ndarray, shape (4, 4), float): the affine matrix from the scan's world mm to the template's world mm.
        """
        rigid_template_to_moving = np.linalg.inv(self.rigid_registration.run())
        parameters = minimize(self._cost, np.zeros(12), args=(rigid_template_to_moving,), method="Powell", options={"xtol": 1e-3, "ftol": 1e-5, "maxiter": 20}).x
        return np.linalg.inv(rigid_template_to_moving @ self._perturbation(parameters))
