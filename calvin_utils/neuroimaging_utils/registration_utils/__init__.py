"""Registration utilities: patients' scans registered to MNI and skull-stripped, at their own resolution, for viewing.

Each module is a folder holding the class it is named for, a schema.py of the data structures that class
takes or returns (where it has any), and <subclass>_utils.py files for the helper classes it offloads to.

    run_registration.py, schema.py      RunRegistration (every scan in a CSV, saved beside each scan) and RegistrationOutputs
    mutual_information/                 MutualInformation: how well a scan, through a candidate matrix, matches the template
        intensity_mutual_information_utils.py    IntensityMutualInformation: mutual information of two sets of intensities
    rigid_registration/                 RigidRegistration: the one implementation of rigid registration
        rotation_utils.py               Rotation: a rotation matrix from three angles
    affine_registration/                AffineRegistration: rigid, then full affine
    skull_strip/                        SkullStrip: the brain, masked, in the frame of the registration asked for; schema.py: SkullStripResult
        brain_mask_transfer_utils.py    BrainMaskTransfer: the MNI brain mask on the scan's own grid
        rigid_from_affine_utils.py      RigidFromAffine: an affine's rotation and translation, its scaling and shear removed

One scan, e.g. in a viewer:
    from calvin_utils.neuroimaging_utils.registration_utils.skull_strip.skull_strip import SkullStrip
    result = SkullStrip("sub-01_T1w.nii.gz", affine=False).run()     # result.brain, result.brain_affine
    from calvin_utils.neuroimaging_utils.registration_utils.rigid_registration.rigid_registration import RigidRegistration
    rigid = RigidRegistration("sub-01_T1w.nii.gz", template_path, template_mask_path).run()     # 4x4, no skull strip

The name is loaded the first time it is asked for, as in tract_utils.
"""

from importlib import import_module

# name -> the module it lives in
_LAZY = {"RunRegistration": "calvin_utils.neuroimaging_utils.registration_utils.run_registration"}

__all__ = list(_LAZY)


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_LAZY[name]), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
