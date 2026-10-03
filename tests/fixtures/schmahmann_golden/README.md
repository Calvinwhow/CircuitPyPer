# Schmahmann golden test cohort

This directory contains five small, deidentified cases derived from the
Schmahmann SCA atrophy analysis tree. It exists solely as stable integration
test data for the real NIfTI, fiber-value, regression, and cross-validation
pipelines.

Contents:

- `nifti/`: derived, normalized composite atrophy maps. They are not raw T1
  scans. Each image was re-saved with a fresh NIfTI header so source subject,
  session, description, and auxiliary fields are not retained.
- `fibers/`: one-dimensional `float32` patient values in the ordered
  `Atlas_all30_MNI` fiber space. These files contain values only, not tract
  geometry.
- `outcomes.csv`: anonymous IDs, relative paths, configured higher-is-worse
  outcomes, and SHA-256 checksums.

The full fiber atlas is intentionally excluded because it is hundreds of
megabytes. These vectors can be read and used for regression without the
atlas. Reconstructing or visualizing tract geometry still requires the
matching `Atlas_all30_MNI` atlas.

Five cases are included because five is the minimum accepted by
`cross_validated_map_damage`. This cohort is a software fixture and is not
large enough for scientific inference. Do not add raw anatomy, identifying
columns, original subject IDs, or additional cases without a specific test
need.
