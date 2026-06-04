# Custom Yabplot Atlas Builder

Builder source:

```text
calvin_utils/neuroimaging_utils/nifti_utils/generate_atlas.py
```

Import:

```python
from calvin_utils.neuroimaging_utils.nifti_utils.generate_atlas import (
    CustomYabplotAtlasBuilder,
)
```

Example:

```python
builder = CustomYabplotAtlasBuilder(
    parcel_dir="/Volumes/HowExp2/resources/atlases/atlases/AAL/combined",
    out_dir="/Volumes/HowExp2/resources/atlases/atlases/AAL/yabplot_custom_from_combined",
    atlas_name="aal_combined",
)
builder.run()
```

Outputs:

```text
source_volumes/aal_combined_all_labels.nii.gz
source_volumes/aal_combined_cortical_labels.nii.gz
source_volumes/aal_combined_subcortical_labels.nii.gz
source_volumes/aal_combined_wb_labels.txt
cortical/aal_combined_conte69.csv
cortical/aal_combined_LUT.txt
subcortical/*.vtk
subcortical/atlas_LUT.txt
```

Method:

- Region names come from parcel filenames after removing `.nii` / `.nii.gz`
  and trailing underscores.
- Subcortical parcels are selected by
  `CustomYabplotAtlasBuilder.DEFAULT_SUBCORTICAL_KEYWORDS`.
- Cortical atlas generation combines cortical masks into a labeled NIfTI and
  nearest-neighbor samples that volume onto yabplot's fsLR32k midthickness
  vertices using `vol2surf`.
- Subcortical atlas generation converts each original named subcortical NIfTI
  mask to a smoothed `.vtk` mesh using marching cubes.

Caveat:

The cortical atlas path is not Workbench ribbon-constrained. If Workbench is
installed later, use `source_volumes/aal_combined_wb_labels.txt` with
`yabplot.build_cortical_atlas` for stricter cortical projection.
