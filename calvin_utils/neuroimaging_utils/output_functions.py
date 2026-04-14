import os
import json
import glob
import numpy as np
import nibabel as nib

from calvin_utils.neuroimaging_utils.nifti_utils.volume_io import NiftiIO
from calvin_utils.neuroimaging_utils.surface_utils.surface_io import SurfaceIO
from calvin_utils.neuroimaging_utils.tract_utils.fiber_io import FiberIO

from calvin_utils.neuroimaging_utils.nifti_utils.volume_io import NiftiIO, VolumetricTimeSeriesIO
from calvin_utils.neuroimaging_utils.surface_utils.surface_io import SurfaceIO
from calvin_utils.neuroimaging_utils.tract_utils.fiber_io import FiberIO


class NeuroimageFileOutporter:
    def __init__(self, output_ftype, mask_path=None):
        self.output_ftype = output_ftype
        self.mask_path = mask_path

        if output_ftype == "nii":
            self.io = NiftiIO(mask_path=mask_path)
        elif output_ftype == "surface":
            self.io = SurfaceIO(mask_path=mask_path)
        elif output_ftype == "fiber":
            self.io = FiberIO(mask_path=mask_path)
        elif output_ftype == "nii_timeseries":
            self.io = VolumetricTimeSeriesIO(mask_path=mask_path)
        else:
            raise ValueError(f"Unknown output_ftype: {output_ftype}")

    def save_map(self, map_data, file_name, out_dir):
        """
        Save one statistical vector using the underlying I/O class.
        """
        fake_target = os.path.join(out_dir, file_name)
        self.io.save_files(
            arr=np.asarray(map_data),
            file_paths=[fake_target],
            dry_run=False,
            file_suffix=""
        )