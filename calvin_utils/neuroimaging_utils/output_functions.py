import os
import numpy as np
from nilearn import plotting

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

    def save_map(self, map_data, file_name, out_dir, visualize=False):
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
        if visualize and hasattr(self.io, "_map_to_image") and hasattr(self.io, "_visualize_map"):
            img = self.io._map_to_image(np.asarray(map_data))
            self.io._visualize_map(img, title=os.path.basename(file_name))

    def view_map(self, map_data, file_name):
        """
        View one statistical vector using the underlying I/O class.
        """
        if hasattr(self.io, "_map_to_image"):
            img = self.io._map_to_image(np.asarray(map_data))
            return plotting.view_img(img, title=os.path.basename(file_name))
        return None
