import os
import numpy as np
if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int": [np.int8, np.int16, np.int32, np.int64],
        "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
        "float": [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others": [np.bool_, np.bytes_, np.str_, np.object_],
    }
if not hasattr(np, "maximum_sctype"):
    def _maximum_sctype(t):
        dtype = np.dtype(t)
        if np.issubdtype(dtype, np.complexfloating):
            return np.complex128
        if np.issubdtype(dtype, np.floating):
            return np.float64
        if np.issubdtype(dtype, np.unsignedinteger):
            return np.uint64
        if np.issubdtype(dtype, np.integer):
            return np.int64
        return dtype.type

    np.maximum_sctype = _maximum_sctype

class NeuroimageFileOutporter:
    def __init__(self, output_ftype, mask_path=None):
        self.output_ftype = output_ftype
        self.mask_path = mask_path

        if output_ftype == "nii":
            from calvin_utils.neuroimaging_utils.nifti_utils.volume_io import NiftiIO

            self.io = NiftiIO(mask_path=mask_path)
        elif output_ftype == "surface":
            from calvin_utils.neuroimaging_utils.surface_utils.surface_io import SurfaceIO

            self.io = SurfaceIO(mask_path=mask_path)
        elif output_ftype == "fiber":
            from calvin_utils.neuroimaging_utils.tract_utils.fiber_io import FiberIO

            self.io = FiberIO(mask_path=mask_path)
        elif output_ftype == "nii_timeseries":
            from calvin_utils.neuroimaging_utils.nifti_utils.volume_io import VolumetricTimeSeriesIO

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
            from nilearn import plotting

            img = self.io._map_to_image(np.asarray(map_data))
            return plotting.view_img(img, title=os.path.basename(file_name))
        return None
