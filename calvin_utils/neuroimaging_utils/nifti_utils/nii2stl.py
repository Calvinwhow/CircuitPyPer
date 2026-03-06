import numpy as np
import nibabel as nib
from skimage.measure import marching_cubes
import trimesh

def nifti_mask_to_stl(nifti_path: str,
                      stl_path: str,
                      level: float = 0.5,
                      step_size: int = 1,
                      keep_largest_component: bool = False,
                      smooth_iters: int = 0) -> None:
    img = nib.load(nifti_path)
    data = img.get_fdata(dtype=np.float32)

    mask = data > 0.5  # binary
    if not np.any(mask):
        raise ValueError("Mask is empty (no nonzero voxels).")

    vol = mask.astype(np.uint8)

    if keep_largest_component:
        from scipy.ndimage import label
        lbl, n = label(vol)
        if n > 1:
            counts = np.bincount(lbl.ravel())
            counts[0] = 0
            largest = counts.argmax()
            vol = (lbl == largest).astype(np.uint8)

    verts_ijk, faces, normals, values = marching_cubes(
        vol,
        level=level,
        step_size=step_size,
        allow_degenerate=False
    )

    affine = img.affine
    verts_h = np.c_[verts_ijk, np.ones(len(verts_ijk), dtype=np.float32)]
    verts_xyz = (affine @ verts_h.T).T[:, :3]

    mesh = trimesh.Trimesh(vertices=verts_xyz, faces=faces, process=True)

    if smooth_iters > 0:
        trimesh.smoothing.filter_laplacian(mesh, iterations=smooth_iters)

    mesh.export(stl_path)