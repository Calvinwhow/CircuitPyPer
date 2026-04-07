import os
import json
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from nilearn import image


class FiberVoxelIndexer:
    """
    Build a voxel index for a canonical fiber library on a reference NIfTI grid.

    Core representation:
        self.fibers[fiber_idx]            -> ndarray, shape (n_vertices, 3) or (n_vertices, 4)
        self.fiber_voxel_indices[fiber_idx] -> 1D ndarray of unique linear voxel indices touched by that fiber

    The index is built in the voxel lattice of reference_nifti_path.
    """

    def __init__(self, reference_nifti_path, fiber_mask=None, step_size_vox=0.5):
        self.reference_nifti_path = reference_nifti_path
        self.reference_img = nib.load(reference_nifti_path)
        self.reference_affine = self.reference_img.affine
        self.reference_shape = self.reference_img.shape[:3]
        self.inv_affine = np.linalg.inv(self.reference_affine)

        self.fiber_mask = fiber_mask
        self.step_size_vox = float(step_size_vox)

        self.fibers = None
        self.fiber_voxel_indices = None

    @staticmethod
    def _identify_fiber_file_type(path):
        p = Path(path)
        suffixes = [s.lower() for s in p.suffixes]

        if suffixes[-1:] == ['.trk']:
            return 'trk'
        if suffixes[-1:] == ['.tck']:
            return 'tck'
        if suffixes[-1:] == ['.trx']:
            return 'trx'
        if suffixes[-1:] == ['.npy']:
            return 'npy'
        if suffixes[-1:] == ['.npz']:
            return 'npz'
        if suffixes[-1:] == ['.json']:
            return 'json'

        return 'unknown'

    @classmethod
    def from_fiber_file(cls, fiber_file_path, reference_nifti_path, fiber_mask=None, step_size_vox=0.5):
        obj = cls(
            reference_nifti_path=reference_nifti_path,
            fiber_mask=fiber_mask,
            step_size_vox=step_size_vox,
        )
        fibers = obj.load_fibers(fiber_file_path)
        obj.build_index(fibers)
        return obj

    def load_fibers(self, fiber_file_path):
        """
        Return list of fibers, each shaped (n_vertices, 3) or (n_vertices, 4).
        For streamline files, only xyz are loaded.
        """
        ftype = self._identify_fiber_file_type(fiber_file_path)

        if ftype in {'trk', 'tck', 'trx'}:
            tractogram = nib.streamlines.load(fiber_file_path).tractogram
            fibers = [np.asarray(sl, dtype=np.float32) for sl in tractogram.streamlines]

        elif ftype == 'npy':
            obj = np.load(fiber_file_path, allow_pickle=True)
            if isinstance(obj, np.ndarray) and obj.dtype == object:
                fibers = obj.tolist()
            else:
                raise ValueError(f"Expected object-array of fibers in {fiber_file_path}")

        elif ftype == 'npz':
            obj = np.load(fiber_file_path, allow_pickle=True)
            if 'fibers' not in obj:
                raise ValueError(f"NPZ fiber file missing 'fibers' key: {fiber_file_path}")
            fibers = obj['fibers'].tolist()

        elif ftype == 'json':
            with open(fiber_file_path, 'r') as f:
                obj = json.load(f)
            if 'fibers' not in obj:
                raise ValueError(f"JSON fiber file missing 'fibers' key: {fiber_file_path}")
            fibers = obj['fibers']

        else:
            raise RuntimeError(f"Unsupported fiber file type: {fiber_file_path}")

        fibers = [np.asarray(f, dtype=np.float32) for f in fibers]

        for i, fiber in enumerate(fibers):
            if fiber.ndim != 2 or fiber.shape[1] not in (3, 4):
                raise ValueError(
                    f"Fiber {i} has invalid shape {fiber.shape}. "
                    f"Expected (n_vertices, 3) or (n_vertices, 4)."
                )

        if self.fiber_mask is not None:
            mask = np.asarray(self.fiber_mask).astype(bool).flatten()
            if mask.shape[0] != len(fibers):
                raise ValueError(
                    f"fiber_mask length ({mask.shape[0]}) does not match number of fibers ({len(fibers)})."
                )
            fibers = [f for f, keep in zip(fibers, mask) if keep]

        return fibers

    def world_to_voxel(self, xyz_world):
        """
        xyz_world: ndarray (..., 3) in world/MNI coordinates
        returns voxel coordinates in floating point index space
        """
        xyz_world = np.asarray(xyz_world, dtype=np.float32)
        orig_shape = xyz_world.shape
        flat = xyz_world.reshape(-1, 3)
        hom = np.concatenate([flat, np.ones((flat.shape[0], 1), dtype=np.float32)], axis=1)
        vox = hom @ self.inv_affine.T
        return vox[:, :3].reshape(orig_shape)

    def _segment_voxel_indices(self, p0_world, p1_world):
        """
        Rasterize a single segment by dense sampling in voxel space.
        Returns unique linear voxel indices crossed by the segment.
        """
        p0_vox = self.world_to_voxel(np.asarray(p0_world))[0:3]
        p1_vox = self.world_to_voxel(np.asarray(p1_world))[0:3]

        delta = p1_vox - p0_vox
        dist = float(np.linalg.norm(delta))

        if dist == 0:
            pts = np.asarray([p0_vox], dtype=np.float32)
        else:
            n_steps = max(2, int(np.ceil(dist / self.step_size_vox)) + 1)
            t = np.linspace(0.0, 1.0, n_steps, dtype=np.float32)[:, None]
            pts = p0_vox[None, :] + t * delta[None, :]

        ijk = np.round(pts).astype(np.int32)

        in_bounds = (
            (ijk[:, 0] >= 0) & (ijk[:, 0] < self.reference_shape[0]) &
            (ijk[:, 1] >= 0) & (ijk[:, 1] < self.reference_shape[1]) &
            (ijk[:, 2] >= 0) & (ijk[:, 2] < self.reference_shape[2])
        )

        ijk = ijk[in_bounds]
        if ijk.shape[0] == 0:
            return np.empty(0, dtype=np.int64)

        lin = np.ravel_multi_index(
            (ijk[:, 0], ijk[:, 1], ijk[:, 2]),
            dims=self.reference_shape
        )

        return np.unique(lin)

    def _fiber_voxel_index(self, fiber_xyz):
        """
        Convert one fiber polyline into the unique set of voxel indices it traverses.
        """
        fiber_xyz = np.asarray(fiber_xyz, dtype=np.float32)
        if fiber_xyz.shape[0] == 0:
            return np.empty(0, dtype=np.int64)
        if fiber_xyz.shape[0] == 1:
            p = self.world_to_voxel(fiber_xyz[:, :3])[0]
            p = np.round(p).astype(np.int32)
            if (
                0 <= p[0] < self.reference_shape[0] and
                0 <= p[1] < self.reference_shape[1] and
                0 <= p[2] < self.reference_shape[2]
            ):
                return np.asarray([np.ravel_multi_index((p[0], p[1], p[2]), self.reference_shape)], dtype=np.int64)
            return np.empty(0, dtype=np.int64)

        chunks = []
        xyz = fiber_xyz[:, :3]

        for i in range(xyz.shape[0] - 1):
            seg_lin = self._segment_voxel_indices(xyz[i], xyz[i + 1])
            if seg_lin.size > 0:
                chunks.append(seg_lin)

        if len(chunks) == 0:
            return np.empty(0, dtype=np.int64)

        return np.unique(np.concatenate(chunks))

    def build_index(self, fibers):
        """
        fibers: list of arrays, each (n_vertices, 3) or (n_vertices, 4)
        """
        self.fibers = [np.asarray(f, dtype=np.float32) for f in fibers]
        self.fiber_voxel_indices = []

        for fiber in tqdm(self.fibers, desc='Indexing fibers into voxel space'):
            self.fiber_voxel_indices.append(self._fiber_voxel_index(fiber[:, :3]))

        return self

    def query_image_hits(self, voxel_data):
        """
        voxel_data must already be on the reference grid.

        Returns:
            fiber_hit_flags: bool array, shape (n_fibers,)
            hit_linear_indices_per_fiber: list of 1D arrays
        """
        flat = voxel_data.reshape(-1)
        active = flat != 0

        n_fibers = len(self.fiber_voxel_indices)
        hit_flags = np.zeros(n_fibers, dtype=bool)
        hit_linear_indices_per_fiber = [None] * n_fibers

        for i, lin_idx in enumerate(self.fiber_voxel_indices):
            if lin_idx.size == 0:
                hit_linear_indices_per_fiber[i] = np.empty(0, dtype=np.int64)
                continue

            local_hits = lin_idx[active[lin_idx]]
            if local_hits.size > 0:
                hit_flags[i] = True
            hit_linear_indices_per_fiber[i] = local_hits

        return hit_flags, hit_linear_indices_per_fiber