"""Invert target trajectories onto precomputed voxel-seeded tract bundles."""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from calvin_utils.neuroimaging_utils.nifti_utils.volume_io import NiftiIO
from calvin_utils.neuroimaging_utils.tract_utils.fiber_geometry_registration import (
    GeodesicFiberRegistration,
)
from calvin_utils.neuroimaging_utils.tract_utils.fiber_intersection import (
    FiberVoxelIndexer,
)
from calvin_utils.neuroimaging_utils.tract_utils.streaming_fiber_inversion import (
    StreamingFiberInverter,
)
from calvin_utils.neuroimaging_utils.tract_utils.sampled_voxel_fiber_store import (
    SampledVoxelFiberStore,
)
from calvin_utils.neuroimaging_utils.tract_utils.target_trajectory_averaging import (
    TargetTrajectoryAverager,
)
from calvin_utils.neuroimaging_utils.tract_utils.voxel_seeded_connectome import (
    VoxelSeededFiberStore,
)


class InvertedFiberConnectivity:
    """Assign each voxel the mean target similarity of fibers seeded from it."""

    def __init__(
        self,
        voxel_seeded_connectome_path,
        reference_nifti_path,
        mask_path=None,
        sample_interval_mm=1.0,
        max_length_mm=None,
        memory_budget_mb=256,
        max_fiber_batch_size=16_384,
    ):
        self.connectome_path = Path(voxel_seeded_connectome_path).expanduser()
        self.reference_nifti_path = Path(reference_nifti_path).expanduser()
        self.mask_path = Path(mask_path or reference_nifti_path).expanduser()
        self.registration = GeodesicFiberRegistration(
            sample_interval_mm=sample_interval_mm,
            max_length_mm=max_length_mm,
        )
        self.reference_indexer = FiberVoxelIndexer(str(self.reference_nifti_path))
        self._load_mask()
        if SampledVoxelFiberStore.is_store(self.connectome_path):
            self.connectome = SampledVoxelFiberStore(self.connectome_path)
            self._sampled_connectome = True
            self.connectome.validate_sampling(self.registration)
        else:
            self.connectome = VoxelSeededFiberStore(self.connectome_path)
            self._sampled_connectome = False
        self.connectome.validate_reference(self.reference_indexer.reference_img)
        self.target_averager = TargetTrajectoryAverager(
            registration=self.registration,
            raw_fiber_loader=self.reference_indexer.load_fibers,
            selected_fiber_loader=self.reference_indexer.load_selected_fibers,
        )
        self.inverter = StreamingFiberInverter(
            registration=self.registration,
            seed_voxel_mapper=self._masked_seed_voxel,
            seed_voxel_batch_mapper=self._masked_seed_voxels,
            n_voxels=self.n_voxels,
            memory_budget_mb=memory_budget_mb,
            max_fiber_batch_size=max_fiber_batch_size,
        )

    def _load_mask(self):
        import nibabel as nib

        reference = self.reference_indexer.reference_img
        mask = nib.load(str(self.mask_path))
        if mask.shape[:3] != reference.shape[:3] or not np.allclose(
            mask.affine, reference.affine
        ):
            raise ValueError(
                "mask_path and reference_nifti_path must have matching shape "
                "and affine."
            )
        mask_flat = mask.get_fdata().reshape(-1) > 0
        if not np.any(mask_flat):
            raise ValueError("The inversion mask is empty.")
        self._mask_indices = np.flatnonzero(mask_flat)
        self._full_to_mask = np.full(mask_flat.size, -1, dtype=np.int64)
        self._full_to_mask[self._mask_indices] = np.arange(
            self._mask_indices.size, dtype=np.int64
        )

    @property
    def n_voxels(self):
        return int(self._mask_indices.size)

    def _masked_seed_voxel(self, full_voxel_index):
        full_voxel_index = int(full_voxel_index)
        if full_voxel_index < 0 or full_voxel_index >= self._full_to_mask.size:
            raise ValueError(
                f"Seed voxel {full_voxel_index} is outside the reference grid."
            )
        return int(self._full_to_mask[full_voxel_index])

    def _masked_seed_voxels(self, full_voxel_indices):
        full_voxel_indices = np.asarray(full_voxel_indices, dtype=np.int64)
        if np.any(full_voxel_indices < 0) or np.any(
            full_voxel_indices >= self._full_to_mask.size
        ):
            raise ValueError("One or more seed voxels are outside the reference grid.")
        return self._full_to_mask[full_voxel_indices]

    @staticmethod
    def _safe_stem(path):
        name = Path(path).name
        for suffix in (".nii.gz", ".fib.gz", ".fib.npy", ".fib.json"):
            if name.lower().endswith(suffix):
                return name[: -len(suffix)]
        return Path(name).stem

    def average_target_trajectory(
        self,
        target_fiber_path,
        weighting="binary",
        min_abs_value=None,
        top_percent=None,
        fiber_atlas_path=None,
    ):
        """Return one registered average of outcome-positive target fibers."""
        return self.target_averager.from_path(
            target_fiber_path,
            weighting=weighting,
            sign="positive",
            min_abs_value=min_abs_value,
            top_percent=top_percent,
            fiber_atlas_path=fiber_atlas_path,
        )

    def _prepare_targets(
        self,
        target_fiber_paths,
        reduction,
        weighting,
        min_abs_value,
        top_percent,
        fiber_atlas_path,
    ):
        reduction = str(reduction).strip().lower()
        if reduction not in {"average_trajectory", "pairwise_mean"}:
            raise ValueError(
                "target_bundle_reduction must be 'average_trajectory' or "
                "'pairwise_mean'."
            )
        targets = []
        target_weights = []
        for target_path in tqdm(target_fiber_paths, desc="Preparing targets"):
            if reduction == "average_trajectory":
                targets.append(
                    self.average_target_trajectory(
                        target_path,
                        weighting=weighting,
                        min_abs_value=min_abs_value,
                        top_percent=top_percent,
                        fiber_atlas_path=fiber_atlas_path,
                    )
                )
                target_weights.append(None)
            else:
                trajectories, weights = self.target_averager.sampled_from_path(
                    target_path,
                    weighting=weighting,
                    sign="positive",
                    min_abs_value=min_abs_value,
                    top_percent=top_percent,
                    fiber_atlas_path=fiber_atlas_path,
                )
                targets.append(trajectories)
                target_weights.append(weights)
        return targets, target_weights

    def generate_inverted_profiles_streaming(
        self,
        target_fiber_paths,
        similarity="valid_sample_cosine",
        target_weighting="binary",
        target_min_abs_value=None,
        target_top_percent=None,
        target_fiber_atlas_path=None,
        target_bundle_reduction="average_trajectory",
    ):
        """Broadcast target matching over seeded fibers and average by voxel."""
        target_fiber_paths = list(target_fiber_paths)
        targets, target_weights = self._prepare_targets(
            target_fiber_paths,
            target_bundle_reduction,
            target_weighting,
            target_min_abs_value,
            target_top_percent,
            target_fiber_atlas_path,
        )
        if self._sampled_connectome:
            return self.inverter.run_sampled_batches(
                sampled_batches=self.connectome.iter_batches(
                    max_fibers_per_batch=self.inverter.max_fiber_batch_size
                ),
                target_trajectories=targets,
                target_weights=target_weights,
                similarity=similarity,
            )
        return self.inverter.run(
            seeded_fibers=self.connectome.iter_seeded_fibers(),
            target_trajectories=targets,
            target_weights=target_weights,
            similarity=similarity,
        )

    def generate_inverted_connectivity_profile(
        self,
        target_fiber_path,
        **kwargs,
    ):
        return self.generate_inverted_profiles_streaming(
            [target_fiber_path], **kwargs
        )[0]

    def save_profile(self, profile, out_path):
        out_path = Path(out_path)
        if out_path.name.endswith(".nii.gz"):
            output_base = out_path.with_name(out_path.name[:-7])
        elif out_path.suffix == ".nii":
            output_base = out_path.with_suffix("")
        else:
            output_base = out_path
        output_base.parent.mkdir(parents=True, exist_ok=True)
        NiftiIO(mask_path=str(self.mask_path)).save_files(
            arr=np.asarray(profile, dtype=np.float32),
            file_paths=[str(output_base)],
            dry_run=False,
        )
        return output_base.with_name(f"{output_base.name}.nii.gz")

    def save_inverted_profiles_from_fibers(
        self,
        target_fiber_paths,
        out_dir,
        suffix=None,
        **kwargs,
    ):
        target_fiber_paths = list(target_fiber_paths)
        similarity = kwargs.get("similarity", "valid_sample_cosine")
        suffix = suffix or f"_{similarity}_fiber_inversion"
        profiles = self.generate_inverted_profiles_streaming(
            target_fiber_paths, **kwargs
        )
        saved = []
        for target_path, profile in zip(target_fiber_paths, profiles):
            output_path = Path(out_dir) / (
                f"{self._safe_stem(target_path)}{suffix}.nii.gz"
            )
            saved.append(self.save_profile(profile, output_path))
        return saved


FiberTrajectorySimilarity = InvertedFiberConnectivity


__all__ = ["FiberTrajectorySimilarity", "InvertedFiberConnectivity"]
