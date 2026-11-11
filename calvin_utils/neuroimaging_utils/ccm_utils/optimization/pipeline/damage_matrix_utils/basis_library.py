"""Loading, validating, projecting, and reconstructing basis maps."""

from collections.abc import Sequence
import json
from pathlib import Path

import numpy as np

from .damage_matrix_schema import BasisShard


FEATURE_DTYPE = np.float32
BASIS_DIR = Path(__file__).resolve().parents[2] / "basis_maps"
SIDECAR_KEYS = ("cohort", "pathology", "input_path", "subject_column", "regressand_list", "voxelwise_vars", "mask_path", "transform", "patient_ids", "bases")
BASIS_KEYS = ("name", "regressand", "voxelwise_var", "n")


class BasisLibrary:
    """Validated basis shards presented as one ordered basis collection.

    shards: tuple[BasisShard] | length shard_count | selected memory-mapped basis shards.
    map_names: tuple[str] | length basis_count | globally qualified basis names in optimization order.
    voxel_count: int | scalar | number of spatial features shared by all basis maps.
    mask_path: str | scalar path | spatial mask defining voxel order.
    sample_sizes: dict[str, int] | length basis_count | patient count used to build each basis map.
    """

    def __init__(self, shards, map_names, voxel_count, mask_path, sample_sizes):
        self.shards = tuple(shards)
        self.map_names = tuple(map_names)
        self.voxel_count = int(voxel_count)
        self.mask_path = str(mask_path)
        self.sample_sizes = dict(sample_sizes)

    @classmethod
    def load(cls, basis_cohorts: Sequence[str] | None = None, basis_dir: str | Path = BASIS_DIR):
        """Load selected shards, or every shard when basis_cohorts is omitted."""
        directory = Path(basis_dir)
        paths = cls._shard_paths(directory, basis_cohorts)
        missing = [path.name for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{missing} not found in {directory}.")
        if not paths:
            raise ValueError(f"No basis shards in {directory}.")
        shards = tuple(cls._load_shard(path) for path in paths)
        voxel_counts = {shard.values.shape[0] for shard in shards}
        if len(voxel_counts) != 1:
            raise ValueError(f"Shards disagree on voxel count: {sorted(voxel_counts)}.")
        mask_paths = {shard.metadata["mask_path"] for shard in shards}
        if len(mask_paths) != 1:
            raise ValueError("Shards disagree on the mask defining feature order.")
        map_names = tuple(name for shard in shards for name in shard.map_names)
        if len(set(map_names)) != len(map_names):
            raise ValueError("Basis names repeat across the selected shards.")
        sample_sizes = {name: int(basis["n"]) for shard in shards for name, basis in zip(shard.map_names, shard.metadata["bases"])}
        return cls(shards, map_names, voxel_counts.pop(), mask_paths.pop(), sample_sizes)

    @staticmethod
    def _shard_paths(directory, basis_cohorts):
        """Resolve selected cohort names to ordered shard paths."""
        if basis_cohorts is None:
            return sorted(directory.glob("shard_*.npy"))
        return [directory / f"shard_{cohort}.npy" for cohort in basis_cohorts]

    @staticmethod
    def _load_shard(path):
        """Memory-map one basis shard after validating its JSON sidecar."""
        path = Path(path)
        sidecar_path = path.with_suffix(".json")
        metadata = json.loads(sidecar_path.read_text())
        missing = [key for key in SIDECAR_KEYS if key not in metadata]
        if missing:
            raise ValueError(f"{sidecar_path.name} is missing {missing}.")
        if path.stem != f"shard_{metadata['cohort']}":
            raise ValueError(f"{path.name} does not match its sidecar cohort {metadata['cohort']!r}.")
        for basis in metadata["bases"]:
            missing = [key for key in BASIS_KEYS if key not in basis]
            if missing:
                raise ValueError(f"A basis in {sidecar_path.name} is missing {missing}.")
        values = np.load(path, mmap_mode="r")
        if values.dtype != FEATURE_DTYPE or values.ndim != 2 or values.shape[1] != len(metadata["bases"]):
            raise ValueError(f"{path.name} must be a {np.dtype(FEATURE_DTYPE).name} voxel-by-basis array with {len(metadata['bases'])} columns; found {values.dtype} {values.shape}.")
        map_names = tuple(f"{metadata['cohort']}:{basis['name']}" for basis in metadata["bases"])
        return BasisShard(path, values, metadata, map_names)

    @property
    def symptoms(self):
        """symptoms: tuple[str] | length unique target count | clinical targets represented by the selected bases."""
        return tuple(dict.fromkeys(basis["regressand"] for shard in self.shards for basis in shard.metadata["bases"]))

    def project(self, patient_maps, voxel_chunk_size):
        """Return raw patient-by-basis dot products and one L2 norm per basis map."""
        products = [np.zeros((len(values), len(self.map_names))) for values in patient_maps]
        squared_norms = np.zeros(len(self.map_names))
        for start in range(0, self.voxel_count, voxel_chunk_size):
            block = np.hstack([shard.values[start:start + voxel_chunk_size] for shard in self.shards])
            if not np.isfinite(block).all():
                raise ValueError(f"Basis shards contain NaN or infinity in voxels {start} to {start + len(block)}.")
            squared_norms += np.square(block, dtype=np.float64).sum(axis=0)
            for product, values in zip(products, patient_maps):
                product += values[:, start:start + voxel_chunk_size] @ block
        return tuple(products), np.sqrt(squared_norms)

    def check_test_leakage(self, test_identifiers):
        """Refuse external-test patients used to build a selected basis cohort."""
        metadata = {shard.metadata["cohort"]: shard.metadata for shard in self.shards}
        leaking = []
        for cohort_name, patient_ids in test_identifiers:
            if cohort_name not in metadata:
                continue
            bare_ids = self._bare_patient_ids(cohort_name, patient_ids)
            if set(bare_ids) & set(map(str, metadata[cohort_name]["patient_ids"])):
                leaking.append(cohort_name)
        if leaking:
            raise ValueError(f"Test datasets {sorted(leaking)} contain patients that built selected basis cohorts; leave those cohorts out of basis_cohorts or test on other patients.")

    @staticmethod
    def _bare_patient_ids(cohort_name, patient_ids):
        """Remove an optional cohort-name prefix from patient identifiers."""
        prefix = f"{cohort_name}:"
        values = np.asarray(patient_ids).astype(str)
        return np.asarray([value[len(prefix):] if value.startswith(prefix) else value for value in values])

    def reconstruct_map(self, weights, basis_l2_norms):
        """Return one voxel-space map from coefficients on normalized bases."""
        normalized_weights = np.asarray(weights, dtype=float).reshape(-1) / basis_l2_norms
        symptom_map = np.zeros(self.voxel_count, dtype=FEATURE_DTYPE)
        offset = 0
        for shard in self.shards:
            width = shard.values.shape[1]
            symptom_map += shard.values @ normalized_weights[offset:offset + width].astype(FEATURE_DTYPE)
            offset += width
        return symptom_map

    def normalized_maps(self, basis_l2_norms):
        """Return basis maps as a basis_count by voxel_count normalized array."""
        matrix = np.hstack([shard.values for shard in self.shards])
        matrix /= basis_l2_norms.astype(FEATURE_DTYPE)
        return matrix.T
