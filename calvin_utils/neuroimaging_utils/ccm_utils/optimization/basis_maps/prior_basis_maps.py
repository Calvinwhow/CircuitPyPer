"""Load existing localization maps into one basis shard, without regression."""

import json
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np

from ..pipeline import basis_damage
from .build import save_atomically, voxelize_fibers
from calvin_utils.neuroimaging_utils.io.importers import GiiNiiFileImport
from calvin_utils.neuroimaging_utils.tract_utils.fiber_converter import FiberFormatConverter
from calvin_utils.neuroimaging_utils.tract_utils.fiber_io import FiberIO


class PriorBasisMaps:
    """Stack already-localized maps into shard_{cohort}.npy, shape (voxels, maps), and its shard_{cohort}.json sidecar.

    cohort: shard name; bases load as "{cohort}:{map name}".
    maps: {map name: {"path": NIfTI, mask-ordered .npy vector, or .fib.json fiber map, "representation": the map's anatomy, local, networks, or fibers}}, plus any basis fields you have for a map (e.g. "n", "regressand", "patient_ids").
    mask_path: the registry's NIfTI mask, which defines the shard's voxel order.
    directory: basis folder the shard is written to.
    """

    def __init__(self, cohort, maps, mask_path, directory):
        self.cohort = cohort
        self.maps = maps
        self.mask_path = mask_path
        self.directory = Path(directory)

    def run(self):
        """Load every map in mask order, stack them as columns, and write the shard."""
        n_voxels = int((np.asarray(nib.load(str(self.mask_path)).dataobj) > 0).sum())
        values = np.column_stack([self.load_map(name, fields["path"], n_voxels) for name, fields in self.maps.items()]).astype(basis_damage.FEATURE_DTYPE)
        return self.save_shard(values)

    def load_map(self, name, path, n_voxels):
        """Load one map as a mask-ordered vector of length n_voxels; a fiber map is rasterized on the atlas its .fib.json names."""
        path = Path(path).expanduser()
        importer = GiiNiiFileImport(import_path=None, mask_path=str(self.mask_path))
        file_type = importer._identify_file_type(path)
        if file_type == "fiber":
            values_path = FiberFormatConverter.values_from_description(path) if path.name.lower().endswith((".fib.json", ".fib.desc.json")) else path
            fiber_atlas = str(FiberFormatConverter.atlas_from_values_description(values_path))
            fiber_values = np.asarray(FiberIO(mask_path=fiber_atlas).load_map_values(path), dtype=np.float32).reshape(-1, 1)
            values = voxelize_fibers(fiber_values, fiber_atlas, self.mask_path)[:, 0]
        elif file_type == "npy":
            values = np.asarray(np.load(path)).reshape(-1)
        else:
            values = np.asarray(importer._import_matrices([str(path)])).reshape(-1)
        if values.size != n_voxels:
            raise ValueError(f"Map {name!r} ({path}) has {values.size} values; the mask has {n_voxels} voxels.")
        return values

    def save_shard(self, values):
        """Write the (voxels, maps) array and a sidecar with every key basis_damage.load_shard requires; a map's own fields override the defaults."""
        bases = [{"name": name, "regressand": name, "voxelwise_var": name, "n": 1, **{key: value for key, value in fields.items() if key != "path"}} for name, fields in self.maps.items()]
        patient_ids = sorted({str(patient) for fields in self.maps.values() for patient in fields.get("patient_ids", [])})
        sidecar = {"cohort": self.cohort, "maps": self.maps, "pathology": self.cohort, "input_path": None, "subject_column": None, "regressand_list": [basis["regressand"] for basis in bases], "voxelwise_vars": [basis["voxelwise_var"] for basis in bases], "mask_path": self.mask_path, "transform": None, "patient_ids": patient_ids, "bases": bases, "n_voxels": int(values.shape[0]), "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        path = self.directory / f"shard_{self.cohort}.npy"
        save_atomically(path, lambda stream: np.save(stream, values))
        save_atomically(path.with_suffix(".json"), lambda stream: stream.write((json.dumps(sidecar, indent=2) + "\n").encode()))
        print(f"shard_{self.cohort}: wrote {values.shape[1]} prior maps x {values.shape[0]} voxels to {path}", flush=True)
        return sidecar
