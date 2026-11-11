"""Per-anatomy visualizations from a finished optimization model."""

from pathlib import Path

import numpy as np

from ..basis_maps.build import fiber_file_atlas
from ..neural_network.anatomy import REPRESENTATIONS, canonical_representation, representation_masks
from ..neural_network.dependencies import require_torch
from ..pipeline.model_fitter_utils.model_fitter_schema import NetworkResult
from calvin_utils.neuroimaging_utils.io.exporters import NeuroimageFileOutporter
from calvin_utils.neuroimaging_utils.tract_utils.fiber_converter import FiberFormatConverter
from calvin_utils.neuroimaging_utils.tract_utils.fiber_io import FiberIO


class AnatomyMaps:
    """Write local, network, and native-fiber visualizations from an in-memory final result."""

    def __init__(self, result, final_model_dir, chunk_voxels=16384):
        self.result = result
        self.final_model_dir = Path(final_model_dir)
        self.chunk_voxels = chunk_voxels

    def run(self):
        """Reconstruct and write each represented anatomy."""
        bases = self.bases()
        weights = self.weights()
        algorithm_maps = self.voxel_algorithm_maps(bases, weights)
        model, scale = self.integration_model(weights, algorithm_maps, bases)
        domain_maps = self.domain_maps(model, algorithm_maps, scale)
        written = []
        for group, anatomy in enumerate(REPRESENTATIONS):
            if any(basis["anatomy"] == anatomy for basis in bases):
                written.append(self.write_nifti(anatomy, domain_maps[group]))
        groups = {}
        for basis in bases:
            if basis["anatomy"] == "fibers":
                groups.setdefault(basis["fiber_atlas_name"], []).append(basis)
        for atlas_name, members in groups.items():
            native_maps = np.zeros((len(REPRESENTATIONS), len(np.load(members[0]["fiber_values_path"], mmap_mode="r").reshape(-1)), weights.shape[1]), dtype=np.float32)
            native_maps[REPRESENTATIONS.index("fibers")] = self.fiber_algorithm_maps(members, weights)
            native_model, native_scale = self.integration_model(weights, native_maps, bases)
            native_fibers = self.domain_maps(native_model, native_maps, native_scale)[REPRESENTATIONS.index("fibers")]
            written.append(self.write_fibers(atlas_name, members[0]["fiber_atlas"], native_fibers))
        return written

    def weights(self):
        """Return the basis-by-algorithm weights used to construct domain maps."""
        if not isinstance(self.result, NetworkResult):
            return np.asarray(self.result.weights, dtype=np.float32).reshape(-1, 1)
        return np.asarray(self.result.parameters.algorithm_weights, dtype=np.float32)

    def integration_model(self, weights, algorithm_maps, bases):
        """Use the fitted network, or its identity initialization for a linear model."""
        if isinstance(self.result, NetworkResult):
            return self.result.model, self.result.training.prediction_scale
        from ..neural_network.model import AnatomicalNetwork
        labels = tuple(basis["anatomy"] for basis in bases)
        model = AnatomicalNetwork(weights, representation_masks(labels), fixed_algorithm_maps=algorithm_maps, mode="frozen")
        return model, 1.0

    def domain_maps(self, model, algorithm_maps, scale):
        """Run the underlying network layers and return their linearly summable domain maps."""
        with require_torch().no_grad():
            _, _, maps = model.integrate_algorithm_maps(algorithm_maps)
        return maps.detach().cpu().numpy().astype(np.float32, copy=False) / np.float32(scale)

    def bases(self):
        """Return lightweight metadata for every fitted basis in weight order."""
        library = self.result.damage.basis_library
        columns = {name: (shard, index) for shard in library.shards for index, name in enumerate(shard.map_names)}
        bases = []
        for weight_index, (name, scale) in enumerate(zip(self.result.damage.map_names, self.result.damage.basis_l2_norms)):
            shard, column = columns[name]
            metadata = shard.metadata["bases"][column]
            anatomy = canonical_representation(metadata["representation"])
            basis = {"name": name, "anatomy": anatomy, "shard": shard, "column": column, "scale": float(scale), "weight_index": weight_index}
            if anatomy == "fibers":
                native = metadata.get("fiber_path") or shard.metadata.get("maps", {}).get(metadata["name"], {}).get("path")
                native = Path(native).expanduser()
                atlas = fiber_file_atlas(native)
                basis["fiber_values_path"] = FiberFormatConverter.values_from_description(native) if native.name.lower().endswith((".fib.json", ".fib.desc.json")) else native
                basis["fiber_atlas"] = atlas
                basis["fiber_atlas_name"] = Path(atlas).stem
            bases.append(basis)
        return bases

    def voxel_algorithm_maps(self, bases, weights):
        """Accumulate every domain's voxelized algorithm maps in bounded chunks."""
        outputs = np.zeros((len(REPRESENTATIONS), self.result.damage.basis_library.voxel_count, weights.shape[1]), dtype=np.float32)
        for group, anatomy in enumerate(REPRESENTATIONS):
            for shard in self.result.damage.basis_library.shards:
                members = [basis for basis in bases if basis["shard"] is shard and basis["anatomy"] == anatomy]
                if not members:
                    continue
                columns = [basis["column"] for basis in members]
                coefficients = np.stack([weights[basis["weight_index"]] / basis["scale"] for basis in members])
                for start in range(0, shard.values.shape[0], self.chunk_voxels):
                    stop = start + self.chunk_voxels
                    outputs[group, start:stop] += shard.values[start:stop, columns] @ coefficients
        return outputs

    def fiber_algorithm_maps(self, bases, weights):
        """Accumulate native fiber maps whose descriptors name the same atlas."""
        first = np.load(bases[0]["fiber_values_path"], mmap_mode="r").reshape(-1)
        output = np.zeros((len(first), weights.shape[1]), dtype=np.float32)
        for basis in bases:
            values = np.load(basis["fiber_values_path"], mmap_mode="r").reshape(-1)
            coefficients = weights[basis["weight_index"]] / basis["scale"]
            for algorithm, coefficient in enumerate(coefficients):
                output[:, algorithm] += values * coefficient
        return output

    def write_nifti(self, anatomy, values):
        """Write one voxel-space anatomy map."""
        exporter = NeuroimageFileOutporter(output_ftype="nifti", mask_path=self.result.damage.basis_library.mask_path)
        exporter.validate_for_output()
        exporter.save_map(map_data=values.astype(np.float32), file_name=f"anatomy_{anatomy}", out_dir=str(self.final_model_dir))
        return self.final_model_dir / f"anatomy_{anatomy}"

    def write_fibers(self, atlas_name, atlas, values):
        """Write one native fiber map for all bases sharing an atlas name."""
        values_path = self.final_model_dir / f"anatomy_fibers_{atlas_name}.fib.npy"
        np.save(values_path, values.astype(np.float32))
        return FiberIO.write_values_description(values_path, atlas)
