"""Per-anatomy localization maps from a finished model: each anatomy's weighted contribution to the model's final map."""

import csv
import json
from pathlib import Path

import numpy as np

from ..basis_maps.build import fiber_file_atlas
from ..neural_network.anatomy import REPRESENTATIONS, canonical_representation
from ..pipeline import basis_damage
from calvin_utils.neuroimaging_utils.io.exporters import NeuroimageFileOutporter
from calvin_utils.neuroimaging_utils.tract_utils.fiber_converter import FiberFormatConverter
from calvin_utils.neuroimaging_utils.tract_utils.fiber_io import FiberIO


class AnatomyMaps:
    """Write the local, networks, and fibers contributions of a finished model, from its saved weights and network parameters.

    final_model_dir: a model's final_model/ folder, holding optimized_weights.csv; a network also holds neural_network_parameters.npz and model.json.
    basis_dir: basis folder holding the shards the model was fitted on.

    Variables (K bases, A algorithms; g is local, networks, or fibers; f indexes voxels, or atlas fibers for the fibers output):
        x_k[f]: basis k's values: its shard column for local and networks, its native .fib.npy for fibers.
        s_k: L2 norm of basis k's shard column; every fit weights x_k / s_k.
        Linear model, weights w_k (already scaled to unit prediction variance by the fit):
            contribution_g[f] = sum over k in g of w_k x_k[f] / s_k
        Network, with W[k, a] = algorithm_weights, b[g, a] = algorithm_bias, alpha[g, a] = algorithm_mix, c[g] = representation_bias, beta[g] = representation_mix, h0[g] = representation_baseline, and sigma = model.json's prediction_scale:
            Z[g, f, a] = sum over k in g of W[k, a] x_k[f] / s_k
            H[g, f] = ReLU(sum_a alpha[g, a] ReLU(Z[g, f, a] + b[g, a]) + c[g])
            contribution_g[f] = beta[g] (H[g, f] - h0[g]) / sigma
    Writes anatomy_local and anatomy_networks as NIfTI, and anatomy_fibers_{atlas}.fib.npy / .fib.json for each fiber atlas.
    """

    def __init__(self, final_model_dir, basis_dir=basis_damage.BASIS_DIR):
        self.final_model_dir = Path(final_model_dir)
        self.basis_dir = Path(basis_dir)

    def run(self):
        """Write one map per anatomy and return the written paths; does nothing when the model saved no weights."""
        if not (self.final_model_dir / "optimized_weights.csv").is_file():
            return []
        network = self.read_network()
        names, weights = self.read_weights(network)
        library = basis_damage.load_library(sorted({name.split(":", 1)[0] for name in names}), self.basis_dir)
        bases = self.read_bases(names, library)
        written = []
        for anatomy in ("local", "networks"):
            members = [index for index, basis in enumerate(bases) if basis["anatomy"] == anatomy]
            if members:
                normalized = np.column_stack([bases[index]["voxels"] / bases[index]["scale"] for index in members])
                written.append(self.write_nifti(anatomy, self.contribution(anatomy, normalized, weights[members], network), library["mask_path"]))
        fiber_members = [index for index, basis in enumerate(bases) if basis["anatomy"] == "fibers"]
        for atlas in dict.fromkeys(bases[index]["fiber_atlas"] for index in fiber_members):
            members = [index for index in fiber_members if bases[index]["fiber_atlas"] == atlas]
            normalized = np.column_stack([np.load(bases[index]["fiber_values_path"]).reshape(-1) / bases[index]["scale"] for index in members])
            written.append(self.write_fibers(atlas, self.contribution("fibers", normalized, weights[members], network)))
        return written

    def read_network(self):
        """The network's saved parameters and prediction scale, or None for a linear model."""
        parameter_path = self.final_model_dir / "neural_network_parameters.npz"
        if not parameter_path.is_file():
            return None
        network = dict(np.load(parameter_path))
        network["prediction_scale"] = float(json.loads((self.final_model_dir / "model.json").read_text())["prediction_scale"])
        return network

    def read_weights(self, network):
        """Basis names in fitted order and their weights, shape (K, A): the CSV weights for a linear model (A = 1), algorithm_weights for the network."""
        with (self.final_model_dir / "optimized_weights.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        names = list(dict.fromkeys(row["map"] for row in rows))
        weights = np.asarray([[float(row["weight"])] for row in rows]) if network is None else np.asarray(network["algorithm_weights"], dtype=float)
        if weights.shape[0] != len(names):
            raise ValueError(f"{len(names)} basis names but {weights.shape[0]} weight rows in {self.final_model_dir}.")
        return names, weights

    def read_bases(self, names, library):
        """For each basis in fitted order: its anatomy, shard column, L2 scale, and for fibers its native values path and atlas."""
        columns = {name: (shard, index) for shard in library["shards"] for index, name in enumerate(shard["names"])}
        bases = []
        for name in names:
            if name not in columns:
                raise ValueError(f"Basis {name!r} is not in the shards in {self.basis_dir}.")
            shard, index = columns[name]
            basis = shard["sidecar"]["bases"][index]
            if basis.get("representation") is None:
                raise ValueError(f"Basis {name!r} has no anatomy label.")
            voxels = np.asarray(shard["values"][:, index], dtype=np.float32)
            record = {"anatomy": canonical_representation(basis["representation"]), "voxels": voxels, "scale": float(np.sqrt(np.square(voxels, dtype=np.float64).sum()))}
            if record["anatomy"] == "fibers":
                native = basis.get("fiber_path") or shard["sidecar"].get("maps", {}).get(basis["name"], {}).get("path")
                if native is None:
                    raise ValueError(f"Fiber basis {name!r} has no native .fib.json.")
                native = Path(native).expanduser()
                record["fiber_values_path"] = FiberFormatConverter.values_from_description(native) if native.name.lower().endswith((".fib.json", ".fib.desc.json")) else native
                record["fiber_atlas"] = fiber_file_atlas(native)
            bases.append(record)
        return bases

    def contribution(self, anatomy, normalized, weights, network):
        """One anatomy's contribution at every voxel or fiber. normalized: (F, n) basis values divided by their scales; weights: (n, A)."""
        algorithm_maps = normalized @ weights
        if network is None:
            return algorithm_maps[:, 0]
        group = REPRESENTATIONS.index(anatomy)
        mixed = np.maximum(algorithm_maps + network["algorithm_bias"][group], 0) @ network["algorithm_mix"][group]
        anatomy_map = np.maximum(mixed + network["representation_bias"][group], 0)
        return network["representation_mix"][group] * (anatomy_map - network["representation_baseline"][group]) / network["prediction_scale"]

    def write_nifti(self, anatomy, values, mask_path):
        """Write one voxel contribution as anatomy_{anatomy} NIfTI in final_model_dir."""
        exporter = NeuroimageFileOutporter(output_ftype="nifti", mask_path=mask_path)
        exporter.validate_for_output()
        exporter.save_map(map_data=values.astype(np.float32), file_name=f"anatomy_{anatomy}", out_dir=str(self.final_model_dir))
        return self.final_model_dir / f"anatomy_{anatomy}"

    def write_fibers(self, atlas, values):
        """Write one native fiber contribution as anatomy_fibers_{atlas}.fib.npy with its .fib.json descriptor."""
        values_path = self.final_model_dir / f"anatomy_fibers_{Path(atlas).stem}.fib.npy"
        np.save(values_path, values.astype(np.float32))
        return FiberIO.write_values_description(values_path, atlas)
