from __future__ import annotations

import json
from pathlib import Path

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
import nibabel as nib

from calvin_utils.neuroimaging_utils.tract_utils.fiber_intersection import FiberVoxelIndexer


class FiberAtlasConverter:
    """
    Convert external tract/fiber files into Calvin's internal ``.npz`` atlas.

    The intended architecture is:

    - external formats enter once through this converter
    - internal fiber filtering uses ``.npz`` atlases only
    - visualization/export tools convert ``.npz`` back out to ``.tck``, MAT,
      NIfTI density maps, or Lead-DBS-compatible outputs

    Supported inputs
    ----------------
    ``.tck`` / ``.tt.tck``
        Standard MRtrix TCK files plus DSI Studio TCK-like files that omit the
        final MRtrix EOF marker. The DSI HCP1065 voxel-space transform already
        handled by ``FiberVoxelIndexer`` is applied during load.

    ``.trk`` / ``.trx``
        Streamline files readable by nibabel.

    ``.npy`` / ``.npz`` / ``.json``
        Existing Calvin-style fiber containers. ``.npz`` must contain a
        ``fibers`` key.

    Output schema
    -------------
    The output is a compressed ``.npz`` with:

    ``fibers``
        Object array, one ``N_i x 3`` float32 array per fiber in MNI/world mm.
    ``idx``
        Per-fiber vertex counts.
    ``global_fiber_ids``
        Zero-based source-order fiber IDs.
    ``metadata_json``
        JSON string describing source, counts, and coordinate assumptions.

    ``run()`` returns the output path as a string.
    """

    def __init__(
        self,
        *,
        input_path: str | Path,
        out_path: str | Path,
        dtype=np.float32,
        overwrite: bool = True,
    ):
        self.input_path = Path(input_path)
        self.out_path = Path(out_path)
        self.dtype = dtype
        self.overwrite = bool(overwrite)

        self.fibers = None
        self.metadata = None

    def run(self):
        self.validate_inputs()
        self.load_fibers()
        self.validate_fibers()
        self.build_metadata()
        self.save_npz()
        return str(self.out_path)

    def validate_inputs(self):
        if not self.input_path.exists():
            raise FileNotFoundError(f"input_path does not exist: {self.input_path}")
        if self.out_path.suffix.lower() != ".npz":
            raise ValueError(f"out_path must end with .npz, got: {self.out_path}")
        if self.out_path.exists() and not self.overwrite:
            raise FileExistsError(f"out_path already exists: {self.out_path}")

    def load_fibers(self):
        ftype = FiberVoxelIndexer._identify_fiber_file_type(self.input_path)

        if ftype in {"trk", "trx"}:
            tractogram = nib.streamlines.load(str(self.input_path)).tractogram
            self.fibers = [np.asarray(streamline, dtype=self.dtype)[:, :3] for streamline in tractogram.streamlines]
            return

        if ftype == "tck":
            try:
                tractogram = nib.streamlines.load(str(self.input_path)).tractogram
                self.fibers = [np.asarray(streamline, dtype=self.dtype)[:, :3] for streamline in tractogram.streamlines]
            except Exception:
                self.fibers = FiberVoxelIndexer._load_tck_tolerant(str(self.input_path))
            return

        if ftype == "npz":
            obj = np.load(self.input_path, allow_pickle=True)
            if "fibers" not in obj:
                raise ValueError(f"NPZ input is missing a 'fibers' key: {self.input_path}")
            self.fibers = [np.asarray(fiber, dtype=self.dtype)[:, :3] for fiber in obj["fibers"].tolist()]
            return

        if ftype == "npy":
            obj = np.load(self.input_path, allow_pickle=True)
            if not (isinstance(obj, np.ndarray) and obj.dtype == object):
                raise ValueError(f"Expected object-array fiber file: {self.input_path}")
            self.fibers = [np.asarray(fiber, dtype=self.dtype)[:, :3] for fiber in obj.tolist()]
            return

        if ftype == "json":
            obj = json.loads(self.input_path.read_text())
            if "fibers" not in obj:
                raise ValueError(f"JSON input is missing a 'fibers' key: {self.input_path}")
            self.fibers = [np.asarray(fiber, dtype=self.dtype)[:, :3] for fiber in obj["fibers"]]
            return

        raise ValueError(f"Unsupported input fiber format: {self.input_path}")

    def validate_fibers(self):
        if self.fibers is None:
            raise RuntimeError("No fibers loaded.")
        cleaned = []
        for i, fiber in enumerate(self.fibers):
            fiber = np.asarray(fiber, dtype=self.dtype)
            if fiber.ndim != 2 or fiber.shape[1] < 3:
                raise ValueError(f"Fiber {i} has invalid shape {fiber.shape}; expected N x 3 or N x 4.")
            fiber = fiber[:, :3]
            finite = np.isfinite(fiber).all(axis=1)
            fiber = fiber[finite]
            if fiber.shape[0] > 1:
                cleaned.append(fiber.astype(self.dtype, copy=False))
        self.fibers = cleaned

    def build_metadata(self):
        idx = self._fiber_lengths(self.fibers)
        self.metadata = {
            "source_path": str(self.input_path),
            "output_path": str(self.out_path),
            "source_suffixes": self.input_path.suffixes,
            "format": "calvin_fiber_atlas_npz",
            "coordinate_space": "MNI/world mm",
            "n_fibers": int(len(self.fibers)),
            "n_vertices": int(idx.sum()),
            "dtype": str(np.dtype(self.dtype)),
            "notes": (
                "External tract formats should be converted once to this NPZ "
                "before use in filter_fibers or regression-prep workflows."
            ),
        }

    def save_npz(self):
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        fiber_array = np.empty(len(self.fibers), dtype=object)
        for i, fiber in enumerate(self.fibers):
            fiber_array[i] = fiber

        np.savez_compressed(
            self.out_path,
            fibers=fiber_array,
            idx=self._fiber_lengths(self.fibers),
            global_fiber_ids=np.arange(len(self.fibers), dtype=np.int64),
            metadata_json=json.dumps(self.metadata),
        )

    @staticmethod
    def _fiber_lengths(fibers):
        return np.asarray([fiber.shape[0] for fiber in fibers], dtype=np.int64)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert external fiber/tract files into Calvin .npz fiber atlas format.")
    parser.add_argument("--input", required=True, help="Input .tck/.tt.tck/.trk/.trx/.npy/.npz/.json fiber file.")
    parser.add_argument("--out", required=True, help="Output Calvin .npz fiber atlas.")
    parser.add_argument("--no-overwrite", action="store_true", help="Fail if output already exists.")
    args = parser.parse_args()

    out = FiberAtlasConverter(
        input_path=args.input,
        out_path=args.out,
        overwrite=not args.no_overwrite,
    ).run()
    print(out)
