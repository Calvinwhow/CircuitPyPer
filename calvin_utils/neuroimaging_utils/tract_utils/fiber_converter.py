import os
import json
import shutil
import numpy as np
import nibabel as nib
from pathlib import Path
from nibabel.streamlines import Tractogram
from nibabel.streamlines.trk import TrkFile


class FiberFormatConverter:
    """
    Convert fiber objects shaped like:
        fiber_i = ndarray, shape (n_vertices, 3) or (n_vertices, 4)

    into tractogram formats using a reference tract file/header.

    Current guaranteed output:
        - .trk

    Planned / stub only:
        - .fib / .fibfilt if you later add a known writer
    """

    def __init__(self, reference_path):
        self.reference_path = reference_path
        self.reference_ftype = self._identify_file_type(reference_path)

        if self.reference_ftype not in {"trk", "tck"}:
            raise ValueError(
                f"Reference file must currently be .trk or .tck, got: {reference_path}"
            )

        self.reference_obj = nib.streamlines.load(reference_path)
        self.reference_tractogram = self.reference_obj.tractogram

        self.reference_header = None
        if self.reference_ftype == "trk":
            self.reference_header = self.reference_obj.header.copy()

    @staticmethod
    def _identify_file_type(path):
        suffix = Path(path).suffix.lower()
        if suffix == ".trk":
            return "trk"
        if suffix == ".tck":
            return "tck"
        if suffix == ".fib":
            return "fib"
        if suffix == ".fibfilt":
            return "fibfilt"
        if suffix == ".npy":
            return "npy"
        if suffix == ".npz":
            return "npz"
        if suffix == ".json":
            return "json"
        return "unknown"

    @staticmethod
    def _load_fibers(path):
        """
        Load saved fibers from:
        - .npy object array
        - .npz with key 'fibers'
        - .json with key 'fibers'

        Expected each fiber:
            (n_vertices, 3) or (n_vertices, 4)
        """
        ftype = FiberFormatConverter._identify_file_type(path)

        if ftype == "npy":
            obj = np.load(path, allow_pickle=True)
            if not (isinstance(obj, np.ndarray) and obj.dtype == object):
                raise ValueError(f"Expected object-array fiber file in {path}")
            fibers = obj.tolist()

        elif ftype == "npz":
            obj = np.load(path, allow_pickle=True)
            if "fibers" not in obj:
                raise ValueError(f"NPZ file missing 'fibers' key: {path}")
            fibers = obj["fibers"].tolist()

        elif ftype == "json":
            with open(path, "r") as f:
                obj = json.load(f)
            if "fibers" not in obj:
                raise ValueError(f"JSON file missing 'fibers' key: {path}")
            fibers = obj["fibers"]

        else:
            raise ValueError(f"Unsupported fiber container for loading: {path}")

        fibers = [np.asarray(f, dtype=np.float32) for f in fibers]

        for i, fiber in enumerate(fibers):
            if fiber.ndim != 2 or fiber.shape[1] not in (3, 4):
                raise ValueError(
                    f"Fiber {i} has invalid shape {fiber.shape}. "
                    f"Expected (n_vertices, 3) or (n_vertices, 4)."
                )

        return fibers

    @staticmethod
    def _split_fibers_xyz_and_values(fibers):
        """
        Returns:
            streamlines: list of (n_vertices, 3)
            data_per_streamline: dict[str, list[np.ndarray]]
            data_per_point: dict[str, list[np.ndarray]]

        If a fiber has 4 columns, col 4 is treated as magnitude.
        """
        streamlines = []
        magnitude_per_streamline = []
        magnitude_per_point = []

        has_magnitude = False

        for fiber in fibers:
            xyz = np.asarray(fiber[:, :3], dtype=np.float32)
            streamlines.append(xyz)

            if fiber.shape[1] == 4:
                has_magnitude = True
                mag = np.asarray(fiber[:, 3], dtype=np.float32)

                if mag.shape[0] != xyz.shape[0]:
                    raise ValueError("Magnitude vector length does not match number of vertices.")

                magnitude_per_point.append(mag[:, None])

                unique_mag = np.unique(mag)
                if unique_mag.shape[0] == 1:
                    magnitude_per_streamline.append(np.asarray([unique_mag[0]], dtype=np.float32))
                else:
                    magnitude_per_streamline.append(np.asarray([np.max(mag)], dtype=np.float32))

        data_per_streamline = {}
        data_per_point = {}

        if has_magnitude:
            data_per_streamline["magnitude"] = magnitude_per_streamline
            data_per_point["magnitude"] = magnitude_per_point

        return streamlines, data_per_streamline, data_per_point

    def _make_tractogram(self, fibers):
        """
        Build a tractogram in the same spatial convention as the reference tractogram.
        """
        streamlines, data_per_streamline, data_per_point = self._split_fibers_xyz_and_values(fibers)

        tractogram = Tractogram(
            streamlines=streamlines,
            data_per_streamline=data_per_streamline if len(data_per_streamline) > 0 else None,
            data_per_point=data_per_point if len(data_per_point) > 0 else None,
            affine_to_rasmm=self.reference_tractogram.affine_to_rasmm,
        )

        return tractogram

    def save_trk(self, fibers, out_path):
        """
        Save fibers to .trk using header metadata stolen from the reference .trk when available.
        """
        tractogram = self._make_tractogram(fibers)

        if self.reference_ftype == "trk":
            header = self.reference_header.copy()
            trk = TrkFile(tractogram, header=header)
            nib.streamlines.save(trk, out_path)
            return out_path

        nib.streamlines.save(tractogram, out_path)
        return out_path

    def convert_fiber_file_to_trk(self, fiber_file_path, out_path):
        fibers = self._load_fibers(fiber_file_path)
        return self.save_trk(fibers, out_path)

    def convert_fibers_to_reference_format(self, fibers, out_path):
        out_ftype = self._identify_file_type(out_path)

        if out_ftype == "trk":
            return self.save_trk(fibers, out_path)

        if out_ftype in {"fib", "fibfilt"}:
            raise NotImplementedError(
                "Writing .fib/.fibfilt is not implemented because the file spec is not yet defined here."
            )

        raise ValueError(f"Unsupported output format: {out_path}")

    def batch_convert_fiber_files_to_trk(self, fiber_file_paths, out_dir, suffix="_viz"):
        os.makedirs(out_dir, exist_ok=True)
        out_paths = []

        for fiber_file_path in fiber_file_paths:
            p = Path(fiber_file_path)
            stem = p.name[:-4] if p.name.endswith(".npy") else p.stem
            out_path = os.path.join(out_dir, f"{stem}{suffix}.trk")
            self.convert_fiber_file_to_trk(fiber_file_path, out_path)
            out_paths.append(out_path)

        return out_paths
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert fiber files to .trk using a reference tractogram header.")

    parser.add_argument(
        "--reference_trk",
        required=True,
        help="Reference .trk file to steal header and spatial metadata from"
    )

    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input fiber files (.npy, .npz, .json)"
    )

    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for .trk files"
    )

    parser.add_argument(
        "--suffix",
        default="_viz",
        help="Suffix to append to output filenames"
    )

    args = parser.parse_args()

    converter = FiberFormatConverter(reference_path=args.reference_trk)

    os.makedirs(args.out_dir, exist_ok=True)

    for in_path in args.inputs:
        p = Path(in_path)
        stem = p.name[:-4] if p.name.endswith(".npy") else p.stem
        out_path = os.path.join(args.out_dir, f"{stem}{args.suffix}.trk")

        print(f"Converting {in_path} -> {out_path}")
        converter.convert_fiber_file_to_trk(in_path, out_path)