import os
import json
import hashlib
import shutil
import numpy as np
import nibabel as nib
from pathlib import Path
from nibabel.streamlines import Tractogram
from nibabel.streamlines.trk import TrkFile
from scipy.io import loadmat


class FiberFormatConverter:
    """Convert native/Lead-DBS fiber containers to tractogram files.

    Instance methods write TRK using a reference tractogram header. The
    ``convert_leaddbs_mat_to_tck`` class method needs no reference file because
    Lead-DBS geometry is already stored in world-millimeter coordinates.

    Fiber objects are shaped like:
        fiber_i = ndarray, shape (n_vertices, 3) or (n_vertices, 4)

    Native results use a paired-file contract::

        <stem>.fib.npy  one scalar per canonical atlas fiber
        <stem>.fib.json descriptor identifying that atlas and ordering

    ``load_fib_npy`` and ``convert_fib_npy_to_tck`` require and resolve the
    descriptor for new numeric ``*.fib.npy`` results. The descriptor itself may
    also be supplied as the input path because it records the values vector.
    The loaders verify that the descriptor belongs to the vector, validate its
    SHA-256, check the atlas file size, and finally require equal atlas/value
    counts. The relative atlas path is tried before the
    absolute path so a result tree remains usable after being moved together
    with its resources. An explicit atlas may identify a relocated copy but
    does not replace the descriptor.

    Legacy ``*.fib.values.npy``/``*.fib.desc.json`` pairs and self-contained
    geometry-bearing ``*.fib.npy`` files remain readable. Bare one-dimensional
    ``*.fib.npy`` files also remain readable with an explicit atlas.
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
        if suffix == ".trx":
            return "trx"
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
        if suffix == ".mat":
            return "mat"
        return "unknown"

    @staticmethod
    def _load_fibers(path):
        """
        Load saved fibers from:
        - .trk/.tck/.trx tractogram
        - .npy object array
        - .npz with key 'fibers'
        - .json with key 'fibers'

        Expected each fiber:
            (n_vertices, 3) or (n_vertices, 4)
        """
        ftype = FiberFormatConverter._identify_file_type(path)

        if ftype in {"trk", "tck", "trx"}:
            tractogram = nib.streamlines.load(str(path)).tractogram
            fibers = [
                np.asarray(streamline, dtype=np.float32)[:, :3]
                for streamline in tractogram.streamlines
            ]

        elif ftype == "npy":
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
    def _as_streamline(value):
        fiber = np.asarray(value, dtype=np.float32)
        if fiber.ndim != 2:
            raise ValueError(f"Expected a 2D fiber array, got shape {fiber.shape}.")
        if fiber.shape[1] not in (3, 4) and fiber.shape[0] in (3, 4):
            fiber = fiber.T
        if fiber.shape[1] not in (3, 4):
            raise ValueError(f"Expected an N x 3 or N x 4 fiber, got shape {fiber.shape}.")
        if not len(fiber):
            raise ValueError("Empty fibers cannot be written to TCK.")
        return fiber[:, :3]

    @staticmethod
    def values_description_path(values_path):
        """Return the required sibling descriptor for a fiber values file.

        New ``map.fib.npy`` maps to ``map.fib.json``. Legacy
        ``map.fib.values.npy`` and ``map.values.npy`` continue to map to
        ``map.fib.desc.json``.
        """
        values_path = Path(values_path).expanduser()
        name = values_path.name
        lower_name = name.lower()
        if lower_name.endswith(".fib.npy"):
            return values_path.with_name(f"{name[:-4]}.json")
        if not lower_name.endswith(".values.npy"):
            raise ValueError(
                f"Expected a .fib.npy or .values.npy path, got: {values_path}"
            )
        base = name[:-11]
        if not base.lower().endswith(".fib"):
            base += ".fib"
        return values_path.with_name(f"{base}.desc.json")

    @classmethod
    def values_from_description(cls, description_path):
        """Resolve the values vector recorded by a fiber JSON descriptor.

        New descriptors contain absolute and relative paths; the relative path
        is preferred so a result directory can move as a unit. Older
        descriptors containing only ``values_file`` remain readable.
        """
        description_path = Path(description_path).expanduser().resolve()
        if not description_path.is_file():
            raise FileNotFoundError(description_path)
        description = json.loads(description_path.read_text(encoding="utf-8"))
        if description.get("schema") != "calvin_utils.fiber_values":
            raise ValueError(f"Unrecognized fiber descriptor schema: {description_path}")
        if description.get("schema_version", 1) not in {1, 2}:
            raise ValueError(f"Unsupported fiber descriptor version: {description_path}")

        values = description.get("values") or {}
        candidates = []
        for relative_path in (
            values.get("relative_path"),
            description.get("values_relative_path"),
            description.get("values_file"),
        ):
            if relative_path:
                candidates.append((description_path.parent / relative_path).resolve())
        for absolute_path in (values.get("path"), description.get("values_path")):
            if absolute_path:
                candidates.append(Path(absolute_path).expanduser())

        expected_name = description.get("values_file")
        expected_size = values.get("size_bytes")
        for candidate in dict.fromkeys(candidates):
            if not candidate.is_file():
                continue
            if expected_name and candidate.name != expected_name:
                continue
            if expected_size is not None and candidate.stat().st_size != expected_size:
                raise ValueError(
                    f"Fiber values size does not match {description_path}: {candidate}"
                )
            expected_digest = values.get("sha256")
            if expected_digest:
                digest = hashlib.sha256()
                with candidate.open("rb") as values_file:
                    for chunk in iter(lambda: values_file.read(1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != expected_digest:
                    raise ValueError(
                        f"Fiber values SHA-256 does not match {description_path}."
                    )
            return candidate
        raise FileNotFoundError(
            f"The values vector recorded in {description_path} could not be found: "
            f"{candidates}"
        )

    @classmethod
    def atlas_from_values_description(cls, values_path, atlas_override=None):
        """Resolve and validate the atlas paired with a fiber values vector.

        Validation covers the descriptor schema, paired value filename,
        value-file SHA-256, recorded atlas size, and atlas existence. The
        returned path supplies geometry only; ``load_fib_npy`` subsequently
        verifies that its fiber count equals the vector length.

        ``atlas_override`` may point to a relocated copy of the recorded atlas.
        It does not replace the descriptor: the descriptor and value checksum
        are still required, and the override must match the recorded file size.

        Raises
        ------
        FileNotFoundError
            If the descriptor or every recorded atlas location is missing.
        ValueError
            If the descriptor is for another vector, has an unknown schema,
            or fails an integrity check.
        """
        values_path = Path(values_path).expanduser().resolve()
        description_path = cls.values_description_path(values_path)
        if not description_path.is_file():
            raise FileNotFoundError(
                f"Fiber values require the paired descriptor: {description_path}"
            )
        description = json.loads(description_path.read_text(encoding="utf-8"))
        if description.get("schema") != "calvin_utils.fiber_values":
            raise ValueError(f"Unrecognized fiber descriptor schema: {description_path}")
        if description.get("schema_version", 1) not in {1, 2}:
            raise ValueError(f"Unsupported fiber descriptor version: {description_path}")
        if description.get("values_file") != values_path.name:
            raise ValueError(
                f"Descriptor {description_path} belongs to "
                f"{description.get('values_file')!r}, not {values_path.name!r}."
            )
        expected_digest = (description.get("values") or {}).get("sha256")
        if expected_digest:
            digest = hashlib.sha256()
            with values_path.open("rb") as values_file:
                for chunk in iter(lambda: values_file.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected_digest:
                raise ValueError(
                    f"Fiber values SHA-256 does not match {description_path}."
                )
        atlas = description.get("fiber_atlas") or {}
        candidates = []
        if atlas_override is not None:
            candidates.append(Path(atlas_override).expanduser().resolve())
        if atlas.get("relative_path"):
            candidates.append((description_path.parent / atlas["relative_path"]).resolve())
        if atlas.get("path"):
            candidates.append(Path(atlas["path"]).expanduser())
        for candidate in candidates:
            if candidate.is_file():
                expected_size = atlas.get("size_bytes")
                if expected_size is not None and candidate.stat().st_size != expected_size:
                    raise ValueError(
                        f"Fiber atlas size does not match {description_path}: {candidate}"
                    )
                return candidate
        raise FileNotFoundError(
            f"The atlas recorded in {description_path} could not be found: {candidates}"
        )

    @classmethod
    def _fibcell_streamlines(cls, value):
        streamlines = []

        def collect(item):
            array = np.asarray(item)
            if array.dtype == object:
                for child in array.flat:
                    collect(child)
            elif array.size:
                streamlines.append(cls._as_streamline(array))

        collect(value)
        return streamlines

    @classmethod
    def _fiber_matrix_streamlines(cls, value, idx=None):
        matrix = np.asarray(value, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"Expected a 2D Lead-DBS fibers matrix, got {matrix.shape}.")
        if matrix.shape[1] not in (3, 4) and matrix.shape[0] in (3, 4):
            matrix = matrix.T
        if matrix.shape[1] not in (3, 4):
            raise ValueError(f"Expected an N x 3 or N x 4 fibers matrix, got {matrix.shape}.")

        if idx is not None:
            lengths = np.asarray(idx, dtype=np.int64).reshape(-1)
            if np.any(lengths <= 0) or lengths.sum() != len(matrix):
                raise ValueError("Lead-DBS idx must contain positive lengths summing to the fibers rows.")
            stops = np.cumsum(lengths)
            starts = np.r_[0, stops[:-1]]
            return [matrix[start:stop, :3] for start, stop in zip(starts, stops)]

        if matrix.shape[1] == 3:
            return [matrix]

        fiber_ids = matrix[:, 3]
        starts = np.r_[0, np.flatnonzero(np.diff(fiber_ids)) + 1]
        stops = np.r_[starts[1:], len(matrix)]
        return [matrix[start:stop, :3] for start, stop in zip(starts, stops)]

    @staticmethod
    def _hdf5_mat_fields(path):
        import h5py

        with h5py.File(path, "r") as mat:
            fields = {}
            for name in ("fibers", "idx", "vals"):
                if name in mat:
                    fields[name] = np.asarray(mat[name])
            if "fibcell" in mat:
                cells = []
                for reference in np.asarray(mat["fibcell"]).flat:
                    cells.append(np.asarray(mat[reference]))
                fields["fibcell"] = cells
            return fields

    @classmethod
    def load_leaddbs_mat(cls, mat_path):
        """Load Lead-DBS FTR or discriminative-fiber MAT geometry."""
        mat_path = Path(mat_path).expanduser()
        if not mat_path.is_file():
            raise FileNotFoundError(mat_path)
        if mat_path.suffix.lower() != ".mat":
            raise ValueError(f"Expected a .mat input, got: {mat_path}")

        try:
            fields = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        except NotImplementedError:
            fields = cls._hdf5_mat_fields(mat_path)

        if "fibcell" in fields:
            streamlines = cls._fibcell_streamlines(fields["fibcell"])
        elif "fibers" in fields:
            streamlines = cls._fiber_matrix_streamlines(
                fields["fibers"], fields.get("idx")
            )
        else:
            raise ValueError(f"Lead-DBS MAT has neither 'fibcell' nor 'fibers': {mat_path}")

        if not streamlines:
            raise ValueError(f"Lead-DBS MAT contains no fibers: {mat_path}")

        lengths = np.asarray([len(fiber) for fiber in streamlines], dtype=np.int64)
        values = fields.get("vals")
        if values is not None:
            values = np.asarray(values, dtype=np.float32).reshape(-1)
            if len(values) != len(streamlines):
                raise ValueError(
                    f"Lead-DBS MAT has {len(values)} vals for {len(streamlines)} fibers."
                )
        return streamlines, lengths, values

    @classmethod
    def load_fib_npy(
        cls,
        fib_path,
        fiber_atlas_path=None,
        sign="both",
        min_abs_value=None,
        top_percent=None,
    ):
        """Load and filter a geometry-bearing or value-only fiber result.

        ``fib_path`` may be a new ``*.fib.json``/``*.fib.npy`` pair, a legacy
        ``*.fib.desc.json``/``*.fib.values.npy`` pair, a self-contained
        geometry-bearing ``*.fib.npy``, or a bare numeric ``*.fib.npy`` plus an
        explicit ``fiber_atlas_path``. An explicit atlas may point to a
        relocated copy recorded by a descriptor.

        Returns ``(streamlines, selected_values, n_input_fibers)`` after
        applying ``sign``, ``min_abs_value``, and ``top_percent``. Geometry and
        value counts must match before filtering.
        """
        fib_path = Path(fib_path).expanduser()
        if not fib_path.is_file():
            raise FileNotFoundError(fib_path)
        if fib_path.name.lower().endswith((".fib.json", ".fib.desc.json")):
            fib_path = cls.values_from_description(fib_path)
        if not fib_path.name.lower().endswith((".fib.npy", ".values.npy")):
            raise ValueError(
                "Expected a .fib.json, .fib.npy, legacy .fib.desc.json/"
                f".fib.values.npy, or .values.npy input, got: {fib_path}"
            )

        stored = np.load(fib_path, allow_pickle=True)
        if stored.dtype == object:
            fibers = [np.asarray(fiber, dtype=np.float32) for fiber in stored.tolist()]
            if not fibers or any(
                fiber.ndim != 2 or fiber.shape[1] != 4 for fiber in fibers
            ):
                raise ValueError(
                    "A geometry-bearing .fib.npy must contain N x 4 fibers."
                )
            streamlines = [cls._as_streamline(fiber) for fiber in fibers]
            values = np.asarray(
                [np.nanmedian(fiber[:, 3]) for fiber in fibers], dtype=np.float32
            )
        else:
            if stored.ndim != 1:
                raise ValueError(
                    f"A fiber value vector must be one-dimensional, got {stored.shape}."
                )
            description_path = cls.values_description_path(fib_path)
            if (
                fib_path.name.lower().endswith(".values.npy")
                or description_path.is_file()
            ):
                fiber_atlas_path = cls.atlas_from_values_description(
                    fib_path, atlas_override=fiber_atlas_path
                )
            elif fiber_atlas_path is None:
                raise ValueError(
                    f"Fiber values require the paired descriptor: {description_path}. "
                    "For a legacy bare vector, explicitly supply the canonical "
                    ".npz/.npy fiber atlas."
                )
            streamlines = [
                cls._as_streamline(fiber)
                for fiber in cls._load_fibers(Path(fiber_atlas_path).expanduser())
            ]
            values = np.asarray(stored, dtype=np.float32)

        if len(streamlines) != len(values):
            raise ValueError(
                f"Fiber/value length mismatch: the atlas has {len(streamlines)} "
                f"fibers but {fib_path} has {len(values)} values."
            )
        keep = cls._fiber_value_mask(values, sign, min_abs_value, top_percent)
        return (
            [fiber for fiber, selected in zip(streamlines, keep) if selected],
            values[keep],
            int(len(values)),
        )

    @staticmethod
    def _fiber_value_mask(values, sign, min_abs_value=None, top_percent=None):
        aliases = {"both": "both", "positive": "positive", "pos": "positive",
                   "negative": "negative", "neg": "negative"}
        normalized_sign = aliases.get(str(sign).strip().lower())
        if normalized_sign is None:
            raise ValueError("sign must be 'both', 'positive', or 'negative'.")

        keep = np.isfinite(values)
        if normalized_sign == "positive":
            keep &= values > 0
        elif normalized_sign == "negative":
            keep &= values < 0
        else:
            keep &= values != 0

        if min_abs_value is not None:
            min_abs_value = float(min_abs_value)
            if min_abs_value < 0:
                raise ValueError("min_abs_value must be nonnegative.")
            keep &= np.abs(values) >= min_abs_value

        if top_percent is not None:
            top_percent = float(top_percent)
            if not 0 < top_percent <= 100:
                raise ValueError("top_percent must be in (0, 100].")
            if np.any(keep):
                cutoff = np.nanpercentile(np.abs(values[keep]), 100 - top_percent)
                keep &= np.abs(values) >= cutoff

        if not np.any(keep):
            raise ValueError("No fibers survive the tract visualization filters.")
        return keep

    @staticmethod
    def _write_tck(streamlines, out_path):
        out_path = Path(out_path).expanduser()
        if out_path.suffix.lower() != ".tck":
            raise ValueError(f"TCK output must end in .tck, got: {out_path}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tractogram = Tractogram(streamlines=streamlines, affine_to_rasmm=np.eye(4))
        nib.streamlines.save(tractogram, str(out_path))
        return out_path

    @classmethod
    def convert_leaddbs_mat_to_tck(
        cls,
        mat_path,
        out_path,
        sign="both",
        min_abs_value=None,
        top_percent=None,
    ):
        """Write MAT streamlines to TCK and return values for later plotting."""
        streamlines, lengths, values = cls.load_leaddbs_mat(mat_path)
        n_input = len(streamlines)
        if values is not None:
            keep = cls._fiber_value_mask(
                values, sign, min_abs_value, top_percent
            )
            streamlines = [
                fiber for fiber, selected in zip(streamlines, keep) if selected
            ]
            lengths = lengths[keep]
            values = values[keep]
        out_path = cls._write_tck(streamlines, out_path)
        return {
            "tck_path": str(out_path),
            "idx": lengths,
            "vals": values,
            "n_fibers": len(streamlines),
            "n_input_fibers": n_input,
        }

    @classmethod
    def convert_fib_npy_to_tck(
        cls,
        fib_path,
        out_path,
        fiber_atlas_path=None,
        sign="both",
        min_abs_value=None,
        top_percent=None,
    ):
        """Filter a native fiber result and materialize it as TCK.

        Input and atlas-resolution rules are identical to ``load_fib_npy``.
        TCK stores geometry but not these scalar values, so the returned
        dictionary includes ``vals`` and per-streamline lengths for callers
        that need to color or annotate the temporary tractogram.
        """
        streamlines, values, n_input = cls.load_fib_npy(
            fib_path,
            fiber_atlas_path=fiber_atlas_path,
            sign=sign,
            min_abs_value=min_abs_value,
            top_percent=top_percent,
        )
        out_path = cls._write_tck(streamlines, out_path)
        lengths = np.asarray([len(fiber) for fiber in streamlines], dtype=np.int64)
        return {
            "tck_path": str(out_path),
            "idx": lengths,
            "vals": values,
            "n_fibers": len(streamlines),
            "n_input_fibers": n_input,
        }

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

    def convert_fiber_file_to_trk(
        self,
        fiber_file_path,
        out_path,
        fiber_atlas_path=None,
        sign="both",
        min_abs_value=None,
        top_percent=None,
    ):
        """Convert atlas geometry or a native result directly to TrackVis.

        New ``*.fib.npy``/``*.fib.json`` and legacy
        ``*.fib.values.npy``/``*.fib.desc.json`` inputs are resolved through
        their descriptor, then combined with the atlas only in memory. No
        intermediate geometry file is written. The selected statistic is
        attached as TrackVis per-streamline and per-point ``magnitude`` data.
        Legacy geometry files and bare atlas containers remain supported.
        """
        fiber_file_path = Path(fiber_file_path).expanduser()
        if fiber_file_path.name.lower().endswith(
            (".fib.npy", ".values.npy", ".fib.json", ".fib.desc.json")
        ):
            streamlines, values, _ = self.load_fib_npy(
                fiber_file_path,
                fiber_atlas_path=fiber_atlas_path,
                sign=sign,
                min_abs_value=min_abs_value,
                top_percent=top_percent,
            )
            fibers = [
                np.column_stack(
                    [fiber, np.full(len(fiber), value, dtype=np.float32)]
                )
                for fiber, value in zip(streamlines, values)
            ]
        else:
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
            lower_name = p.name.lower()
            if lower_name.endswith(".fib.values.npy"):
                stem = p.name[:-15]
            elif lower_name.endswith(".values.npy"):
                stem = p.name[:-11]
            elif lower_name.endswith(".fib.json"):
                stem = p.name[:-9]
            elif lower_name.endswith(".fib.desc.json"):
                stem = p.name[:-14]
            elif lower_name.endswith(".fib.npy"):
                stem = p.name[:-8]
            else:
                stem = p.name[:-4] if lower_name.endswith(".npy") else p.stem
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
        help="Input fiber files (.fib.json/.fib.npy, legacy .fib.desc.json/.fib.values.npy, .npz, .npy, .json)"
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
