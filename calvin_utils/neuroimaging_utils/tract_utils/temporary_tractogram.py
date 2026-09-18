from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from calvin_utils.neuroimaging_utils.tract_utils.fiber_converter import (
    FiberFormatConverter,
)


class TemporaryTractogram:
    """Materialize a MAT or native fiber result as a temporary TCK.

    Supported sources are Lead-DBS ``.mat``, self-contained ``*.fib.npy``, and
    lightweight ``*.fib.values.npy``/``*.fib.desc.json`` pairs. The descriptor
    can be passed directly and is used to locate and validate both the values
    and canonical geometry atlas automatically. ``fiber_atlas_path`` may point
    to a relocated copy of the described atlas and remains necessary for legacy
    numeric ``*.fib.npy``; it never makes the values descriptor optional.

    The values vector and descriptor are not temporary and are never modified.
    Only the derived TCK is placed in a temporary directory, which is removed
    on context exit or conversion failure. Selected scalar values remain
    available through ``vals`` and ``point_values()`` because TCK itself does
    not preserve per-fiber statistics.

    Use as a context manager so cleanup is deterministic::

        with TemporaryTractogram("map.fib.desc.json") as tractogram:
            render(tractogram.tck_path, data=tractogram.point_values())
    """

    def __init__(
        self,
        source_path,
        temp_root=None,
        fiber_atlas_path=None,
        sign="both",
        min_abs_value=None,
        top_percent=None,
    ):
        self.source_path = Path(source_path).expanduser()
        self.temp_root = (
            Path(temp_root).expanduser() if temp_root is not None else None
        )
        self.fiber_atlas_path = fiber_atlas_path
        self.sign = sign
        self.min_abs_value = min_abs_value
        self.top_percent = top_percent
        self.tck_path = None
        self.idx = None
        self.vals = None
        self.n_fibers = None
        self.n_input_fibers = None
        self._directory = None

    def create(self):
        """Create the temporary TCK and return ``self``.

        Descriptor and atlas integrity are validated by
        ``FiberFormatConverter`` before a values vector is materialized.
        Repeated calls during the same context return the existing result.
        """
        if self._directory is not None:
            return self

        self._directory = TemporaryDirectory(
            prefix="neuro_plotter_tracts_",
            dir=str(self.temp_root) if self.temp_root is not None else None,
        )
        try:
            name = self.source_path.name
            lower_name = name.lower()
            if lower_name.endswith(".fib.npy"):
                stem = name[:-8]
            elif lower_name.endswith(".fib.values.npy"):
                stem = name[:-15]
            elif lower_name.endswith(".values.npy"):
                stem = name[:-11]
            elif lower_name.endswith(".fib.desc.json"):
                stem = name[:-14]
            else:
                stem = self.source_path.stem
            out_path = Path(self._directory.name) / f"{stem}.tck"
            if lower_name.endswith(
                (".fib.npy", ".values.npy", ".fib.desc.json")
            ):
                result = FiberFormatConverter.convert_fib_npy_to_tck(
                    self.source_path,
                    out_path,
                    fiber_atlas_path=self.fiber_atlas_path,
                    sign=self.sign,
                    min_abs_value=self.min_abs_value,
                    top_percent=self.top_percent,
                )
            elif self.source_path.suffix.lower() == ".mat":
                result = FiberFormatConverter.convert_leaddbs_mat_to_tck(
                    self.source_path,
                    out_path,
                    sign=self.sign,
                    min_abs_value=self.min_abs_value,
                    top_percent=self.top_percent,
                )
            else:
                raise ValueError(
                    "TemporaryTractogram requires .fib.desc.json, "
                    ".fib.values.npy, legacy .fib.npy/.values.npy, or .mat, "
                    f"got: {self.source_path}"
                )
        except Exception:
            self.cleanup()
            raise

        self.tck_path = Path(result["tck_path"])
        self.idx = result["idx"]
        self.vals = result["vals"]
        self.n_fibers = result["n_fibers"]
        self.n_input_fibers = result["n_input_fibers"]
        return self

    def cleanup(self):
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None

    def point_values(self):
        """Repeat each selected fiber value once per streamline vertex."""
        if self.vals is None:
            return None
        return np.repeat(self.vals, self.idx)

    def __enter__(self):
        return self.create()

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()
        return False


__all__ = ["TemporaryTractogram"]
