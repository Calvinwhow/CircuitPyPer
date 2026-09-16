from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from calvin_utils.neuroimaging_utils.tract_utils.fiber_converter import (
    FiberFormatConverter,
)


class TemporaryTractogram:
    """Materialize a MAT or native fiber result as TCK within a context."""

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
        if self._directory is not None:
            return self

        self._directory = TemporaryDirectory(
            prefix="neuro_plotter_tracts_",
            dir=str(self.temp_root) if self.temp_root is not None else None,
        )
        try:
            name = self.source_path.name
            stem = name[:-8] if name.lower().endswith(".fib.npy") else self.source_path.stem
            out_path = Path(self._directory.name) / f"{stem}.tck"
            if name.lower().endswith(".fib.npy"):
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
                    f"TemporaryTractogram requires .fib.npy or .mat, got: {self.source_path}"
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
        if self.vals is None:
            return None
        return np.repeat(self.vals, self.idx)

    def __enter__(self):
        return self.create()

    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup()
        return False


__all__ = ["TemporaryTractogram"]
