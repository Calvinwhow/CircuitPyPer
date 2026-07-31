from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from scipy.io import loadmat, savemat
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

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


class _NullProgress:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def update(self, n=1):
        return None

    def set_postfix(self, *args, **kwargs):
        return None


@dataclass(frozen=True)
class MaskRule:
    """
    Selection rule applied to a structural connectome.

    rule:
        intersects_any   keep fibers touching any voxel in masks
        intersects_all   keep fibers touching every mask
        endpoint_in      keep fibers with either endpoint in any mask
        connects         keep fibers with one endpoint in each of two masks
        excludes         remove fibers touching any voxel in masks
    """

    rule: str
    masks: tuple[str, ...]
    min_vertices: int = 1

    @classmethod
    def from_dict(cls, value: dict) -> "MaskRule":
        masks = value.get("masks", value.get("mask"))
        if isinstance(masks, (str, Path)):
            masks = (str(masks),)
        return cls(
            rule=str(value["rule"]),
            masks=tuple(str(mask) for mask in masks),
            min_vertices=int(value.get("min_vertices", 1)),
        )


@dataclass(frozen=True)
class FiberSubset:
    fibers: list[np.ndarray]
    idx: np.ndarray
    global_fiber_ids: np.ndarray
    source_path: str
    rules: tuple[MaskRule, ...]
    selected_mask: np.ndarray


class ConnectomeSeed:
    """
    Derive a reusable fiber atlas from a larger structural connectome.

    This ports the core Lead-DBS ROI filtering idea used by
    `ea_filterfiber_roi.m`: test fiber vertices against a seed/ROI mask,
    select the corresponding fiber IDs, and emit a smaller connectome.

    Unlike Lead-DBS' helper, this preserves the original global fiber IDs
    after renumbering the selected atlas fibers locally.
    """

    def __init__(
        self,
        connectome_path: str | Path,
        seed_mask: str | Path | Iterable[str | Path] | None = None,
        *,
        intersect_mask: str | Path | Iterable[str | Path] | None = None,
        exclude_mask: str | Path | Iterable[str | Path] | None = None,
        out: str | Path | None = None,
        output_path: str | Path | None = None,
        method: str = "intersects_any",
        intersect_method: str = "intersects_all",
        min_vertices: int = 1,
        rule: MaskRule | dict | None = None,
        rules: Iterable[MaskRule | dict] | None = None,
        default_rule: str | None = None,
        combine: str = "all",
        max_fibers: int | None = 100_000,
        fiber_count: int | None = None,
        random_seed: int = 42,
        chunk_size: int = 1_000_000,
        fa_threshold: float = 0.03,
        angle_threshold: float = 60.0,
        step_size: float = 1.0,
        max_steps: int = 250,
        min_length: float = 10.0,
        show_progress: bool = True,
        save_mat: bool | None = None,
        auto_run: bool = True,
    ):
        """
        Parameters
        ----------
        connectome_path:
            Source structural connectome or reconstruction. Supported:
            DSI Studio .fib/.fib.gz, Lead-DBS .mat, .trk/.tck/.trx,
            object-array .npy, or .npz with a ``fibers`` key.
        seed_mask:
            One mask path or multiple mask paths. If ``rules`` is not supplied,
            this becomes one ``default_rule`` over these masks.
        intersect_mask:
            Optional mask path or mask paths that selected fibers must also
            intersect. Multiple masks default to requiring every mask.
        exclude_mask:
            Optional mask path or mask paths that selected fibers must avoid.
        out:
            Student-facing alias for ``output_path``.
        output_path:
            Optional output atlas path. ``run()`` saves automatically when set.
        method:
            Plain-language seed rule for normal use. Options include
            ``intersects_any``, ``intersects_all``, ``endpoint_in``,
            ``connects``, and ``excludes``.
        intersect_method:
            Rule used for ``intersect_mask``. Defaults to ``intersects_all`` so
            a list of target masks means the fiber must touch each target.
        min_vertices:
            Minimum number of fiber vertices that must touch a mask for
            intersection-based methods.
        rule:
            Advanced singular rule, e.g.
            ``{"rule": "intersects_any", "mask": "/path/to/mask.nii.gz"}``.
        rules:
            Explicit mask rules. Use this for multi-mask logic such as
            ``connects`` or include/exclude rule chains.
        default_rule:
            Deprecated alias for ``method``.
        combine:
            How to combine multiple rules: ``all`` or ``any``.
        max_fibers:
            Maximum number of selected atlas fibers to keep. Defaults to 100,000.
            If more fibers pass the seed rules, a reproducible random subset is
            sampled while preserving global fiber IDs.
        fiber_count:
            Number of streamlines to generate when ``connectome_path`` is a DSI
            Studio ``.fib`` reconstruction. Defaults to ``max_fibers``.
        random_seed:
            Seed used only when ``max_fibers`` requires downsampling.
        chunk_size:
            Number of fiber vertices tested per vectorized mask pass.
        fa_threshold:
            Minimum local anisotropy value used when tracking from ``.fib``.
        angle_threshold:
            Maximum turn angle in degrees for ``.fib`` tracking.
        step_size:
            Tracking step size in voxel units for ``.fib`` sources.
        max_steps:
            Maximum propagation steps in each direction for ``.fib`` tracking.
        min_length:
            Minimum streamline length in mm for generated ``.fib`` fibers.
        show_progress:
            If True, show tqdm progress bars during fiber generation and mask
            intersection scans.
        save_mat:
            Force Lead-DBS-like .mat output when saving. If None, inferred from
            ``output_path`` suffix.
        auto_run:
            If True, immediately derive/save the subset when a rule or seed mask
            is supplied. The created subset is available as ``self.subset``.
        """
        if out is not None and output_path is not None and str(out) != str(output_path):
            raise ValueError("Use either out or output_path, not both with different values.")
        if rule is not None and rules is not None:
            raise ValueError("Use either rule or rules, not both.")

        if default_rule is not None:
            method = default_rule

        output_path = output_path if output_path is not None else out
        rules = (rule,) if rule is not None else rules

        self.connectome_path = str(connectome_path)
        self.seed_mask = self._normalize_mask_paths(seed_mask)
        self.intersect_mask = self._normalize_mask_paths(intersect_mask)
        self.exclude_mask = self._normalize_mask_paths(exclude_mask)
        self.output_path = str(output_path) if output_path is not None else None
        self.method = method
        self.intersect_method = intersect_method
        self.min_vertices = int(min_vertices)
        self.rules = self._normalize_rules(
            rules=rules,
            seed_mask=self.seed_mask,
            intersect_mask=self.intersect_mask,
            exclude_mask=self.exclude_mask,
            method=method,
            intersect_method=intersect_method,
            min_vertices=self.min_vertices,
        )
        self.combine = combine
        self.max_fibers = max_fibers
        self.fiber_count = int(fiber_count) if fiber_count is not None else (
            int(max_fibers) if max_fibers is not None else 100_000
        )
        self.random_seed = int(random_seed)
        self.chunk_size = int(chunk_size)
        self.fa_threshold = float(fa_threshold)
        self.angle_threshold = float(angle_threshold)
        self.step_size = float(step_size)
        self.max_steps = int(max_steps)
        self.min_length = float(min_length)
        self.show_progress = bool(show_progress)
        self.save_mat = save_mat
        self._stream_connectome = self._can_stream_connectome(connectome_path) and bool(self.rules)
        self._n_fibers = None
        if self._stream_connectome:
            self.fibers = []
            self.global_fiber_ids = np.empty(0, dtype=np.int64)
            self.idx = np.empty(0, dtype=np.int64)
        else:
            self.fibers, self.global_fiber_ids = self._load_or_generate_connectome(connectome_path)
            self.idx = np.asarray([fiber.shape[0] for fiber in self.fibers], dtype=np.int64)
        self._flat_points = None
        self._flat_fiber_ids = None
        self.subset = None

        if auto_run and self.rules:
            self.subset = self.run()

    @staticmethod
    def _normalize_mask_paths(mask_paths: str | Path | Iterable[str | Path] | None) -> tuple[str, ...]:
        if mask_paths is None:
            return ()
        if isinstance(mask_paths, (str, Path)):
            return (str(mask_paths),)
        return tuple(str(mask) for mask in mask_paths)

    @staticmethod
    def _normalize_rules(
        *,
        rules: Iterable[MaskRule | dict] | None,
        seed_mask: tuple[str, ...],
        intersect_mask: tuple[str, ...],
        exclude_mask: tuple[str, ...],
        method: str,
        intersect_method: str,
        min_vertices: int,
    ) -> tuple[MaskRule, ...]:
        if rules is not None:
            return tuple(rule if isinstance(rule, MaskRule) else MaskRule.from_dict(rule) for rule in rules)
        mask_rules = []
        if seed_mask:
            mask_rules.append(MaskRule(rule=method, masks=seed_mask, min_vertices=min_vertices))
        if intersect_mask:
            mask_rules.append(MaskRule(rule=intersect_method, masks=intersect_mask, min_vertices=min_vertices))
        if exclude_mask:
            mask_rules.append(MaskRule(rule="excludes", masks=exclude_mask, min_vertices=min_vertices))
        return tuple(mask_rules)

    @property
    def n_fibers(self) -> int:
        if self._n_fibers is not None:
            return self._n_fibers
        return len(self.fibers)

    def _progress(self, iterable=None, **kwargs):
        if not self.show_progress or tqdm is None:
            return iterable if iterable is not None else _NullProgress()
        if iterable is None:
            return tqdm(**kwargs)
        return tqdm(iterable, **kwargs)

    def _load_or_generate_connectome(self, path: str | Path) -> tuple[list[np.ndarray], np.ndarray]:
        suffixes = self._suffixes(path)
        if suffixes[-1:] == [".fib"] or suffixes[-2:] == [".fib", ".gz"]:
            if not self.seed_mask:
                raise ValueError("A DSI Studio .fib source requires seed_mask to generate fibers.")
            return self._generate_from_fib(path)
        return self.load_connectome(path)

    @property
    def flat_points(self) -> np.ndarray:
        if self._flat_points is None:
            self._flat_points = np.vstack([fiber[:, :3] for fiber in self.fibers]).astype(np.float32, copy=False)
        return self._flat_points

    @property
    def flat_fiber_ids(self) -> np.ndarray:
        if self._flat_fiber_ids is None:
            self._flat_fiber_ids = np.repeat(np.arange(self.n_fibers, dtype=np.int64), self.idx)
        return self._flat_fiber_ids

    @staticmethod
    def _suffixes(path: str | Path) -> list[str]:
        return [suffix.lower() for suffix in Path(path).suffixes]

    @classmethod
    def _can_stream_connectome(cls, path: str | Path) -> bool:
        return cls._suffixes(path)[-1:] in ([".trk"], [".tck"], [".trx"])

    @staticmethod
    def _loadmat(path: str | Path, **kwargs):
        path = Path(path)
        if [suffix.lower() for suffix in path.suffixes][-2:] == [".fib", ".gz"]:
            with gzip.open(path, "rb") as f:
                return loadmat(f, squeeze_me=True, **kwargs)
        return loadmat(path, squeeze_me=True, **kwargs)

    @staticmethod
    def _fiber_length_mm(fiber: np.ndarray) -> float:
        if fiber.shape[0] < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(fiber[:, :3], axis=0), axis=1).sum())

    def _load_fib_fields(self, path: str | Path) -> dict:
        variable_names = [
            "dimension",
            "voxel_size",
            "trans",
            "odf_vertices",
            "fa0",
            "fa1",
            "fa2",
            "fa3",
            "fa4",
            "index0",
            "index1",
            "index2",
            "index3",
            "index4",
        ]
        mat = self._loadmat(path, variable_names=variable_names)
        dim = tuple(int(v) for v in np.asarray(mat["dimension"]).ravel())
        n_voxels = int(np.prod(dim))
        affine = np.asarray(mat["trans"], dtype=np.float32).T
        vertices = np.asarray(mat["odf_vertices"], dtype=np.float32)
        vertices /= np.linalg.norm(vertices, axis=0, keepdims=True) + 1e-8

        fa = []
        indices = []
        for i in range(5):
            fa_key = f"fa{i}"
            idx_key = f"index{i}"
            if fa_key not in mat or idx_key not in mat:
                continue
            fa.append(np.asarray(mat[fa_key], dtype=np.float32).reshape(-1, order="F"))
            indices.append(np.asarray(mat[idx_key], dtype=np.int64).reshape(-1, order="F"))

        if not fa or any(arr.size != n_voxels for arr in fa) or any(arr.size != n_voxels for arr in indices):
            raise ValueError(f"Unexpected DSI Studio .fib orientation field shape: {path}")

        return {
            "dim": dim,
            "affine": affine,
            "inv_affine": np.linalg.inv(affine),
            "vertices": vertices,
            "fa": fa,
            "indices": indices,
        }

    def _seed_points_in_fib_voxels(self, fib: dict) -> np.ndarray:
        seed_points = []
        for seed_mask in self.seed_mask:
            mask = self._load_mask(seed_mask)
            active = np.where(mask["active_flat"])[0]
            ijk = np.column_stack(np.unravel_index(active, mask["shape"]))
            world = nib.affines.apply_affine(mask["img"].affine, ijk)
            fib_vox = nib.affines.apply_affine(fib["inv_affine"], world)
            in_bounds = self._fib_in_bounds(fib_vox, fib["dim"])
            seed_points.append(fib_vox[in_bounds].astype(np.float32, copy=False))

        if not seed_points:
            return np.empty((0, 3), dtype=np.float32)
        points = np.vstack(seed_points)
        if points.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        return np.unique(np.round(points, decimals=3), axis=0).astype(np.float32, copy=False)

    @staticmethod
    def _fib_in_bounds(points: np.ndarray, dim: tuple[int, int, int]) -> np.ndarray:
        return (
            (points[:, 0] >= 0) & (points[:, 0] < dim[0]) &
            (points[:, 1] >= 0) & (points[:, 1] < dim[1]) &
            (points[:, 2] >= 0) & (points[:, 2] < dim[2])
        )

    @staticmethod
    def _fib_linear_index(voxel: np.ndarray, dim: tuple[int, int, int]) -> int:
        return int(voxel[0] + voxel[1] * dim[0] + voxel[2] * dim[0] * dim[1])

    def _directions_at(self, fib: dict, pos: np.ndarray) -> np.ndarray:
        voxel = np.rint(pos).astype(np.int64)
        dim = fib["dim"]
        if (
            voxel[0] < 0 or voxel[0] >= dim[0] or
            voxel[1] < 0 or voxel[1] >= dim[1] or
            voxel[2] < 0 or voxel[2] >= dim[2]
        ):
            return np.empty((0, 3), dtype=np.float32)

        lin = self._fib_linear_index(voxel, dim)
        dirs = []
        for fa, indices in zip(fib["fa"], fib["indices"]):
            if fa[lin] < self.fa_threshold:
                continue
            dirs.append(fib["vertices"][:, indices[lin]])
        if not dirs:
            return np.empty((0, 3), dtype=np.float32)
        return np.vstack(dirs).astype(np.float32, copy=False)

    def _initial_direction(self, fib: dict, pos: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
        dirs = self._directions_at(fib, pos)
        if dirs.shape[0] == 0:
            return None
        return dirs[int(rng.integers(0, dirs.shape[0]))]

    def _next_direction(self, fib: dict, pos: np.ndarray, previous: np.ndarray, cos_threshold: float) -> np.ndarray | None:
        dirs = self._directions_at(fib, pos)
        if dirs.shape[0] == 0:
            return None

        dots = dirs @ previous
        signed_dirs = dirs.copy()
        flip = dots < 0
        signed_dirs[flip] *= -1
        dots = np.abs(dots)
        best = int(np.argmax(dots))
        if dots[best] < cos_threshold:
            return None
        return signed_dirs[best]

    def _propagate_fib(self, fib: dict, start: np.ndarray, direction: np.ndarray, cos_threshold: float) -> list[np.ndarray]:
        points = [start.astype(np.float32, copy=True)]
        current = start.astype(np.float32, copy=True)
        direction = direction.astype(np.float32, copy=True)
        direction /= np.linalg.norm(direction) + 1e-8

        for _ in range(self.max_steps):
            next_pos = current + direction * self.step_size
            if not self._fib_in_bounds(next_pos[None, :], fib["dim"])[0]:
                break
            next_direction = self._next_direction(fib, next_pos, direction, cos_threshold)
            if next_direction is None:
                break
            points.append(next_pos.astype(np.float32, copy=True))
            current = next_pos
            direction = next_direction / (np.linalg.norm(next_direction) + 1e-8)

        return points

    def _track_fib_streamline(self, fib: dict, start: np.ndarray, rng: np.random.Generator) -> np.ndarray | None:
        direction = self._initial_direction(fib, start, rng)
        if direction is None:
            return None

        cos_threshold = float(np.cos(np.deg2rad(self.angle_threshold)))
        forward = self._propagate_fib(fib, start, direction, cos_threshold)
        backward = self._propagate_fib(fib, start, -direction, cos_threshold)
        voxel_points = list(reversed(backward[1:])) + forward
        if len(voxel_points) < 2:
            return None

        world = nib.affines.apply_affine(fib["affine"], np.vstack(voxel_points)).astype(np.float32, copy=False)
        if self._fiber_length_mm(world) < self.min_length:
            return None
        return world

    def _generate_from_fib(self, path: str | Path) -> tuple[list[np.ndarray], np.ndarray]:
        fib = self._load_fib_fields(path)
        seed_points = self._seed_points_in_fib_voxels(fib)
        if seed_points.shape[0] == 0:
            raise ValueError("seed_mask has no voxels inside the DSI Studio .fib grid.")

        rng = np.random.default_rng(self.random_seed)
        target = self.fiber_count
        fibers = []

        with self._progress(total=target, desc="Generating fibers", unit="fiber") as pbar:
            for attempt in range(1, self.fiber_count + 1):
                if len(fibers) >= target:
                    break
                base = seed_points[int(rng.integers(0, seed_points.shape[0]))]
                start = base + rng.uniform(-0.5, 0.5, size=3).astype(np.float32)
                streamline = self._track_fib_streamline(fib, start, rng)
                if streamline is not None:
                    fibers.append(streamline)
                    pbar.update(1)
                if attempt == 1 or attempt % 1000 == 0:
                    pbar.set_postfix(attempts=attempt, accepted=len(fibers))

        if not fibers:
            raise ValueError(
                "No fibers were generated from the .fib source. Try lowering fa_threshold, "
                "checking seed_mask space, or using a larger seed."
            )

        return fibers, np.arange(1, len(fibers) + 1, dtype=np.int64)

    @classmethod
    def load_connectome(cls, path: str | Path) -> tuple[list[np.ndarray], np.ndarray]:
        path = Path(path)
        suffixes = cls._suffixes(path)

        if suffixes[-1:] in ([".trk"], [".tck"], [".trx"]):
            streamlines = nib.streamlines.load(str(path)).tractogram.streamlines
            fibers = [np.asarray(streamline, dtype=np.float32)[:, :3] for streamline in streamlines]
            return fibers, np.arange(1, len(fibers) + 1, dtype=np.int64)

        if suffixes[-1:] == [".mat"]:
            return cls._load_mat_connectome(path)

        if suffixes[-1:] == [".npy"]:
            obj = np.load(path, allow_pickle=True)
            return cls._coerce_loaded_connectome(obj)

        if suffixes[-1:] == [".npz"]:
            obj = np.load(path, allow_pickle=True)
            if "fibers" not in obj:
                raise ValueError(f"NPZ connectome missing 'fibers' key: {path}")
            global_ids = np.asarray(obj["global_fiber_ids"], dtype=np.int64) if "global_fiber_ids" in obj else None
            idx = np.asarray(obj["idx"], dtype=np.int64).flatten() if "idx" in obj else None
            return cls._coerce_loaded_connectome(obj["fibers"], idx=idx, global_ids=global_ids)

        raise ValueError(f"Unsupported connectome format: {path}")

    @classmethod
    def _load_mat_connectome(cls, path: Path) -> tuple[list[np.ndarray], np.ndarray]:
        try:
            mat = loadmat(path, squeeze_me=True, struct_as_record=False)
            if "fibers" not in mat:
                raise ValueError(f"MAT connectome missing 'fibers': {path}")
            fibers = np.asarray(mat["fibers"])
            idx = np.asarray(mat["idx"]).flatten() if "idx" in mat else None
        except NotImplementedError:
            with h5py.File(path, "r") as f:
                if "fibers" not in f:
                    raise ValueError(f"HDF5 MAT connectome missing 'fibers': {path}")
                fibers = np.asarray(f["fibers"]).T
                idx = np.asarray(f["idx"]).flatten() if "idx" in f else None

        return cls._coerce_loaded_connectome(fibers, idx=idx)

    @staticmethod
    def _coerce_loaded_connectome(
        obj,
        idx: np.ndarray | None = None,
        global_ids: np.ndarray | None = None,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        arr = np.asarray(obj, dtype=object if getattr(obj, "dtype", None) == object else None)

        if arr.dtype == object:
            fibers = [np.asarray(fiber, dtype=np.float32)[:, :3] for fiber in arr.tolist()]
            ids = global_ids if global_ids is not None else np.arange(1, len(fibers) + 1, dtype=np.int64)
            return fibers, np.asarray(ids, dtype=np.int64)

        numeric = np.asarray(obj, dtype=np.float32)
        if numeric.ndim != 2:
            raise ValueError(f"Expected 2D fiber matrix or object-array fibers, got shape {numeric.shape}")

        if numeric.shape[1] < numeric.shape[0] and numeric.shape[0] in (3, 4):
            numeric = numeric.T

        if numeric.shape[1] >= 4:
            fiber_ids = numeric[:, 3].astype(np.int64)
            old_ids = np.unique(fiber_ids)
            fibers = [numeric[fiber_ids == old_id, :3].astype(np.float32, copy=False) for old_id in old_ids]
            ids = global_ids if global_ids is not None else old_ids
            return fibers, np.asarray(ids, dtype=np.int64)

        if idx is None:
            raise ValueError("Fiber matrix without a fourth ID column requires idx lengths.")

        lengths = np.asarray(idx, dtype=np.int64).flatten()
        starts = np.r_[0, np.cumsum(lengths[:-1])]
        stops = np.cumsum(lengths)
        fibers = [numeric[start:stop, :3].astype(np.float32, copy=False) for start, stop in zip(starts, stops)]
        ids = global_ids if global_ids is not None else np.arange(1, len(fibers) + 1, dtype=np.int64)
        return fibers, np.asarray(ids, dtype=np.int64)

    @staticmethod
    def _load_mask(mask_path: str | Path):
        img = nib.load(str(mask_path))
        data = img.get_fdata()
        active = np.isfinite(data) & (data != 0)
        if not np.any(active):
            raise ValueError(f"Mask has no nonzero voxels: {mask_path}")

        active_indices = np.column_stack(np.where(active))
        active_world = nib.affines.apply_affine(img.affine, active_indices)
        half_voxel_world_extent = 0.5 * np.sum(np.abs(img.affine[:3, :3]), axis=1)
        bbox_min = active_world.min(axis=0) - half_voxel_world_extent
        bbox_max = active_world.max(axis=0) + half_voxel_world_extent

        return {
            "path": str(mask_path),
            "img": img,
            "active_flat": active.reshape(-1),
            "shape": active.shape,
            "inv_affine": np.linalg.inv(img.affine),
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
        }

    def _point_membership(self, points: np.ndarray, mask) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        inside_bbox = np.all((points >= mask["bbox_min"]) & (points <= mask["bbox_max"]), axis=1)
        out = np.zeros(points.shape[0], dtype=bool)
        if not np.any(inside_bbox):
            return out

        candidate_idx = np.where(inside_bbox)[0]
        candidate_points = points[candidate_idx]
        vox = nib.affines.apply_affine(mask["inv_affine"], candidate_points)
        ijk = np.round(vox).astype(np.int64)

        shape = mask["shape"]
        in_bounds = (
            (ijk[:, 0] >= 0) & (ijk[:, 0] < shape[0]) &
            (ijk[:, 1] >= 0) & (ijk[:, 1] < shape[1]) &
            (ijk[:, 2] >= 0) & (ijk[:, 2] < shape[2])
        )
        if not np.any(in_bounds):
            return out

        valid_idx = candidate_idx[in_bounds]
        valid_ijk = ijk[in_bounds]
        lin = np.ravel_multi_index(
            (valid_ijk[:, 0], valid_ijk[:, 1], valid_ijk[:, 2]),
            dims=shape,
        )
        out[valid_idx] = mask["active_flat"][lin]
        return out

    def _fiber_hits_for_mask(self, mask_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        mask = self._load_mask(mask_path)
        counts = np.zeros(self.n_fibers, dtype=np.int64)
        points = self.flat_points
        fiber_ids = self.flat_fiber_ids
        starts = range(0, points.shape[0], self.chunk_size)
        total_chunks = (points.shape[0] + self.chunk_size - 1) // self.chunk_size
        desc = f"Scanning {Path(mask_path).name}"

        for start in self._progress(starts, total=total_chunks, desc=desc, unit="chunk", leave=False):
            stop = min(start + self.chunk_size, points.shape[0])
            member = self._point_membership(points[start:stop], mask)
            if not np.any(member):
                continue
            hit_fiber_ids = fiber_ids[start:stop][member]
            counts += np.bincount(hit_fiber_ids, minlength=self.n_fibers)

        return counts > 0, counts

    def _endpoint_hits_for_mask(self, mask_path: str | Path) -> np.ndarray:
        mask = self._load_mask(mask_path)
        first = np.vstack([fiber[0, :3] for fiber in self.fibers]).astype(np.float32)
        last = np.vstack([fiber[-1, :3] for fiber in self.fibers]).astype(np.float32)
        return self._point_membership(first, mask) | self._point_membership(last, mask)

    def _streamline_count_for_mask(self, fiber: np.ndarray, mask) -> int:
        if fiber.size == 0:
            return 0
        return int(np.count_nonzero(self._point_membership(fiber[:, :3], mask)))

    def _streamline_endpoint_hit_for_mask(self, fiber: np.ndarray, mask) -> bool:
        if fiber.shape[0] == 0:
            return False
        endpoints = fiber[[0, -1], :3].astype(np.float32, copy=False)
        return bool(np.any(self._point_membership(endpoints, mask)))

    def _evaluate_streamline_rule(self, fiber: np.ndarray, rule: MaskRule, mask_cache: dict[str, dict]) -> bool:
        if rule.rule == "intersects_any":
            return any(
                self._streamline_count_for_mask(fiber, mask_cache[mask_path]) >= rule.min_vertices
                for mask_path in rule.masks
            )

        if rule.rule == "intersects_all":
            return all(
                self._streamline_count_for_mask(fiber, mask_cache[mask_path]) >= rule.min_vertices
                for mask_path in rule.masks
            )

        if rule.rule == "endpoint_in":
            return any(
                self._streamline_endpoint_hit_for_mask(fiber, mask_cache[mask_path])
                for mask_path in rule.masks
            )

        if rule.rule == "connects":
            if len(rule.masks) != 2:
                raise ValueError("'connects' requires exactly two masks.")
            return (
                self._streamline_endpoint_hit_for_mask(fiber, mask_cache[rule.masks[0]]) and
                self._streamline_endpoint_hit_for_mask(fiber, mask_cache[rule.masks[1]])
            )

        if rule.rule == "excludes":
            return not any(
                self._streamline_count_for_mask(fiber, mask_cache[mask_path]) >= rule.min_vertices
                for mask_path in rule.masks
            )

        raise ValueError(f"Unsupported rule: {rule.rule}")

    def _streamline_selected(self, fiber: np.ndarray, rules: tuple[MaskRule, ...], mask_cache: dict[str, dict]) -> bool:
        rule_results = [self._evaluate_streamline_rule(fiber, rule, mask_cache) for rule in rules]
        if self.combine == "all":
            return all(rule_results)
        if self.combine == "any":
            return any(rule_results)
        raise ValueError("combine must be 'all' or 'any'.")

    def _select_streaming(self) -> FiberSubset:
        if not self.rules:
            raise ValueError("At least one rule or seed_mask is required.")

        mask_paths = []
        for rule in self.rules:
            mask_paths.extend(rule.masks)
        mask_cache = {mask_path: self._load_mask(mask_path) for mask_path in dict.fromkeys(mask_paths)}

        selected_fibers = []
        selected_global_ids = []
        n_seen = 0
        n_matched = 0
        rng = np.random.default_rng(self.random_seed)
        max_fibers = None if self.max_fibers is None else int(self.max_fibers)

        tractogram = nib.streamlines.load(str(self.connectome_path), lazy_load=True).tractogram
        streamlines = tractogram.streamlines
        for source_index, streamline in enumerate(
            self._progress(streamlines, desc="Scanning streamlines", unit="fiber"),
            start=1,
        ):
            n_seen = source_index
            fiber = np.asarray(streamline, dtype=np.float32)[:, :3]
            if not self._streamline_selected(fiber, self.rules, mask_cache):
                continue

            n_matched += 1
            if max_fibers is None or len(selected_fibers) < max_fibers:
                selected_fibers.append(fiber)
                selected_global_ids.append(source_index)
                continue

            replace_at = int(rng.integers(0, n_matched))
            if replace_at < max_fibers:
                selected_fibers[replace_at] = fiber
                selected_global_ids[replace_at] = source_index

        order = np.argsort(selected_global_ids)
        selected_fibers = [selected_fibers[i] for i in order]
        selected_global_ids = np.asarray([selected_global_ids[i] for i in order], dtype=np.int64)
        idx = np.asarray([fiber.shape[0] for fiber in selected_fibers], dtype=np.int64)
        selected_mask = np.zeros(n_seen, dtype=bool)
        if selected_global_ids.size:
            selected_mask[selected_global_ids - 1] = True

        self.fibers = selected_fibers
        self.global_fiber_ids = selected_global_ids
        self.idx = idx
        self._n_fibers = n_seen

        return FiberSubset(
            fibers=selected_fibers,
            idx=idx,
            global_fiber_ids=selected_global_ids,
            source_path=self.connectome_path,
            rules=self.rules,
            selected_mask=selected_mask,
        )

    def evaluate_rule(self, rule: MaskRule) -> np.ndarray:
        if rule.rule == "intersects_any":
            selected = np.zeros(self.n_fibers, dtype=bool)
            for mask_path in rule.masks:
                hits, counts = self._fiber_hits_for_mask(mask_path)
                selected |= hits & (counts >= rule.min_vertices)
            return selected

        if rule.rule == "intersects_all":
            selected = np.ones(self.n_fibers, dtype=bool)
            for mask_path in rule.masks:
                hits, counts = self._fiber_hits_for_mask(mask_path)
                selected &= hits & (counts >= rule.min_vertices)
            return selected

        if rule.rule == "endpoint_in":
            selected = np.zeros(self.n_fibers, dtype=bool)
            for mask_path in rule.masks:
                selected |= self._endpoint_hits_for_mask(mask_path)
            return selected

        if rule.rule == "connects":
            if len(rule.masks) != 2:
                raise ValueError("'connects' requires exactly two masks.")
            return self._endpoint_hits_for_mask(rule.masks[0]) & self._endpoint_hits_for_mask(rule.masks[1])

        if rule.rule == "excludes":
            excluded = np.zeros(self.n_fibers, dtype=bool)
            for mask_path in rule.masks:
                hits, counts = self._fiber_hits_for_mask(mask_path)
                excluded |= hits & (counts >= rule.min_vertices)
            return ~excluded

        raise ValueError(f"Unsupported rule: {rule.rule}")

    def select(
        self,
        rules: Iterable[MaskRule | dict] | None = None,
        combine: str | None = None,
        max_fibers: int | None = None,
        random_seed: int | None = None,
    ) -> FiberSubset:
        rules = self.rules if rules is None else tuple(
            rule if isinstance(rule, MaskRule) else MaskRule.from_dict(rule)
            for rule in rules
        )
        if not rules:
            raise ValueError("At least one rule or seed_mask is required.")

        combine = self.combine if combine is None else combine
        max_fibers = self.max_fibers if max_fibers is None else max_fibers
        random_seed = self.random_seed if random_seed is None else int(random_seed)

        rule_masks = [self.evaluate_rule(rule) for rule in rules]

        if combine == "all":
            selected_mask = np.logical_and.reduce(rule_masks)
        elif combine == "any":
            selected_mask = np.logical_or.reduce(rule_masks)
        else:
            raise ValueError("combine must be 'all' or 'any'.")

        selected_indices = np.where(selected_mask)[0]
        if max_fibers is not None and selected_indices.shape[0] > int(max_fibers):
            rng = np.random.default_rng(random_seed)
            selected_indices = np.sort(rng.choice(selected_indices, size=int(max_fibers), replace=False))
            sampled_mask = np.zeros_like(selected_mask)
            sampled_mask[selected_indices] = True
            selected_mask = sampled_mask

        fibers = [self.fibers[i].astype(np.float32, copy=False) for i in selected_indices]
        global_ids = self.global_fiber_ids[selected_indices]
        idx = np.asarray([fiber.shape[0] for fiber in fibers], dtype=np.int64)

        return FiberSubset(
            fibers=fibers,
            idx=idx,
            global_fiber_ids=global_ids,
            source_path=self.connectome_path,
            rules=rules,
            selected_mask=selected_mask,
        )

    def run(self) -> FiberSubset:
        subset = self._select_streaming() if self._stream_connectome else self.select()
        if self.output_path is not None:
            self.save_subset(
                subset,
                self.output_path,
                save_mat=bool(self.save_mat) if self.save_mat is not None else False,
            )
        return subset

    @staticmethod
    def _lead_matrix_from_subset(subset: FiberSubset) -> np.ndarray:
        chunks = []
        for local_id, fiber in enumerate(subset.fibers, start=1):
            local_col = np.full((fiber.shape[0], 1), local_id, dtype=np.float32)
            chunks.append(np.hstack([fiber[:, :3], local_col]))
        if not chunks:
            return np.empty((0, 4), dtype=np.float32)
        return np.vstack(chunks).astype(np.float32, copy=False)

    @classmethod
    def save_subset(cls, subset: FiberSubset, out_path: str | Path, save_mat: bool = False) -> str:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            "source_path": subset.source_path,
            "rules": [
                {
                    "rule": rule.rule,
                    "masks": list(rule.masks),
                    "min_vertices": rule.min_vertices,
                }
                for rule in subset.rules
            ],
            "n_selected_fibers": int(len(subset.fibers)),
            "n_source_fibers": int(subset.selected_mask.shape[0]),
        }

        if out_path.suffix.lower() == ".mat":
            save_mat = True

        if save_mat:
            mat_path = out_path if out_path.suffix.lower() == ".mat" else out_path.with_suffix(".mat")
            savemat(
                mat_path,
                {
                    "ea_fibformat": "1.1",
                    "fourindex": np.asarray(1),
                    "fibers": cls._lead_matrix_from_subset(subset),
                    "idx": subset.idx[:, None],
                    "vals": np.ones((subset.idx.shape[0], 1), dtype=np.float32),
                    "global_fiber_ids": subset.global_fiber_ids[:, None],
                    "source_connectome": subset.source_path,
                    "metadata_json": json.dumps(metadata),
                },
                do_compression=True,
            )
            return str(mat_path)

        np.savez_compressed(
            out_path,
            fibers=np.asarray(subset.fibers, dtype=object),
            idx=subset.idx,
            global_fiber_ids=subset.global_fiber_ids,
            selected_mask=subset.selected_mask,
            metadata_json=json.dumps(metadata),
        )
        return str(out_path)


def _parse_rules(rule_args: list[str]) -> list[MaskRule]:
    rules = []
    for raw in rule_args:
        obj = json.loads(raw)
        rules.append(MaskRule.from_dict(obj))
    return rules


def main():
    parser = argparse.ArgumentParser(description="Derive a seed/ROI-filtered fiber atlas from a structural connectome.")
    parser.add_argument("--connectome", required=True, help="Input connectome/reconstruction (.fib, .mat, .trk, .tck, .npy, .npz).")
    parser.add_argument("--out", required=True, help="Output .npz or .mat path.")
    parser.add_argument("--seed-mask", action="append", help="Seed/ROI NIfTI mask. May be supplied multiple times.")
    parser.add_argument("--intersect-mask", action="append", help="Target NIfTI mask that fibers must also intersect. May be supplied multiple times.")
    parser.add_argument("--exclude-mask", action="append", help="NIfTI mask that selected fibers must avoid. May be supplied multiple times.")
    parser.add_argument("--method", default="intersects_any", help="Seed method: intersects_any, intersects_all, endpoint_in, connects, excludes.")
    parser.add_argument("--intersect-method", default="intersects_all", help="Intersection rule for --intersect-mask. Default: intersects_all.")
    parser.add_argument("--min-vertices", type=int, default=1, help="Minimum touching vertices required for intersection methods.")
    parser.add_argument("--combine", default="all", choices=["all", "any"], help="How to combine multiple rules.")
    parser.add_argument("--max-fibers", type=int, default=100_000, help="Maximum selected fibers to keep. Default: 100000.")
    parser.add_argument("--fiber-count", type=int, help="Number of fibers to generate from a DSI Studio .fib source. Defaults to --max-fibers.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed used if selected fibers exceed --max-fibers.")
    parser.add_argument("--chunk-size", type=int, default=1_000_000, help="Number of fiber vertices processed per chunk.")
    parser.add_argument("--fa-threshold", type=float, default=0.03, help="Minimum anisotropy for .fib tracking.")
    parser.add_argument("--angle-threshold", type=float, default=60.0, help="Maximum turn angle in degrees for .fib tracking.")
    parser.add_argument("--step-size", type=float, default=1.0, help="Tracking step size in voxel units for .fib sources.")
    parser.add_argument("--max-steps", type=int, default=250, help="Maximum tracking steps in each direction for .fib sources.")
    parser.add_argument("--min-length", type=float, default=10.0, help="Minimum generated fiber length in mm for .fib sources.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument(
        "--rule",
        action="append",
        help='JSON rule, e.g. {"rule":"intersects_any","mask":"cerebellum.nii.gz"}',
    )
    parser.add_argument("--mat", action="store_true", help="Save Lead-DBS-like .mat instead of .npz.")
    args = parser.parse_args()

    if not args.rule and not args.seed_mask and not args.intersect_mask:
        parser.error("Provide --seed-mask, --intersect-mask, or at least one --rule.")

    seeder = ConnectomeSeed(
        connectome_path=args.connectome,
        seed_mask=args.seed_mask,
        intersect_mask=args.intersect_mask,
        exclude_mask=args.exclude_mask,
        output_path=args.out,
        rules=_parse_rules(args.rule) if args.rule else None,
        method=args.method,
        intersect_method=args.intersect_method,
        min_vertices=args.min_vertices,
        combine=args.combine,
        max_fibers=args.max_fibers,
        fiber_count=args.fiber_count,
        random_seed=args.random_seed,
        chunk_size=args.chunk_size,
        fa_threshold=args.fa_threshold,
        angle_threshold=args.angle_threshold,
        step_size=args.step_size,
        max_steps=args.max_steps,
        min_length=args.min_length,
        show_progress=not args.no_progress,
        save_mat=args.mat,
        auto_run=False,
    )
    subset = seeder.run()
    out_path = args.out
    print(f"Selected {len(subset.fibers)} of {seeder.n_fibers} fibers.")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
