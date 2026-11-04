#!/usr/bin/env python3
"""Rewrite a precomputed connectome's T and Combo chunks as row-major float8 e4m3fn bytes (uint8 .npy).

T chunks (float32, column-major in GSP1000, row-major in yeo1000_dil) are stored row-major so FunctionalConnectivity
reads only the rows a seed touches. Combo chunks (float32 BOLD cross-products, up to ~1e7) are stored as the pooled
correlation Combo_ik / (norm_i * norm_k) scaled by COMBO_FP8_SCALE. Combo is symmetric, so a corrupted stretch of a
row (non-finite values or |correlation| > 1) is rebuilt from the mirror entries Combo[k, i] in other rows. Remaining
NaN entries are stored as 0. Counts go to conversion_log.csv. Each source chunk is read once, sequentially;
finished chunks are skipped on rerun.
"""

import csv
import fcntl
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path: sys.path.insert(0, str(PROJECT_DIR))

from calvin_utils.neuroimaging_utils.fmri_utils.compute_connectivity import FunctionalConnectivity
from calvin_utils.neuroimaging_utils.fmri_utils.fp8 import COMBO_FP8_SCALE, E4M3_MAX, decode_e4m3, encode_e4m3


# =============================================================================
# CONFIG
# =============================================================================

# Folder holding float32 T/ and Combo/; T_fp8/ and Combo_fp8/ are written beside them.
CONNECTOME_ROOT = Path("/Volumes/HowExp2/resources/connectome_rs_fMRI/GSP1000")
NORM_MASK_PATH = CONNECTOME_ROOT / "GSP1000_norm_weighted_mask.nii.gz"  # GSP1000: GSP1000_norm_weighted_mask.nii.gz
T_PATTERN = "{chunk}_T.npy"
COMBO_PATTERN = "{chunk}_Combo.npy"


def _read_block(file, shape):
    block = np.fromfile(file, dtype=np.float32, count=int(np.prod(shape)))
    if block.size != np.prod(shape): raise IOError(f"Short read: expected {int(np.prod(shape))} values, got {block.size}.")
    return block.reshape(shape)


def _check_encoded(expected, decoded, label):
    """Every value must decode to within e4m3's rounding bound (2**-4 relative, 2**-10 absolute near zero)."""
    target = np.clip(np.asarray(expected, dtype=np.float64), -E4M3_MAX, E4M3_MAX)
    bound = np.maximum(np.abs(target) * 2.0**-4, 2.0**-10) * (1 + 1e-5)
    if np.any(np.abs(np.asarray(decoded, dtype=np.float64) - target) > bound): raise RuntimeError(f"Encoding check failed at {label}.")


def _flush_to_disk(file):
    """fsync, plus macOS F_FULLFSYNC: plain fsync there does not make the drive write its cache, so USB disks can lose blocks."""
    file.flush()
    os.fsync(file.fileno())
    if hasattr(fcntl, "F_FULLFSYNC"): fcntl.fcntl(file.fileno(), fcntl.F_FULLFSYNC)


def _write_atomic(codes, out_path):
    partial_path = out_path.with_name(f"{out_path.name}.partial")
    with open(partial_path, "wb") as file:
        np.save(file, codes)
        _flush_to_disk(file)
    os.replace(partial_path, out_path)


def _is_valid_chunk(path):
    """Return True when path is a readable row-major uint8 .npy (catches files whose header block was lost)."""
    try:
        chunk = np.load(path, mmap_mode="r", allow_pickle=False)
    except (ValueError, OSError):
        return False
    return chunk.dtype == np.uint8 and chunk.ndim == 2 and chunk.flags.c_contiguous


def _check_written(out_path, codes, seed=1):
    written = np.load(out_path, mmap_mode="r")
    if written.dtype != np.uint8 or written.shape != codes.shape or not written.flags.c_contiguous: raise RuntimeError(f"{out_path} was not written as a row-major uint8 array.")
    row = int(np.random.default_rng(seed).integers(codes.shape[0]))
    if not np.array_equal(written[row], codes[row]): raise RuntimeError(f"{out_path} row {row} does not match the encoded data.")


def make_combo_repair(rows, chunk_labels, norms, combo_dir, pattern, max_correlation=1.0001, margin=4096):
    """Return repair(start, block) for one Combo chunk whose rows are the voxels `rows`.

    Rows containing impossible values are rebuilt from mirror entries over the corrupted column span (plus a margin),
    replacing every value that disagrees with its mirror. Returns the number of values replaced.
    """
    local_index = np.empty(chunk_labels.size, dtype=np.int64)
    for chunk in np.unique(chunk_labels): local_index[chunk_labels == chunk] = np.arange(np.count_nonzero(chunk_labels == chunk))

    def mirror_values(voxel, columns):
        values = np.empty(columns.size, dtype=np.float32)
        for chunk in np.unique(chunk_labels[columns]):
            in_chunk = chunk_labels[columns] == chunk
            values[in_chunk] = np.load(combo_dir / pattern.format(chunk=int(chunk)), mmap_mode="r")[local_index[columns[in_chunk]], voxel]
        values[columns == voxel] = norms[voxel] ** 2
        return values

    def repair(start, block):
        voxels = rows[start : start + block.shape[0]]
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"): correlation = block / (norms[voxels][:, None] * norms[None, :])
        bad = ~np.isfinite(correlation) | (np.abs(correlation) > max_correlation)
        replaced = 0
        for row in np.flatnonzero(bad.any(axis=1)):
            columns = np.flatnonzero(bad[row])
            span = np.arange(max(0, columns[0] - margin), min(block.shape[1], columns[-1] + 1 + margin))
            mirror = mirror_values(voxels[row], span)
            if np.any(~np.isfinite(mirror)) or np.any(np.abs(mirror / (norms[voxels[row]] * norms[span])) > max_correlation): raise RuntimeError(f"Voxel {voxels[row]}: its Combo row and the mirror entries are both corrupted.")
            changed = ~np.isclose(block[row, span], mirror, rtol=1e-5, atol=0)
            block[row, span[changed]] = mirror[changed]
            replaced += int(np.count_nonzero(changed))
            tqdm.write(f"Voxel {voxels[row]}: rebuilt {int(np.count_nonzero(changed))} corrupted Combo values (columns {span[0]}-{span[-1]}) from mirror entries.")
        return replaced

    return repair


def convert_chunk(source_path, out_path, row_scale=None, column_scale=None, repair=None, block_bytes=2**27, n_checks=32, seed=0):
    """Encode source * row_scale[:, None] * column_scale[None, :] as row-major e4m3 bytes.

    Column-major sources are transposed on the way. repair(start_row, block), for row-major sources, fixes a block in
    place and returns how many values it replaced. Returns (max |encoded value|, n saturated, n NaN set to 0, n repaired, seconds).
    """
    started = time.time()
    source = np.load(source_path, mmap_mode="r")
    if source.dtype != np.float32 or source.ndim != 2: raise ValueError(f"{source_path} is not a 2-D float32 array.")
    n_rows, n_columns = source.shape
    column_major = not source.flags.c_contiguous
    if column_major and repair is not None: raise ValueError("repair is only supported for row-major sources.")
    row_scale = np.ones(n_rows, dtype=np.float32) if row_scale is None else np.asarray(row_scale, dtype=np.float32)
    column_scale = np.ones(n_columns, dtype=np.float32) if column_scale is None else np.asarray(column_scale, dtype=np.float32)
    # Blocks follow the on-disk order: columns of a column-major source, rows of a row-major one.
    n_units, unit_size = (n_columns, n_rows) if column_major else (n_rows, n_columns)
    unit_scale, inner_scale = (column_scale, row_scale) if column_major else (row_scale, column_scale)
    units_per_block = max(1, block_bytes // (unit_size * 4))
    codes = np.empty((n_rows, n_columns), dtype=np.uint8)
    max_abs = 0.0
    n_saturated = 0
    n_nan = 0
    n_repaired = 0
    # The next block is read in a background thread while the current one is encoded.
    with open(source_path, "rb") as file, ThreadPoolExecutor(max_workers=1) as pool:
        file.seek(source.offset)
        pending = pool.submit(_read_block, file, (min(units_per_block, n_units), unit_size))
        for start in range(0, n_units, units_per_block):
            stop = min(start + units_per_block, n_units)
            block = pending.result()
            if stop < n_units: pending = pool.submit(_read_block, file, (min(units_per_block, n_units - stop), unit_size))
            if repair is not None: n_repaired += repair(start, block)
            missing = np.isnan(block)
            n_nan += int(np.count_nonzero(missing))
            scaled = np.where(missing, np.float32(0), block) * unit_scale[start:stop, None] * inner_scale[None, :]
            magnitude = np.abs(scaled)
            max_abs = max(max_abs, float(magnitude.max()))
            n_saturated += int(np.count_nonzero(magnitude > E4M3_MAX))
            encoded = encode_e4m3(scaled)
            if column_major: codes[:, start:stop] = encoded.T
            else: codes[start:stop] = encoded
    # Random on-disk units are contiguous, so they are cheap to re-read for the check.
    # Repaired values are not on disk, so the re-read check is skipped for chunks that needed a repair.
    for unit in np.random.default_rng(seed).choice(n_units, size=0 if n_repaired else min(n_checks, n_units), replace=False):
        original = np.asarray(source[:, unit] if column_major else source[unit], dtype=np.float64)
        decoded = decode_e4m3(codes[:, unit] if column_major else codes[unit])
        _check_encoded(np.nan_to_num(original, nan=0.0) * unit_scale[unit] * inner_scale, decoded, f"{'column' if column_major else 'row'} {unit}")
    _write_atomic(codes, out_path)
    _check_written(out_path, codes)
    return max_abs, n_saturated, n_nan, n_repaired, time.time() - started


def convert_directory(source_dir, out_dir, pattern, encoding, convert):
    """Convert every chunk in source_dir with convert(chunk, source_path, out_path), logging to conversion_log.csv."""
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = sorted(int(path.name.split("_")[0]) for path in source_dir.glob(pattern.format(chunk="[0-9]*")))
    if not chunks: raise FileNotFoundError(f"No chunks matching {pattern} in {source_dir}.")
    log_path = out_dir / "conversion_log.csv"
    write_header = not log_path.is_file()
    with log_path.open("a", newline="") as file:
        writer = csv.writer(file)
        if write_header: writer.writerow(["chunk", "source", "encoding", "max_abs_encoded", "n_saturated", "n_nan_set_to_0", "n_repaired", "seconds"])
        for chunk in tqdm(chunks, desc=f"Converting {source_dir.parent.name}/{source_dir.name} chunks"):
            source_path = source_dir / pattern.format(chunk=chunk)
            out_path = out_dir / pattern.format(chunk=chunk)
            if out_path.is_file():
                if _is_valid_chunk(out_path): continue
                tqdm.write(f"Chunk {chunk}: {out_path.name} is not a valid .npy file; rebuilding it.")
            max_abs, n_saturated, n_nan, n_repaired, seconds = convert(chunk, source_path, out_path)
            writer.writerow([chunk, source_path, encoding, f"{max_abs:.3f}", n_saturated, n_nan, n_repaired, f"{seconds:.1f}"])
            file.flush()
            if n_saturated: tqdm.write(f"Chunk {chunk}: {n_saturated} encoded values above {E4M3_MAX} were saturated to {E4M3_MAX}.")
    print(f"Converted chunks written to: {out_dir}")


def main():
    t_dir = CONNECTOME_ROOT / "T"
    convert_directory(t_dir, CONNECTOME_ROOT / "T_fp8", T_PATTERN, "float8_e4m3fn bytes (uint8), row-major", lambda chunk, source_path, out_path: convert_chunk(source_path, out_path))
    # Combo encoding needs each voxel's BOLD norm, which equals sqrt of Combo's diagonal.
    mapper = FunctionalConnectivity(connectome_dir=t_dir, chunk_pattern=T_PATTERN, norm_mask_path=NORM_MASK_PATH)
    chunk_rows = dict(mapper.chunk_rows)
    norms = mapper.voxel_norm
    encoding = f"Combo_ik / (norm_i * norm_k) * {COMBO_FP8_SCALE:g} as float8_e4m3fn bytes (uint8), row-major"
    combo_dir = CONNECTOME_ROOT / "Combo"
    convert_directory(combo_dir, CONNECTOME_ROOT / "Combo_fp8", COMBO_PATTERN, encoding, lambda chunk, source_path, out_path: convert_chunk(source_path, out_path, row_scale=COMBO_FP8_SCALE / norms[chunk_rows[chunk]], column_scale=1 / norms, repair=make_combo_repair(chunk_rows[chunk], mapper.chunk_labels, norms, combo_dir, COMBO_PATTERN)))


if __name__ == "__main__":
    main()
