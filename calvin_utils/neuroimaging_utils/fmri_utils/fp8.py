"""Store float32 connectome values as float8 e4m3fn bytes (uint8) and decode them back to float32.

e4m3fn: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa bits, no infinities, NaN at 0x7F/0xFF.
Largest finite value is 448, smallest normal is 2**-6, subnormal step is 2**-9.
Rounding is to nearest, ties to even, so per-value error is at most 2**-4 of |x| (2**-10 for |x| < 2**-6).
"""

import numpy as np

E4M3_MAX = 448.0
# Combo chunks are stored as Combo_ik / (norm_i * norm_k) * COMBO_FP8_SCALE: a pooled correlation (|value| <= 1) scaled into e4m3's range.
COMBO_FP8_SCALE = 256.0


def _decode_table():
    codes = np.arange(256, dtype=np.int64)
    sign = np.where(codes & 0x80, -1.0, 1.0)
    exponent = (codes >> 3) & 0xF
    mantissa = codes & 0x7
    values = np.where(exponent == 0, sign * mantissa * 2.0**-9, sign * (1 + mantissa / 8) * 2.0 ** (exponent - 7))
    values[[0x7F, 0xFF]] = np.nan
    return values.astype(np.float32)


_DECODE_TABLE = _decode_table()


def decode_e4m3(codes):
    """Return float32 values for an array of e4m3fn bytes."""
    return np.take(_DECODE_TABLE, codes)


def encode_e4m3(values):
    """Round float32 values to e4m3fn bytes, saturating magnitudes above 448 to +/-448."""
    values = np.asarray(values, dtype=np.float32)
    if np.isnan(values).any(): raise ValueError("NaN values cannot be encoded as e4m3fn connectome data.")
    magnitude = np.minimum(np.abs(values), np.float32(E4M3_MAX))
    bits = magnitude.view(np.uint32)
    rounded = bits + np.uint32(0x7FFFF) + ((bits >> np.uint32(20)) & np.uint32(1))
    normal = ((rounded >> np.uint32(23)).astype(np.int32) - 120) * 8 + ((rounded >> np.uint32(20)) & np.uint32(7)).astype(np.int32)
    subnormal = np.rint(magnitude * np.float32(512)).astype(np.int32)
    codes = np.where(magnitude < np.float32(2.0**-6), subnormal, normal).astype(np.uint8)
    codes[np.signbit(values)] |= np.uint8(0x80)
    return codes
