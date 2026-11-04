import numpy as np
import pytest

from calvin_utils.neuroimaging_utils.fmri_utils.fp8 import E4M3_MAX, decode_e4m3, encode_e4m3


def test_every_finite_code_round_trips():
    codes = np.array([code for code in range(256) if code not in (0x7F, 0xFF)], dtype=np.uint8)
    assert np.array_equal(encode_e4m3(decode_e4m3(codes)), codes)


def test_error_is_within_half_step_and_saturates():
    values = np.concatenate([np.random.default_rng(0).normal(0, 30, 100000), [1e-4, -3e-3, 447.9, 500.0, -1e6]]).astype(np.float32)
    decoded = decode_e4m3(encode_e4m3(values)).astype(np.float64)
    target = np.clip(values.astype(np.float64), -E4M3_MAX, E4M3_MAX)
    assert np.all(np.abs(decoded - target) <= np.maximum(np.abs(target) * 2.0**-4, 2.0**-10))


def test_nan_is_rejected():
    with pytest.raises(ValueError): encode_e4m3(np.array([1.0, np.nan], dtype=np.float32))
