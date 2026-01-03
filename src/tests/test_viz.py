"""Tests for visualization functions."""

import numpy as np

from dtm_differ.viz import (
    _finite_1d,
    _masked,
    _robust_limits,
)


def test_masked_regular_array():
    """Test _masked function with regular array."""
    data = np.array([1.0, 2.0, 3.0, np.nan])
    mask = np.array([False, False, True, False])

    result = _masked(data, mask)

    assert hasattr(result, "mask")
    assert np.ma.isMaskedArray(result)
    assert bool(result.mask[2]) is True  # Manually masked
    assert bool(result.mask[3]) is False  # NaN not automatically masked by this function


def test_masked_masked_array():
    """Test _masked function with already masked array."""
    data = np.ma.array([1.0, 2.0, 3.0], mask=[False, True, False])
    mask = np.array([True, False, False])

    result = _masked(data, mask)

    assert np.ma.isMaskedArray(result)
    # Combined mask: first element should be masked (from new mask)
    assert bool(result.mask[0]) is True
    # Second element should be masked (from original mask)
    assert bool(result.mask[1]) is True
    assert bool(result.mask[2]) is False


def test_finite_1d_regular_array():
    """Test _finite_1d with regular array."""
    data = np.array([1.0, 2.0, np.nan, np.inf, -np.inf, 3.0])

    result = _finite_1d(data)

    assert len(result) == 3  # Only 1.0, 2.0, 3.0 are finite
    assert np.all(np.isfinite(result))
    assert 1.0 in result
    assert 2.0 in result
    assert 3.0 in result


def test_finite_1d_masked_array():
    """Test _finite_1d with masked array."""
    data = np.ma.array([1.0, 2.0, 3.0], mask=[False, True, False])

    result = _finite_1d(data)

    # Should only return unmasked finite values
    assert len(result) == 2
    assert 1.0 in result
    assert 3.0 in result
    assert 2.0 not in result


def test_finite_1d_empty():
    """Test _finite_1d with all invalid data."""
    data = np.array([np.nan, np.inf, -np.inf])

    result = _finite_1d(data)

    assert len(result) == 0


def test_robust_limits_normal():
    """Test _robust_limits with normal data."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])  # One outlier

    vmin, vmax = _robust_limits(data, pct=98.0)

    assert vmin == 0.0
    assert vmax > 5.0  # Should be above the normal values
    assert vmax < 100.0  # Should exclude the outlier at 98th percentile


def test_robust_limits_empty():
    """Test _robust_limits with empty data."""
    data = np.array([])

    vmin, vmax = _robust_limits(data)

    assert vmin == 0.0
    assert vmax == 1.0  # Default fallback


def test_robust_limits_all_negative():
    """Test _robust_limits with all negative values."""
    data = np.array([-5.0, -4.0, -3.0, -2.0, -1.0])

    vmin, vmax = _robust_limits(data)

    assert vmin == 0.0
    # vmax should be positive (using max if percentile <= 0)
    assert vmax >= 0.0
