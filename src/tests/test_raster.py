import numpy as np
import pytest

from dtm_differ.raster import (
    generate_change_direction_raster,
    generate_change_magnitude_raster,
    generate_elevation_change_raster,
    generate_ranked_movement_raster,
    generate_slope_degrees_raster,
)

# -----------------------------------------------------------------------------
# Change direction
# -----------------------------------------------------------------------------


def test_change_direction_basic(create_test_dem):
    data = np.array([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]], dtype=np.float32)
    diff = create_test_dem(data) - create_test_dem(np.zeros_like(data))

    direction = generate_change_direction_raster(diff)

    assert direction[0, 0] == -1
    assert direction[0, 1] == 0
    assert direction[0, 2] == 1
    assert direction.dtype == np.int8


def test_change_direction_tiny_values_not_rounded(create_test_dem):
    # Regression: early version used atol in comparison, eating small changes
    tiny = np.array([[1e-7, -1e-7]], dtype=np.float32)
    diff = create_test_dem(tiny)

    direction = generate_change_direction_raster(diff)

    assert direction[0, 0] == 1, "tiny positive should still be +1"
    assert direction[0, 1] == -1, "tiny negative should still be -1"


# -----------------------------------------------------------------------------
# Change magnitude
# -----------------------------------------------------------------------------


def test_change_magnitude_returns_absolute(create_test_dem):
    data = np.array([[-2.0, 0.0, 2.0]], dtype=np.float32)
    diff = create_test_dem(data)

    mag = generate_change_magnitude_raster(diff)

    np.testing.assert_allclose(mag, [[2.0, 0.0, 2.0]])


def test_change_magnitude_nodata_becomes_nan(create_test_dem):
    data = np.array([[1.0, -9999.0]], dtype=np.float32)
    diff = create_test_dem(data, nodata=-9999.0)

    mag = generate_change_magnitude_raster(diff)

    assert mag[0, 0] == 1.0
    assert np.isnan(mag[0, 1])


# -----------------------------------------------------------------------------
# Elevation change
# -----------------------------------------------------------------------------


def test_elevation_change_preserves_sign(create_test_dem):
    diff = create_test_dem(np.array([[-5.0, 5.0]], dtype=np.float32))
    result = generate_elevation_change_raster(diff)
    assert result[0, 0] == -5.0 and result[0, 1] == 5.0


def test_elevation_change_returns_plain_array(create_test_dem):
    # We want plain arrays with NaN, not numpy masked arrays
    diff = create_test_dem(np.array([[1.0]], dtype=np.float32))
    result = generate_elevation_change_raster(diff)
    assert not isinstance(result, np.ma.MaskedArray)


# -----------------------------------------------------------------------------
# Movement ranking
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected_rank",
    [
        (0.5, 0),  # below green
        (1.0, 1),  # exactly at green -> included
        (2.9, 1),  # just under amber
        (3.0, 2),  # exactly at amber -> included
        (5.9, 2),  # just under red
        (6.0, 3),  # exactly at red -> included
        (100.0, 3),
    ],
)
def test_movement_rank_thresholds(value, expected_rank):
    mag = np.array([[value]], dtype=np.float32)
    rank = generate_ranked_movement_raster(mag, t_green=1.0, t_amber=3.0, t_red=6.0)
    assert rank[0, 0] == expected_rank


def test_movement_rank_invalid_threshold_order():
    with pytest.raises(ValueError, match="Thresholds must satisfy"):
        generate_ranked_movement_raster(
            np.array([[1.0]]), t_green=5.0, t_amber=3.0, t_red=6.0
        )


# -----------------------------------------------------------------------------
# Slope
# -----------------------------------------------------------------------------


def test_slope_flat_surface(create_test_dem):
    flat = np.ones((10, 10), dtype=np.float32) * 100.0
    slope = generate_slope_degrees_raster(create_test_dem(flat))
    assert np.nanmax(slope) < 0.01, "flat surface should have ~0 slope"


def test_slope_steep_ramp(create_test_dem):
    # 45° slope: rise equals run
    ramp = np.array([[0, 10, 20], [0, 10, 20], [0, 10, 20]], dtype=np.float32)
    dem = create_test_dem(ramp, bounds=(0, 0, 30, 30))  # 10m pixels

    slope = generate_slope_degrees_raster(dem)

    center = slope[1, 1]
    assert 40 < center < 50, f"expected ~45°, got {center}"


def test_slope_nodata_doesnt_bleed(create_test_dem):
    # Found this: nodata cell was corrupting neighbor gradients
    data = np.array(
        [
            [10.0, 10.0, 10.0],
            [10.0, -9999.0, 10.0],
            [10.0, 10.0, 10.0],
        ],
        dtype=np.float32,
    )
    dem = create_test_dem(data, nodata=-9999.0)

    slope = generate_slope_degrees_raster(dem)

    assert np.isnan(slope[1, 1]), "center nodata should be NaN"
    assert np.isfinite(slope[0, 0]), "corners should still compute"
    assert np.isfinite(slope[2, 2])
