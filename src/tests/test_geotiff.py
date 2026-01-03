"""Tests for GeoTIFF handling functions."""

import numpy as np
from rasterio.transform import from_bounds

from dtm_differ.geotiff import (
    _bounds_overlap,
    check_raster_compatability,
    validate_dem_data,
)
from dtm_differ.types import GeotiffInformation


def test_validate_dem_data_sufficient(create_test_dem):
    """Test validation passes for DEM with sufficient valid data."""
    data = np.ones((10, 10), dtype=np.float32)
    dem = create_test_dem(data)

    is_valid, message = validate_dem_data(dem)

    assert is_valid is True
    assert "100.0%" in message or "100%" in message


def test_validate_dem_data_insufficient(create_test_dem):
    """Test validation fails for DEM with insufficient valid data."""
    data = np.full((10, 10), -9999.0, dtype=np.float32)
    data[0, 0] = 1.0  # Only one valid pixel
    dem = create_test_dem(data, nodata=-9999.0)

    is_valid, message = validate_dem_data(dem, min_valid_pixels=0.1)  # Require 10%

    assert is_valid is False
    assert "minimum" in message.lower() or "only" in message.lower()


def test_validate_dem_data_with_nodata(create_test_dem):
    """Test validation handles nodata correctly."""
    data = np.ones((10, 10), dtype=np.float32)
    data[0:5, :] = -9999.0  # Half nodata
    dem = create_test_dem(data, nodata=-9999.0)

    is_valid, message = validate_dem_data(dem, min_valid_pixels=0.4)

    assert is_valid is True  # 50% valid > 40% minimum


def test_validate_dem_data_no_nodata(create_test_dem):
    """Test validation works when DEM has no nodata value."""
    import warnings

    data = np.ones((10, 10), dtype=np.float32)
    data[0, 0] = np.nan  # NaN but no nodata attribute
    # create_test_dem will convert NaN to NaN (no nodata specified)
    # xdem will set a default nodata and warn about it - this is expected behavior
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module="geoutils",
            message=".*default nodata.*",
        )
        dem = create_test_dem(data, nodata=None)

    is_valid, message = validate_dem_data(dem)

    # Should still pass as most values are finite
    assert is_valid is True


def test_bounds_overlap():
    """Test bounds overlap detection."""
    # Overlapping bounds
    a = (0.0, 0.0, 10.0, 10.0)  # xmin, ymin, xmax, ymax
    b = (5.0, 5.0, 15.0, 15.0)
    assert _bounds_overlap(a, b) is True

    # Non-overlapping bounds
    a = (0.0, 0.0, 5.0, 5.0)
    b = (10.0, 10.0, 15.0, 15.0)
    assert _bounds_overlap(a, b) is False

    # Touching bounds (should overlap)
    a = (0.0, 0.0, 5.0, 5.0)
    b = (5.0, 5.0, 10.0, 10.0)
    assert _bounds_overlap(a, b) is True


def test_check_raster_compatability_same():
    """Test compatibility check for identical rasters."""
    transform = from_bounds(0, 0, 10, 10, 10, 10)
    a_info = GeotiffInformation(
        path=None,  # type: ignore
        crs="EPSG:4326",
        bounds=(0, 0, 10, 10),
        width=10,
        height=10,
        transform=transform,
        dtype="float32",
        nodata=-9999.0,
    )
    b_info = GeotiffInformation(
        path=None,  # type: ignore
        crs="EPSG:4326",
        bounds=(0, 0, 10, 10),
        width=10,
        height=10,
        transform=transform,
        dtype="float32",
        nodata=-9999.0,
    )

    compat = check_raster_compatability(a_info, b_info)

    assert compat.same_crs is True
    assert compat.same_grid is True
    assert compat.same_transform is True
    assert compat.same_shape is True
    assert compat.overlaps is True
    assert compat.reason is None


def test_check_raster_compatability_different_crs():
    """Test compatibility check for different CRS."""
    transform = from_bounds(0, 0, 10, 10, 10, 10)
    a_info = GeotiffInformation(
        path=None,  # type: ignore
        crs="EPSG:4326",
        bounds=(0, 0, 10, 10),
        width=10,
        height=10,
        transform=transform,
        dtype="float32",
        nodata=-9999.0,
    )
    b_info = GeotiffInformation(
        path=None,  # type: ignore
        crs="EPSG:3857",  # Different CRS
        bounds=(0, 0, 10, 10),
        width=10,
        height=10,
        transform=transform,
        dtype="float32",
        nodata=-9999.0,
    )

    compat = check_raster_compatability(a_info, b_info)

    assert compat.same_crs is False
    assert compat.reason is not None
    assert "reprojection" in compat.reason.lower()
