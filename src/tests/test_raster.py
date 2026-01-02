"""Tests for raster processing functions."""
import numpy as np
import pytest
import xdem
from rasterio.transform import from_bounds
from dtm_differ.raster import (
    generate_change_direction_raster,
    generate_change_magnitude_raster,
    generate_elevation_change_raster,
    generate_ranked_movement_raster,
    generate_slope_degrees_raster,
)


def create_test_dem(data: np.ndarray, nodata: float | None = None) -> xdem.DEM:
    """Helper to create a test DEM from array data, properly handling nodata."""
    transform = from_bounds(0, 0, 10, 10, data.shape[1], data.shape[0])
    
    # Prepare data: convert nodata to NaN to avoid warnings
    prepared_data = data.copy().astype(np.float32)
    
    # Handle non-finite values first (NaN, inf)
    non_finite_mask = ~np.isfinite(prepared_data)
    if np.any(non_finite_mask):
        # If nodata is specified, use it; otherwise convert to NaN
        if nodata is not None:
            prepared_data[non_finite_mask] = nodata
        else:
            prepared_data[non_finite_mask] = np.nan
    
    # Convert nodata values to NaN before passing to xdem
    if nodata is not None:
        # Use np.isclose for floating point comparison to handle precision issues
        nodata_mask = np.isclose(prepared_data, nodata, equal_nan=False)
        prepared_data[nodata_mask] = np.nan
    
    return xdem.DEM.from_array(
        prepared_data,
        transform=transform,
        crs="EPSG:4326",
        nodata=nodata,  # Still pass nodata so xdem knows what value to use when saving
    )


def test_generate_change_direction():
    """Test change direction raster generation."""
    # Create test data: negative, zero, positive values
    data = np.array([[-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]], dtype=np.float32)
    dem_a = create_test_dem(data)
    dem_b = create_test_dem(np.zeros_like(data))
    diff = dem_a - dem_b
    
    direction = generate_change_direction_raster(diff)
    
    assert direction.dtype == np.int8
    assert np.all(direction[0, 0] == -1)  # negative change
    assert np.all(direction[0, 1] == 0)    # no change
    assert np.all(direction[0, 2] == 1)    # positive change


def test_generate_change_magnitude():
    """Test change magnitude raster generation."""
    data = np.array([[-2.0, 0.0, 2.0], [1.0, 0.0, -1.0]], dtype=np.float32)
    dem_a = create_test_dem(data)
    dem_b = create_test_dem(np.zeros_like(data))
    diff = dem_a - dem_b
    
    magnitude = generate_change_magnitude_raster(diff)
    
    assert np.issubdtype(magnitude.dtype, np.floating)
    assert np.allclose(magnitude, np.abs(data))
    # Check that all values are non-negative
    assert np.all(magnitude[~np.isnan(magnitude)] >= 0)


def test_generate_change_magnitude_with_nodata():
    """Test change magnitude handles nodata correctly."""
    data = np.array([[1.0, 2.0], [3.0, -9999.0]], dtype=np.float32)
    dem_a = create_test_dem(data, nodata=-9999.0)
    dem_b = create_test_dem(np.zeros_like(data), nodata=-9999.0)
    diff = dem_a - dem_b
    
    magnitude = generate_change_magnitude_raster(diff)
    
    # Nodata should be converted to NaN
    # Function should return regular array (not masked), with NaN for nodata
    assert not np.ma.isMaskedArray(magnitude), "Should return regular array, not masked array"
    assert np.isnan(magnitude[1, 1])


def test_generate_elevation_change():
    """Test elevation change raster generation."""
    data = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
    dem_a = create_test_dem(data)
    dem_b = create_test_dem(np.zeros_like(data))
    diff = dem_a - dem_b
    
    elevation_change = generate_elevation_change_raster(diff)
    
    assert np.issubdtype(elevation_change.dtype, np.floating)
    # Function should return regular array (not masked)
    assert not np.ma.isMaskedArray(elevation_change), "Should return regular array, not masked array"
    assert np.allclose(elevation_change, data)


def test_generate_ranked_movement_raster():
    """Test ranked movement raster generation."""
    # Create magnitude array with values below, at, and above thresholds
    magnitude = np.array([
        [0.5, 1.0, 2.0],   # below, at green, between green/amber
        [3.0, 5.0, 7.0],   # at amber, between amber/red, above red
    ], dtype=np.float32)
    
    rank = generate_ranked_movement_raster(
        magnitude,
        t_green=1.0,
        t_amber=3.0,
        t_red=6.0,
    )
    
    assert rank.dtype == np.uint8
    assert rank[0, 0] == 0  # below threshold
    assert rank[0, 1] == 1  # green
    assert rank[0, 2] == 1  # green
    assert rank[1, 0] == 2  # amber
    assert rank[1, 1] == 2  # amber
    assert rank[1, 2] == 3  # red


def test_generate_ranked_movement_raster_invalid_thresholds():
    """Test that invalid thresholds raise ValueError."""
    magnitude = np.array([[1.0, 2.0]], dtype=np.float32)
    
    with pytest.raises(ValueError, match="Thresholds must satisfy"):
        generate_ranked_movement_raster(
            magnitude,
            t_green=3.0,  # Invalid: green > amber
            t_amber=2.0,
            t_red=6.0,
        )


def test_generate_slope_degrees_raster():
    """Test slope calculation."""
    # Create a simple ramp DEM
    data = np.array([
        [0.0, 1.0, 2.0],
        [0.0, 1.0, 2.0],
    ], dtype=np.float32)
    dem = create_test_dem(data)
    
    slope = generate_slope_degrees_raster(dem)
    
    assert np.issubdtype(slope.dtype, np.floating)
    # Function should return regular array (not masked)
    assert not np.ma.isMaskedArray(slope), "Should return regular array, not masked array"
    assert np.all(slope >= 0)  # Slope should be non-negative
    assert np.all(slope <= 90)  # Slope should be <= 90 degrees

