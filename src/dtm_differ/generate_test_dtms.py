"""
Generate sample DTMs for testing edge cases in nodata handling and pipeline processing.

This script creates various test scenarios:
- Nodata handling (with/without nodata, different nodata values)
- Movement thresholds (boundary conditions)
- Slope variations
- Data quality edge cases
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xdem
from rasterio.transform import from_bounds


def create_test_dem(
    data: np.ndarray,
    *,
    nodata: float | None = None,
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 1000.0, 1000.0),
    crs: str = "EPSG:4326",
) -> xdem.DEM:
    """Create a test DEM from array data."""
    height, width = data.shape
    xmin, ymin, xmax, ymax = bounds
    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

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

    return xdem.DEM.from_array(prepared_data, transform=transform, crs=crs, nodata=nodata)


def generate_scenario_1_basic_movement(out_dir: Path) -> None:
    """
    Scenario 1: Basic movement patterns with no nodata.

    Tests:
    - Basic differencing
    - Movement at threshold boundaries
    - Mixed subsidence/uplift
    """
    size = 100
    a_data = np.zeros((size, size), dtype=np.float32)
    b_data = np.zeros((size, size), dtype=np.float32)

    # Create movement patterns
    # Top-left: subsidence (negative change)
    a_data[0:30, 0:30] = 10.0
    b_data[0:30, 0:30] = 8.0  # -2.0m change

    # Top-right: uplift (positive change)
    a_data[0:30, 70:100] = 10.0
    b_data[0:30, 70:100] = 13.0  # +3.0m change (at amber threshold)

    # Bottom-left: no change
    a_data[70:100, 0:30] = 10.0
    b_data[70:100, 0:30] = 10.0  # 0.0m change

    # Bottom-right: large subsidence (red threshold)
    a_data[70:100, 70:100] = 10.0
    b_data[70:100, 70:100] = 3.0  # -7.0m change (exceeds red threshold)

    # Center: green threshold boundary
    a_data[40:60, 40:60] = 10.0
    b_data[40:60, 40:60] = 11.0  # +1.0m change (at green threshold)

    a_dem = create_test_dem(a_data, nodata=None)
    b_dem = create_test_dem(b_data, nodata=None)

    a_dem.save(out_dir / "scenario1_dem_a.tif")
    b_dem.save(out_dir / "scenario1_dem_b.tif")
    print("Generated Scenario 1: Basic movement patterns")


def generate_scenario_2_nodata_edges(out_dir: Path) -> None:
    """
    Scenario 2: Nodata at edges and boundaries.

    Tests:
    - Nodata handling in input DEMs
    - Nodata propagation to difference
    - Nodata at array boundaries
    """
    size = 100
    nodata_value = -9999.0

    a_data = np.ones((size, size), dtype=np.float32) * 10.0
    b_data = np.ones((size, size), dtype=np.float32) * 12.0

    # Add nodata at edges
    a_data[0, :] = nodata_value  # Top edge
    a_data[:, 0] = nodata_value  # Left edge
    a_data[-1, :] = nodata_value  # Bottom edge
    a_data[:, -1] = nodata_value  # Right edge

    # Add nodata in center (overlapping)
    a_data[45:55, 45:55] = nodata_value
    b_data[45:55, 45:55] = nodata_value

    # Add nodata only in DEM A (non-overlapping)
    a_data[20:30, 20:30] = nodata_value

    # Add nodata only in DEM B (non-overlapping)
    b_data[70:80, 70:80] = nodata_value

    a_dem = create_test_dem(a_data, nodata=nodata_value)
    b_dem = create_test_dem(b_data, nodata=nodata_value)

    a_dem.save(out_dir / "scenario2_dem_a.tif")
    b_dem.save(out_dir / "scenario2_dem_b.tif")
    print("Generated Scenario 2: Nodata at edges")


def generate_scenario_3_mixed_nodata_values(out_dir: Path) -> None:
    """
    Scenario 3: Different nodata values in DEM A and B.

    Tests:
    - Different nodata values in input DEMs
    - How xdem handles mismatched nodata
    """
    size = 100
    nodata_a = -9999.0
    nodata_b = -32768.0

    a_data = np.ones((size, size), dtype=np.float32) * 10.0
    b_data = np.ones((size, size), dtype=np.float32) * 12.0

    # Add nodata with different values
    a_data[0:20, 0:20] = nodata_a
    b_data[0:20, 0:20] = nodata_b

    # Overlapping nodata areas
    a_data[40:60, 40:60] = nodata_a
    b_data[40:60, 40:60] = nodata_b

    a_dem = create_test_dem(a_data, nodata=nodata_a)
    b_dem = create_test_dem(b_data, nodata=nodata_b)

    a_dem.save(out_dir / "scenario3_dem_a.tif")
    b_dem.save(out_dir / "scenario3_dem_b.tif")
    print("Generated Scenario 3: Mixed nodata values")


def generate_scenario_4_slope_variations(out_dir: Path) -> None:
    """
    Scenario 4: Various slope conditions.

    Tests:
    - Flat areas
    - Gentle slopes
    - Steep slopes
    - Nodata affecting slope calculation
    """
    size = 100
    a_data = np.zeros((size, size), dtype=np.float32)
    b_data = np.zeros((size, size), dtype=np.float32)

    # Create a ramp (gentle slope)
    for i in range(size):
        a_data[i, :] = i * 0.1  # 0.1m per pixel
        b_data[i, :] = i * 0.1 + 1.0  # Same slope, shifted up

    # Add steep slope in center
    center = size // 2
    for i in range(size):
        for j in range(size):
            dist = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            if dist < 20:
                a_data[i, j] += dist * 0.5  # Steep slope
                b_data[i, j] += dist * 0.5 + 2.0  # Same slope, shifted

    # Add nodata that will affect slope calculation
    nodata_value = -9999.0
    a_data[10:15, 10:15] = nodata_value
    b_data[10:15, 10:15] = nodata_value

    a_dem = create_test_dem(a_data, nodata=nodata_value)
    b_dem = create_test_dem(b_data, nodata=nodata_value)

    a_dem.save(out_dir / "scenario4_dem_a.tif")
    b_dem.save(out_dir / "scenario4_dem_b.tif")
    print("Generated Scenario 4: Slope variations")


def generate_scenario_5_threshold_boundaries(out_dir: Path) -> None:
    """
    Scenario 5: Movement values exactly at thresholds.

    Tests:
    - Values exactly at green threshold (1.0m)
    - Values exactly at amber threshold (3.0m)
    - Values exactly at red threshold (6.0m)
    - Values just below/above thresholds
    """
    size = 100
    a_data = np.ones((size, size), dtype=np.float32) * 10.0
    b_data = np.ones((size, size), dtype=np.float32) * 10.0

    # Green threshold: exactly 1.0m
    b_data[0:20, 0:20] = 11.0  # +1.0m

    # Just below green: 0.99m
    b_data[0:20, 20:40] = 10.99  # +0.99m

    # Just above green: 1.01m
    b_data[0:20, 40:60] = 11.01  # +1.01m

    # Amber threshold: exactly 3.0m
    b_data[20:40, 0:20] = 13.0  # +3.0m

    # Just below amber: 2.99m
    b_data[20:40, 20:40] = 12.99  # +2.99m

    # Just above amber: 3.01m
    b_data[20:40, 40:60] = 13.01  # +3.01m

    # Red threshold: exactly 6.0m
    b_data[40:60, 0:20] = 16.0  # +6.0m

    # Just below red: 5.99m
    b_data[40:60, 20:40] = 15.99  # +5.99m

    # Just above red: 6.01m
    b_data[40:60, 40:60] = 16.01  # +6.01m

    # Negative changes (subsidence)
    b_data[60:80, 0:20] = 9.0  # -1.0m (green threshold)
    b_data[60:80, 20:40] = 7.0  # -3.0m (amber threshold)
    b_data[60:80, 40:60] = 4.0  # -6.0m (red threshold)

    a_dem = create_test_dem(a_data, nodata=None)
    b_dem = create_test_dem(b_data, nodata=None)

    a_dem.save(out_dir / "scenario5_dem_a.tif")
    b_dem.save(out_dir / "scenario5_dem_b.tif")
    print("Generated Scenario 5: Threshold boundaries")


def generate_scenario_6_sparse_nodata(out_dir: Path) -> None:
    """
    Scenario 6: Sparse nodata (should pass validation).

    Tests:
    - DEMs with some nodata but sufficient valid data
    - Nodata scattered throughout
    """
    size = 100
    nodata_value = -9999.0

    a_data = np.ones((size, size), dtype=np.float32) * 10.0
    b_data = np.ones((size, size), dtype=np.float32) * 12.0

    # Scatter nodata randomly (~10% nodata)
    np.random.seed(42)  # For reproducibility
    nodata_mask = np.random.random((size, size)) < 0.1
    a_data[nodata_mask] = nodata_value
    b_data[nodata_mask] = nodata_value

    # Add some movement in valid areas
    b_data[30:50, 30:50] = 13.0  # +3.0m change

    a_dem = create_test_dem(a_data, nodata=nodata_value)
    b_dem = create_test_dem(b_data, nodata=nodata_value)

    a_dem.save(out_dir / "scenario6_dem_a.tif")
    b_dem.save(out_dir / "scenario6_dem_b.tif")
    print("Generated Scenario 6: Sparse nodata")


def generate_scenario_7_mostly_nodata(out_dir: Path) -> None:
    """
    Scenario 7: Mostly nodata (should fail validation).

    Tests:
    - DEMs with >99% nodata (should fail validation)
    - Edge case for data quality checks
    """
    size = 100
    nodata_value = -9999.0

    a_data = np.full((size, size), nodata_value, dtype=np.float32)
    b_data = np.full((size, size), nodata_value, dtype=np.float32)

    # Only a few valid pixels
    a_data[50:52, 50:52] = 10.0
    b_data[50:52, 50:52] = 12.0

    a_dem = create_test_dem(a_data, nodata=nodata_value)
    b_dem = create_test_dem(b_data, nodata=nodata_value)

    a_dem.save(out_dir / "scenario7_dem_a.tif")
    b_dem.save(out_dir / "scenario7_dem_b.tif")
    print("Generated Scenario 7: Mostly nodata (should fail validation)")


def generate_scenario_8_no_nodata_attribute(out_dir: Path) -> None:
    """
    Scenario 8: DEMs without nodata attribute but with NaN/inf.

    Tests:
    - DEMs with NaN values but no nodata attribute
    - DEMs with infinite values

    Note: NaN/inf will be converted to NaN by create_test_dem, which is fine
    for testing how the pipeline handles non-finite values.
    """
    size = 100
    a_data = np.ones((size, size), dtype=np.float32) * 10.0
    b_data = np.ones((size, size), dtype=np.float32) * 12.0

    # Add NaN (no nodata attribute)
    a_data[0:10, 0:10] = np.nan
    b_data[0:10, 0:10] = np.nan

    # Add inf
    a_data[10:20, 10:20] = np.inf
    b_data[10:20, 10:20] = np.inf

    # create_test_dem will convert NaN/inf to NaN, which is fine for testing
    a_dem = create_test_dem(a_data, nodata=None)
    b_dem = create_test_dem(b_data, nodata=None)

    a_dem.save(out_dir / "scenario8_dem_a.tif")
    b_dem.save(out_dir / "scenario8_dem_b.tif")
    print("Generated Scenario 8: No nodata attribute (NaN/inf)")


def generate_all_scenarios(out_dir: Path) -> None:
    """Generate all test scenarios."""
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating test DTM scenarios...")
    print("=" * 50)

    generate_scenario_1_basic_movement(out_dir)
    generate_scenario_2_nodata_edges(out_dir)
    generate_scenario_3_mixed_nodata_values(out_dir)
    generate_scenario_4_slope_variations(out_dir)
    generate_scenario_5_threshold_boundaries(out_dir)
    generate_scenario_6_sparse_nodata(out_dir)
    generate_scenario_7_mostly_nodata(out_dir)
    generate_scenario_8_no_nodata_attribute(out_dir)

    print("=" * 50)
    print(f"All scenarios generated in: {out_dir}")
    print("\nGenerated files:")
    for f in sorted(out_dir.glob("*.tif")):
        print(f"  - {f.name}")


def main() -> None:
    """CLI entry point for generating test DTMs."""
    import sys

    if len(sys.argv) > 1:
        out_dir = Path(sys.argv[1])
    else:
        out_dir = Path("test_data") / "sample_dtms"

    generate_all_scenarios(out_dir)


if __name__ == "__main__":
    main()
