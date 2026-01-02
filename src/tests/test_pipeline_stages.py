from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import numpy as np


from pathlib import Path


_repo_root = Path(__file__).resolve().parents[2]
_test_cache_root = _repo_root / ".tmp" / "test-cache"
(_test_cache_root / "mplconfig").mkdir(parents=True, exist_ok=True)
(_test_cache_root / "xdg-cache").mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_test_cache_root / "mplconfig")
os.environ["XDG_CACHE_HOME"] = str(_test_cache_root / "xdg-cache")


import xdem  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

from dtm_differ.pipeline.stages import compute_rasters  # noqa: E402
from dtm_differ.pipeline.types import AlignedDems, ProcessingConfig, ReprojectionInfo  # noqa: E402


def _dem_from_array(array: np.ndarray, *, nodata: float | None = None) -> xdem.DEM:
    """Create a test DEM from array, properly handling nodata."""
    # Prepare data: convert nodata to NaN to avoid warnings
    prepared_data = array.copy().astype(np.float32)
    
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
        transform=from_origin(0, 0, 1, 1),
        crs="EPSG:4326",
        nodata=nodata,  # Still pass nodata so xdem knows what value to use when saving
    )


def test_compute_rasters_ranking() -> None:
    a_dem = _dem_from_array(np.zeros((2, 2), dtype=float))
    b_dem = _dem_from_array(np.zeros((2, 2), dtype=float))
    diff = _dem_from_array(np.array([[0.5, -1.5], [3.5, -6.5]], dtype=float))

    reproj_info = ReprojectionInfo(occurred=False)
    with patch("dtm_differ.pipeline.stages.generate_difference_raster", return_value=diff):
        rasters = compute_rasters(AlignedDems(a_dem=a_dem, b_dem=b_dem, reprojection_info=reproj_info), ProcessingConfig())

    # Default config enables uncertainty and suppresses ranks within noise,
    # so the -1.5 m cell is suppressed (|dh| <= k*sigma_dh).
    expected = np.array([[0, 0], [2, 3]], dtype=np.uint8)
    assert np.array_equal(rasters.movement_rank, expected)
    assert rasters.movement_rank.dtype == np.uint8
    assert rasters.output_mask.dtype == np.bool_

def test_compute_rasters_masks_nodata() -> None:
    a_dem = _dem_from_array(np.zeros((2, 2), dtype=float))
    b_dem = _dem_from_array(np.zeros((2, 2), dtype=float))
    diff = _dem_from_array(np.array([[1.0, -9999.0], [2.0, 3.0]], dtype=float), nodata=-9999.0)

    reproj_info = ReprojectionInfo(occurred=False)
    with patch("dtm_differ.pipeline.stages.generate_difference_raster", return_value=diff):
        rasters = compute_rasters(AlignedDems(a_dem=a_dem, b_dem=b_dem, reprojection_info=reproj_info), ProcessingConfig())

    assert bool(rasters.output_mask[0, 1])
    assert np.isnan(rasters.elevation_change[0, 1])
    assert np.isnan(rasters.change_magnitude[0, 1])
    assert int(rasters.change_direction[0, 1]) == 0
    assert int(rasters.movement_rank[0, 1]) == 0
