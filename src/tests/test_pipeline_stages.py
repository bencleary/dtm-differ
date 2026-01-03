from __future__ import annotations

from unittest.mock import patch

import numpy as np

from dtm_differ.pipeline.stages import compute_rasters  # noqa: E402
from dtm_differ.pipeline.types import (  # noqa: E402
    AlignedDems,
    ProcessingConfig,
    ReprojectionInfo,
)


def test_compute_rasters_ranking(create_test_dem) -> None:
    a_dem = create_test_dem(np.zeros((2, 2), dtype=float))
    b_dem = create_test_dem(np.zeros((2, 2), dtype=float))
    diff = create_test_dem(np.array([[0.5, -1.5], [3.5, -6.5]], dtype=float))

    reproj_info = ReprojectionInfo(occurred=False)
    with patch("dtm_differ.pipeline.stages.generate_difference_raster", return_value=diff):
        rasters = compute_rasters(
            AlignedDems(a_dem=a_dem, b_dem=b_dem, reprojection_info=reproj_info),
            ProcessingConfig(),
        )

    # Default config enables uncertainty and suppresses ranks within noise,
    # so the -1.5 m cell is suppressed (|dh| <= k*sigma_dh).
    expected = np.array([[0, 0], [2, 3]], dtype=np.uint8)
    assert np.array_equal(rasters.movement_rank, expected)
    assert rasters.movement_rank.dtype == np.uint8
    assert rasters.output_mask.dtype == np.bool_


def test_compute_rasters_masks_nodata(create_test_dem) -> None:
    a_dem = create_test_dem(np.zeros((2, 2), dtype=float))
    b_dem = create_test_dem(np.zeros((2, 2), dtype=float))
    diff = create_test_dem(np.array([[1.0, -9999.0], [2.0, 3.0]], dtype=float), nodata=-9999.0)

    reproj_info = ReprojectionInfo(occurred=False)
    with patch("dtm_differ.pipeline.stages.generate_difference_raster", return_value=diff):
        rasters = compute_rasters(
            AlignedDems(a_dem=a_dem, b_dem=b_dem, reprojection_info=reproj_info),
            ProcessingConfig(),
        )

    assert bool(rasters.output_mask[0, 1])
    assert np.isnan(rasters.elevation_change[0, 1])
    assert np.isnan(rasters.change_magnitude[0, 1])
    assert int(rasters.change_direction[0, 1]) == 0
    assert int(rasters.movement_rank[0, 1]) == 0


def test_elevation_masking_min_elevation(create_test_dem) -> None:
    """Test that min_elevation masks areas below the threshold."""
    # Create DEM A with elevations: [[1.0, 3.0], [5.0, 10.0]]
    a_dem = create_test_dem(np.array([[1.0, 3.0], [5.0, 10.0]], dtype=float))
    b_dem = create_test_dem(np.array([[1.5, 3.5], [5.5, 10.5]], dtype=float))
    diff = create_test_dem(np.array([[0.5, 0.5], [0.5, 0.5]], dtype=float))

    reproj_info = ReprojectionInfo(occurred=False)
    with patch("dtm_differ.pipeline.stages.generate_difference_raster", return_value=diff):
        rasters = compute_rasters(
            AlignedDems(a_dem=a_dem, b_dem=b_dem, reprojection_info=reproj_info),
            ProcessingConfig(min_elevation=2.0),  # Mask areas < 2m
        )

    # Pixel [0,0] has elevation 1.0 < 2.0, should be masked
    assert bool(rasters.output_mask[0, 0])
    assert np.isnan(rasters.elevation_change[0, 0])

    # Pixel [0,1] has elevation 3.0 >= 2.0, should NOT be masked
    assert not bool(rasters.output_mask[0, 1])
    assert np.isfinite(rasters.elevation_change[0, 1])

    # Pixels [1,0] and [1,1] have elevations 5.0 and 10.0, should NOT be masked
    assert not bool(rasters.output_mask[1, 0])
    assert not bool(rasters.output_mask[1, 1])


def test_elevation_masking_max_elevation(create_test_dem) -> None:
    """Test that max_elevation masks areas above the threshold."""
    # Create DEM A with elevations: [[1.0, 3.0], [5.0, 10.0]]
    a_dem = create_test_dem(np.array([[1.0, 3.0], [5.0, 10.0]], dtype=float))
    b_dem = create_test_dem(np.array([[1.5, 3.5], [5.5, 10.5]], dtype=float))
    diff = create_test_dem(np.array([[0.5, 0.5], [0.5, 0.5]], dtype=float))

    reproj_info = ReprojectionInfo(occurred=False)
    with patch("dtm_differ.pipeline.stages.generate_difference_raster", return_value=diff):
        rasters = compute_rasters(
            AlignedDems(a_dem=a_dem, b_dem=b_dem, reprojection_info=reproj_info),
            ProcessingConfig(max_elevation=8.0),  # Mask areas > 8m
        )

    # Pixel [1,1] has elevation 10.0 > 8.0, should be masked
    assert bool(rasters.output_mask[1, 1])
    assert np.isnan(rasters.elevation_change[1, 1])

    # Other pixels have elevations <= 8.0, should NOT be masked
    assert not bool(rasters.output_mask[0, 0])
    assert not bool(rasters.output_mask[0, 1])
    assert not bool(rasters.output_mask[1, 0])


def test_elevation_masking_both_bounds(create_test_dem) -> None:
    """Test that both min and max elevation bounds work together."""
    # Create DEM A with elevations: [[1.0, 3.0], [5.0, 10.0]]
    a_dem = create_test_dem(np.array([[1.0, 3.0], [5.0, 10.0]], dtype=float))
    b_dem = create_test_dem(np.array([[1.5, 3.5], [5.5, 10.5]], dtype=float))
    diff = create_test_dem(np.array([[0.5, 0.5], [0.5, 0.5]], dtype=float))

    reproj_info = ReprojectionInfo(occurred=False)
    with patch("dtm_differ.pipeline.stages.generate_difference_raster", return_value=diff):
        rasters = compute_rasters(
            AlignedDems(a_dem=a_dem, b_dem=b_dem, reprojection_info=reproj_info),
            ProcessingConfig(min_elevation=2.0, max_elevation=8.0),  # Focus on 2-8m range
        )

    # Pixel [0,0]: elevation 1.0 < 2.0, masked
    assert bool(rasters.output_mask[0, 0])

    # Pixel [0,1]: elevation 3.0 in range [2,8], not masked
    assert not bool(rasters.output_mask[0, 1])

    # Pixel [1,0]: elevation 5.0 in range [2,8], not masked
    assert not bool(rasters.output_mask[1, 0])

    # Pixel [1,1]: elevation 10.0 > 8.0, masked
    assert bool(rasters.output_mask[1, 1])


def test_elevation_masking_with_existing_nodata(create_test_dem) -> None:
    """Test that elevation masking combines with existing nodata areas."""
    # Create DEM A with one nodata pixel and varying elevations
    a_dem = create_test_dem(np.array([[1.0, -9999.0], [5.0, 10.0]], dtype=float), nodata=-9999.0)
    b_dem = create_test_dem(np.array([[1.5, -9999.0], [5.5, 10.5]], dtype=float), nodata=-9999.0)
    diff = create_test_dem(np.array([[0.5, -9999.0], [0.5, 0.5]], dtype=float), nodata=-9999.0)

    reproj_info = ReprojectionInfo(occurred=False)
    with patch("dtm_differ.pipeline.stages.generate_difference_raster", return_value=diff):
        rasters = compute_rasters(
            AlignedDems(a_dem=a_dem, b_dem=b_dem, reprojection_info=reproj_info),
            ProcessingConfig(min_elevation=2.0),  # Also mask areas < 2m
        )

    # Pixel [0,0]: elevation 1.0 < 2.0, masked by elevation threshold
    assert bool(rasters.output_mask[0, 0])

    # Pixel [0,1]: nodata in source, masked by existing nodata
    assert bool(rasters.output_mask[0, 1])

    # Pixel [1,0]: elevation 5.0 >= 2.0, not masked
    assert not bool(rasters.output_mask[1, 0])

    # Pixel [1,1]: elevation 10.0 >= 2.0, not masked
    assert not bool(rasters.output_mask[1, 1])
