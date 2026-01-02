from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xdem  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402

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
    with patch(
        "dtm_differ.pipeline.stages.generate_difference_raster", return_value=diff
    ):
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
    diff = create_test_dem(
        np.array([[1.0, -9999.0], [2.0, 3.0]], dtype=float), nodata=-9999.0
    )

    reproj_info = ReprojectionInfo(occurred=False)
    with patch(
        "dtm_differ.pipeline.stages.generate_difference_raster", return_value=diff
    ):
        rasters = compute_rasters(
            AlignedDems(a_dem=a_dem, b_dem=b_dem, reprojection_info=reproj_info),
            ProcessingConfig(),
        )

    assert bool(rasters.output_mask[0, 1])
    assert np.isnan(rasters.elevation_change[0, 1])
    assert np.isnan(rasters.change_magnitude[0, 1])
    assert int(rasters.change_direction[0, 1]) == 0
    assert int(rasters.movement_rank[0, 1]) == 0
