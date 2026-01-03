from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import xdem
from numpy.typing import NDArray


@dataclass(frozen=True)
class ProcessingConfig:
    resample: Literal["nearest", "bilinear"] = "bilinear"
    align: Literal["to-a", "to-b"] = "to-a"
    t_green: float = 1.0
    t_amber: float = 3.0
    t_red: float = 6.0
    generate_report: bool = True
    generate_polygons: bool = True
    uncertainty_mode: Literal["none", "constant"] = "constant"
    sigma_a: float = 0.5
    sigma_b: float = 0.5
    sigma_coreg: float = 0.3
    k_sigma: float = 1.96
    suppress_within_noise_rank: bool = True
    min_elevation: float | None = None
    max_elevation: float | None = None


@dataclass(frozen=True)
class Workspace:
    job_id: str
    out_dir: Path
    map_layers_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class Inputs:
    a_path: Path
    b_path: Path


@dataclass(frozen=True)
class ReprojectionInfo:
    """Metadata about reprojection/resampling operations."""

    occurred: bool
    source_crs_a: str | None = None
    source_crs_b: str | None = None
    target_crs: str | None = None
    resampling_method: str | None = None
    reference_grid: str | None = None  # "to-a" or "to-b"
    reason: str | None = None


@dataclass(frozen=True)
class AlignedDems:
    a_dem: xdem.DEM
    b_dem: xdem.DEM
    reprojection_info: ReprojectionInfo


@dataclass(frozen=True)
class DerivedRasters:
    diff: xdem.DEM
    elevation_change: NDArray[np.floating]
    change_direction: NDArray[np.int8]
    change_magnitude: NDArray[np.floating]
    slope_deg: NDArray[np.floating]
    movement_rank: NDArray[np.uint8]
    output_mask: NDArray[np.bool_]
    sigma_dh: NDArray[np.floating] | None
    z_score: NDArray[np.floating] | None
    within_noise_mask: NDArray[np.uint8] | None
    reprojection_info: ReprojectionInfo


@dataclass(frozen=True)
class ProcessingResult:
    job_id: str
    diff: xdem.DEM
    elevation_change: NDArray[np.floating]
    change_direction: NDArray[np.int8]
    change_magnitude: NDArray[np.floating]
    slope_deg: NDArray[np.floating]
    movement_rank: NDArray[np.uint8]
    output_mask: NDArray[np.bool_]
    transform: Any
    crs: Any
    map_layers_dir: Path
    reports_dir: Path
    sigma_dh: NDArray[np.floating] | None
    z_score: NDArray[np.floating] | None
    within_noise_mask: NDArray[np.uint8] | None
    deferred_output: str | None = None
