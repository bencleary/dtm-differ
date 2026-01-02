from dataclasses import dataclass
from pathlib import Path

from rasterio.transform import Affine

type GeoTiffBounds = tuple[float, float, float, float]


@dataclass(frozen=True)
class GeotiffInformation:
    path: Path
    crs: str
    bounds: GeoTiffBounds
    width: int
    height: int
    transform: Affine
    dtype: str
    nodata: float | None
    linear_unit_name: str | None = None
    linear_unit_factor: float | None = None


@dataclass(frozen=True)
class RasterCompatability:
    same_crs: bool
    same_grid: bool  # transform + width/height
    same_transform: bool
    same_shape: bool
    overlaps: bool
    reason: str | None = None
