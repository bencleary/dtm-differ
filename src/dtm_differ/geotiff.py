from typing import Literal
import rasterio
from pathlib import Path
from rasterio.errors import RasterioIOError
from dtm_differ.types import GeotiffInformation, RasterCompatability, GeoTiffBounds
import xdem
import numpy as np


def validate_geotiff(geotiff_path: str):
    """
    Validate a GeoTIFF file.

    Throws:
        ValueError: If the GeoTIFF file is not valid
    """

    file_path = Path(geotiff_path)

    if not file_path.exists():
        raise ValueError("File does not exist")
    if not file_path.is_file():
        raise ValueError("File is not a file")
    if not file_path.suffix.lower() == ".tif":
        raise ValueError("File is not a GeoTIFF file")

    try:
        with rasterio.open(geotiff_path) as src:
            if src.meta.get("driver", "") != "GTiff":
                raise ValueError("Not a GeoTIFF file")
            if src.count < 1:
                raise ValueError("Has no bands")
            if src.width <= 0 or src.height <= 0:
                raise ValueError("Width or height is less than or equal to 0")
            if src.crs is None:
                raise ValueError("No CRS")
    except RasterioIOError as e:
        raise ValueError(f"Invalid GeoTIFF file: {geotiff_path} - {e}")


def get_geotiff_metadata(geotiff_path: str) -> tuple[GeotiffInformation, xdem.DEM]:
    """
    Get the metadata of a GeoTIFF file.

    Returns:
        GeotiffInformation: The metadata of the GeoTIFF file
    """
    file_path = Path(geotiff_path)
    with rasterio.open(file_path) as src:
        return GeotiffInformation(
            path=file_path,
            crs=src.crs.to_string(),
            bounds=src.bounds,
            width=src.width,
            height=src.height,
            transform=src.transform,
            dtype=str(src.dtypes[0]),
            nodata=src.nodata,
        ), xdem.DEM(geotiff_path)


def _bounds_overlap(a: GeoTiffBounds, b: GeoTiffBounds) -> bool:
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def check_raster_compatability(a: GeotiffInformation, b: GeotiffInformation) -> RasterCompatability:
    """
    Check if two rasters are compatible for differencing.

    Returns:
        RasterCompatability: The compatibility of the two rasters
    
    Throws:
        ValueError: If the two rasters are not compatible
    """
    same_crs = a.crs == b.crs
    same_transform = a.transform == b.transform
    same_shape = a.width == b.width and a.height == b.height
    same_grid = same_transform and same_shape
    overlaps = _bounds_overlap(a.bounds, b.bounds) if same_crs else False

    reason = None
    if not same_crs:
        reason = "Different CRS, alignment will require reprojection"
    elif not same_grid:
        reason = "Different grid, alignment will require resampling to reference grid"
    elif not overlaps:
        reason = "No overlap, alignment will require resampling"

    return RasterCompatability(
        same_crs=same_crs,
        same_grid=same_grid,
        same_transform=same_transform,
        same_shape=same_shape,
        overlaps=overlaps,
        reason=reason,
    )


def reproject_raster(
    direction: Literal["to-a", "to-b"],
    a: xdem.DEM,
    b: xdem.DEM,
    *,
    resampling: Literal["nearest", "bilinear"] = "bilinear",
) -> xdem.DEM:
    """
    Reproject and resample one raster to match the other's CRS/grid.

    Returns:
        xdem.DEM: The reprojected raster

    Throws:
        ValueError: If the direction is invalid
    """    
    match direction:
        case "to-a":
            return b.reproject(ref=a, resampling=resampling)
        case "to-b":
            return a.reproject(ref=b, resampling=resampling)
        case _:
            raise ValueError(f"Invalid direction: {direction}")



def validate_dem_data(dem: xdem.DEM, min_valid_pixels: float = 0.01) -> tuple[bool, str]:
    """
          Validate that DEM has sufficient valid data.
          
          Args:
              dem: DEM to validate
              min_valid_pixels: Minimum fraction of pixels that must be valid (default 1%)
          
          Returns:
              Tuple of (is_valid, message)
    """
    total_pixels = dem.data.size
    if dem.nodata is not None:
        valid_pixels = np.sum(dem.data != dem.nodata)
    else:
        valid_pixels = np.sum(np.isfinite(dem.data))
          
    valid_fraction = valid_pixels / total_pixels
    if valid_fraction < min_valid_pixels:
        return False, f"Only {valid_fraction:.1%} of pixels are valid (minimum {min_valid_pixels:.1%})"
    return True, f"{valid_fraction:.1%} of pixels are valid"