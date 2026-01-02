import numpy as np
import xdem
from numpy.typing import NDArray


def generate_difference_raster(a: xdem.DEM, b: xdem.DEM) -> xdem.DEM:
    """
    Generate a difference raster between two DEMs.

    Returns:
        xdem.DEM: The difference raster
    """
    diff = a - b
    diff.info(stats=True)
    return diff


def generate_difference_sigma_constant(
    shape: tuple[int, int],
    *,
    output_mask: NDArray[np.bool_],
    sigma_a: float,
    sigma_b: float,
    sigma_coreg: float,
) -> NDArray[np.floating]:
    sigma = float(np.sqrt(sigma_a**2 + sigma_b**2 + sigma_coreg**2))
    sigma_dh = np.full(shape, sigma, dtype=np.float32)
    sigma_dh[output_mask] = np.nan
    return sigma_dh


def generate_z_score(
    dh: NDArray[np.floating],
    sigma_dh: NDArray[np.floating],
    *,
    output_mask: NDArray[np.bool_],
) -> NDArray[np.floating]:
    z = np.full_like(dh, np.nan, dtype=np.float32)
    valid = (~output_mask) & np.isfinite(dh) & np.isfinite(sigma_dh) & (sigma_dh > 0)
    z[valid] = (dh[valid] / sigma_dh[valid]).astype(np.float32, copy=False)
    return z


def generate_within_noise_mask_u8(
    dh: NDArray[np.floating],
    sigma_dh: NDArray[np.floating],
    *,
    output_mask: NDArray[np.bool_],
    k_sigma: float,
) -> NDArray[np.uint8]:
    within = np.zeros(dh.shape, dtype=np.uint8)
    valid = (~output_mask) & np.isfinite(dh) & np.isfinite(sigma_dh) & (sigma_dh > 0)
    within[valid] = (np.abs(dh[valid]) <= (float(k_sigma) * sigma_dh[valid])).astype(
        np.uint8
    )
    return within


def generate_elevation_change_raster(diff: xdem.DEM) -> NDArray[np.floating]:
    """
    Generate an elevation change raster from a difference raster.

    Returns:
        NDArray[np.floating]: The elevation change array (nodata converted to NaN)
    """
    data = diff.data
    # Convert masked array to regular array with NaN
    if np.ma.isMaskedArray(data):
        elevation_change = np.ma.filled(data.astype(float), np.nan)
    else:
        elevation_change = data.astype(float)

    nodata = diff.nodata
    if nodata is not None:
        elevation_change[elevation_change == nodata] = np.nan
    return elevation_change


def generate_change_direction_raster(diff: xdem.DEM) -> NDArray[np.int8]:
    """
    Generate a change direction raster from a difference raster.

    Returns:
        NDArray[np.int8]: The change direction array (-1, 0, 1)
    """
    direction = np.zeros_like(diff.data, dtype=np.int8)
    if diff.nodata is None:
        direction[diff.data > 0] = 1
        direction[diff.data < 0] = -1
        return direction

    nodata_mask = diff.data == diff.nodata
    direction[(diff.data > 0) & ~nodata_mask] = 1
    direction[(diff.data < 0) & ~nodata_mask] = -1
    return direction


def generate_change_magnitude_raster(diff: xdem.DEM) -> NDArray[np.floating]:
    """
    Generate a change magnitude raster from a difference raster.

    Returns:
        NDArray[np.floating]: The absolute elevation change array (nodata converted to NaN)
    """
    data = diff.data
    # Convert masked array to regular array with NaN
    if np.ma.isMaskedArray(data):
        data_float = np.ma.filled(data.astype(float), np.nan)
    else:
        data_float = data.astype(float)

    magnitude = np.abs(data_float)
    if diff.nodata is not None:
        magnitude[diff.data == diff.nodata] = np.nan
    return magnitude


def generate_ranked_movement_raster(
    movement_magnitude: NDArray[np.floating],
    *,
    t_green: float = 1.0,
    t_amber: float = 3.0,
    t_red: float = 6.0,
) -> NDArray[np.uint8]:
    """
    Rank movement magnitude into Green/Amber/Red classes.

    Class meanings:
        0: below thresholds / unclassified
        1: green  (t_green <= mag < t_amber)
        2: amber  (t_amber <= mag < t_red)
        3: red    (mag >= t_red)
    """
    if not (0 <= t_green <= t_amber <= t_red):
        raise ValueError("Thresholds must satisfy 0 <= t_green <= t_amber <= t_red")

    class_raster = np.zeros_like(movement_magnitude, dtype=np.uint8)
    class_raster[(movement_magnitude >= t_green) & (movement_magnitude < t_amber)] = 1
    class_raster[(movement_magnitude >= t_amber) & (movement_magnitude < t_red)] = 2
    class_raster[movement_magnitude >= t_red] = 3
    return class_raster


def generate_slope_degrees_raster(dem: xdem.DEM) -> NDArray[np.floating]:
    """
    Estimate slope angle (degrees) from a DEM using finite differences.
    """
    try:
        dx, dy = dem.res
    except (AttributeError, TypeError, ValueError) as e:
        import warnings

        warnings.warn(f"Could not get DEM resolution, using 1.0: {e}")
        dx = dy = 1.0

    dx = float(abs(dx))
    dy = float(abs(dy))

    # Convert masked array to regular array with NaN
    data = dem.data
    if np.ma.isMaskedArray(data):
        z = np.ma.filled(data.astype(float), np.nan)
    else:
        z = data.astype(float)

    if dem.nodata is not None:
        z[z == dem.nodata] = np.nan

    dzdy, dzdx = np.gradient(z, dy, dx)
    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    slope_deg = np.degrees(slope_rad)

    slope_deg[~np.isfinite(z)] = np.nan

    return slope_deg
