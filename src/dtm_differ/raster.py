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


def generate_change_direction_from_dh(
    dh: NDArray[np.floating],
    *,
    output_mask: NDArray[np.bool_],
    sigma_dh: NDArray[np.floating] | None = None,
    k_sigma: float | None = None,
) -> NDArray[np.int8]:
    """
    Generate a direction raster (-1, 0, 1) from a dh array.

    If sigma is provided, only changes that are *detectable* at the chosen k threshold
    are assigned a direction; otherwise direction is 0.

    Args:
        dh: Elevation change array in meters (NaN for nodata).
        output_mask: Boolean mask where outputs should be treated as nodata.
        sigma_dh: Per-pixel 1σ uncertainty for dh (meters).
        k_sigma: Detectability multiplier k; changes are detectable if |dh| > k·σ.

    Returns:
        Direction array: -1 (negative), 0 (not detectable / no change), 1 (positive).
    """
    direction = np.zeros(dh.shape, dtype=np.int8)
    valid = (~output_mask) & np.isfinite(dh)

    if sigma_dh is not None and k_sigma is not None and float(k_sigma) > 0:
        sigma_ok = np.isfinite(sigma_dh) & (sigma_dh > 0)
        detectable = valid & sigma_ok & (np.abs(dh) > (float(k_sigma) * sigma_dh))
    else:
        detectable = valid & (dh != 0)

    direction[detectable & (dh > 0)] = 1
    direction[detectable & (dh < 0)] = -1
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

    Notes:
        This implementation is nodata-aware: it avoids differencing across NaN/nodata
        boundaries by falling back to forward/backward differences when needed.
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

    # Build 4-neighbour arrays with NaN padding at edges.
    left = np.empty_like(z)
    right = np.empty_like(z)
    up = np.empty_like(z)
    down = np.empty_like(z)

    left[:, 0] = np.nan
    left[:, 1:] = z[:, :-1]
    right[:, -1] = np.nan
    right[:, :-1] = z[:, 1:]
    up[0, :] = np.nan
    up[1:, :] = z[:-1, :]
    down[-1, :] = np.nan
    down[:-1, :] = z[1:, :]

    z_finite = np.isfinite(z)
    left_f = np.isfinite(left)
    right_f = np.isfinite(right)
    up_f = np.isfinite(up)
    down_f = np.isfinite(down)

    dzdx = np.full(z.shape, np.nan, dtype=np.float32)
    dzdy = np.full(z.shape, np.nan, dtype=np.float32)

    # Central difference when both neighbours exist; otherwise fallback to forward/backward.
    central_x = left_f & right_f
    forward_x = (~left_f) & right_f & z_finite
    backward_x = left_f & (~right_f) & z_finite

    dzdx[central_x] = ((right - left) / (2.0 * dx))[central_x].astype(
        np.float32, copy=False
    )
    dzdx[forward_x] = ((right - z) / dx)[forward_x].astype(np.float32, copy=False)
    dzdx[backward_x] = ((z - left) / dx)[backward_x].astype(np.float32, copy=False)

    central_y = up_f & down_f
    forward_y = (~up_f) & down_f & z_finite
    backward_y = up_f & (~down_f) & z_finite

    dzdy[central_y] = ((down - up) / (2.0 * dy))[central_y].astype(
        np.float32, copy=False
    )
    dzdy[forward_y] = ((down - z) / dy)[forward_y].astype(np.float32, copy=False)
    dzdy[backward_y] = ((z - up) / dy)[backward_y].astype(np.float32, copy=False)

    slope_rad = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    slope_deg = np.degrees(slope_rad)

    slope_deg[~z_finite] = np.nan

    return slope_deg
