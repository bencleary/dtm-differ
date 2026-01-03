import numpy as np
import xdem
from numpy.typing import NDArray


def _to_float_array(data: NDArray, nodata: float | None = None) -> NDArray[np.floating]:
    """
    Convert a possibly-masked array to a plain float array with NaN for missing values.

    This centralizes the masked-array handling that was previously duplicated
    across several functions.
    """
    if np.ma.isMaskedArray(data):
        arr = np.ma.filled(data.astype(float), np.nan)
    else:
        arr = data.astype(float)

    if nodata is not None:
        arr[arr == nodata] = np.nan

    return arr


def generate_difference_raster(a: xdem.DEM, b: xdem.DEM) -> xdem.DEM:
    """
    Generate a difference raster between two DEMs.

    Returns:
        xdem.DEM: The difference raster
    """
    return a - b


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
    within[valid] = (np.abs(dh[valid]) <= (float(k_sigma) * sigma_dh[valid])).astype(np.uint8)
    return within


def generate_elevation_change_raster(diff: xdem.DEM) -> NDArray[np.floating]:
    """Extract elevation change as a float array (nodata becomes NaN)."""
    return _to_float_array(diff.data, nodata=diff.nodata)


def generate_change_direction_raster(diff: xdem.DEM) -> NDArray[np.int8]:
    """
    Classify each cell as uplift (+1), subsidence (-1), or no change (0).
    """
    direction = np.zeros_like(diff.data, dtype=np.int8)
    nodata_mask = np.zeros_like(diff.data, dtype=bool)
    if diff.nodata is not None:
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
    Direction from dh array, optionally filtered by detectability threshold.

    When sigma_dh and k_sigma are provided, only changes exceeding k·σ
    are assigned a direction; everything else is marked as 0.
    """
    direction = np.zeros(dh.shape, dtype=np.int8)
    valid = (~output_mask) & np.isfinite(dh)

    if sigma_dh is not None and k_sigma is not None and float(k_sigma) > 0:
        sigma_ok = np.isfinite(sigma_dh) & (sigma_dh > 0)
        detectable = valid & sigma_ok & (np.abs(dh) > float(k_sigma) * sigma_dh)
    else:
        detectable = valid & (dh != 0)

    direction[detectable & (dh > 0)] = 1
    direction[detectable & (dh < 0)] = -1
    return direction


def generate_change_magnitude_raster(diff: xdem.DEM) -> NDArray[np.floating]:
    """Absolute elevation change (nodata becomes NaN)."""
    return np.abs(_to_float_array(diff.data, nodata=diff.nodata))


def generate_ranked_movement_raster(
    movement_magnitude: NDArray[np.floating],
    *,
    t_green: float = 1.0,
    t_amber: float = 3.0,
    t_red: float = 6.0,
) -> NDArray[np.uint8]:
    """
    Classify magnitude into risk tiers: 0=below threshold, 1=green, 2=amber, 3=red.
    """
    if not (0 <= t_green <= t_amber <= t_red):
        raise ValueError("Thresholds must satisfy 0 <= t_green <= t_amber <= t_red")

    rank = np.zeros_like(movement_magnitude, dtype=np.uint8)
    rank[(movement_magnitude >= t_green) & (movement_magnitude < t_amber)] = 1
    rank[(movement_magnitude >= t_amber) & (movement_magnitude < t_red)] = 2
    rank[movement_magnitude >= t_red] = 3
    return rank


def generate_slope_degrees_raster(dem: xdem.DEM) -> NDArray[np.floating]:
    """
    Compute slope angle in degrees using finite differences.

    Uses central differences where possible, falling back to forward/backward
    differences at edges or near nodata cells.
    """
    try:
        dx, dy = dem.res
    except (AttributeError, TypeError, ValueError):
        dx = dy = 1.0

    dx, dy = float(abs(dx)), float(abs(dy))

    z = _to_float_array(dem.data, nodata=dem.nodata)

    # Build neighbor arrays (NaN-padded at edges)
    left = np.empty_like(z)
    left[:, 0] = np.nan
    left[:, 1:] = z[:, :-1]

    right = np.empty_like(z)
    right[:, -1] = np.nan
    right[:, :-1] = z[:, 1:]

    up = np.empty_like(z)
    up[0, :] = np.nan
    up[1:, :] = z[:-1, :]

    down = np.empty_like(z)
    down[-1, :] = np.nan
    down[:-1, :] = z[1:, :]

    z_ok = np.isfinite(z)
    l_ok, r_ok = np.isfinite(left), np.isfinite(right)
    u_ok, d_ok = np.isfinite(up), np.isfinite(down)

    dzdx = np.full(z.shape, np.nan, dtype=np.float32)
    dzdy = np.full(z.shape, np.nan, dtype=np.float32)

    # X gradient
    dzdx[l_ok & r_ok] = ((right - left) / (2 * dx))[l_ok & r_ok]
    dzdx[~l_ok & r_ok & z_ok] = ((right - z) / dx)[~l_ok & r_ok & z_ok]
    dzdx[l_ok & ~r_ok & z_ok] = ((z - left) / dx)[l_ok & ~r_ok & z_ok]

    # Y gradient
    dzdy[u_ok & d_ok] = ((down - up) / (2 * dy))[u_ok & d_ok]
    dzdy[~u_ok & d_ok & z_ok] = ((down - z) / dy)[~u_ok & d_ok & z_ok]
    dzdy[u_ok & ~d_ok & z_ok] = ((z - up) / dy)[u_ok & ~d_ok & z_ok]

    slope_deg = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))
    slope_deg[~z_ok] = np.nan
    return slope_deg
