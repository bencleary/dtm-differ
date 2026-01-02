from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xdem
from numpy.typing import NDArray

from dtm_differ.constants import (
    NODATA_DIRECTION,
    NODATA_FLOAT,
    NODATA_MASK,
    NODATA_RANK,
)
from dtm_differ.geotiff import (
    check_raster_compatability,
    get_geotiff_metadata,
    reproject_raster,
    validate_dem_data,
    validate_geotiff,
)
from dtm_differ.raster import (
    generate_change_direction_from_dh,
    generate_change_magnitude_raster,
    generate_difference_raster,
    generate_difference_sigma_constant,
    generate_elevation_change_raster,
    generate_ranked_movement_raster,
    generate_slope_degrees_raster,
    generate_within_noise_mask_u8,
    generate_z_score,
)
from dtm_differ.viz import (
    ReportImage,
    save_confidence_weighted_magnitude_png,
    save_direction_png,
    save_elevation_change_diverging_png,
    save_movement_magnitude_viridis_png,
    save_rank_png,
    save_slope_png,
    save_within_noise_mask_png,
    save_z_score_diverging_png,
    write_html_report,
)

from .types import AlignedDems, DerivedRasters, Inputs, ProcessingConfig, Workspace


def _finite(values: NDArray[np.floating]) -> NDArray[np.floating]:
    arr = np.asarray(values, dtype=np.float32)
    return arr[np.isfinite(arr)]


def _compute_dh_metrics(
    dh: NDArray[np.floating], *, valid_mask: NDArray[np.bool_]
) -> dict[str, float | int]:
    total_count = int(valid_mask.size)
    valid_count = int(np.sum(valid_mask))
    valid_fraction = (valid_count / total_count) if total_count else 0.0

    finite_dh = _finite(dh[valid_mask])
    metrics: dict[str, float | int] = {
        "n_total": total_count,
        "n_valid": valid_count,
        "valid_fraction": float(valid_fraction),
    }

    if finite_dh.size == 0:
        return metrics

    abs_dh = np.abs(finite_dh)
    median = float(np.median(finite_dh))
    mad = float(np.median(np.abs(finite_dh - median)))

    metrics.update(
        {
            "mean_m": float(np.mean(finite_dh)),
            "median_m": median,
            "std_m": float(np.std(finite_dh)),
            "rmse_m": float(np.sqrt(np.mean(finite_dh**2))),
            "mae_m": float(np.mean(abs_dh)),
            "nmad_m": float(1.4826 * mad),
            "min_m": float(np.min(finite_dh)),
            "max_m": float(np.max(finite_dh)),
            "p25_m": float(np.percentile(finite_dh, 25)),
            "p50_m": float(np.percentile(finite_dh, 50)),
            "p75_m": float(np.percentile(finite_dh, 75)),
            "p95_m": float(np.percentile(finite_dh, 95)),
            "p99_m": float(np.percentile(finite_dh, 99)),
        }
    )
    return metrics


def _save_metrics_json(out_path: Path, payload: dict) -> None:
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

def _build_metrics_payload(
    r: DerivedRasters,
    config: ProcessingConfig,
    *,
    a_info=None,
    b_info=None,
) -> dict:
    valid_mask = ~r.output_mask
    dh_metrics = _compute_dh_metrics(r.elevation_change, valid_mask=valid_mask)

    coreg_metrics: dict[str, float | int] | None = None
    if r.reprojection_info.occurred:
        coreg_metrics = _estimate_planar_trend_metrics(
            r.elevation_change, valid_mask=valid_mask, transform=r.diff.transform
        )

    return {
        "dh_metrics": dh_metrics,
        "coregistration_check": coreg_metrics,
        "thresholds_m": {
            "green": config.t_green,
            "amber": config.t_amber,
            "red": config.t_red,
        },
        "uncertainty": {
            "mode": config.uncertainty_mode,
            "sigma_a_m": config.sigma_a,
            "sigma_b_m": config.sigma_b,
            "sigma_coreg_m": config.sigma_coreg,
            "k_sigma": config.k_sigma,
            "suppress_within_noise_rank": config.suppress_within_noise_rank,
        },
        "alignment": {
            "occurred": r.reprojection_info.occurred,
            "align": config.align,
            "resample": config.resample,
            "reason": r.reprojection_info.reason,
            "source_crs_a": r.reprojection_info.source_crs_a,
            "source_crs_b": r.reprojection_info.source_crs_b,
            "target_crs": r.reprojection_info.target_crs,
            "reference_grid": r.reprojection_info.reference_grid,
        },
        "inputs": {
            "a_name": getattr(getattr(a_info, "path", None), "name", None),
            "b_name": getattr(getattr(b_info, "path", None), "name", None),
            "a_crs": getattr(a_info, "crs", None),
            "b_crs": getattr(b_info, "crs", None),
            "a_linear_unit_name": getattr(a_info, "linear_unit_name", None),
            "b_linear_unit_name": getattr(b_info, "linear_unit_name", None),
            "a_linear_unit_factor": getattr(a_info, "linear_unit_factor", None),
            "b_linear_unit_factor": getattr(b_info, "linear_unit_factor", None),
        },
    }


def save_metrics(
    r: DerivedRasters, ws: Workspace, config: ProcessingConfig, a_info=None, b_info=None
) -> dict:
    """
    Write machine-readable processing metrics to `reports/metrics.json`.

    This is created even when `generate_report=False` so downstream workflows can
    consume QA metrics without parsing HTML.
    """
    payload = _build_metrics_payload(r, config, a_info=a_info, b_info=b_info)
    _save_metrics_json(ws.reports_dir / "metrics.json", payload)
    return payload


def warn_if_threshold_units_may_be_wrong(a_info, b_info) -> None:
    """
    Best-effort warning about CRS linear units vs. meter-based threshold assumptions.

    Notes:
        GeoTIFFs rarely declare vertical units explicitly. We therefore check CRS *linear*
        units as a proxy signal and warn when they are clearly non-metric (e.g., feet or degrees).
    """

    def describe(info) -> str:
        unit = getattr(info, "linear_unit_name", None)
        factor = getattr(info, "linear_unit_factor", None)
        if unit is None:
            return "unknown"
        if factor is None:
            return str(unit)
        return f"{unit} (factor {factor:g})"

    def is_meter_like(info) -> bool:
        unit = getattr(info, "linear_unit_name", None)
        factor = getattr(info, "linear_unit_factor", None)
        if unit is None:
            return False
        unit_l = str(unit).lower()
        if unit_l in {"metre", "meter", "metres", "meters", "m"}:
            return True
        if factor is not None and abs(float(factor) - 1.0) < 1e-6:
            return True
        return False

    def is_degree_like(info) -> bool:
        unit = getattr(info, "linear_unit_name", None)
        if unit is None:
            return False
        return "degree" in str(unit).lower()

    if is_degree_like(a_info) or is_degree_like(b_info):
        print("⚠️  WARNING: One or both DEM CRSs use degree-based units.")
        print(f"   DEM A CRS units: {describe(a_info)}")
        print(f"   DEM B CRS units: {describe(b_info)}")
        print(
            "   Thresholds are specified in meters and slope/distance calculations assume linear units; "
            "consider reprojecting to a projected CRS (meters) and verifying vertical units."
        )
        return

    if not (is_meter_like(a_info) and is_meter_like(b_info)):
        print("⚠️  WARNING: Could not confirm both DEM CRSs use meter-based units.")
        print(f"   DEM A CRS units: {describe(a_info)}")
        print(f"   DEM B CRS units: {describe(b_info)}")
        print(
            "   Thresholds are specified in meters. GeoTIFF CRS units are a proxy for horizontal units; "
            "vertical units may still differ. Verify your DEM vertical units before interpreting thresholds."
        )


def _estimate_planar_trend_metrics(
    dh: NDArray[np.floating], *, valid_mask: NDArray[np.bool_], transform
) -> dict[str, float | int] | None:
    """
    Estimate a best-fit plane (a*x + b*y + c) through dh for co-registration diagnostics.

    Returns:
        A dict of plane coefficients and residual metrics, or None if insufficient data.
    """
    valid = valid_mask & np.isfinite(dh)
    flat = np.flatnonzero(valid)
    if flat.size < 3:
        return None

    max_samples = 200_000
    step = max(1, int(flat.size // max_samples))
    flat = flat[::step]
    rows, cols = np.unravel_index(flat, dh.shape)

    a = float(getattr(transform, "a"))
    b = float(getattr(transform, "b"))
    c = float(getattr(transform, "c"))
    d = float(getattr(transform, "d"))
    e = float(getattr(transform, "e"))
    f = float(getattr(transform, "f"))

    x = (a * cols + b * rows + c).astype(np.float64, copy=False)
    y = (d * cols + e * rows + f).astype(np.float64, copy=False)
    z = np.asarray(dh[rows, cols], dtype=np.float64)

    design = np.column_stack([x, y, np.ones_like(x)])
    coeffs, *_ = np.linalg.lstsq(design, z, rcond=None)
    plane = design @ coeffs
    residual = z - plane

    ax, by, intercept = (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]))
    tilt = float(np.sqrt(ax**2 + by**2))
    tilt_angle_deg = float(np.degrees(np.arctan(tilt)))

    return {
        "n_samples": int(z.size),
        "plane_ax_m_per_unit": ax,
        "plane_by_m_per_unit": by,
        "plane_intercept_m": intercept,
        "tilt_m_per_unit": tilt,
        "tilt_angle_deg": tilt_angle_deg,
        "residual_mean_m": float(np.mean(residual)),
        "residual_rmse_m": float(np.sqrt(np.mean(residual**2))),
    }


def make_workspace(out_dir: Path) -> Workspace:
    out_dir.mkdir(parents=True, exist_ok=True)
    map_layers_dir = out_dir / "map_layers"
    reports_dir = out_dir / "reports"
    map_layers_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    return Workspace(
        out_dir=out_dir, map_layers_dir=map_layers_dir, reports_dir=reports_dir
    )


def validate_inputs(inputs: Inputs) -> None:
    validate_geotiff(str(inputs.a_path))
    validate_geotiff(str(inputs.b_path))


def load_dems(inputs: Inputs):
    a_info, a_dem = get_geotiff_metadata(str(inputs.a_path))
    b_info, b_dem = get_geotiff_metadata(str(inputs.b_path))
    return a_info, a_dem, b_info, b_dem


def validate_data_quality(a_dem: xdem.DEM, b_dem: xdem.DEM) -> None:
    a_valid, a_msg = validate_dem_data(a_dem)
    b_valid, b_msg = validate_dem_data(b_dem)
    if not a_valid:
        raise ValueError(f"DEM A has insufficient valid data: {a_msg}")
    if not b_valid:
        raise ValueError(f"DEM B has insufficient valid data: {b_msg}")


def align_dems(
    a_info,
    a_dem: xdem.DEM,
    b_info,
    b_dem: xdem.DEM,
    config: ProcessingConfig,
) -> AlignedDems:
    from .types import ReprojectionInfo
    
    compat = check_raster_compatability(a_info, b_info)
    reprojection_info = ReprojectionInfo(
        occurred=False,
        source_crs_a=a_info.crs,
        source_crs_b=b_info.crs,
        target_crs=None,
        resampling_method=None,
        reference_grid=None,
        reason=None,
    )
    
    if not compat.same_crs or not compat.same_grid or not compat.overlaps:
        if config.align == "to-a":
            target_crs = a_info.crs
            print(f"Reprojecting DEM B to match DEM A:")
            print(f"  Source CRS (B): {b_info.crs}")
            print(f"  Target CRS (A): {a_info.crs}")
            print(f"  Resampling method: {config.resample}")
            b_dem = reproject_raster("to-a", a_dem, b_dem, resampling=config.resample)
            reprojection_info = ReprojectionInfo(
                occurred=True,
                source_crs_a=a_info.crs,
                source_crs_b=b_info.crs,
                target_crs=target_crs,
                resampling_method=config.resample,
                reference_grid="to-a",
                reason=compat.reason,
            )
        elif config.align == "to-b":
            target_crs = b_info.crs
            print(f"Reprojecting DEM A to match DEM B:")
            print(f"  Source CRS (A): {a_info.crs}")
            print(f"  Target CRS (B): {b_info.crs}")
            print(f"  Resampling method: {config.resample}")
            a_dem = reproject_raster("to-b", a_dem, b_dem, resampling=config.resample)
            reprojection_info = ReprojectionInfo(
                occurred=True,
                source_crs_a=a_info.crs,
                source_crs_b=b_info.crs,
                target_crs=target_crs,
                resampling_method=config.resample,
                reference_grid="to-b",
                reason=compat.reason,
            )
        else:
            raise ValueError(f"Invalid alignment: {config.align}")
    else:
        print("DEMs are compatible - no reprojection needed.")
    
    return AlignedDems(a_dem=a_dem, b_dem=b_dem, reprojection_info=reprojection_info)


def compute_rasters(aligned: AlignedDems, config: ProcessingConfig) -> DerivedRasters:
    diff = generate_difference_raster(aligned.a_dem, aligned.b_dem)

    data = diff.data
    if np.ma.isMaskedArray(data):
        output_mask = np.ma.getmaskarray(data).astype(bool, copy=True)
    else:
        output_mask = np.zeros_like(data, dtype=bool)

    if diff.nodata is not None:
        output_mask |= np.asarray(data) == diff.nodata

    elevation_change = generate_elevation_change_raster(diff)
    elevation_change[output_mask] = np.nan

    sigma_dh = None
    z_score = None
    within_noise_mask = None

    if config.uncertainty_mode == "constant":
        sigma_dh = generate_difference_sigma_constant(
            elevation_change.shape,
            output_mask=output_mask,
            sigma_a=config.sigma_a,
            sigma_b=config.sigma_b,
            sigma_coreg=config.sigma_coreg,
        )
        z_score = generate_z_score(elevation_change, sigma_dh, output_mask=output_mask)
        within_noise_mask = generate_within_noise_mask_u8(
            elevation_change, sigma_dh, output_mask=output_mask, k_sigma=config.k_sigma
        )

    change_direction = generate_change_direction_from_dh(
        elevation_change,
        output_mask=output_mask,
        sigma_dh=sigma_dh if config.uncertainty_mode != "none" else None,
        k_sigma=config.k_sigma if config.uncertainty_mode != "none" else None,
    )

    change_magnitude = generate_change_magnitude_raster(diff)
    change_magnitude[output_mask] = np.nan

    slope_deg = generate_slope_degrees_raster(aligned.a_dem)
    slope_deg[output_mask] = np.nan

    movement_rank = generate_ranked_movement_raster(
        change_magnitude,
        t_green=config.t_green,
        t_amber=config.t_amber,
        t_red=config.t_red,
    )
    movement_rank[output_mask] = 0

    if (
        config.uncertainty_mode != "none"
        and config.suppress_within_noise_rank
        and within_noise_mask is not None
    ):
        movement_rank[within_noise_mask == 1] = 0

    return DerivedRasters(
        diff=diff,
        elevation_change=elevation_change,
        change_direction=change_direction,
        change_magnitude=change_magnitude,
        slope_deg=slope_deg,
        movement_rank=movement_rank,
        output_mask=output_mask,
        sigma_dh=sigma_dh,
        z_score=z_score,
        within_noise_mask=within_noise_mask,
        reprojection_info=aligned.reprojection_info,
    )


def save_geotiff_outputs(
    r: DerivedRasters, ws: Workspace, *, generate_polygons: bool
) -> None:
    r.diff.save(ws.map_layers_dir / "diff.tif")

    def save_array(
        array: NDArray,
        *,
        filename: str,
        nodata: float | int,
        dtype: np.dtype,
        fill_nonfinite: bool = False,
    ) -> None:
        filled = np.asarray(array).copy()
        mask = r.output_mask
        if fill_nonfinite:
            mask = np.logical_or(mask, ~np.isfinite(filled))
        filled[mask] = nodata

        # Convert nodata values to NaN before passing to xdem to avoid warnings
        # xdem expects NaN for nodata, not the nodata value itself
        # For integer types, we need to convert to float first to allow NaN
        prepared = (
            filled.astype(np.float32)
            if np.issubdtype(dtype, np.integer)
            else filled.astype(dtype)
        )
        if nodata is not None:
            # Use np.isclose for floating point comparison
            nodata_mask = np.isclose(prepared, nodata, equal_nan=False)
            prepared[nodata_mask] = np.nan

        xdem.DEM.from_array(
            prepared,
            transform=r.diff.transform,
            crs=r.diff.crs,
            nodata=nodata,
        ).save(ws.map_layers_dir / filename)

    save_array(
        r.elevation_change,
        filename="elevation_change.tif",
        nodata=NODATA_FLOAT,
        dtype=np.float32,
        fill_nonfinite=True,
    )
    save_array(
        r.change_magnitude,
        filename="change_magnitude.tif",
        nodata=NODATA_FLOAT,
        dtype=np.float32,
        fill_nonfinite=True,
    )
    save_array(
        r.change_direction,
        filename="change_direction.tif",
        nodata=NODATA_DIRECTION,
        dtype=np.int16,
        fill_nonfinite=False,
    )
    save_array(
        r.slope_deg,
        filename="slope_degrees.tif",
        nodata=NODATA_FLOAT,
        dtype=np.float32,
        fill_nonfinite=True,
    )
    save_array(
        r.movement_rank,
        filename="movement_rank.tif",
        nodata=NODATA_RANK,
        dtype=np.uint8,
        fill_nonfinite=False,
    )

    if r.sigma_dh is not None:
        save_array(
            r.sigma_dh,
            filename="sigma_dh.tif",
            nodata=NODATA_FLOAT,
            dtype=np.float32,
            fill_nonfinite=True,
        )

    if r.z_score is not None:
        save_array(
            r.z_score,
            filename="z_score.tif",
            nodata=NODATA_FLOAT,
            dtype=np.float32,
            fill_nonfinite=True,
        )

    if r.within_noise_mask is not None:
        save_array(
            r.within_noise_mask,
            filename="within_noise_mask.tif",
            nodata=NODATA_MASK,
            dtype=np.uint8,
            fill_nonfinite=False,
        )

    if generate_polygons:
        try:
            # Convert nodata to NaN before passing to xdem to avoid warnings
            rank_data = r.movement_rank.astype(np.float32)
            if NODATA_RANK is not None:
                nodata_mask = np.isclose(rank_data, NODATA_RANK, equal_nan=False)
                rank_data[nodata_mask] = np.nan

            polys = xdem.DEM.from_array(
                rank_data,
                transform=r.diff.transform,
                crs=r.diff.crs,
                nodata=NODATA_RANK,
            ).polygonize(target_values=[1, 2, 3], data_column_name="class")
            polys.save(ws.map_layers_dir / "movement_ranked_polygons.shp")
        except Exception:
            pass


def generate_report(
    r: DerivedRasters, ws: Workspace, config: ProcessingConfig, a_info=None, b_info=None
) -> None:
    images: list[ReportImage] = []
    summary_html: str | None = None
    processing_html: str | None = None

    # Build processing metadata HTML
    processing_items = []

    metrics_payload = _build_metrics_payload(r, config, a_info=a_info, b_info=b_info)
    dh_metrics = metrics_payload["dh_metrics"]
    coreg_metrics = metrics_payload["coregistration_check"]
    
    # Input file information
    if a_info and b_info:
        processing_items.append(f"<li><strong>Input DEM A:</strong> {a_info.path.name} (CRS: {a_info.crs})</li>")
        processing_items.append(f"<li><strong>Input DEM B:</strong> {b_info.path.name} (CRS: {b_info.crs})</li>")
        if getattr(a_info, "linear_unit_name", None):
            processing_items.append(
                f"<li><strong>DEM A linear units:</strong> {a_info.linear_unit_name}</li>"
            )
        if getattr(b_info, "linear_unit_name", None):
            processing_items.append(
                f"<li><strong>DEM B linear units:</strong> {b_info.linear_unit_name}</li>"
            )
    
    # Output CRS
    if r.diff.crs:
        output_crs = str(r.diff.crs) if hasattr(r.diff.crs, '__str__') else repr(r.diff.crs)
        processing_items.append(f"<li><strong>Output CRS:</strong> {output_crs}</li>")
    
    # Reprojection information
    if r.reprojection_info.occurred:
        processing_items.append("<li><strong>Reprojection:</strong> Yes</li>")
        if r.reprojection_info.source_crs_a:
            processing_items.append(f"<li><strong>Source CRS (A):</strong> {r.reprojection_info.source_crs_a}</li>")
        if r.reprojection_info.source_crs_b:
            processing_items.append(f"<li><strong>Source CRS (B):</strong> {r.reprojection_info.source_crs_b}</li>")
        if r.reprojection_info.target_crs:
            processing_items.append(f"<li><strong>Target CRS:</strong> {r.reprojection_info.target_crs}</li>")
        if r.reprojection_info.resampling_method:
            processing_items.append(f"<li><strong>Resampling method:</strong> {r.reprojection_info.resampling_method}</li>")
        if r.reprojection_info.reference_grid:
            processing_items.append(f"<li><strong>Reference grid:</strong> {r.reprojection_info.reference_grid}</li>")
        if r.reprojection_info.reason:
            processing_items.append(f"<li><strong>Reason:</strong> {r.reprojection_info.reason}</li>")
    else:
        processing_items.append("<li><strong>Reprojection:</strong> No (DEMs were compatible)</li>")
    
    # Processing configuration
    processing_items.append(f"<li><strong>Resampling method:</strong> {config.resample}</li>")
    processing_items.append(f"<li><strong>Alignment reference:</strong> {config.align}</li>")
    processing_items.append(f"<li><strong>Movement thresholds:</strong> Green={config.t_green:.1f}m, Amber={config.t_amber:.1f}m, Red={config.t_red:.1f}m</li>")
    
    # Uncertainty configuration
    if config.uncertainty_mode == "constant":
        processing_items.append(f"<li><strong>Uncertainty mode:</strong> Constant</li>")
        processing_items.append(f"<li><strong>σ<sub>A</sub>:</strong> {config.sigma_a:.3f} m</li>")
        processing_items.append(f"<li><strong>σ<sub>B</sub>:</strong> {config.sigma_b:.3f} m</li>")
        processing_items.append(f"<li><strong>σ<sub>coreg</sub>:</strong> {config.sigma_coreg:.3f} m</li>")
        processing_items.append(f"<li><strong>k (significance threshold):</strong> {config.k_sigma:g}</li>")
        processing_items.append(f"<li><strong>Suppress within-noise ranks:</strong> {'Yes' if config.suppress_within_noise_rank else 'No'}</li>")
    else:
        processing_items.append("<li><strong>Uncertainty mode:</strong> None ⚠️</li>")
        processing_items.append("<li><em>Uncertainty analysis not performed. Results may include noise-level changes.</em></li>")
    
    # Add statistical summary
    valid_mask = ~r.output_mask
    valid_pct = float(dh_metrics["valid_fraction"]) * 100.0
    processing_items.append(
        f"<li><strong>Valid pixels:</strong> {valid_pct:.1f}% ({dh_metrics['n_valid']:,} of {dh_metrics['n_total']:,})</li>"
    )

    if coreg_metrics is not None:
        processing_items.append(
            "<li><strong>Co-registration check (planar trend in dh):</strong></li>"
        )
        processing_items.append(
            "<li style='margin-left: 20px;'>"
            f"Samples: {coreg_metrics['n_samples']:,}, "
            f"Residual RMSE: {coreg_metrics['residual_rmse_m']:.3f} m"
            "</li>"
        )
        processing_items.append(
            "<li style='margin-left: 20px;'>"
            f"Tilt magnitude: {coreg_metrics['tilt_m_per_unit']:.3e} m/unit "
            f"(tilt angle ~ {coreg_metrics['tilt_angle_deg']:.4f}°)"
            "</li>"
        )

    if "mean_m" in dh_metrics:
        processing_items.append("<li><strong>Elevation change statistics (dh):</strong></li>")
        processing_items.append(
            f"<li style='margin-left: 20px;'>Mean (bias): {dh_metrics['mean_m']:+.3f} m</li>"
        )
        processing_items.append(
            f"<li style='margin-left: 20px;'>Median: {dh_metrics['median_m']:+.3f} m</li>"
        )
        processing_items.append(
            f"<li style='margin-left: 20px;'>Std dev: {dh_metrics['std_m']:.3f} m</li>"
        )
        processing_items.append(
            f"<li style='margin-left: 20px;'>RMSE: {dh_metrics['rmse_m']:.3f} m</li>"
        )
        processing_items.append(
            f"<li style='margin-left: 20px;'>MAE: {dh_metrics['mae_m']:.3f} m</li>"
        )
        processing_items.append(
            f"<li style='margin-left: 20px;'>NMAD: {dh_metrics['nmad_m']:.3f} m</li>"
        )
        processing_items.append(
            "<li style='margin-left: 20px;'>"
            f"Percentiles: 25th={dh_metrics['p25_m']:+.3f}, 50th={dh_metrics['p50_m']:+.3f}, "
            f"75th={dh_metrics['p75_m']:+.3f}, 95th={dh_metrics['p95_m']:+.3f}, 99th={dh_metrics['p99_m']:+.3f} m"
            "</li>"
        )
    
    # Movement rank distribution
    valid_rank = r.movement_rank[valid_mask]
    rank_0 = int(np.sum(valid_rank == 0))
    rank_1 = int(np.sum(valid_rank == 1))
    rank_2 = int(np.sum(valid_rank == 2))
    rank_3 = int(np.sum(valid_rank == 3))
    if int(dh_metrics["n_valid"]) > 0:
        processing_items.append("<li><strong>Movement rank distribution:</strong></li>")
        valid_n = int(dh_metrics["n_valid"])
        processing_items.append(
            f"<li style='margin-left: 20px;'>Unclassified (0): {rank_0:,} ({rank_0/valid_n*100:.1f}%)</li>"
        )
        processing_items.append(
            f"<li style='margin-left: 20px;'>Green (1): {rank_1:,} ({rank_1/valid_n*100:.1f}%)</li>"
        )
        processing_items.append(
            f"<li style='margin-left: 20px;'>Amber (2): {rank_2:,} ({rank_2/valid_n*100:.1f}%)</li>"
        )
        processing_items.append(
            f"<li style='margin-left: 20px;'>Red (3): {rank_3:,} ({rank_3/valid_n*100:.1f}%)</li>"
        )
    
    if processing_items:
        processing_html = f"<ul>{''.join(processing_items)}</ul>"

    movement_png = "movement_magnitude_green_to_red.png"
    save_movement_magnitude_viridis_png(
        r.change_magnitude,
        nodata_mask=r.output_mask,
        out_path=ws.reports_dir / movement_png,
    )
    images.append(ReportImage(movement_png, "Movement magnitude (Green → Red)"))

    dh_png = "elevation_change_diverging.png"
    save_elevation_change_diverging_png(
        r.elevation_change,
        nodata_mask=r.output_mask,
        out_path=ws.reports_dir / dh_png,
    )
    images.append(ReportImage(dh_png, "Elevation change (diverging)"))

    slope_png = "slope_degrees.png"
    save_slope_png(
        r.slope_deg, nodata_mask=r.output_mask, out_path=ws.reports_dir / slope_png
    )
    images.append(ReportImage(slope_png, "Slope (degrees)"))

    dir_png = "movement_direction.png"
    zero_label = "No change (dh = 0)"
    if config.uncertainty_mode != "none":
        zero_label = f"Not detectable (|z| < {config.k_sigma:g})"
    save_direction_png(
        r.change_direction,
        nodata_mask=r.output_mask,
        out_path=ws.reports_dir / dir_png,
        zero_label=zero_label,
    )
    images.append(ReportImage(dir_png, "Direction of movement"))

    rank_png = "movement_rank.png"
    save_rank_png(
        r.movement_rank,
        nodata_mask=r.output_mask,
        out_path=ws.reports_dir / rank_png,
        t_green=config.t_green,
        t_amber=config.t_amber,
        t_red=config.t_red,
    )
    images.append(ReportImage(rank_png, "Ranked movement (G/A/R)"))

    if (
        r.z_score is not None
        and r.within_noise_mask is not None
        and r.sigma_dh is not None
        and config.uncertainty_mode != "none"
    ):
        valid = (~r.output_mask) & np.isfinite(r.z_score)
        if np.any(valid):
            within_noise = r.within_noise_mask == 1
            within_noise_pct = float(np.mean(within_noise[valid]) * 100.0)

            absz = np.abs(r.z_score)
            detectable_pct = float(np.mean(absz[valid] >= config.k_sigma) * 100.0)
            high_pct = float(np.mean(absz[valid] >= 3.0) * 100.0)
            sigma_med = float(np.nanmedian(r.sigma_dh))

            summary_html = f"""
            <ul>
              <li>Assumed σ<sub>dh</sub> (median): {sigma_med:.3f} m</li>
              <li>Detectable if |z| ≥ {config.k_sigma:g}</li>
              <li>Within noise: {within_noise_pct:.1f}%</li>
              <li>Detectable: {detectable_pct:.1f}%</li>
              <li>High confidence (|z| ≥ 3): {high_pct:.1f}%</li>
            </ul>
            """.strip()

        conf_mag_png = "movement_magnitude_confidence_weighted.png"
        save_confidence_weighted_magnitude_png(
            r.change_magnitude,
            r.z_score,
            nodata_mask=r.output_mask,
            out_path=ws.reports_dir / conf_mag_png,
        )
        images.append(
            ReportImage(
                conf_mag_png,
                "Movement magnitude (confidence-weighted)",
                "Same magnitude map, but low |z| areas are faded (less reliable).",
            )
        )

        z_png = "z_score.png"
        save_z_score_diverging_png(
            r.z_score,
            nodata_mask=r.output_mask,
            out_path=ws.reports_dir / z_png,
            k_sigma=config.k_sigma,
        )
        images.append(
            ReportImage(
                z_png,
                "Significance (z-score)",
                f"z = dh/σ. Detectable changes satisfy |z| ≥ {config.k_sigma:g}.",
            )
        )

        wn_png = "within_noise_mask.png"
        save_within_noise_mask_png(
            r.within_noise_mask,
            nodata_mask=r.output_mask,
            out_path=ws.reports_dir / wn_png,
            k_sigma=config.k_sigma,
        )
        images.append(
            ReportImage(
                wn_png,
                "Within noise mask",
                "Grey pixels indicate changes that are not detectable at the chosen k threshold.",
            )
        )

    write_html_report(
        ws.reports_dir,
        title="dtm-differ report",
        images=images,
        map_layers_dir=ws.map_layers_dir,
        summary_html=summary_html,
        processing_html=processing_html,
    )
