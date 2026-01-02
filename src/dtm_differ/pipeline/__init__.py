from __future__ import annotations

from pathlib import Path

from . import stages
from .types import Inputs, ProcessingConfig, ProcessingResult


def run_pipeline(
    a_path: Path,
    b_path: Path,
    out_dir: Path,
    config: ProcessingConfig | None = None,
) -> ProcessingResult:
    config = config or ProcessingConfig()
    ws = stages.make_workspace(out_dir)
    inputs = Inputs(a_path=a_path, b_path=b_path)

    stages.validate_inputs(inputs)
    a_info, a_dem, b_info, b_dem = stages.load_dems(inputs)
    stages.warn_if_threshold_units_may_be_wrong(a_info, b_info)
    stages.validate_data_quality(a_dem, b_dem)
    aligned = stages.align_dems(a_info, a_dem, b_info, b_dem, config)
    rasters = stages.compute_rasters(aligned, config)

    stages.save_geotiff_outputs(rasters, ws, generate_polygons=config.generate_polygons)
    stages.save_metrics(rasters, ws, config, a_info, b_info)

    if config.generate_report:
        try:
            stages.generate_report(rasters, ws, config, a_info, b_info)
        except Exception as e:
            import warnings
            warnings.warn(
                f"Report generation failed: {e}. Check matplotlib installation and dependencies.",
                UserWarning,
                stacklevel=2
            )
            print(f"⚠️  WARNING: Report generation failed: {e}")
            print("   Check matplotlib installation and dependencies.")

    return ProcessingResult(
        diff=rasters.diff,
        elevation_change=rasters.elevation_change,
        change_direction=rasters.change_direction,
        change_magnitude=rasters.change_magnitude,
        slope_deg=rasters.slope_deg,
        movement_rank=rasters.movement_rank,
        output_mask=rasters.output_mask,
        transform=rasters.diff.transform,
        crs=rasters.diff.crs,
        map_layers_dir=ws.map_layers_dir,
        reports_dir=ws.reports_dir,
        sigma_dh=rasters.sigma_dh,
        z_score=rasters.z_score,
        within_noise_mask=rasters.within_noise_mask,
    )
