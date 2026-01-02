from __future__ import annotations

import os
import sys
from pathlib import Path

from . import stages
from .types import Inputs, ProcessingConfig, ProcessingResult


def run_pipeline(
    a_path: Path,
    b_path: Path,
    out_dir: Path,
    config: ProcessingConfig | None = None,
    *,
    progress: bool | None = None,
) -> ProcessingResult:
    config = config or ProcessingConfig()
    ws = stages.make_workspace(out_dir)
    inputs = Inputs(a_path=a_path, b_path=b_path)

    progress_enabled = False
    if progress is True:
        progress_enabled = True
    elif progress is None:
        progress_enabled = sys.stderr.isatty() and ("PYTEST_CURRENT_TEST" not in os.environ)

    pbar = None
    if progress_enabled:
        from tqdm.auto import tqdm

        # Stage-level progress; some stages (reprojection/polygonize/report) can take a long time.
        total_steps = 8
        pbar = tqdm(total=total_steps, desc="dtm-differ", unit="step")

    def step(label: str) -> None:
        if pbar is None:
            return
        pbar.set_description(label)
        pbar.update(1)

    try:
        stages.validate_inputs(inputs)
        step("Validate inputs")

        a_info, a_dem, b_info, b_dem = stages.load_dems(inputs)
        step("Load DEMs")

        stages.warn_if_threshold_units_may_be_wrong(a_info, b_info)
        stages.validate_data_quality(a_dem, b_dem)
        step("Validate data quality")

        aligned = stages.align_dems(a_info, a_dem, b_info, b_dem, config)
        step("Align DEMs")

        rasters = stages.compute_rasters(aligned, config)
        step("Compute rasters")

        stages.save_geotiff_outputs(
            rasters, ws, generate_polygons=config.generate_polygons
        )
        step("Write GeoTIFF outputs")

        stages.save_metrics(rasters, ws, config, a_info, b_info)
        step("Write metrics")

        if config.generate_report:
            try:
                stages.generate_report(rasters, ws, config, a_info, b_info)
            except Exception as e:
                import warnings

                warnings.warn(
                    f"Report generation failed: {e}. Check matplotlib installation and dependencies.",
                    UserWarning,
                    stacklevel=2,
                )
                print(f"⚠️  WARNING: Report generation failed: {e}")
                print("   Check matplotlib installation and dependencies.")
            step("Write report")
        else:
            step("Skip report")
    finally:
        if pbar is not None:
            pbar.close()

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
