from __future__ import annotations

import os
import sys
import warnings
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from dtm_differ.db import Database

from . import stages
from .types import Inputs, ProcessingConfig, ProcessingResult


def run_pipeline(
    db: Database,
    job_id: str,
    a_path: Path,
    b_path: Path,
    out_dir: Path,
    config: ProcessingConfig | None = None,
    *,
    progress: bool | None = None,
    defer_output: bool | None = None,
) -> ProcessingResult:
    config = config or ProcessingConfig()
    ws = stages.make_workspace(out_dir, job_id=job_id)
    inputs = Inputs(a_path=a_path, b_path=b_path)

    progress_enabled = False
    if progress is True:
        progress_enabled = True
    elif progress is None:
        progress_enabled = sys.stderr.isatty() and (
            "PYTEST_CURRENT_TEST" not in os.environ
        )

    if defer_output is True:
        defer_output_enabled = True
    elif defer_output is None:
        defer_output_enabled = progress_enabled
    else:
        defer_output_enabled = False

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    captured_stdout = StringIO()
    captured_stderr = StringIO()
    warnings_list: list[warnings.WarningMessage] | None = None
    deferred_output_text: str | None = None
    exc: BaseException | None = None

    pbar = None
    if progress_enabled:
        from tqdm.auto import tqdm

        # Stage-level progress; some stages (reprojection/polygonize/report) can take a long time.
        total_steps = 8
        pbar = tqdm(
            total=total_steps, desc="dtm-differ", unit="step", file=original_stderr
        )

    def step(label: str) -> None:
        if pbar is None:
            return
        pbar.set_description(label)
        pbar.update(1)

    try:
        if defer_output_enabled:
            with (
                warnings.catch_warnings(record=True) as w,
                redirect_stdout(captured_stdout),
                redirect_stderr(captured_stderr),
            ):
                warnings_list = w
                warnings.simplefilter("default")
                db.update_job_status(job_id, status="running")
                rasters = _run_pipeline_steps(
                    inputs,
                    ws,
                    config,
                    step=step,
                )
        else:
            with nullcontext():
                rasters = _run_pipeline_steps(
                    inputs,
                    ws,
                    config,
                    step=step,
                )
                db.update_job_status(job_id, status="running")
    except BaseException as e:
        exc = e
        db.update_job_status(job_id, status="failed")
        raise
    finally:
        if pbar is not None:
            pbar.close()
        if defer_output_enabled:
            # Safety: ensure stdout/stderr are restored even if a third-party library
            # mutates them during processing.
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            out = captured_stdout.getvalue().strip()
            err = captured_stderr.getvalue().strip()
            warn_lines = []
            if warnings_list is not None:
                for wm in warnings_list:
                    warn_lines.append(
                        warnings.formatwarning(
                            wm.message,
                            wm.category,
                            wm.filename,
                            wm.lineno,
                            line=wm.line,
                        ).rstrip()
                    )

            sections: list[str] = []
            if warn_lines:
                sections.append("Captured warnings:\n" + "\n".join(warn_lines))
            if out:
                sections.append("Captured stdout:\n" + out)
            if err:
                sections.append("Captured stderr:\n" + err)

            if sections:
                deferred_output_text = "\n\n".join(sections)
                if exc is not None:
                    print(deferred_output_text, file=original_stderr)

            if exc is None:
                db.update_job_status(job_id, status="completed")

    return ProcessingResult(
        job_id=job_id,
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
        deferred_output=deferred_output_text,
    )


def _run_pipeline_steps(
    inputs: Inputs,
    ws,
    config: ProcessingConfig,
    *,
    step,
):
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

    stages.save_geotiff_outputs(rasters, ws, generate_polygons=config.generate_polygons)
    step("Write GeoTIFF outputs")

    stages.save_metrics(rasters, ws, config, a_info, b_info)
    step("Write metrics")

    if config.generate_report:
        try:
            stages.generate_report(
                rasters, ws, config, a_info, b_info, job_id=ws.job_id
            )
        except Exception as e:
            warnings.warn(
                f"Report generation failed: {e}. Check matplotlib installation and dependencies.",
                UserWarning,
                stacklevel=2,
            )
        step("Write report")
    else:
        step("Skip report")

    return rasters
