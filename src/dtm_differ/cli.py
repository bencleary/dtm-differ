from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from dtm_differ.geotiff import (
    check_raster_compatability,
    get_geotiff_metadata,
    reproject_raster,
    validate_geotiff,
    validate_dem_data,
)
from dtm_differ.raster import (
    generate_change_direction_raster,
    generate_change_magnitude_raster,
    generate_difference_raster,
    generate_elevation_change_raster,
    generate_ranked_movement_raster,
    generate_slope_degrees_raster,
)
from dtm_differ.viz import (
    ReportImage,
    save_direction_png,
    save_elevation_change_diverging_png,
    save_movement_magnitude_viridis_png,
    save_rank_png,
    save_slope_png,
    write_html_report,
)

from dtm_differ.constants import NODATA_FLOAT, NODATA_DIRECTION, NODATA_RANK

import xdem

def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="dtm-differ",
        description="Simple DTM differencing tool for GeoTIFFs"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run a DTM differencing job")
    run.add_argument("--a", required=True, help="Path to DTM A (before)")
    run.add_argument("--b", required=True, help="Path to DTM B (after)")
    run.add_argument("--out", required=True, help="Output directory")

    run.add_argument("--resample", choices=["nearest", "bilinear"], default="bilinear")
    run.add_argument("--align", choices=["to-a", "to-b"], default="to-a")
    run.add_argument("--thresholds", help="Thresholds in metres: green,amber,red (or amber,red)")
    run.add_argument("--style", choices=["diverging", "terrain", "greyscale"], default="diverging")

    status = subparsers.add_parser("status", help="Check job status")
    status.add_argument("job_id")

    return parser

def main():
    parser = build_parser()
    args = parser.parse_args()

    def parse_thresholds(raw: str | None) -> tuple[float, float, float]:
        if raw is None:
            return 1.0, 3.0, 6.0
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        values = [float(p) for p in parts]
        if len(values) == 2:
            return 1.0, values[0], values[1]
        if len(values) == 3:
            return values[0], values[1], values[2]
        raise ValueError("Invalid --thresholds; expected 'amber,red' or 'green,amber,red'")


    match args.command:
        case "run":

            a_path = Path(args.a)
            b_path = Path(args.b)
            out_dir = Path(args.out)
            out_dir.mkdir(parents=True, exist_ok=True)
            map_layers_dir = out_dir / "map_layers"
            reports_dir = out_dir / "reports"
            map_layers_dir.mkdir(parents=True, exist_ok=True)
            reports_dir.mkdir(parents=True, exist_ok=True)

            # Dev convenience: `make run` passes `example.tif`, so fall back to xdem's bundled examples
            if not a_path.exists() and not b_path.exists() and args.a == "example.tif" and args.b == "example.tif":
                a_path = Path(xdem.examples.get_path("longyearbyen_ref_dem"))
                b_path = Path(xdem.examples.get_path("longyearbyen_tba_dem_coreg"))

            validate_geotiff(str(a_path))
            validate_geotiff(str(b_path))

            a_info, a_dem = get_geotiff_metadata(str(a_path))
            b_info, b_dem = get_geotiff_metadata(str(b_path))


            a_valid, a_valid_message = validate_dem_data(a_dem)
            b_valid, b_valid_message = validate_dem_data(b_dem)


            compatability = check_raster_compatability(a_info, b_info)

            if compatability.reason is not None:
                print(compatability.reason)

            if not compatability.same_crs or not compatability.same_grid or not compatability.overlaps:
                match args.align:
                    case "to-a":
                        b_dem = reproject_raster("to-a", a_dem, b_dem, resampling=args.resample)
                    case "to-b":
                        a_dem = reproject_raster("to-b", a_dem, b_dem, resampling=args.resample)
                    case _:
                        raise ValueError(f"Invalid alignment: {args.align}")


            diff = generate_difference_raster(a_dem, b_dem)
            elevation_change = generate_elevation_change_raster(diff)
            change_direction = generate_change_direction_raster(diff)
            change_magnitude = generate_change_magnitude_raster(diff)
            slope_deg = generate_slope_degrees_raster(a_dem)

            t_green, t_amber, t_red = parse_thresholds(args.thresholds)
            movement_rank = generate_ranked_movement_raster(
                change_magnitude, t_green=t_green, t_amber=t_amber, t_red=t_red
            )

            diff_path = map_layers_dir / "diff.tif"
            diff.save(diff_path)

            output_nodata = NODATA_FLOAT
            output_mask = np.zeros_like(diff.data, dtype=bool)
            if diff.nodata is not None:
                output_mask = diff.data == diff.nodata

            elevation_change_filled = elevation_change.copy()
            elevation_change_filled[output_mask] = output_nodata
            xdem.DEM.from_array(
                elevation_change_filled,
                transform=diff.transform,
                crs=diff.crs,
                nodata=output_nodata,
            ).save(map_layers_dir / "elevation_change.tif")

            change_magnitude_filled = change_magnitude.copy()
            change_magnitude_filled[output_mask] = output_nodata
            xdem.DEM.from_array(
                change_magnitude_filled,
                transform=diff.transform,
                crs=diff.crs,
                nodata=output_nodata,
            ).save(map_layers_dir / "change_magnitude.tif")

            direction_nodata = NODATA_DIRECTION
            change_direction_filled = change_direction.copy()
            change_direction_filled[output_mask] = direction_nodata
            xdem.DEM.from_array(
                change_direction_filled.astype(np.float32),
                transform=diff.transform,
                crs=diff.crs,
                nodata=direction_nodata,
            ).save(map_layers_dir / "change_direction.tif")

            slope_filled = slope_deg.copy()
            slope_filled[output_mask] = output_nodata
            xdem.DEM.from_array(
                slope_filled.astype(np.float32),
                transform=diff.transform,
                crs=diff.crs,
                nodata=output_nodata,
            ).save(map_layers_dir / "slope_degrees.tif")

            xdem.DEM.from_array(
                movement_rank.astype(np.float32),
                transform=diff.transform,
                crs=diff.crs,
                nodata=NODATA_RANK,
            ).save(map_layers_dir / "movement_rank.tif")

            try:
                polys = xdem.DEM.from_array(
                    movement_rank.astype(np.float32),
                    transform=diff.transform,
                    crs=diff.crs,
                    nodata=NODATA_RANK,
                ).polygonize(target_values=[1, 2, 3], data_column_name="class")
                polys.save(map_layers_dir / "movement_ranked_polygons.shp")
            except Exception as e:
                print(f"Warning: polygonize failed: {e}")

            # PNG previews + report (best-effort; GeoTIFFs remain the main outputs)
            try:
                images: list[ReportImage] = []

                movement_png = "movement_magnitude_green_to_red.png"
                save_movement_magnitude_viridis_png(
                    change_magnitude,
                    nodata_mask=output_mask,
                    out_path=reports_dir / movement_png,
                )
                images.append(ReportImage(movement_png, "Movement magnitude (Green → Red)"))

                dh_png = "elevation_change_diverging.png"
                save_elevation_change_diverging_png(
                    elevation_change,
                    nodata_mask=output_mask,
                    out_path=reports_dir / dh_png,
                )
                images.append(ReportImage(dh_png, "Elevation change (diverging)"))

                slope_png = "slope_degrees.png"
                save_slope_png(slope_deg, nodata_mask=output_mask, out_path=reports_dir / slope_png)
                images.append(ReportImage(slope_png, "Slope (degrees)"))

                dir_png = "movement_direction.png"
                save_direction_png(change_direction, nodata_mask=output_mask, out_path=reports_dir / dir_png)
                images.append(ReportImage(dir_png, "Direction of movement"))

                rank_png = "movement_rank.png"
                save_rank_png(
                    movement_rank,
                    nodata_mask=output_mask,
                    out_path=reports_dir / rank_png,
                    t_green=t_green,
                    t_amber=t_amber,
                    t_red=t_red,
                )
                images.append(ReportImage(rank_png, "Ranked movement (G/A/R)"))

                report_path = write_html_report(
                    reports_dir,
                    title="dtm-differ report",
                    images=images,
                    map_layers_dir=map_layers_dir,
                )
                print(f"Wrote report to {report_path}")
            except Exception as e:
                print(f"Warning: report generation failed: {e}")

            print(f"Wrote map layers to {map_layers_dir}")
            print(f"Wrote reports to {reports_dir}")
            print(f"Rank thresholds: {t_green},{t_amber},{t_red} m")
            return 0
        case "status":
            pass
        case _:
            print("Invalid command")
            parser.print_help()
            return 1
    

if __name__ == "__main__":
    main()
