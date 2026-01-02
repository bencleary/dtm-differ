from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import xdem

from dtm_differ.pipeline import run_pipeline
from dtm_differ.pipeline.types import ProcessingConfig


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="dtm-differ", description="Simple DTM differencing tool for GeoTIFFs"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run a DTM differencing job")
    run.add_argument("--a", required=True, help="Path to DTM A (before)")
    run.add_argument("--b", required=True, help="Path to DTM B (after)")
    run.add_argument("--out", required=True, help="Output directory")

    run.add_argument("--resample", choices=["nearest", "bilinear"], default="bilinear")
    run.add_argument("--align", choices=["to-a", "to-b"], default="to-a")
    run.add_argument(
        "--thresholds",
        help="Thresholds in meters: green,amber,red (or amber,red). "
        "⚠️  WARNING: Assumes input DEMs use meter-based vertical units. "
        "Verify your DEM vertical units match this assumption."
    )
    run.add_argument(
        "--style", choices=["diverging", "terrain", "greyscale"], default="diverging"
    )

    run.add_argument(
        "--uncertainty",
        choices=["none", "constant"],
        default="constant",
        help="Uncertainty handling: constant (default, recommended) or none (not recommended). Constant mode propagates sigma_a/b/coreg.",
    )
    run.add_argument(
        "--sigma-a",
        type=float,
        default=0.5,
        help="1σ vertical uncertainty of DEM A (m). Default: 0.5 m (conservative estimate). "
        "Must be >= 0. Typical values: 0.1-0.5 m for high-quality surveys, 0.5-2.0 m for lower quality.",
    )
    run.add_argument(
        "--sigma-b",
        type=float,
        default=0.5,
        help="1σ vertical uncertainty of DEM B (m). Default: 0.5 m (conservative estimate). "
        "Must be >= 0. Typical values: 0.1-0.5 m for high-quality surveys, 0.5-2.0 m for lower quality.",
    )
    run.add_argument(
        "--sigma-coreg",
        type=float,
        default=0.3,
        help="1σ co-registration uncertainty (m). Default: 0.3 m (conservative estimate). "
        "Must be >= 0. Typical values: 0.1-0.5 m depending on alignment quality.",
    )
    run.add_argument(
        "--k-sigma",
        type=float,
        default=1.96,
        help="Significance multiplier k (must be > 0). Default: 1.96 (~95% confidence). "
        "Typical values: 1.96 (~95%), 2.58 (~99%), 3.0 (~99.7%).",
    )

    run.add_argument(
        "--no-suppress-within-noise-rank",
        action="store_true",
        help="Do not suppress movement_rank where |dh| <= k*sigma_dh.",
    )
    run.add_argument(
        "--progress",
        action=BooleanOptionalAction,
        default=None,
        help="Show progress bar (default: auto when interactive).",
    )

    status = subparsers.add_parser("status", help="Check job status")
    status.add_argument("job_id")

    return parser


def main() -> int:
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
        raise ValueError(
            "Invalid --thresholds; expected 'amber,red' or 'green,amber,red'"
        )

    match args.command:
        case "run":
            a_path = Path(args.a)
            b_path = Path(args.b)
            out_dir = Path(args.out)

            # Dev convenience: `make run` passes `example.tif`, so fall back to xdem's bundled examples
            if (
                not a_path.exists()
                and not b_path.exists()
                and args.a == "example.tif"
                and args.b == "example.tif"
            ):
                a_path = Path(xdem.examples.get_path("longyearbyen_ref_dem"))
                b_path = Path(xdem.examples.get_path("longyearbyen_tba_dem_coreg"))

            t_green, t_amber, t_red = parse_thresholds(args.thresholds)

            # Validate uncertainty parameters
            if args.uncertainty == "constant":
                if args.sigma_a < 0 or args.sigma_b < 0 or args.sigma_coreg < 0:
                    print("ERROR: Uncertainty sigma values must be >= 0")
                    return 1
                if args.k_sigma <= 0:
                    print("ERROR: k_sigma must be > 0")
                    return 1
                if args.k_sigma > 5:
                    print(
                        f"⚠️  WARNING: k_sigma={args.k_sigma} is unusually high. "
                        "Typical values are 1.96 (~95% confidence) or 3.0 (~99.7% confidence)."
                    )

            # Warn if uncertainty is explicitly disabled
            if args.uncertainty == "none":
                print(
                    "⚠️  WARNING: Uncertainty analysis disabled. Results may include noise-level changes."
                )
                print(
                    "   Consider using --uncertainty constant with appropriate sigma values for defensible results."
                )

            result = run_pipeline(
                a_path=a_path,
                b_path=b_path,
                out_dir=out_dir,
                config=ProcessingConfig(
                    resample=args.resample,
                    align=args.align,
                    t_green=t_green,
                    t_amber=t_amber,
                    t_red=t_red,
                    uncertainty_mode=args.uncertainty,
                    sigma_a=args.sigma_a,
                    sigma_b=args.sigma_b,
                    sigma_coreg=args.sigma_coreg,
                    k_sigma=args.k_sigma,
                    suppress_within_noise_rank=not args.no_suppress_within_noise_rank,
                ),
                progress=args.progress,
            )

            print(f"Wrote map layers to {result.map_layers_dir}")
            print(f"Wrote reports to {result.reports_dir}")
            print(f"Rank thresholds: {t_green},{t_amber},{t_red} m")
            return 0
        case "status":
            return 0
        case _:
            print("Invalid command")
            parser.print_help()
            return 1


if __name__ == "__main__":
    main()
