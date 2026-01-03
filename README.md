# DTM Differ

A command-line tool for detecting elevation changes between Digital Terrain Models (DTMs). Built for geotechnical monitoring workflows—subsidence tracking, erosion detection, cut/fill analysis.

## Installation

Requires Python 3.13+. Install with [uv](https://github.com/astral-sh/uv) (recommended) or pip:

```bash
git clone <repository-url>
cd dtm-differ
uv pip install -e .
```

## Quick Start

```bash
# Basic usage
dtm-differ run --a before.tif --b after.tif --out output/

# Custom movement thresholds (meters)
dtm-differ run --a before.tif --b after.tif --out output/ --thresholds "0.5,2.0,5.0"
```

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--a`, `--b` | required | Input DTM paths (GeoTIFF) |
| `--out` | required | Output directory |
| `--thresholds` | `1.0,3.0,6.0` | Green/amber/red thresholds in meters |
| `--resample` | `bilinear` | Resampling method (`nearest`, `bilinear`) |
| `--align` | `to-a` | Reference grid (`to-a`, `to-b`) |
| `--uncertainty` | `constant` | Uncertainty mode (`constant`, `none`) |
| `--sigma-a`, `--sigma-b` | `0.5` | DEM vertical uncertainty (1σ, meters) |
| `--sigma-coreg` | `0.3` | Co-registration uncertainty (1σ, meters) |
| `--k-sigma` | `1.96` | Significance threshold (~95% confidence) |

## Outputs

```
output/
├── map_layers/
│   ├── diff.tif                 # Raw difference (A - B)
│   ├── elevation_change.tif     # Difference with nodata handling
│   ├── change_magnitude.tif     # Absolute change
│   ├── change_direction.tif     # -1 (subsidence), 0, +1 (uplift)
│   ├── slope_degrees.tif        # Slope from DEM A
│   ├── movement_rank.tif        # 0=below threshold, 1=green, 2=amber, 3=red
│   ├── sigma_dh.tif             # Combined uncertainty (if enabled)
│   ├── z_score.tif              # Significance score (if enabled)
│   └── within_noise_mask.tif    # Areas below detection threshold
└── reports/
    ├── report.html              # Visual summary
    └── metrics.json             # Machine-readable QA metrics
```

## Python API

```python
from pathlib import Path
from dtm_differ.pipeline import run_pipeline
from dtm_differ.pipeline.types import ProcessingConfig

result = run_pipeline(
    a_path=Path("before.tif"),
    b_path=Path("after.tif"),
    out_dir=Path("output/"),
    config=ProcessingConfig(t_green=1.0, t_amber=3.0, t_red=6.0),
)
```

## Movement Classification

Changes are classified by absolute magnitude:

| Rank | Label | Default Range |
|------|-------|---------------|
| 0 | Below threshold | < 1.0 m |
| 1 | Green | 1.0 – 3.0 m |
| 2 | Amber | 3.0 – 6.0 m |
| 3 | Red | ≥ 6.0 m |

Thresholds are inclusive on the lower bound: a 3.0m change is amber, not green.

## Input Requirements

- **Format**: GeoTIFF with valid CRS
- **Units**: Thresholds assume meters—verify your vertical units match
- **Memory**: ~8× input file size at peak (loaded into memory)

The tool auto-reprojects mismatched CRS and resamples different grid alignments.

## Development

```bash
uv pip install -e ".[test]"
pytest src/tests/
```

## Why I Built This

I needed a repeatable way to compare survey DEMs for a mining monitoring project. Existing tools either required a full GIS setup or didn't handle uncertainty propagation. This fills the gap: a focused CLI that produces defensible outputs with confidence intervals.

See [docs/methodology.md](docs/methodology.md) for technical details on the difference calculation, slope algorithm, and uncertainty propagation.

## License

MIT

## Acknowledgments

Built on [xdem](https://github.com/GlacioHack/xdem), [rasterio](https://rasterio.readthedocs.io/), and [NumPy](https://numpy.org/).
