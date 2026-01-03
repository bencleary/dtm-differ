# DTM Differ

DTM Differ is a command-line utility for quantifying elevation change between Digital Terrain Models. It is designed for geotechnical and environmental monitoring workflows, including subsidence assessment, erosion analysis, and cut and fill evaluation.

The tool focuses on producing interpretable, defensible outputs rather than exploratory GIS analysis.

## Installation

Python 3.13 or later is required. Installation via `uv` is recommended, though `pip` is also supported.

```bash
git clone <repository-url>
cd dtm-differ
uv pip install -e .
```

## Quick Start

```bash
# Basic comparison
dtm-differ run --a before.tif --b after.tif --out output/

# Define custom movement thresholds (metres)
dtm-differ run --a before.tif --b after.tif --out output/ --thresholds "0.5,2.0,5.0"

# Coastal use case: exclude low-lying or marine areas
dtm-differ run --a before.tif --b after.tif --out output/ --min-elevation 2.0

# Constrain analysis to a specific elevation band
dtm-differ run --a before.tif --b after.tif --out output/ --min-elevation 2.0 --max-elevation 80.0
```

## Command-Line Options

| Option                   | Default       | Description                                                   |
| ------------------------ | ------------- | ------------------------------------------------------------- |
| `--a`, `--b`             | required      | Input DTM GeoTIFFs                                            |
| `--out`                  | required      | Output directory                                              |
| `--thresholds`           | `1.0,3.0,6.0` | Green, amber, red thresholds in metres                        |
| `--resample`             | `bilinear`    | Resampling method                                             |
| `--align`                | `to-a`        | Reference grid alignment                                      |
| `--min-elevation`        | `None`        | Mask areas below this elevation                               |
| `--max-elevation`        | `None`        | Mask areas above this elevation                               |
| `--uncertainty`          | `constant`    | Uncertainty handling mode                                     |
| `--sigma-a`, `--sigma-b` | `0.5`         | Vertical uncertainty per DEM (1σ, metres)                     |
| `--sigma-coreg`          | `0.3`         | Co-registration uncertainty (1σ, metres)                      |
| `--k-sigma`              | `1.96`        | Significance multiplier (approximately 95 percent confidence) |

## Outputs

```
output/
├── map_layers/
│   ├── diff.tif
│   ├── elevation_change.tif
│   ├── change_magnitude.tif
│   ├── change_direction.tif
│   ├── slope_degrees.tif
│   ├── movement_rank.tif
│   ├── sigma_dh.tif
│   ├── z_score.tif
│   └── within_noise_mask.tif
└── reports/
    ├── report.html
    └── metrics.json
```

The outputs are structured for both visual inspection and downstream automation. Raster layers are GIS-ready, while summary metrics support quality assurance and reporting.

## Python API

DTM Differ can also be used programmatically for batch processing or integration into larger pipelines.

```python
from pathlib import Path
from uuid import uuid4
from dtm_differ.db import Database
from dtm_differ.pipeline import run_pipeline
from dtm_differ.pipeline.types import ProcessingConfig

db = Database("jobs.sqlite")
db.initialise()

job_id = str(uuid4())
db.create_job(job_id)

result = run_pipeline(
    db=db,
    job_id=job_id,
    a_path=Path("before.tif"),
    b_path=Path("after.tif"),
    out_dir=Path("output/"),
    config=ProcessingConfig(
        t_green=1.0,
        t_amber=3.0,
        t_red=6.0,
    ),
)
```

## Movement Classification

Elevation change is categorised by absolute magnitude.

| Rank | Classification  | Default Range |
| ---- | --------------- | ------------- |
| 0    | Below threshold | < 1.0 m       |
| 1    | Green           | 1.0 to 3.0 m  |
| 2    | Amber           | 3.0 to 6.0 m  |
| 3    | Red             | ≥ 6.0 m       |

Thresholds are inclusive at the lower bound. A change of exactly 3.0 metres is classified as amber.

## Input Assumptions

* Inputs must be single-band GeoTIFFs with a valid CRS
* Thresholds assume vertical units are metres
* Peak memory usage is approximately eight times the input raster size

Mismatched projections and grid alignments are handled automatically through reprojection and resampling.

## Development and Testing

```bash
uv pip install -e ".[test]"
pytest src/tests/
```

## Rationale

This tool exists to bridge the gap between heavyweight GIS workflows and overly simplistic differencing scripts. Many existing solutions either require full desktop GIS environments or fail to account for uncertainty and detection limits.

DTM Differ provides a narrow, well-defined pipeline that produces traceable results suitable for engineering and decision-support contexts.

Further details on the underlying methodology, including differencing logic, slope calculation, and uncertainty propagation, are available in `docs/methodology.md`.

## Acknowledgements

DTM Differ builds on the excellent work of the open-source geospatial community, particularly xdem, rasterio, and NumPy.
