# DTM Differ

A Python tool for differencing Digital Terrain Models (DTMs) to detect and analyze elevation changes over time. Designed for geotechnical engineering applications, it identifies areas of subsidence, uplift, erosion, and deposition.

## Features

- **DTM Differencing**: Compute elevation changes between two DTMs
- **Automatic Alignment**: Handles different CRS and grid alignments with automatic reprojection
- **Movement Classification**: Categorises changes into Green/Amber/Red risk levels based on configurable thresholds
- **Comprehensive Outputs**: Generates GeoTIFF rasters, visualisations, and HTML reports
- **Slope Analysis**: Calculates slope angles from input DEMs
- **Polygon Generation**: Creates shapefiles for movement-ranked areas

## Installation

### Requirements

- Python >= 3.13
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Install with uv

```bash
# Clone the repository
git clone <repository-url>
cd dtm-differ

# Install in editable mode
uv pip install -e .
```

### Install with pip

```bash
pip install -e .
```

## Quick Start

```bash
# Basic usage
dtm-differ run --a before.tif --b after.tif --out output/

# With custom thresholds (in meters)
dtm-differ run --a before.tif --b after.tif --out output/ \
  --thresholds "1.0,3.0,6.0"

# Using example data (for testing)
make run
```

## Usage

### Command Line Interface

```bash
dtm-differ run [OPTIONS]
```

#### Required Arguments

- `--a PATH`: Path to DTM A (before/earlier survey)
- `--b PATH`: Path to DTM B (after/later survey)
- `--out PATH`: Output directory for results

#### Optional Arguments

- `--resample {nearest,bilinear}`: Resampling method for alignment (default: `bilinear`)
- `--align {to-a,to-b}`: Which DEM to use as reference grid (default: `to-a`)
- `--thresholds THRESHOLDS`: Movement thresholds in meters
  - Format: `green,amber,red` or `amber,red` (green defaults to 1.0m)
  - Default: `1.0,3.0,6.0`
  - Example: `--thresholds "2.0,5.0"` (green=1.0, amber=2.0, red=5.0)
  - ⚠️ **IMPORTANT**: Thresholds are assumed to be in **meters**. Verify your input DEMs use meter-based vertical units. If your DEMs use feet or other units, convert thresholds accordingly.
- `--style {diverging,terrain,greyscale}`: Visualisation style (default: `diverging`)

### Examples

#### Basic Analysis

```bash
dtm-differ run \
  --a survey_2020.tif \
  --b survey_2023.tif \
  --out results/
```

#### Custom Thresholds for Mining Application

```bash
dtm-differ run \
  --a pre_mining.tif \
  --b post_mining.tif \
  --out mining_analysis/ \
  --thresholds "0.5,2.0,5.0"
```

#### Different Coordinate System Alignment

```bash
dtm-differ run \
  --a dem_a_utm.tif \
  --b dem_b_latlon.tif \
  --out output/ \
  --align to-b \
  --resample nearest
```

## Output Structure

The tool generates outputs in the following structure:

```
output/
├── map_layers/          # GeoTIFF raster files
│   ├── diff.tif                    # Raw difference (A - B)
│   ├── elevation_change.tif         # Elevation change (with nodata handling)
│   ├── change_magnitude.tif        # Absolute elevation change
│   ├── change_direction.tif        # Direction: -1 (subsidence), 0 (not detectable/no change), 1 (uplift)
│   ├── slope_degrees.tif           # Slope angle in degrees
│   ├── movement_rank.tif            # Risk classification (0=unclassified, 1=green, 2=amber, 3=red)
│   └── movement_ranked_polygons.shp # Shapefile of ranked areas (if generated)
└── reports/             # Visualisations and HTML report
	    ├── report.html                  # Interactive HTML report
	    ├── metrics.json                 # Machine-readable QA metrics + run configuration
	    ├── movement_magnitude_green_to_red.png
	    ├── elevation_change_diverging.png
	    ├── slope_degrees.png
	    ├── movement_direction.png
    └── movement_rank.png
```

### Output Files Explained

#### GeoTIFF Rasters

- **`diff.tif`**: Raw difference raster (DTM A - DTM B). Positive values indicate uplift/deposition, negative values indicate subsidence/erosion.
- **`elevation_change.tif`**: Same as diff but with proper nodata handling. Use this for analysis.
- **`change_magnitude.tif`**: Absolute value of elevation change. Useful for identifying areas of significant movement regardless of direction.
- **`change_direction.tif`**: Direction of change:
  - `-1`: Subsidence/erosion (elevation decreased)
  - `0`: Not detectable at the chosen uncertainty threshold (or exactly zero when uncertainty is disabled)
  - `1`: Uplift/deposition (elevation increased)
- **`slope_degrees.tif`**: Slope angle in degrees, calculated from DTM A.
- **`movement_rank.tif`**: Risk classification based on magnitude thresholds:
  - `0`: Below thresholds / unclassified
  - `1`: Green (low risk): `t_green <= magnitude < t_amber`
  - `2`: Amber (medium risk): `t_amber <= magnitude < t_red`
  - `3`: Red (high risk): `magnitude >= t_red`

#### Visualisations

All PNG files are high-resolution (150 DPI) visualisations suitable for reports and presentations. The HTML report (`report.html`) provides an interactive view of all visualisations with links to download the GeoTIFF files.

## Python API

The tool can also be used programmatically:

```python
from pathlib import Path
from dtm_differ.pipeline import run_pipeline
from dtm_differ.pipeline.types import ProcessingConfig

# Configure processing
config = ProcessingConfig(
    resample="bilinear",
    align="to-a",
    t_green=1.0,
    t_amber=3.0,
    t_red=6.0,
    generate_report=True,
    generate_polygons=True,
)

# Run pipeline
result = run_pipeline(
    a_path=Path("before.tif"),
    b_path=Path("after.tif"),
    out_dir=Path("output/"),
    config=config,
)

# Access results
print(f"Output directory: {result.map_layers_dir}")
print(f"Report directory: {result.reports_dir}")
```

## Input Requirements

### DEM Format

- **Format**: GeoTIFF (.tif, .tiff)
- **Coordinate System**: Any valid CRS (tool handles reprojection automatically)
- **Data Type**: Floating point elevation values
- **Nodata**: Properly set nodata values (tool will detect and handle)

### DEM Compatibility

The tool automatically:
- Validates input GeoTIFFs
- Checks for sufficient valid data (>1% by default)
- Detects CRS mismatches
- Handles different grid alignments
- Reprojects and resamples as needed

### Memory Requirements

- **Minimum**: 2x the size of input DEMs in RAM
- **Peak usage**: ~8x the size of input DEMs
- **Example**: For a 10,000 x 10,000 float32 DEM (~400 MB file), expect ~3.2 GB peak memory usage

For very large datasets, ensure adequate system memory or consider processing subsets.

## Development

### Setup

```bash
# Install with development dependencies
uv pip install -e ".[test]"

# Or using Make
make install
```

### Running Tests

```bash
# Run all tests
pytest src/tests/

# Run with coverage
pytest src/tests/ --cov=dtm_differ

# Run specific test file
pytest src/tests/test_raster.py
```

### Project Structure

```
dtm-differ/
├── src/
│   └── dtm_differ/
│       ├── cli.py              # Command-line interface
│       ├── constants.py        # Configuration constants
│       ├── geotiff.py          # GeoTIFF I/O and validation
│       ├── raster.py           # Raster processing functions
│       ├── types.py            # Type definitions
│       ├── viz.py              # Visualisation functions
│       └── pipeline/           # Processing pipeline
│           ├── __init__.py
│           ├── stages.py      # Pipeline stages
│           └── types.py        # Pipeline types
├── src/tests/                 # Test suite
├── output/                    # Example outputs (gitignored)
└── pyproject.toml             # Project configuration
```

### Running Example

```bash
# Uses xdem example data if files don't exist
make run
```

## Methodology

### Difference Calculation

The tool computes elevation change as:

```
elevation_change = DTM_A - DTM_B
```

- **Positive values**: Uplift or deposition (elevation increased)
- **Negative values**: Subsidence or erosion (elevation decreased)

### Slope Calculation

Slope is calculated using finite differences:

```
slope = arctan(√(dz/dx)² + (dz/dy)²)
```

Where `dz/dx` and `dz/dy` are computed using `numpy.gradient()`.

### Movement Ranking

Movement is classified based on absolute magnitude with **inclusive lower bounds and exclusive upper bounds**:

- **Green** (rank 1): Low movement (`t_green ≤ magnitude < t_amber`)
- **Amber** (rank 2): Medium movement (`t_amber ≤ magnitude < t_red`)
- **Red** (rank 3): High movement (`magnitude ≥ t_red`)
- **Unclassified** (rank 0): Below green threshold (`magnitude < t_green`)

**Examples with default thresholds (1.0, 3.0, 6.0 m):**
- A change of 0.5 m → Unclassified (0)
- A change of 1.0 m → Green (1) - exactly at threshold, included
- A change of 2.0 m → Green (1) - within range
- A change of 3.0 m → Amber (2) - exactly at threshold, included
- A change of 5.0 m → Amber (2) - within range
- A change of 6.0 m → Red (3) - exactly at threshold, included
- A change of 10.0 m → Red (3) - above threshold

Default thresholds (1.0, 3.0, 6.0 meters) are suitable for many geotechnical applications but should be adjusted based on:
- Measurement precision
- Expected natural variation
- Project-specific requirements
- Uncertainty analysis results (see Uncertainty Analysis section)

### Interpreting "No Change" Results

The `change_direction.tif` and `movement_rank.tif` files use value `0` to indicate a "no-change / not-detectable" outcome, but the interpretation depends on whether uncertainty analysis was performed:

**When uncertainty analysis is enabled** (recommended):
- `change_direction.tif == 0` indicates the change is **not detectable** at the chosen k threshold (|z| < k)
- `movement_rank.tif == 0` means either the magnitude is below the green threshold, or the rank was suppressed because it is within noise
- Check `z_score.tif` and `within_noise_mask.tif` to interpret confidence and detectability

**When uncertainty analysis is disabled** (not recommended):
- "No change" (rank 0) simply means the magnitude is below the green threshold
- These results may include noise-level variations that appear as "no change"
- Consider enabling uncertainty analysis for defensible results

**Best practice**: Always review the `z_score.tif` and `within_noise_mask.tif` outputs (when uncertainty is enabled) to understand the confidence level of "no change" classifications.

## Uncertainty Analysis

The tool supports uncertainty propagation to help distinguish meaningful terrain changes from measurement noise. By default, uncertainty analysis is **enabled** with conservative default values.

### Default Uncertainty Parameters

- **σ<sub>A</sub>** (DEM A uncertainty): 0.5 m (1σ)
- **σ<sub>B</sub>** (DEM B uncertainty): 0.5 m (1σ)
- **σ<sub>coreg</sub>** (co-registration uncertainty): 0.3 m (1σ)
- **k** (significance threshold): 1.96 (~95% confidence)

The combined uncertainty σ<sub>dh</sub> is calculated as:
```
σ_dh = √(σ_A² + σ_B² + σ_coreg²)
```

### Uncertainty Outputs

When uncertainty analysis is enabled, additional outputs are generated:

- **`sigma_dh.tif`**: Combined uncertainty for each pixel
- **`z_score.tif`**: Significance score (z = dh / σ_dh). Changes with |z| ≥ k are considered detectable
- **`within_noise_mask.tif`**: Mask indicating areas where changes are within noise (|dh| ≤ k·σ_dh)

The HTML report includes a confidence summary showing:
- Percentage of area within noise
- Percentage of area with detectable changes
- Percentage of area with high confidence changes (|z| ≥ 3)

### Disabling Uncertainty Analysis

⚠️ **Not recommended**: Disabling uncertainty analysis means results may include noise-level changes. If you must disable it, use `--uncertainty none` and be aware that:
- Movement classifications may include measurement noise
- Results may not be defensible in regulatory or technical reviews
- The tool will display a warning when uncertainty is disabled

## Limitations

- **Memory**: Entire rasters are loaded into memory. Very large datasets (>50M pixels) may require significant RAM.
- **Chunking**: Chunked processing for very large files is not yet implemented.
- **Spatial Uncertainty**: Currently supports only constant (spatially uniform) uncertainty. Spatially-varying uncertainty is not yet supported.
- **Volume Calculation**: Cut/fill volume calculations are not included (planned for future release).
- **Unit Validation**: The tool assumes meter-based vertical units for thresholds but does not automatically validate this from CRS metadata. Users must verify unit compatibility.

## License

[Add your license here]

## Citation

If you use this tool in your research, please cite:

```
[Add citation information]
```

## Acknowledgments

Built with:
- [xdem](https://github.com/GlacioHack/xdem) - DEM processing
- [rasterio](https://rasterio.readthedocs.io/) - GeoTIFF I/O
- [NumPy](https://numpy.org/) - Array operations
- [Matplotlib](https://matplotlib.org/) - Visualizations
