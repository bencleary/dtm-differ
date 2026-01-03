# Methodology

Technical details on how DTM Differ processes elevation data.

## Difference Calculation

Elevation change is computed as:

```
dh = DEM_A - DEM_B
```

- **Positive values**: Uplift or deposition (surface rose)
- **Negative values**: Subsidence or erosion (surface lowered)

Convention: A is "before", B is "after". If your data is reversed, negate results or swap inputs.

## Slope Calculation

Slope uses finite differences with adaptive stencils:

```
slope = arctan(√((∂z/∂x)² + (∂z/∂y)²))
```

The algorithm uses central differences where both neighbors exist, falling back to forward/backward differences at edges or near nodata cells. This avoids the artifacts you get from naive `np.gradient` calls near missing data.

## Uncertainty Propagation

When `--uncertainty constant` is enabled (default), the tool propagates measurement uncertainty through the difference calculation.

### Combined Uncertainty

```
σ_dh = √(σ_A² + σ_B² + σ_coreg²)
```

Where:
- `σ_A`, `σ_B`: Vertical uncertainty of each DEM (default 0.5m)
- `σ_coreg`: Co-registration uncertainty (default 0.3m)

### Significance Testing

A z-score is computed for each pixel:

```
z = dh / σ_dh
```

Changes are considered **detectable** when `|z| ≥ k` (default k=1.96, ~95% confidence).

### Movement Rank Suppression

By default, `movement_rank` is set to 0 for pixels where `|dh| ≤ k·σ_dh`. This prevents noise from being classified as movement. Disable with `--no-suppress-within-noise-rank`.

## Interpreting "No Change"

The value 0 in `change_direction.tif` and `movement_rank.tif` means different things depending on uncertainty mode:

| Mode | Meaning of 0 |
|------|--------------|
| `--uncertainty constant` | Change not detectable at chosen confidence level |
| `--uncertainty none` | Magnitude below green threshold (may include noise) |

Always check `z_score.tif` and `within_noise_mask.tif` to understand confidence.

## Co-registration Diagnostics

When reprojection occurs, the tool fits a plane to the dh surface:

```
dh ≈ a·x + b·y + c
```

A non-zero tilt (large `a` or `b`) suggests residual misalignment between the DEMs. Check `metrics.json` for:

- `tilt_m_per_unit`: Gradient magnitude
- `tilt_angle_deg`: Tilt as an angle
- `residual_rmse_m`: Scatter after removing trend

## Memory Usage

The tool loads entire rasters into memory. Approximate peak usage:

```
Peak RAM ≈ 8 × (input file size)
```

For a 10,000 × 10,000 float32 DEM (~400 MB), expect ~3.2 GB peak usage.

## Limitations

- No chunked/streaming processing for very large files
- Uncertainty is spatially constant (no per-pixel error maps)
- Volume calculations not yet implemented
- Vertical unit validation relies on CRS metadata (often missing)
