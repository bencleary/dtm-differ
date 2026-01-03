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

## Elevation Masking

Elevation-based spatial filtering allows restricting analysis to terrain zones of interest. This is particularly useful for coastal, estuarine, or variable-terrain monitoring.

### Implementation

When `--min-elevation` or `--max-elevation` are specified:

1. Extract elevations from DEM **A** (the "before" reference)
2. Create a boolean mask for pixels outside the elevation range:
   ```
   mask = (z_A < min_elev) | (z_A > max_elev)
   ```
3. Combine with existing nodata mask using logical OR
4. Apply combined mask to all derived rasters

### Use Cases

#### Coastal Cliff Monitoring

Exclude tidal zones and sea surface (which varies between surveys):

```bash
# Mask everything below Mean High Water (~2m ODN for UK south coast)
dtm-differ run --a 2023.tif --b 2024.tif --out output/ --min-elevation 2.0
```

**Why?** LiDAR captures the water surface at survey time. Different tidal states and wave conditions create false "changes" of tens of meters that obscure real cliff erosion signals.

#### Focused Analysis Zones

Combine `--min-elevation` and `--max-elevation` to analyze specific terrain features:

```bash
# Analyze only the cliff zone (2-80m) excluding hilltops and sea
dtm-differ run --a before.tif --b after.tif --out output/ \
  --min-elevation 2.0 --max-elevation 80.0 \
  --thresholds "0.5,1.0,2.0"  # Fine thresholds for cliff work
```

**Typical applications:**
- **Cliff erosion:** Focus on the active cliff face, excluding beach and inland areas
- **River terraces:** Isolate specific terrace levels for flood/erosion analysis
- **Mine monitoring:** Analyze individual benches by elevation range
- **Estuarine zones:** Separate subtidal, intertidal, and supratidal areas

### Choosing Elevation Thresholds

**For UK coastal work:**
- Use local Mean High Water Spring (MHWS) for `--min-elevation`
- Check Ordnance Datum (ODN) or local tide gauge data
- Typical MHWS: 2-3m ODN for south coast, 4-5m for areas with higher tidal range

**For other regions:**
- Use regional vertical datum + typical high tide
- NAVD88 (North America), AHD (Australia), EGM2008 (global geoid)
- Check national mapping agencies for tidal datums

### Technical Notes

- Masking uses only DEM A elevations—ensures consistent reference frame
- Nodata values in DEM A are handled correctly (excluded before elevation comparison)
- Statistics in `metrics.json` reflect only the unmasked valid area
- Edge effects at mask boundaries are minimal—no spatial filtering involved
- Works alongside existing nodata handling—both masks combine using logical OR

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
