# Methodology

This document explains how DTM Differ processes elevation data and how each derived output should be interpreted. The intent is to make the calculations transparent and to clarify what each step represents in practical terms.

## Elevation Difference

### What this does

The core operation is a direct comparison between two terrain surfaces to quantify how elevation has changed over time.

### How it is calculated

Elevation change is computed on a cell-by-cell basis:

```
dh = DEM_A − DEM_B
```

DEM A is treated as the earlier surface and DEM B as the later surface.

* Positive values indicate uplift or material accumulation
* Negative values indicate subsidence or erosion

### How to interpret it

Each pixel represents the vertical change at that location between surveys. Large contiguous areas of change typically indicate real geomorphic or engineered movement, while isolated single-pixel changes are more likely to be noise.

If your inputs are temporally reversed, swap the inputs or negate the output to restore the expected sign convention.

## Slope Calculation

### What this does

Slope describes how steep the terrain is at each pixel and is derived from the reference surface. It is used to provide context for change interpretation and to support downstream analysis.

### How it is calculated

Slope is calculated using finite-difference gradients:

```
slope = arctan( √((∂z/∂x)² + (∂z/∂y)²) )
```

Central differences are used where possible. At raster edges or near nodata cells, forward or backward differences are applied instead.

### Why this approach is used

Simple gradient operators often produce artefacts near nodata boundaries. The adaptive stencil approach used here reduces edge effects and produces more stable slope estimates in real-world datasets.

## Uncertainty Propagation

### What this does

Uncertainty propagation quantifies how confident you can be that an observed elevation change represents real movement rather than measurement noise.

### Why it matters

LiDAR- and photogrammetry-derived DTMs contain vertical error. When differencing two surfaces, these errors accumulate. Treating all observed change as real can lead to false positives, particularly for small-magnitude movements.

## Combined Uncertainty

### What this represents

Combined uncertainty expresses the expected error in the elevation difference at each pixel.

### How it is calculated

```
σ_dh = √(σ_A² + σ_B² + σ_coreg²)
```

* σ_A and σ_B represent vertical uncertainty in each DEM
* σ_coreg represents residual alignment error between surfaces

### How to interpret it

Higher combined uncertainty means a larger change is required before movement can be considered detectable. Default values are intentionally conservative and suitable for most airborne survey data.

## Significance Assessment

### What this does

Significance testing converts raw elevation change into a confidence measure.

### How it is calculated

A z-score is computed per pixel:

```
z = dh / σ_dh
```

### How to interpret it

The z-score expresses change relative to expected noise. Values exceeding the configured threshold indicate that the change is unlikely to be caused by measurement uncertainty alone.

The default threshold corresponds to approximately 95 percent confidence.

## Noise Suppression

### What this does

Noise suppression prevents small, statistically insignificant changes from being classified as movement.

### How it works

Pixels where the absolute elevation change falls within the uncertainty envelope are assigned a movement rank of zero by default.

### Why this is important

Without suppression, noise can be misclassified as green or amber movement, particularly in flat terrain. Suppression ensures classification maps highlight only defensible change.

## Elevation-Based Masking

### What this does

Elevation masking limits analysis to specific vertical zones of interest.

### Why it is useful

In many environments, certain elevation ranges are expected to vary independently of true terrain change. Masking removes these areas before differencing results are interpreted.

## Masking Logic

### How it is applied

When minimum or maximum elevation limits are provided:

1. Elevation values are taken from DEM A
2. Pixels outside the specified range are flagged
3. This mask is combined with nodata handling
4. The combined mask is applied to all derived outputs

### Why DEM A is used

Using the earlier surface as the reference ensures a consistent spatial frame and avoids circular logic when interpreting change.

## Practical Applications

### Coastal and Cliff Monitoring

Water surfaces captured during surveys vary with tide and wave conditions. Masking low elevations removes false changes that can otherwise dominate the results.

### Targeted Terrain Analysis

Combining minimum and maximum elevation thresholds allows isolation of specific features such as cliff faces, terraces, or engineered benches.

This enables finer movement thresholds to be applied where smaller changes are operationally meaningful.

## Selecting Elevation Thresholds

### Guidance

Thresholds should be chosen using local vertical datums and known environmental conditions.

For UK coastal work, Mean High Water Spring relative to Ordnance Datum is a practical lower bound. Other regions should use equivalent local tidal or vertical references.

## Interpreting Zero Values

### What zero means

A value of zero does not always imply no movement.

| Mode                 | Meaning                                |
| -------------------- | -------------------------------------- |
| Uncertainty enabled  | Change is not statistically detectable |
| Uncertainty disabled | Change is below the green threshold    |

### Best practice

Always consult the z-score and noise mask outputs when interpreting areas with zero classification.

## Co-registration Diagnostics

### What this does

When surfaces are reprojected or aligned, a planar trend is fitted to the difference surface:

```
dh ≈ a·x + b·y + c
```

### How to interpret it

Significant tilt terms indicate residual misalignment rather than real terrain change. Diagnostic metrics help identify when further alignment or quality control is required.

## Memory Considerations

### How the tool behaves

DTM Differ loads full rasters into memory to simplify processing and ensure predictable behaviour.

### What to expect

Peak memory usage is approximately eight times the input raster size. Large datasets therefore require adequate system memory.

## Known Limitations

### Current constraints

* No streaming or tiled processing
* Uncertainty assumed spatially constant
* No volumetric change calculations
* Vertical unit checks depend on CRS metadata

These limitations are deliberate and prioritise clarity, reproducibility, and defensibility over maximum automation.
