from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ReportImage:
    filename: str
    title: str
    description: str | None = None


def _prepare_matplotlib_cache(out_dir: Path) -> None:
    """
    Keep Matplotlib caches inside the output directory so report generation works
    in sandboxed/CI environments.
    """
    import os

    mpl_dir = out_dir / ".mplconfig"
    cache_dir = out_dir / ".cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))


def _masked(data: NDArray[Any], mask: NDArray[np.bool_]) -> np.ma.MaskedArray:
    if np.ma.isMaskedArray(data):
        combined_mask = np.logical_or(np.ma.getmaskarray(data), mask)
        return np.ma.array(np.asarray(data), mask=combined_mask)
    return np.ma.array(data, mask=mask)

def _finite_1d(values: NDArray[Any]) -> NDArray[np.floating]:
    if np.ma.isMaskedArray(values):
        arr = values.compressed().astype(float, copy=False)
    else:
        arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _robust_limits(values: NDArray[np.floating], *, pct: float = 98.0) -> tuple[float, float]:
    finite = _finite_1d(values)
    if finite.size == 0:
        return 0.0, 1.0
    vmax = float(np.percentile(finite, pct))
    if vmax <= 0:
        vmax = float(np.max(finite)) if np.max(finite) > 0 else 1.0
    return 0.0, vmax


def save_movement_magnitude_viridis_png(
    movement_magnitude_m: NDArray[np.floating],
    *,
    nodata_mask: NDArray[np.bool_],
    out_path: Path,
    title: str = "Movement magnitude (m)",
) -> None:
    _prepare_matplotlib_cache(out_path.parent)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Green (good/low) -> Red (bad/high)
    cmap = plt.get_cmap("RdYlGn_r").copy()
    cmap.set_bad(color=(1, 1, 1, 0))  # transparent for nodata

    vmin, vmax = _robust_limits(movement_magnitude_m)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    im = ax.imshow(_masked(movement_magnitude_m, nodata_mask), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Magnitude (m)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_elevation_change_diverging_png(
    elevation_change_m: NDArray[np.floating],
    *,
    nodata_mask: NDArray[np.bool_],
    out_path: Path,
    title: str = "Elevation change (m)",
) -> None:
    _prepare_matplotlib_cache(out_path.parent)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color=(1, 1, 1, 0))

    finite = _finite_1d(elevation_change_m)
    if finite.size == 0:
        vlim = 1.0
    else:
        vlim = float(np.percentile(np.abs(finite), 98))
        if vlim <= 0:
            vlim = 1.0

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    im = ax.imshow(_masked(elevation_change_m, nodata_mask), cmap=cmap, vmin=-vlim, vmax=vlim)
    ax.set_title(title)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("dh (m)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_slope_png(
    slope_degrees: NDArray[np.floating],
    *,
    nodata_mask: NDArray[np.bool_],
    out_path: Path,
    title: str = "Slope (degrees)",
) -> None:
    _prepare_matplotlib_cache(out_path.parent)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.cm.inferno.copy()
    cmap.set_bad(color=(1, 1, 1, 0))

    finite = _finite_1d(slope_degrees)
    vmax = float(np.percentile(finite, 98)) if finite.size else 45.0
    vmax = max(vmax, 1.0)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    im = ax.imshow(_masked(slope_degrees, nodata_mask), cmap=cmap, vmin=0.0, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Slope (°)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_direction_png(
    direction: NDArray[np.int8],
    *,
    nodata_mask: NDArray[np.bool_],
    out_path: Path,
    title: str = "Direction of movement",
) -> None:
    _prepare_matplotlib_cache(out_path.parent)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    # Map -1/0/1 to 0/1/2 for display
    display = np.full(direction.shape, np.nan, dtype=float)
    display[direction == -1] = 0
    display[direction == 0] = 1
    display[direction == 1] = 2
    display[nodata_mask] = np.nan

    cmap = ListedColormap(["#1E88E5", "#BDBDBD", "#E53935"])
    cmap.set_bad(color=(1, 1, 1, 0))

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.imshow(display, cmap=cmap, vmin=0, vmax=2)
    ax.set_title(title)
    ax.axis("off")

    legend = [
        Patch(color="#E53935", label="Uplift / deposition (dh > 0)"),
        Patch(color="#1E88E5", label="Subsidence / erosion (dh < 0)"),
        Patch(color="#BDBDBD", label="No change (dh = 0)"),
    ]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=1, frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_rank_png(
    movement_rank: NDArray[np.uint8],
    *,
    nodata_mask: NDArray[np.bool_],
    out_path: Path,
    t_green: float,
    t_amber: float,
    t_red: float,
    title: str = "Ranked movement (Green / Amber / Red)",
) -> None:
    _prepare_matplotlib_cache(out_path.parent)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    display = movement_rank.astype(float)
    display[nodata_mask] = np.nan
    display[display == 0] = np.nan

    # index 1..3 -> colors; index 0 unused in display
    cmap = ListedColormap(["none", "#4CAF50", "#FFC107", "#F44336"])
    cmap.set_bad(color=(1, 1, 1, 0))

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    ax.imshow(display, cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")

    legend = [
        Patch(color="#4CAF50", label=f"Green: {t_green:.1f}–{t_amber:.1f} m"),
        Patch(color="#FFC107", label=f"Amber: {t_amber:.1f}–{t_red:.1f} m"),
        Patch(color="#F44336", label=f"Red: ≥ {t_red:.1f} m"),
    ]
    ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=3, frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_html_report(
    out_dir: Path,
    *,
    title: str,
    images: list[ReportImage],
    map_layers_dir: Path | None = None,
) -> Path:
    report_path = out_dir / "report.html"

    map_layers_html = ""
    if map_layers_dir is not None:
        rel_prefix = Path("..") / map_layers_dir.name
        layer_files = []
        for suffix in (".tif", ".tiff", ".shp", ".gpkg"):
            layer_files.extend(sorted(map_layers_dir.glob(f"*{suffix}")))

        if layer_files:
            items_html = "\n".join(
                f'<li><a href="{(rel_prefix / p.name).as_posix()}">{p.name}</a></li>' for p in layer_files
            )
            map_layers_html = f"""
            <section class="layers">
              <strong>Map layers</strong>
              <ul>
                {items_html}
              </ul>
            </section>
            """.strip()

    items = "\n".join(
        f"""
        <section class="card">
          <h2>{img.title}</h2>
          {f"<p>{img.description}</p>" if img.description else ""}
          <a href="{img.filename}"><img src="{img.filename}" alt="{img.title}"></a>
        </section>
        """.strip()
        for img in images
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; margin: 24px; color: #111; }}
    h1 {{ margin: 0 0 16px 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }}
    .card {{ border: 1px solid #e5e5e5; border-radius: 10px; padding: 12px; background: #fff; }}
    img {{ width: 100%; height: auto; border-radius: 8px; border: 1px solid #eee; }}
    p {{ margin: 8px 0 0 0; color: #333; }}
    .layers {{ margin: 12px 0 18px 0; padding: 12px; border: 1px solid #e5e5e5; border-radius: 10px; background: #fafafa; }}
    .layers ul {{ margin: 8px 0 0 18px; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  {map_layers_html}
  <div class="grid">
    {items}
  </div>
</body>
</html>
"""

    report_path.write_text(html, encoding="utf-8")
    return report_path
