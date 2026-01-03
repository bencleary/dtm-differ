from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape


@dataclass
class RankDistribution:
    unclassified: int
    unclassified_pct: float
    green: int
    green_pct: float
    amber: int
    amber_pct: float
    red: int
    red_pct: float


@dataclass
class ElevationStats:
    mean: float
    median: float
    std: float
    rmse: float
    mae: float
    nmad: float
    min: float
    max: float
    p25: float
    p75: float
    p95: float
    p99: float


@dataclass
class ConfidenceSummary:
    sigma_dh_median: float
    k_sigma: float
    within_noise_pct: float
    detectable_pct: float
    high_confidence_pct: float


@dataclass
class ProcessingInfo:
    input_a: str
    input_b: str
    crs_a: str
    crs_b: str
    output_crs: str
    reprojected: bool
    resample_method: str
    thresholds: tuple[float, float, float]
    uncertainty_mode: str
    sigma_a: float | None = None
    sigma_b: float | None = None
    sigma_coreg: float | None = None
    min_elevation: float | None = None
    max_elevation: float | None = None


@dataclass
class ReportImage:
    filename: str
    title: str
    description: str = ""


@dataclass
class ReportData:
    job_id: str
    generated_at: datetime
    version: str
    processing: ProcessingInfo
    stats: ElevationStats
    ranks: RankDistribution
    confidence: ConfidenceSummary | None
    images: list[ReportImage] = field(default_factory=list)
    valid_pixels: int = 0
    total_pixels: int = 0


def render_report(data: ReportData, output_dir: Path) -> Path:
    """Render the HTML report using Jinja2 templates."""
    env = Environment(
        loader=PackageLoader("dtm_differ", "templates"),
        autoescape=select_autoescape(["html"]),
    )

    template = env.get_template("report.html")
    html = template.render(
        job_id=data.job_id,
        generated_at=data.generated_at.strftime("%Y-%m-%d %H:%M:%S"),
        version=data.version,
        processing=data.processing,
        stats=data.stats,
        ranks=data.ranks,
        confidence=data.confidence,
        images=data.images,
        valid_pixels=data.valid_pixels,
        total_pixels=data.total_pixels,
    )

    report_path = output_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path
