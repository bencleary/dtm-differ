"""Tests for the report generation module."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from dtm_differ.pipeline.stages import make_workspace
from dtm_differ.pipeline.types import Workspace
from dtm_differ.report import (
    ConfidenceSummary,
    ElevationStats,
    ProcessingInfo,
    RankDistribution,
    ReportData,
    ReportImage,
    render_report,
)


@pytest.fixture
def sample_processing_info():
    """Create sample ProcessingInfo for testing."""
    return ProcessingInfo(
        input_a="dem_before.tif",
        input_b="dem_after.tif",
        crs_a="EPSG:32632",
        crs_b="EPSG:32632",
        output_crs="EPSG:32632",
        reprojected=False,
        resample_method="bilinear",
        thresholds=(1.0, 2.0, 5.0),
        uncertainty_mode="constant",
        sigma_a=0.1,
        sigma_b=0.1,
        sigma_coreg=0.05,
    )


@pytest.fixture
def sample_elevation_stats():
    """Create sample ElevationStats for testing."""
    return ElevationStats(
        mean=0.5,
        median=0.3,
        std=1.2,
        rmse=1.3,
        mae=0.8,
        nmad=0.9,
        min=-5.0,
        max=10.0,
        p25=-0.2,
        p75=1.0,
        p95=3.5,
        p99=5.0,
    )


@pytest.fixture
def sample_rank_distribution():
    """Create sample RankDistribution for testing."""
    return RankDistribution(
        unclassified=100,
        unclassified_pct=10.0,
        green=500,
        green_pct=50.0,
        amber=300,
        amber_pct=30.0,
        red=100,
        red_pct=10.0,
    )


@pytest.fixture
def sample_confidence_summary():
    """Create sample ConfidenceSummary for testing."""
    return ConfidenceSummary(
        sigma_dh_median=0.15,
        k_sigma=1.96,
        within_noise_pct=25.0,
        detectable_pct=75.0,
        high_confidence_pct=60.0,
    )


@pytest.fixture
def sample_report_data(
    sample_processing_info,
    sample_elevation_stats,
    sample_rank_distribution,
    sample_confidence_summary,
):
    """Create sample ReportData for testing."""
    return ReportData(
        job_id="test-job-123",
        generated_at=datetime(2024, 1, 15, 10, 30, 0),
        version="0.1.0",
        processing=sample_processing_info,
        stats=sample_elevation_stats,
        ranks=sample_rank_distribution,
        confidence=sample_confidence_summary,
        images=[
            ReportImage("movement.png", "Movement Magnitude", "Shows movement"),
            ReportImage("elevation.png", "Elevation Change"),
        ],
        valid_pixels=1000,
        total_pixels=1200,
    )


# --- Dataclass tests ---


def test_processing_info_creation():
    """Test ProcessingInfo can be created with required fields."""
    info = ProcessingInfo(
        input_a="a.tif",
        input_b="b.tif",
        crs_a="EPSG:4326",
        crs_b="EPSG:4326",
        output_crs="EPSG:4326",
        reprojected=False,
        resample_method="nearest",
        thresholds=(1.0, 2.0, 3.0),
        uncertainty_mode="none",
    )
    assert info.input_a == "a.tif"
    assert info.sigma_a is None
    assert info.sigma_b is None
    assert info.sigma_coreg is None


def test_processing_info_with_uncertainty():
    """Test ProcessingInfo with uncertainty parameters."""
    info = ProcessingInfo(
        input_a="a.tif",
        input_b="b.tif",
        crs_a="EPSG:4326",
        crs_b="EPSG:4326",
        output_crs="EPSG:4326",
        reprojected=True,
        resample_method="bilinear",
        thresholds=(1.0, 2.0, 3.0),
        uncertainty_mode="constant",
        sigma_a=0.1,
        sigma_b=0.15,
        sigma_coreg=0.05,
    )
    assert info.sigma_a == 0.1
    assert info.sigma_b == 0.15
    assert info.sigma_coreg == 0.05


def test_elevation_stats_creation():
    """Test ElevationStats stores all statistics."""
    stats = ElevationStats(
        mean=1.0,
        median=0.9,
        std=0.5,
        rmse=0.6,
        mae=0.4,
        nmad=0.45,
        min=-2.0,
        max=5.0,
        p25=0.5,
        p75=1.5,
        p95=3.0,
        p99=4.0,
    )
    assert stats.mean == 1.0
    assert stats.median == 0.9
    assert stats.min == -2.0
    assert stats.max == 5.0


def test_rank_distribution_percentages():
    """Test RankDistribution stores counts and percentages."""
    ranks = RankDistribution(
        unclassified=50,
        unclassified_pct=5.0,
        green=600,
        green_pct=60.0,
        amber=250,
        amber_pct=25.0,
        red=100,
        red_pct=10.0,
    )
    assert ranks.green == 600
    assert ranks.green_pct == 60.0
    total_pct = (
        ranks.unclassified_pct + ranks.green_pct + ranks.amber_pct + ranks.red_pct
    )
    assert total_pct == 100.0


def test_confidence_summary_creation():
    """Test ConfidenceSummary creation."""
    confidence = ConfidenceSummary(
        sigma_dh_median=0.2,
        k_sigma=2.0,
        within_noise_pct=30.0,
        detectable_pct=70.0,
        high_confidence_pct=50.0,
    )
    assert confidence.sigma_dh_median == 0.2
    assert confidence.k_sigma == 2.0
    assert confidence.within_noise_pct + confidence.detectable_pct == 100.0


def test_report_image_with_description():
    """Test ReportImage with optional description."""
    img = ReportImage(
        filename="test.png",
        title="Test Image",
        description="A description of the image",
    )
    assert img.filename == "test.png"
    assert img.title == "Test Image"
    assert img.description == "A description of the image"


def test_report_image_without_description():
    """Test ReportImage with default empty description."""
    img = ReportImage(filename="test.png", title="Test Image")
    assert img.description == ""


def test_report_data_with_confidence(sample_report_data):
    """Test ReportData with confidence summary."""
    assert sample_report_data.job_id == "test-job-123"
    assert sample_report_data.confidence is not None
    assert len(sample_report_data.images) == 2


def test_report_data_without_confidence(
    sample_processing_info,
    sample_elevation_stats,
    sample_rank_distribution,
):
    """Test ReportData without confidence summary (uncertainty disabled)."""
    data = ReportData(
        job_id="no-confidence-job",
        generated_at=datetime.now(),
        version="0.1.0",
        processing=sample_processing_info,
        stats=sample_elevation_stats,
        ranks=sample_rank_distribution,
        confidence=None,
        images=[],
        valid_pixels=500,
        total_pixels=600,
    )
    assert data.confidence is None
    assert data.images == []


def test_report_data_default_values(
    sample_processing_info,
    sample_elevation_stats,
    sample_rank_distribution,
):
    """Test ReportData default values for optional fields."""
    data = ReportData(
        job_id="defaults-job",
        generated_at=datetime.now(),
        version="0.1.0",
        processing=sample_processing_info,
        stats=sample_elevation_stats,
        ranks=sample_rank_distribution,
        confidence=None,
    )
    assert data.images == []
    assert data.valid_pixels == 0
    assert data.total_pixels == 0


# --- render_report tests ---


def test_render_report_creates_file(sample_report_data):
    """Test that render_report creates an HTML file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        result_path = render_report(sample_report_data, output_dir)

        assert result_path.exists()
        assert result_path.name == "report.html"
        assert result_path.parent == output_dir


def test_render_report_returns_correct_path(sample_report_data):
    """Test that render_report returns the correct path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        result_path = render_report(sample_report_data, output_dir)

        expected_path = output_dir / "report.html"
        assert result_path == expected_path


def test_render_report_content_contains_job_id(sample_report_data):
    """Test that rendered HTML contains the job ID."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        result_path = render_report(sample_report_data, output_dir)

        content = result_path.read_text(encoding="utf-8")
        assert sample_report_data.job_id in content


def test_render_report_content_contains_version(sample_report_data):
    """Test that rendered HTML contains the version."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        result_path = render_report(sample_report_data, output_dir)

        content = result_path.read_text(encoding="utf-8")
        assert sample_report_data.version in content


def test_render_report_content_contains_statistics(sample_report_data):
    """Test that rendered HTML contains elevation statistics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        result_path = render_report(sample_report_data, output_dir)

        content = result_path.read_text(encoding="utf-8")
        assert "0.5" in content  # mean
        assert "0.3" in content  # median


def test_render_report_content_contains_images(sample_report_data):
    """Test that rendered HTML contains image references."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        result_path = render_report(sample_report_data, output_dir)

        content = result_path.read_text(encoding="utf-8")
        for img in sample_report_data.images:
            assert img.filename in content
            assert img.title in content


def test_render_report_content_contains_processing_info(sample_report_data):
    """Test that rendered HTML contains processing information."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        result_path = render_report(sample_report_data, output_dir)

        content = result_path.read_text(encoding="utf-8")
        assert sample_report_data.processing.input_a in content
        assert sample_report_data.processing.input_b in content


def test_render_report_content_contains_confidence_when_present(sample_report_data):
    """Test that rendered HTML contains confidence info when provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        result_path = render_report(sample_report_data, output_dir)

        content = result_path.read_text(encoding="utf-8")
        assert sample_report_data.confidence is not None
        assert "1.96" in content  # k_sigma value


def test_render_report_without_confidence(
    sample_processing_info,
    sample_elevation_stats,
    sample_rank_distribution,
):
    """Test rendering report without confidence summary."""
    data = ReportData(
        job_id="no-confidence",
        generated_at=datetime(2024, 1, 1, 12, 0, 0),
        version="0.1.0",
        processing=sample_processing_info,
        stats=sample_elevation_stats,
        ranks=sample_rank_distribution,
        confidence=None,
        images=[],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        result_path = render_report(data, output_dir)

        assert result_path.exists()
        content = result_path.read_text(encoding="utf-8")
        assert "no-confidence" in content


def test_render_report_is_valid_html(sample_report_data):
    """Test that rendered output is valid HTML structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        result_path = render_report(sample_report_data, output_dir)

        content = result_path.read_text(encoding="utf-8")
        assert "<!doctype html>" in content.lower()
        assert "<html" in content.lower()
        assert "</html>" in content.lower()
        assert "<head>" in content.lower()
        assert "</head>" in content.lower()
        assert "<body" in content.lower()
        assert "</body>" in content.lower()


def test_render_report_has_tailwind_cdn(sample_report_data):
    """Test that rendered HTML includes Tailwind CSS CDN."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        result_path = render_report(sample_report_data, output_dir)

        content = result_path.read_text(encoding="utf-8")
        assert "cdn.tailwindcss.com" in content
        assert "tailwind.config" in content


def test_render_report_has_dark_theme_colors(sample_report_data):
    """Test that rendered HTML contains dark theme color configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        result_path = render_report(sample_report_data, output_dir)

        content = result_path.read_text(encoding="utf-8")
        assert "background:" in content
        assert "foreground:" in content
        assert "primary:" in content
        assert '"muted-foreground":' in content or "muted-foreground:" in content


def test_render_report_has_overview_cards(sample_report_data):
    """Test that rendered HTML contains overview cards section."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        result_path = render_report(sample_report_data, output_dir)

        content = result_path.read_text(encoding="utf-8")
        assert "Valid Pixels" in content
        assert "Mean Bias" in content
        assert "Detectable Changes" in content


def test_render_report_has_rank_distribution_badges(sample_report_data):
    """Test that rendered HTML contains rank distribution with badges."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        result_path = render_report(sample_report_data, output_dir)

        content = result_path.read_text(encoding="utf-8")
        assert "bg-primary" in content  # Green badge
        assert "bg-rank-amber" in content  # Amber badge
        assert "bg-destructive" in content  # Red badge
        assert "Movement Rank Distribution" in content


def test_render_report_has_rank_bar_visualization(sample_report_data):
    """Test that rendered HTML contains the rank bar visualization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        result_path = render_report(sample_report_data, output_dir)

        content = result_path.read_text(encoding="utf-8")
        assert "rank-bar" in content
        assert "bg-rank-green" in content
        assert "bg-rank-amber" in content
        assert "bg-rank-red" in content


def test_render_report_uses_tailwind_classes(sample_report_data):
    """Test that rendered HTML uses Tailwind utility classes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        result_path = render_report(sample_report_data, output_dir)

        content = result_path.read_text(encoding="utf-8")
        # Check for common Tailwind classes used in the template
        assert "bg-card" in content
        assert "border-border" in content
        assert "text-foreground" in content
        assert "text-muted-foreground" in content
        assert "font-mono" in content


def test_render_report_contains_dtm_differ_branding(sample_report_data):
    """Test that report contains DTM Differ branding."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        result_path = render_report(sample_report_data, output_dir)

        content = result_path.read_text(encoding="utf-8")
        assert "DTM Differ Report" in content
        assert "Digital Terrain Model Analysis" in content


def test_render_report_formatted_datetime(sample_report_data):
    """Test that datetime is formatted correctly in output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        result_path = render_report(sample_report_data, output_dir)

        content = result_path.read_text(encoding="utf-8")
        assert "2024-01-15 10:30:00" in content


def test_render_report_overwrites_existing(sample_report_data):
    """Test that render_report overwrites an existing report."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        report_path = output_dir / "report.html"

        report_path.write_text("old content", encoding="utf-8")

        render_report(sample_report_data, output_dir)

        content = report_path.read_text(encoding="utf-8")
        assert "old content" not in content
        assert sample_report_data.job_id in content


def test_render_report_empty_images_list(
    sample_processing_info,
    sample_elevation_stats,
    sample_rank_distribution,
):
    """Test rendering report with empty images list."""
    data = ReportData(
        job_id="empty-images",
        generated_at=datetime.now(),
        version="0.1.0",
        processing=sample_processing_info,
        stats=sample_elevation_stats,
        ranks=sample_rank_distribution,
        confidence=None,
        images=[],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        result_path = render_report(data, output_dir)

        assert result_path.exists()
        content = result_path.read_text(encoding="utf-8")
        assert "empty-images" in content


# --- Workspace integration tests ---


def test_workspace_has_job_id():
    """Test that Workspace dataclass has job_id field."""
    ws = Workspace(
        job_id="test-123",
        out_dir=Path("/tmp/test"),
        map_layers_dir=Path("/tmp/test/map_layers"),
        reports_dir=Path("/tmp/test/reports"),
    )
    assert ws.job_id == "test-123"


def test_make_workspace_accepts_job_id():
    """Test that make_workspace accepts and stores job_id."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ws = make_workspace(Path(tmpdir), job_id="workspace-job-456")
        assert ws.job_id == "workspace-job-456"


def test_make_workspace_default_job_id():
    """Test that make_workspace has default empty job_id."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ws = make_workspace(Path(tmpdir))
        assert ws.job_id == ""


def test_make_workspace_creates_directories():
    """Test that make_workspace creates required directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "output"
        ws = make_workspace(out_dir, job_id="test")

        assert ws.out_dir.exists()
        assert ws.map_layers_dir.exists()
        assert ws.reports_dir.exists()
