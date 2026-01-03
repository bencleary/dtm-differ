import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest
import xdem
from numpy.typing import NDArray
from rasterio.transform import from_bounds, from_origin

from dtm_differ.db import Database

_repo_root = Path(__file__).resolve().parents[2]

# Path to generated test DTMs
TEST_DTM_DIR = _repo_root / "test_data" / "sample_dtms"


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for each test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_dtm_dir():
    """Get the test DTM directory, generating if needed."""
    if not TEST_DTM_DIR.exists() or not list(TEST_DTM_DIR.glob("*.tif")):
        pytest.skip(
            f"Test DTMs not found in {TEST_DTM_DIR}. "
            "Run 'make generate-test-dtms' to generate them."
        )
    return TEST_DTM_DIR


def make_test_dem(
    data: NDArray[np.floating],
    *,
    nodata: float | None = None,
    bounds: tuple[float, float, float, float] = (0, 0, 100, 100),
    crs: str = "EPSG:4326",
    use_origin: bool = False,
) -> xdem.DEM:
    """
    Build an xdem.DEM from a numpy array for testing.

    Handles the nodata/NaN dance that xdem requires: we convert nodata sentinel
    values and non-finite values to NaN before passing to xdem, which avoids
    spurious warnings during tests.
    """
    height, width = data.shape

    if use_origin:
        transform = from_origin(bounds[0], bounds[3], 1.0, 1.0)
    else:
        transform = from_bounds(*bounds, width, height)

    arr = data.astype(np.float32, copy=True)

    bad = ~np.isfinite(arr)
    if nodata is not None:
        bad |= np.isclose(arr, nodata)
    arr[bad] = np.nan

    return xdem.DEM.from_array(arr, transform=transform, nodata=np.nan, crs=crs)


@pytest.fixture
def create_test_dem():
    return make_test_dem


@pytest.fixture
def db():
    """Create a database with a temp file that persists across calls."""
    database = Database(":memory:")
    database.initialise()
    yield database

    # Path(db_path).unlink(missing_ok=True)


# Alias for backwards compatibility with existing tests
test_db = db


@pytest.fixture
def test_job_id():
    """Generate a unique job ID for testing."""
    return str(uuid.uuid4())
