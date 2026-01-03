"""DTM-Differ: Digital Terrain Model comparison with uncertainty propagation.

A command-line tool for detecting elevation changes between DTMs. Built for
geotechnical monitoring workflows—subsidence tracking, erosion detection,
cut/fill analysis.
"""

from dtm_differ.db import Database
from dtm_differ.pipeline import run_pipeline
from dtm_differ.pipeline.types import ProcessingConfig, ProcessingResult

__version__ = "0.1.0"

__all__ = [
    # Main pipeline
    "run_pipeline",
    # Configuration
    "ProcessingConfig",
    "ProcessingResult",
    # Database
    "Database",
    # Version
    "__version__",
]
