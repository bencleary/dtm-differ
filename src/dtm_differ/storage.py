from pathlib import Path


def create_directory(output_dir: Path, job: str) -> None:
    """Create a directory if it does not exist."""
    target_path = output_dir / job
    target_path.mkdir(parents=True, exist_ok=True)


def check_if_job_directory_exists(output_dir: Path, job: str) -> bool:
    """Check if a job directory exists."""
    return (output_dir / job).exists()
