import os
import shutil
from pathlib import Path

from dtm_differ.storage import check_if_job_directory_exists, create_directory


def test_create_directory():
    test_path = Path("test_directory")
    create_directory(test_path, "test-job")
    assert os.path.exists(test_path)
    shutil.rmtree(test_path)


def test_check_if_job_directory_exists():
    test_path = Path("test_directory")
    create_directory(test_path, "test-job")
    assert check_if_job_directory_exists(test_path, "test-job")
    shutil.rmtree(test_path)


def test_check_if_job_directory_does_not_exist():
    test_path = Path("test_directory")
    create_directory(test_path, "test-job")
    assert not check_if_job_directory_exists(test_path, "nonexistent_job")
    shutil.rmtree(test_path)
