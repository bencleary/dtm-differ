from dtm_differ.storage import check_if_job_directory_exists, create_directory


def test_create_directory(temp_output_dir):
    create_directory(temp_output_dir, "test-job")
    assert (temp_output_dir / "test-job").exists()


def test_check_if_job_directory_exists(temp_output_dir):
    create_directory(temp_output_dir, "test-job")
    assert check_if_job_directory_exists(temp_output_dir, "test-job")


def test_check_if_job_directory_does_not_exist(temp_output_dir):
    create_directory(temp_output_dir, "test-job")
    assert not check_if_job_directory_exists(temp_output_dir, "nonexistent_job")
