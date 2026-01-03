import uuid


def test_create_and_get_job_status(db):
    job_id = str(uuid.uuid4())

    db.create_job(job_id)
    status = db.get_job_status(job_id)

    assert status == "pending"


def test_get_job_status_missing_job(db):
    status = db.get_job_status("nonexistent-job-id")
    assert status is None


def test_update_job_status(db):
    job_id = str(uuid.uuid4())
    db.create_job(job_id)

    db.update_job_status(job_id, status="running")
    assert db.get_job_status(job_id) == "running"

    db.update_job_status(job_id, status="completed")
    assert db.get_job_status(job_id) == "completed"


def test_update_nonexistent_job_does_nothing(db):
    """Updating a job that doesn't exist should not create it."""
    db.update_job_status("nonexistent", status="completed")
    status = db.get_job_status("nonexistent")
    assert status is None


def test_delete_job(db):
    job_id = str(uuid.uuid4())
    db.create_job(job_id)

    db.delete_job(job_id)

    assert db.get_job_status(job_id) is None
