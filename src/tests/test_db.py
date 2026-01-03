import uuid

from dtm_differ.db import Database


def test_get_job_status():
    db = Database(":memory:")
    db.initialise()
    job_id = str(uuid.uuid4())
    db.create_job(job_id)
    status = db.get_job_status(job_id)
    assert status == "pending"


def test_update_job_status():
    db = Database(":memory:")
    db.initialise()
    job_id = str(uuid.uuid4())
    db.create_job(job_id)
    db.update_job_status(job_id, status="completed")
    status = db.get_job_status(job_id)
    assert status == "completed"


def test_update_job_status_with_no_job():
    db = Database(":memory:")
    db.initialise()
    job_id = str(uuid.uuid4())
    db.update_job_status(job_id, status="completed")
    status = db.get_job_status(job_id)
    assert status == "completed"


def test_update_job_status_with_invalid_status():
    db = Database(":memory:")
    db.initialise()
    job_id = str(uuid.uuid4())
    db.create_job(job_id)
    db.update_job_status(job_id, status="invalid")
    status = db.get_job_status(job_id)
    assert status == "pending"


def test_update_job_status_with_invalid_status_and_no_job():
    db = Database(":memory:")
    db.initialise()
    job_id = str(uuid.uuid4())
    db.update_job_status(job_id, status="invalid")
    status = db.get_job_status(job_id)
    assert status == "completed"
