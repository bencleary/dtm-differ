import sqlite3
from typing import Literal


class Database:
    _conn: sqlite3.Connection | None = None

    def __init__(self, db_path):
        self.db_path = db_path

    def initialise(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        conn.commit()

        self._conn = conn

    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.commit()
            self._conn.close()

    def create_job(
        self,
        job_id: str,
        status: Literal["pending", "running", "completed", "failed"] = "pending",
    ):
        if not self._conn:
            raise ValueError("Database connection not initialized")

        cursor = self._conn.cursor()

        cursor.execute(
            """
            INSERT INTO jobs (id, status) VALUES (?, ?)
        """,
            (job_id, status),
        )

        self._conn.commit()

    def get_job_status(self, job_id: str) -> str | None:
        if not self._conn:
            raise ValueError("Database connection not initialized")

        cursor = self._conn.cursor()

        cursor.execute(
            """
            SELECT status FROM jobs WHERE id = ?
        """,
            (job_id,),
        )

        row = cursor.fetchone()
        self._conn.commit()
        return row[0] if row else None

    def update_job_status(
        self, job_id: str, status: Literal["pending", "running", "completed", "failed"]
    ):
        if not self._conn:
            raise ValueError("Database connection not initialized")

        cursor = self._conn.cursor()

        cursor.execute(
            """
            UPDATE jobs SET status = ? WHERE id = ?
            """,
            (status, job_id),
        )

        self._conn.commit()

    def delete_job(self, job_id: str):
        if not self._conn:
            raise ValueError("Database connection not initialized")

        cursor = self._conn.cursor()

        cursor.execute(
            """
            DELETE FROM jobs WHERE id = ?
            """,
            (job_id,),
        )
        self._conn.commit()
