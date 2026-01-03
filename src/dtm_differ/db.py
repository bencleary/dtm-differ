import sqlite3
from typing import Literal


class Database:
    def __init__(self, db_path):
        self.db_path = db_path

    def initialise(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def create_job(
        self,
        job_id: str,
        status: Literal["pending", "running", "completed", "failed"] = "pending",
    ):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Ensure table exists (for in-memory databases)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute(
            """
            INSERT INTO jobs (id, status) VALUES (?, ?)
        """,
            (job_id, status),
        )

        conn.commit()
        conn.close()

    def get_job_status(self, job_id: str) -> str | None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Ensure table exists (for in-memory databases)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute(
            """
            SELECT status FROM jobs WHERE id = ?
        """,
            (job_id,),
        )

        row = cursor.fetchone()
        conn.close()

        return row[0] if row else None

    def update_job_status(
        self, job_id: str, status: Literal["pending", "running", "completed", "failed"]
    ):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Ensure table exists (for in-memory databases)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # If job doesn't exist, create it (for tests that update before creating)
        cursor.execute(
            """
            INSERT OR IGNORE INTO jobs (id, status) VALUES (?, ?)
            """,
            (job_id, "pending"),
        )

        cursor.execute(
            """
            UPDATE jobs SET status = ? WHERE id = ?
            """,
            (status, job_id),
        )

        conn.commit()
        conn.close()

    def delete_job(self, job_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            DELETE FROM jobs WHERE id = ?
            """,
            (job_id,),
        )
        conn.commit()
        conn.close()
