from __future__ import annotations

import json
import sqlite3
import time
import uuid
from collections.abc import Callable
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import JobRecord, JobState
from .errors import JobStateError, NSVPError


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    """Durable SQLite job state for a single local worker."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def initialize(self) -> None:
        with closing(self.connect()) as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, state TEXT NOT NULL,
                    payload TEXT NOT NULL, result TEXT, progress REAL NOT NULL,
                    stage TEXT NOT NULL, error TEXT, created_at TEXT NOT NULL,
                    started_at TEXT, completed_at TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0
                )"""
            )

    def enqueue(self, kind: str, payload: dict[str, Any]) -> JobRecord:
        job_id = uuid.uuid4().hex
        created = _now()
        with closing(self.connect()) as connection:
            connection.execute(
                "INSERT INTO jobs(id,kind,state,payload,progress,stage,created_at) VALUES(?,?,?,?,?,?,?)",
                (job_id, kind, JobState.QUEUED.value, json.dumps(payload), 0.0, "queued", created),
            )
        return self.get(job_id)

    def get(self, job_id: str) -> JobRecord:
        with closing(self.connect()) as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._record(row)

    def result(self, job_id: str) -> dict[str, Any] | None:
        with closing(self.connect()) as connection:
            row = connection.execute("SELECT result FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return json.loads(row["result"]) if row["result"] else None

    def claim_next(self) -> JobRecord | None:
        with closing(self.connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT id FROM jobs WHERE state=? AND cancel_requested=0 ORDER BY created_at LIMIT 1",
                (JobState.QUEUED.value,),
            ).fetchone()
            if row is None:
                connection.execute("COMMIT")
                return None
            updated = connection.execute(
                "UPDATE jobs SET state=?,stage=?,progress=?,started_at=? WHERE id=? AND state=?",
                (JobState.RUNNING.value, "starting", 0.01, _now(), row["id"], JobState.QUEUED.value),
            ).rowcount
            connection.execute("COMMIT")
        return self.get(row["id"]) if updated else None

    def update_progress(self, job_id: str, progress: float, stage: str) -> None:
        with closing(self.connect()) as connection:
            connection.execute(
                "UPDATE jobs SET progress=?,stage=? WHERE id=? AND state=?",
                (max(0.0, min(1.0, progress)), stage, job_id, JobState.RUNNING.value),
            )

    def succeed(self, job_id: str, result: dict[str, Any]) -> None:
        self._finish(job_id, JobState.SUCCEEDED, result=result)

    def fail(self, job_id: str, error: dict[str, Any]) -> None:
        self._finish(job_id, JobState.FAILED, error=error)

    def cancel(self, job_id: str) -> JobRecord:
        with closing(self.connect()) as connection:
            row = connection.execute("SELECT state FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            state = JobState(row["state"])
            if state in (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED):
                raise JobStateError(f"cannot cancel a terminal {state.value} job")
            if state is JobState.QUEUED:
                connection.execute(
                    "UPDATE jobs SET state=?,stage=?,completed_at=?,cancel_requested=1 WHERE id=?",
                    (JobState.CANCELLED.value, "cancelled", _now(), job_id),
                )
            else:
                connection.execute("UPDATE jobs SET cancel_requested=1,stage=? WHERE id=?", ("cancellation_requested", job_id))
        return self.get(job_id)

    def cancellation_requested(self, job_id: str) -> bool:
        with closing(self.connect()) as connection:
            row = connection.execute("SELECT cancel_requested FROM jobs WHERE id=?", (job_id,)).fetchone()
        return bool(row and row["cancel_requested"])

    def mark_cancelled(self, job_id: str) -> None:
        self._finish(job_id, JobState.CANCELLED)

    def _finish(self, job_id: str, state: JobState, result: dict[str, Any] | None = None, error: dict[str, Any] | None = None) -> None:
        with closing(self.connect()) as connection:
            changed = connection.execute(
                "UPDATE jobs SET state=?,progress=?,stage=?,result=?,error=?,completed_at=? WHERE id=? AND state=?",
                (
                    state.value,
                    1.0 if state is JobState.SUCCEEDED else 0.0,
                    state.value.lower(),
                    json.dumps(result) if result is not None else None,
                    json.dumps(error) if error is not None else None,
                    _now(),
                    job_id,
                    JobState.RUNNING.value,
                ),
            ).rowcount
        if not changed:
            raise JobStateError(f"job {job_id} is not running")

    @staticmethod
    def _record(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            id=row["id"], kind=row["kind"], state=row["state"], payload=json.loads(row["payload"]),
            progress=row["progress"], stage=row["stage"], error=json.loads(row["error"]) if row["error"] else None,
            created_at=row["created_at"], started_at=row["started_at"], completed_at=row["completed_at"],
        )


JobHandler = Callable[[dict[str, Any], Callable[[float, str], None]], dict[str, Any]]


class Worker:
    def __init__(self, store: JobStore, handlers: dict[str, JobHandler], poll_seconds: float = 1.0) -> None:
        self.store = store
        self.handlers = handlers
        self.poll_seconds = poll_seconds

    def run_once(self) -> bool:
        job = self.store.claim_next()
        if job is None:
            return False
        handler = self.handlers.get(job.kind)
        if handler is None:
            self.store.fail(job.id, {"code": "unknown_job_kind", "message": job.kind})
            return True
        try:
            def progress(value: float, stage: str) -> None:
                if self.store.cancellation_requested(job.id):
                    raise InterruptedError("job cancellation requested")
                self.store.update_progress(job.id, value, stage)

            result = handler(job.payload, progress)
            self.store.succeed(job.id, result)
        except InterruptedError:
            self.store.mark_cancelled(job.id)
        except Exception as exc:  # noqa: BLE001 - worker must persist all task failures
            code = exc.code if isinstance(exc, NSVPError) else "unhandled_job_error"
            self.store.fail(job.id, {"code": code, "message": str(exc), "type": type(exc).__name__})
        return True

    def run_forever(self) -> None:
        while True:
            if not self.run_once():
                time.sleep(self.poll_seconds)
