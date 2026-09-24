from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .contracts import JobRecord, JobState
from .errors import JobStateError, NSVPError
from .execution import ExecutionControl, execution_scope

logger = logging.getLogger(__name__)

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
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, state TEXT NOT NULL,
                    payload TEXT NOT NULL, result TEXT, progress REAL NOT NULL,
                    stage TEXT NOT NULL, error TEXT, created_at TEXT NOT NULL,
                    started_at TEXT, completed_at TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0
                )"""
            )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(jobs)")}
            for name, definition in {
                "attempt_count": "INTEGER NOT NULL DEFAULT 0", "worker_id": "TEXT",
                "heartbeat_at": "TEXT", "process_id": "INTEGER", "process_created_at": "REAL",
            }.items():
                if name not in columns:
                    connection.execute(f"ALTER TABLE jobs ADD COLUMN {name} {definition}")
            connection.execute("CREATE TABLE IF NOT EXISTS nsvp_schema (version INTEGER PRIMARY KEY)")
            connection.execute("INSERT OR IGNORE INTO nsvp_schema(version) VALUES(1)")
            connection.execute("COMMIT")

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

    def list(self, kind: str, limit: int = 50, offset: int = 0) -> list[JobRecord]:
        if not 1 <= limit <= 200 or offset < 0:
            raise ValueError("invalid job pagination")
        with closing(self.connect()) as connection:
            rows = connection.execute(
                "SELECT * FROM jobs WHERE kind=? ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                (kind, limit, offset),
            ).fetchall()
        return [self._record(row) for row in rows]

    def result(self, job_id: str) -> dict[str, Any] | None:
        with closing(self.connect()) as connection:
            row = connection.execute("SELECT result FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return json.loads(row["result"]) if row["result"] else None

    def claim_next(self, worker_id: str | None = None, kinds: tuple[str, ...] = ()) -> JobRecord | None:
        with closing(self.connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            filter_sql = " AND kind IN (" + ",".join("?" for _ in kinds) + ")" if kinds else ""
            row = connection.execute(
                "SELECT id FROM jobs WHERE state=? AND cancel_requested=0" + filter_sql
                + " ORDER BY created_at,id LIMIT 1",
                (JobState.QUEUED.value, *kinds),
            ).fetchone()
            if row is None:
                connection.execute("COMMIT")
                return None
            updated = connection.execute(
                "UPDATE jobs SET state=?,stage=?,progress=?,started_at=?,worker_id=?,heartbeat_at=?,"
                "attempt_count=attempt_count+1 WHERE id=? AND state=?",
                (JobState.RUNNING.value, "starting", 0.01, _now(), worker_id, _now(), row["id"], JobState.QUEUED.value),
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
            connection.execute("BEGIN IMMEDIATE")
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
                connection.execute("UPDATE jobs SET cancel_requested=1,state=?,stage=? WHERE id=?",
                    (JobState.CANCELLING.value, "cancellation_requested", job_id))
            connection.execute("COMMIT")
        return self.get(job_id)

    def cancellation_requested(self, job_id: str) -> bool:
        with closing(self.connect()) as connection:
            row = connection.execute("SELECT cancel_requested FROM jobs WHERE id=?", (job_id,)).fetchone()
        return bool(row and row["cancel_requested"])

    def mark_cancelled(self, job_id: str) -> None:
        self._finish(job_id, JobState.CANCELLED)

    def _finish(self, job_id: str, state: JobState, result: dict[str, Any] | None = None, error: dict[str, Any] | None = None) -> None:
        with closing(self.connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT cancel_requested FROM jobs WHERE id=?", (job_id,)).fetchone()
            if state is JobState.SUCCEEDED and row and row["cancel_requested"]:
                state, result = JobState.CANCELLED, None
            changed = connection.execute(
                "UPDATE jobs SET state=?,progress=?,stage=?,result=?,error=?,completed_at=? "
                "WHERE id=? AND state IN (?,?) AND process_id IS NULL",
                (
                    state.value,
                    1.0 if state is JobState.SUCCEEDED else 0.0,
                    state.value.lower(),
                    json.dumps(result) if result is not None else None,
                    json.dumps(error) if error is not None else None,
                    _now(),
                    job_id,
                    JobState.RUNNING.value,
                    JobState.CANCELLING.value,
                ),
            ).rowcount
            connection.execute("COMMIT")
        if not changed:
            raise JobStateError(f"job {job_id} is not running")

    @staticmethod
    def _record(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            id=row["id"], kind=row["kind"], state=row["state"], payload=json.loads(row["payload"]),
            progress=row["progress"], stage=row["stage"], error=json.loads(row["error"]) if row["error"] else None,
            created_at=row["created_at"], started_at=row["started_at"], completed_at=row["completed_at"],
            attempt_count=row["attempt_count"], worker_id=row["worker_id"], heartbeat_at=row["heartbeat_at"],
        )

    def heartbeat(self, job_id: str, worker_id: str) -> bool:
        with closing(self.connect()) as connection:
            return bool(connection.execute(
                "UPDATE jobs SET heartbeat_at=? WHERE id=? AND worker_id=? AND state IN (?,?)",
                (_now(), job_id, worker_id, JobState.RUNNING.value, JobState.CANCELLING.value),
            ).rowcount)

    def track_process(self, job_id: str, worker_id: str, pid: int, created_at: float) -> None:
        with closing(self.connect()) as connection:
            changed = connection.execute(
                "UPDATE jobs SET process_id=?,process_created_at=? WHERE id=? AND worker_id=? "
                "AND state IN (?,?) AND process_id IS NULL",
                (pid, created_at, job_id, worker_id, JobState.RUNNING.value, JobState.CANCELLING.value),
            ).rowcount
        if not changed:
            raise JobStateError("worker lost ownership before process registration")

    def clear_process(self, job_id: str, worker_id: str, pid: int) -> None:
        with closing(self.connect()) as connection:
            connection.execute(
                "UPDATE jobs SET process_id=NULL,process_created_at=NULL "
                "WHERE id=? AND worker_id=? AND process_id=?", (job_id, worker_id, pid),
            )

    def recover_stale(self, lease_seconds: float = 60) -> int:
        from .adapters.external import terminate_process_tree

        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=lease_seconds)).isoformat()
        recovered = 0
        with closing(self.connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT * FROM jobs WHERE state IN (?,?) AND COALESCE(heartbeat_at,started_at,created_at)<?",
                (JobState.RUNNING.value, JobState.CANCELLING.value, cutoff),
            ).fetchall()
            for row in rows:
                if row["process_id"] is not None and not terminate_process_tree(row["process_id"], row["process_created_at"]):
                    continue
                state = JobState.CANCELLED if row["cancel_requested"] else JobState.FAILED
                connection.execute(
                    "UPDATE jobs SET state=?,stage=?,completed_at=?,error=?,process_id=NULL,"
                    "process_created_at=NULL WHERE id=?",
                    (state.value, "worker_lost", _now(), json.dumps({"code": "worker_lost",
                        "message": "Worker lease expired; job was not automatically retried."}), row["id"]),
                )
                recovered += 1
            connection.execute("COMMIT")
        return recovered


JobHandler = Callable[[dict[str, Any], Callable[[float, str], None]], dict[str, Any]]


class Worker:
    def __init__(self, store: JobStore, handlers: dict[str, JobHandler], poll_seconds: float = 1.0, *, kinds: tuple[str, ...] = ()) -> None:
        self.store = store
        self.handlers = handlers
        self.poll_seconds = poll_seconds
        self.kinds = kinds
        self.worker_id = uuid.uuid4().hex

    def run_once(self) -> bool:
        self.store.recover_stale()
        job = self.store.claim_next(self.worker_id, self.kinds)
        if job is None:
            return False
        handler = self.handlers.get(job.kind)
        if handler is None:
            self.store.fail(job.id, {"code": "unknown_job_kind", "message": job.kind})
            return True
        stopped, lost = threading.Event(), threading.Event()

        def heartbeat() -> None:
            while not stopped.wait(5):
                try:
                    if not self.store.heartbeat(job.id, self.worker_id):
                        lost.set()
                        return
                except Exception:
                    logger.exception("Worker heartbeat failed")
                    lost.set()
                    return

        def check() -> None:
            if lost.is_set():
                raise JobStateError("Worker lease lost")
            if self.store.cancellation_requested(job.id):
                raise InterruptedError("job cancellation requested")

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            def progress(value: float, stage: str) -> None:
                check()
                self.store.update_progress(job.id, value, stage)

            with execution_scope(ExecutionControl(
                check,
                lambda pid, created: self.store.track_process(job.id, self.worker_id, pid, created),
                lambda pid: self.store.clear_process(job.id, self.worker_id, pid),
                self.store.path.parent / "job-logs" / job.id / str(job.attempt_count),
            )):
                check()
                result = handler(job.payload, progress)
            self.store.succeed(job.id, result)
        except InterruptedError:
            self.store.mark_cancelled(job.id)
        except Exception as exc:
            logger.exception("Job failed: %s (%s)", job.id, job.kind)
            code = exc.code if isinstance(exc, NSVPError) else "unhandled_job_error"
            self.store.fail(job.id, {"code": code, "message": str(exc), "type": type(exc).__name__})
        finally:
            stopped.set()
            thread.join(timeout=10)
        return True

    def run_forever(self) -> None:
        while True:
            if not self.run_once():
                time.sleep(self.poll_seconds)
