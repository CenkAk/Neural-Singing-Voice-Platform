from __future__ import annotations

import logging
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import pytest

from nsvp.adapters.external import run_external, terminate_process_tree
from nsvp.contracts import JobState
from nsvp.execution import ExecutionControl, execution_scope
from nsvp.jobs import JobStore, Worker
from nsvp.resources import device_lock
from nsvp.storage import sha256_file


def test_worker_cancels_silent_process_tree(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.db")
    child_file = tmp_path / "child.pid"
    program = (
        "import subprocess,sys,time,pathlib; "
        "print('before cancellation', flush=True); "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        f"pathlib.Path({str(child_file)!r}).write_text(str(p.pid)); time.sleep(60)"
    )

    def handler(payload: object, progress: object) -> dict[str, object]:
        run_external([sys.executable, "-c", program], cwd=tmp_path)
        return {}

    job = store.enqueue("external", {})
    errors: list[Exception] = []

    def work() -> None:
        try:
            Worker(store, {"external": handler}).run_once()
        except Exception as exc:
            logging.getLogger(__name__).exception("Test worker failed")
            errors.append(exc)

    thread = threading.Thread(target=work)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not child_file.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert child_file.exists()
        child = psutil.Process(int(child_file.read_text()))
        cancelled = store.cancel(job.id)
        assert cancelled.state is JobState.CANCELLING
        thread.join(timeout=15)
        assert not thread.is_alive()
        assert not errors
        assert store.get(job.id).state is JobState.CANCELLED
        logs = list((tmp_path / "job-logs" / job.id).rglob("*.stdout.log"))
        assert len(logs) == 1
        assert "before cancellation" in logs[0].read_text()
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
    finally:
        if thread.is_alive():
            store.cancel(job.id)
            thread.join(timeout=15)


@pytest.mark.parametrize("timeout", [False, True])
def test_worker_preserves_failed_and_timed_out_process_logs(tmp_path: Path, timeout: bool) -> None:
    store = JobStore(tmp_path / "jobs.db")
    program = "import sys,time; print('diagnostic output',flush=True); print('diagnostic error',file=sys.stderr,flush=True); "
    program += "time.sleep(30)" if timeout else "sys.exit(3)"

    def handler(payload: object, progress: object) -> dict[str, object]:
        run_external([sys.executable, "-c", program], cwd=tmp_path, timeout=0.5 if timeout else 10)
        return {}

    job = store.enqueue("external", {})
    assert Worker(store, {"external": handler}).run_once()
    assert store.get(job.id).state is JobState.FAILED
    logs = tmp_path / "job-logs" / job.id
    assert "diagnostic output" in next(logs.rglob("*.stdout.log")).read_text()
    assert "diagnostic error" in next(logs.rglob("*.stderr.log")).read_text()


def test_external_output_tail_is_bounded_but_full_log_is_retained(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    control = ExecutionControl(lambda: None, lambda *_: None, lambda _: None, logs)
    with execution_scope(control):
        result = run_external([sys.executable, "-c", "import sys; sys.stdout.write('a'*1000000+'END')"], cwd=tmp_path)
    assert len(result.stdout) == 65536
    assert result.stdout.endswith("END")
    assert next(logs.glob("*.stdout.log")).stat().st_size == 1000003


def test_hashing_can_be_cancelled_between_chunks(tmp_path: Path) -> None:
    import hashlib

    data = b"a" * (3 * 1024 * 1024)
    path = tmp_path / "weights.bin"
    path.write_bytes(data)
    checks = 0

    def cancel() -> None:
        nonlocal checks
        checks += 1
        if checks == 3:
            raise InterruptedError("hash cancelled")

    with execution_scope(ExecutionControl(cancel, lambda *_: None, lambda _: None)), pytest.raises(InterruptedError):
        sha256_file(path)
    assert checks == 3
    assert sha256_file(path) == hashlib.sha256(data).hexdigest()


def test_cancellation_wins_completion_race(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.db")
    job = store.enqueue("test", {})
    store.claim_next()
    store.cancel(job.id)
    store.succeed(job.id, {"should_not_publish": True})
    assert store.get(job.id).state is JobState.CANCELLED
    assert store.result(job.id) is None


def test_stale_jobs_fail_without_retry_and_live_heartbeat_survives(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.db")
    stale = store.enqueue("test", {})
    store.claim_next("dead")
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE jobs SET heartbeat_at='2000-01-01T00:00:00+00:00' WHERE id=?", (stale.id,))
    live = store.enqueue("test", {})
    store.claim_next("live")
    assert store.heartbeat(live.id, "live")
    assert not store.heartbeat(live.id, "other")
    assert store.recover_stale() == 1
    assert store.get(stale.id).state is JobState.FAILED
    assert store.get(stale.id).attempt_count == 1
    assert store.get(live.id).state is JobState.RUNNING
    assert store.claim_next() is None


def test_worker_kind_filter_does_not_claim_other_jobs(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.db")
    gpu = store.enqueue("conversion", {})
    cpu = store.enqueue("dataset_analyze", {})
    selected = store.claim_next("cpu-worker", ("dataset_analyze",))
    assert selected is not None and selected.id == cpu.id
    assert store.get(gpu.id).state is JobState.QUEUED


def test_pid_identity_mismatch_does_not_kill_process() -> None:
    process = psutil.Process()
    assert terminate_process_tree(process.pid, process.create_time() - 1)
    assert process.is_running()


def test_device_lock_wait_is_cancellable_and_cpu_does_not_wait(tmp_path: Path) -> None:
    def cancel() -> None:
        raise InterruptedError("cancelled")

    with device_lock("cuda:0", directory=tmp_path):
        with device_lock("cpu", directory=tmp_path):
            pass
        with execution_scope(ExecutionControl(cancel, lambda *_: None, lambda _: None)), pytest.raises(InterruptedError), device_lock("rocm:0", directory=tmp_path):
            pytest.fail("locked device was entered")
    with device_lock("cuda:0", directory=tmp_path):
        pass


def test_device_lock_serializes_independent_processes(tmp_path: Path) -> None:
    code = """
import sys
from pathlib import Path
from nsvp.resources import device_lock
root = Path(sys.argv[1])
with device_lock('cuda:1', directory=root):
    (root / 'ready').write_text('independent device acquired')
with device_lock('rocm:0', directory=root):
    print('same device acquired', flush=True)
"""
    process = None
    try:
        with device_lock("cuda:0", directory=tmp_path):
            process = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            deadline = time.monotonic() + 10
            while not (tmp_path / "ready").exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            assert (tmp_path / "ready").exists()
            with pytest.raises(subprocess.TimeoutExpired):
                process.communicate(timeout=0.3)
        output, errors = process.communicate(timeout=10)
        assert process.returncode == 0, errors
        assert output.strip() == "same device acquired"
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate(timeout=10)


@pytest.mark.parametrize("backend,device", [("rocm", "cuda:0"), ("cuda", "cuda:0"), ("mps", "mps")])
def test_training_uses_the_selected_gpu_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                            backend: str, device: str) -> None:
    from nsvp.adapters.external import EnvironmentProbe
    from nsvp.contracts import BackendName
    from nsvp.storage import LocalArtifactStore
    from nsvp.training import SeedVCTrainingBridge

    (tmp_path / "train.py").write_text("# synthetic entrypoint", encoding="utf-8")
    monkeypatch.setattr("nsvp.training.probe_environment", lambda *_: EnvironmentProbe(
        python_version="fixture", backends=[BackendName(backend)]))
    calls = []

    def launch(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(kwargs["device"])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("nsvp.training.run_external", launch)
    SeedVCTrainingBridge(tmp_path, LocalArtifactStore(tmp_path / "artifacts")).run(tmp_path / "config.yml", "fixture")
    assert calls == [device]


def test_existing_sqlite_database_is_migrated_without_losing_jobs(tmp_path: Path) -> None:
    path = tmp_path / "old.db"
    with sqlite3.connect(path) as connection:
        connection.execute("""CREATE TABLE jobs (
            id TEXT PRIMARY KEY, kind TEXT NOT NULL, state TEXT NOT NULL, payload TEXT NOT NULL,
            result TEXT, progress REAL NOT NULL, stage TEXT NOT NULL, error TEXT,
            created_at TEXT NOT NULL, started_at TEXT, completed_at TEXT,
            cancel_requested INTEGER NOT NULL DEFAULT 0)""")
        connection.execute("INSERT INTO jobs(id,kind,state,payload,progress,stage,created_at) "
            "VALUES('old','test','QUEUED','{}',0,'queued','2026-01-01')")
    store = JobStore(path)
    assert store.get("old").attempt_count == 0
    assert store.claim_next("new-worker") is not None
    assert JobStore(path).get("old").attempt_count == 1
