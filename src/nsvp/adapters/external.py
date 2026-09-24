from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import psutil
from pydantic import BaseModel, Field

from ..contracts import BackendName, ProcessTelemetry
from ..errors import ConfigurationError, DependencyUnavailableError, NSVPError
from ..execution import check_cancelled, current_execution
from ..resources import device_lock


class ExternalExecutionError(NSVPError):
    code = "external_execution_failed"


def python_executable(configured: Path | None) -> Path:
    # Resolving a venv's Python symlink selects the system interpreter and loses its packages.
    executable = Path(os.path.abspath(configured or sys.executable))
    if not executable.is_file():
        raise DependencyUnavailableError(f"External Python executable does not exist: {executable}")
    return executable


def offline_environment(cache: Path | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    if cache is not None:
        env["HF_HOME"] = str(cache.resolve())
        env["HF_HUB_CACHE"] = str(cache.resolve() / "hub")
    return env


def terminate_process_tree(pid: int, created_at: float, grace_seconds: float = 5) -> bool:
    """Only terminate the recorded process identity, never a reused PID."""
    try:
        parent = psutil.Process(pid)
        if parent.create_time() != created_at:
            return True
        processes = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        return True
    for process in reversed(processes):
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=grace_seconds)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=5)
    return not alive


def run_external(
    command: list[str], *, cwd: Path, timeout: float = 1800,
    env: dict[str, str] | None = None, label: str = "External provider",
    device: str | None = None,
    telemetry_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    if timeout <= 0:
        raise ValueError("external timeout must be positive")
    check_cancelled()
    control = current_execution.get()
    log_directory = control.log_directory if control is not None else None
    if log_directory is not None:
        log_directory.mkdir(parents=True, exist_ok=True)
    prefix = uuid.uuid4().hex + "-"
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NO_WINDOW
    # Files prevent a verbose provider from blocking on full pipes or exhausting RAM.
    with (
        device_lock(device),
        tempfile.NamedTemporaryFile(prefix=prefix, suffix=".stdout.log", dir=log_directory,
                                    delete=log_directory is None) as stdout,
        tempfile.NamedTemporaryFile(prefix=prefix, suffix=".stderr.log", dir=log_directory,
                                    delete=log_directory is None) as stderr,
    ):
        try:
            process = subprocess.Popen(
                command, cwd=cwd.resolve(), stdout=stdout, stderr=stderr,
                env=env if env is not None else offline_environment(),
                creationflags=creationflags,
                start_new_session=os.name != "nt",
            )
        except OSError as exc:
            raise DependencyUnavailableError(f"{label} could not start: {exc}") from exc
        started = time.monotonic()
        peak_rss: int | None = None
        samples = missed_samples = 0
        created_at: float | None = None
        try:
            try:
                created_at = psutil.Process(process.pid).create_time()
            except psutil.NoSuchProcess:
                pass
            if control is not None and created_at is not None:
                control.process_started(process.pid, created_at)
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                check_cancelled()
                if telemetry_path is not None:
                    try:
                        parent = psutil.Process(process.pid)
                        if parent.create_time() != created_at:
                            raise psutil.NoSuchProcess(process.pid)
                        rss = sum(item.memory_info().rss for item in [parent, *parent.children(recursive=True)])
                        peak_rss = max(peak_rss or 0, rss)
                        samples += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        missed_samples += 1
                if time.monotonic() >= deadline:
                    raise ExternalExecutionError(f"{label} exceeded its {timeout:g} second timeout")
                time.sleep(0.1)
            check_cancelled()
        except BaseException:
            if created_at is not None:
                if not terminate_process_tree(process.pid, created_at):
                    raise ExternalExecutionError("External process tree could not be stopped") from None
            elif process.poll() is None:
                process.kill()
            process.wait(timeout=10)
            raise
        finally:
            if control is not None and process.poll() is not None:
                control.process_finished(process.pid)
            if telemetry_path is not None:
                telemetry_path.write_text(ProcessTelemetry(
                    elapsed_seconds=time.monotonic() - started, sampled_peak_rss_bytes=peak_rss,
                    samples=samples, missed_samples=missed_samples, returncode=process.poll(),
                ).model_dump_json(indent=2), encoding="utf-8")
        # Full worker logs stay on disk; only a bounded tail is returned to callers.
        stdout.seek(max(0, stdout.seek(0, os.SEEK_END) - 65536))
        stderr.seek(max(0, stderr.seek(0, os.SEEK_END) - 65536))
        result = subprocess.CompletedProcess(
            command, process.returncode,
            stdout.read().decode("utf-8", errors="replace"),
            stderr.read().decode("utf-8", errors="replace"),
        )
    if result.returncode:
        logs = (result.stdout[-1500:].strip() + "\n" + result.stderr[-3000:].strip()).strip()
        raise ExternalExecutionError(f"{label} exited with code {result.returncode}: {logs}")
    return result


def validate_revision(repository: Path, expected: str | None) -> str:
    if expected is None:
        return "unknown"
    if len(expected) != 40 or any(c not in "0123456789abcdefABCDEF" for c in expected):
        raise ConfigurationError("provider revision must be a full Git commit SHA")
    result = run_external(
        ["git", "rev-parse", "HEAD"], cwd=repository, timeout=15, label="Repository revision check",
    )
    actual = result.stdout.strip()
    if actual.lower() != expected.lower():
        raise ConfigurationError("provider checkout does not match its configured revision")
    dirty = run_external(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repository, timeout=15, label="Repository status check",
    )
    if dirty.stdout.strip():
        raise ConfigurationError("pinned provider checkout has modified tracked files")
    return actual


class EnvironmentProbe(BaseModel):
    python_version: str
    torch_version: str | None = None
    backends: list[BackendName] = Field(default_factory=lambda: [BackendName.CPU])
    device_names: dict[str, str] = Field(default_factory=dict)
    modules: dict[str, bool] = Field(default_factory=dict)


_PROBE = """
import importlib.util, json, platform, sys
result = {'python_version': platform.python_version(), 'backends': ['cpu'],
          'device_names': {'cpu': platform.processor() or 'CPU'}, 'modules': {}}
for name in sys.argv[1:]:
    result['modules'][name] = importlib.util.find_spec(name) is not None
if importlib.util.find_spec('torch') is not None:
    import torch
    result['torch_version'] = str(torch.__version__)
    if torch.cuda.is_available():
        backend = 'rocm' if getattr(torch.version, 'hip', None) else 'cuda'
        result['backends'].append(backend)
        result['device_names'][backend] = torch.cuda.get_device_name(0)
    mps = getattr(torch.backends, 'mps', None)
    if mps is not None and mps.is_available():
        result['backends'].append('mps')
        result['device_names']['mps'] = 'Apple Metal'
if importlib.util.find_spec('torch_directml') is not None:
    result['backends'].append('directml')
print(json.dumps(result))
"""


def probe_environment(executable: Path | None, modules: tuple[str, ...] = ()) -> EnvironmentProbe:
    python = python_executable(executable)
    result = run_external(
        [str(python), "-c", _PROBE, *modules], cwd=python.parent,
        timeout=60, label="Provider environment probe",
    )
    try:
        return EnvironmentProbe.model_validate(json.loads(result.stdout.strip().splitlines()[-1]))
    except (ValueError, IndexError) as exc:
        raise ExternalExecutionError("Provider environment probe returned invalid metadata") from exc
