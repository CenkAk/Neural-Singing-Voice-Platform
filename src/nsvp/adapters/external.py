from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from ..contracts import BackendName
from ..errors import ConfigurationError, DependencyUnavailableError, NSVPError


class ExternalExecutionError(NSVPError):
    code = "external_execution_failed"


def python_executable(configured: Path | None) -> Path:
    executable = (configured or Path(sys.executable)).resolve()
    if not executable.is_file():
        raise DependencyUnavailableError(f"External Python executable does not exist: {executable}")
    return executable


def offline_environment(cache: Path | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})
    if cache is not None:
        env["HF_HOME"] = str(cache.resolve())
    return env


def run_external(
    command: list[str], *, cwd: Path, timeout: float = 1800,
    env: dict[str, str] | None = None, label: str = "External provider",
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            command, cwd=cwd.resolve(), capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=False, timeout=timeout,
            env=env if env is not None else offline_environment(),
        )
    except subprocess.TimeoutExpired as exc:
        raise ExternalExecutionError(f"{label} exceeded its {timeout:g} second timeout") from exc
    except OSError as exc:
        raise DependencyUnavailableError(f"{label} could not start: {exc}") from exc
    if result.returncode:
        logs = (result.stdout[-1500:] + "\n" + result.stderr[-3000:]).strip()
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
