from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from test_dataset_pipeline import fixture_audio

from nsvp.adapters.external import ExternalExecutionError, python_executable, run_external
from nsvp.adapters.seed_vc import SeedVCConverter, SeedVCSettings
from nsvp.audio.io import save_audio
from nsvp.errors import NSVPError


def test_external_python_default_is_explicit() -> None:
    assert python_executable(None) == Path(sys.executable).resolve()


def test_external_timeout_and_failed_logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired("fixture", 1)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(ExternalExecutionError, match="timeout"):
        run_external(["fixture"], cwd=tmp_path, timeout=1)
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess("fixture", 2, "output", "error"))
    with pytest.raises(ExternalExecutionError, match="output\nerror"):
        run_external(["fixture"], cwd=tmp_path)


def test_seed_adapter_arguments_and_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repository = tmp_path / "external repository"
    repository.mkdir()
    (repository / "inference.py").write_text("# fixture", encoding="utf-8")
    checkpoint, config = repository / "weights.pt", repository / "model.yaml"
    checkpoint.write_bytes(b"fixture")
    config.write_text("fixture: true", encoding="utf-8")
    audio = fixture_audio()
    commands: list[list[str]] = []

    def execute(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        save_audio(Path(command[command.index("--output") + 1]) / "generated.wav", audio)
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr("nsvp.adapters.seed_vc.run_external", execute)
    converter = SeedVCConverter(SeedVCSettings(repository_root=repository, checkpoint_path=checkpoint,
        config_path=config, python_executable=Path(sys.executable), device="cpu"))
    output = converter.convert(audio, audio, 3, tmp_path / "work space")
    command = commands[0]
    assert command[0] == str(Path(sys.executable).resolve())
    assert command[command.index("--semi-tone-shift") + 1] == "3"
    assert command[command.index("--auto-f0-adjust") + 1] == "False"
    assert command[command.index("--f0-condition") + 1] == "True"
    assert Path(command[command.index("--source") + 1]).is_absolute()
    assert output.samples == audio.samples
    monkeypatch.setattr("nsvp.adapters.seed_vc.run_external", lambda *a, **kw: None)
    with pytest.raises(NSVPError, match="without a WAV"):
        converter.convert(audio, audio, 0, tmp_path / "empty-output")
