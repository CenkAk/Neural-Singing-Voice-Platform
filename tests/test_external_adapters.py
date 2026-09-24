from __future__ import annotations

import subprocess
import sys
import venv
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from test_dataset_pipeline import fixture_audio

from nsvp.adapters.external import (
    ExternalExecutionError,
    offline_environment,
    python_executable,
    run_external,
)
from nsvp.adapters.seed_vc import SeedVCConverter, SeedVCSettings
from nsvp.audio.io import save_audio
from nsvp.contracts import ProcessTelemetry
from nsvp.errors import AudioValidationError, ConfigurationError, NSVPError
from nsvp.evaluation import evaluate_audio


def test_external_python_default_is_explicit() -> None:
    assert python_executable(None) == Path(sys.executable).absolute()


def test_external_python_preserves_virtual_environment(tmp_path: Path) -> None:
    environment = tmp_path / "isolated"
    venv.EnvBuilder(with_pip=False, symlinks=sys.platform != "win32").create(environment)
    executable = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    result = subprocess.run([str(python_executable(executable)), "-c", "import sys; print(sys.prefix)"],
        check=True, capture_output=True, text=True)
    assert Path(result.stdout.strip()).resolve() == environment.resolve()


def test_seed_runner_restores_cache_and_resolves_auxiliary_files_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nsvp.adapters.runners import seed_vc

    env = offline_environment(tmp_path)
    assert env["HF_HUB_OFFLINE"] == "1"
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    downloads: list[dict[str, object]] = []
    hub = ModuleType("huggingface_hub")

    def download(**kwargs: object) -> str:
        downloads.append(kwargs)
        return str(tmp_path / str(kwargs["filename"]))

    hub.__dict__["hf_hub_download"] = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(manual_seed=lambda _: None, device=str))
    upstream = SimpleNamespace(torchaudio=SimpleNamespace())

    def import_module(name: str) -> object:
        if name == "huggingface_hub":
            return hub
        if name == "huggingface_hub.constants":
            return SimpleNamespace(HF_HUB_CACHE=env["HF_HUB_CACHE"])
        assert name == "inference"
        monkeypatch.setenv("HF_HUB_CACHE", "./checkpoints/hf_cache")
        return upstream

    def run(args: object) -> None:
        import numpy as np
        import soundfile as sf

        assert seed_vc.os.environ["HF_HUB_CACHE"] == env["HF_HUB_CACHE"]
        values = np.array([[1.25, -1.25, 0.123456], [0.0, 0.5, -0.5]], dtype=np.float32)
        tensor = SimpleNamespace()
        tensor.detach = lambda: tensor
        tensor.cpu = lambda: tensor
        tensor.numpy = lambda: values
        output = tmp_path / "output.wav"
        upstream.torchaudio.save(str(output), tensor, 44100)
        samples, rate = sf.read(output, dtype="float32", always_2d=True)
        assert rate == 44100
        np.testing.assert_array_equal(samples, values.T)
        assert sf.info(output).subtype == "FLOAT"
        loader = upstream.load_custom_model_from_hf
        assert loader("fixture/weights", "model.bin") == str(tmp_path / "model.bin")
        assert loader("fixture/weights", "model.bin", "config.json") == (
            str(tmp_path / "model.bin"), str(tmp_path / "config.json"))

    upstream.main = run
    monkeypatch.setattr(seed_vc.importlib, "import_module", import_module)
    monkeypatch.setattr(sys, "argv", ["seed_vc.py", "--source", "source.wav", "--target", "target.wav",
        "--output", "output", "--checkpoint", "model.bin", "--config", "config.json", "--device", "cpu"])
    seed_vc.main()
    assert len(downloads) == 3
    assert all(call["local_files_only"] is True and call["cache_dir"] == env["HF_HUB_CACHE"] for call in downloads)


def test_external_timeout_and_failed_logs(tmp_path: Path) -> None:
    with pytest.raises(ExternalExecutionError, match="timeout"):
        run_external([sys.executable, "-c", "import time; time.sleep(30)"], cwd=tmp_path, timeout=0.2)
    with pytest.raises(ExternalExecutionError, match="output\nerror"):
        run_external([sys.executable, "-c", "import sys; print('output'); print('error',file=sys.stderr); sys.exit(2)"], cwd=tmp_path)


def test_process_telemetry_survives_timeout_and_reaches_evaluation(tmp_path: Path) -> None:
    telemetry = tmp_path / "process.json"
    program = "import time; data=bytearray(16*1024*1024); time.sleep(2)"
    run_external([sys.executable, "-c", program], cwd=tmp_path, telemetry_path=telemetry)
    observed = ProcessTelemetry.model_validate_json(telemetry.read_text(encoding="utf-8"))
    assert observed.returncode == 0 and observed.elapsed_seconds >= 2
    assert observed.samples > 0
    assert observed.sampled_peak_rss_bytes is not None and observed.sampled_peak_rss_bytes >= 16 * 1024 * 1024
    report = evaluate_audio(fixture_audio(), fixture_audio(), provider_process=observed)
    metrics = report.families["performance"]
    assert metrics["provider_sampled_peak_rss_bytes"].value == observed.sampled_peak_rss_bytes
    assert metrics["peak_vram_bytes"].status == "not_measured"
    with pytest.raises(ExternalExecutionError, match="timeout"):
        run_external([sys.executable, "-c", program], cwd=tmp_path, timeout=0.3, telemetry_path=telemetry)
    failed = ProcessTelemetry.model_validate_json(telemetry.read_text(encoding="utf-8"))
    assert failed.returncode is not None and failed.returncode != 0
    assert failed.elapsed_seconds >= 0.3


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
    assert command[0] == str(Path(sys.executable).absolute())
    assert command[command.index("--semi-tone-shift") + 1] == "3"
    assert command[command.index("--auto-f0-adjust") + 1] == "False"
    assert command[command.index("--f0-condition") + 1] == "True"
    assert Path(command[command.index("--source") + 1]).is_absolute()
    assert output.samples == audio.samples
    with pytest.raises(ConfigurationError, match="stale output reuse"):
        converter.convert(audio, audio, 3, tmp_path / "work space")
    assert converter.diagnostic_artifacts == {}
    assert len(commands) == 1
    def truncated(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        short = audio.model_copy(update={"waveform": audio.waveform[:, :audio.samples // 2]})
        save_audio(Path(command[command.index("--output") + 1]) / "generated.wav", short)
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr("nsvp.adapters.seed_vc.run_external", truncated)
    with pytest.raises(AudioValidationError, match="alignment tolerance"):
        converter.convert(audio, audio, 0, tmp_path / "truncated")
    assert converter.diagnostic_artifacts["model_output_original.wav"].is_file()
    assert '"accepted": false' in converter.diagnostic_artifacts["model_alignment.json"].read_text()
    monkeypatch.setattr("nsvp.adapters.seed_vc.run_external", lambda *a, **kw: None)
    with pytest.raises(NSVPError, match="without a WAV"):
        converter.convert(audio, audio, 0, tmp_path / "empty-output")
