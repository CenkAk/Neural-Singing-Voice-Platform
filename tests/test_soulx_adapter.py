from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from test_dataset_pipeline import fixture_audio

from nsvp.adapters.soulx_singer import SoulXSingerSettings, SoulXSingerSVCConverter
from nsvp.audio.io import save_audio
from nsvp.errors import AudioValidationError, ConfigurationError, NSVPError


@pytest.fixture
def soulx_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SoulXSingerSettings:
    root = tmp_path / "soulx"
    for name in ("cli/inference_svc.py", "preprocess/tools/f0_extraction.py", "svc.pt", "rmvpe.pt", "config.yaml"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    revision = "a" * 40
    cache = tmp_path / "hf"
    whisper = cache / "hub/models--openai--whisper-base"
    (whisper / "refs").mkdir(parents=True)
    (whisper / "refs/main").write_text(revision)
    snapshot = whisper / "snapshots" / revision
    snapshot.mkdir(parents=True)
    for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
        (snapshot / name).write_bytes(b"fixture")
    monkeypatch.setattr("nsvp.adapters.soulx_singer.validate_revision", lambda *args: revision)
    return SoulXSingerSettings(
        repository_root=root, checkpoint_path=root / "svc.pt", config_path=root / "config.yaml",
        rmvpe_checkpoint_path=root / "rmvpe.pt", huggingface_cache=cache,
        revision=revision, whisper_revision=revision, python_executable=Path(sys.executable),
    )


def test_soulx_requires_all_assets_before_execution(soulx_settings: SoulXSingerSettings) -> None:
    converter = SoulXSingerSVCConverter(soulx_settings)
    converter.validate_installation()
    soulx_settings.rmvpe_checkpoint_path.unlink()
    with pytest.raises(ConfigurationError, match="required file"):
        converter.validate_installation()


def test_soulx_requires_exact_whisper_revision(soulx_settings: SoulXSingerSettings) -> None:
    soulx_settings.whisper_revision = "b" * 40
    with pytest.raises(ConfigurationError, match="whisper_revision"):
        SoulXSingerSVCConverter(soulx_settings).validate_installation()


@pytest.mark.parametrize("output_seconds,expected", [(3.0, None), (2.0, AudioValidationError)])
def test_soulx_invocation_offline_and_duration_validation(
    soulx_settings: SoulXSingerSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    output_seconds: float, expected: type[Exception] | None,
) -> None:
    captured: list[list[str]] = []

    def execute(command: list[str], **kwargs: object) -> None:
        captured.append(command)
        env = kwargs["env"]
        assert isinstance(env, dict) and env["HF_HUB_OFFLINE"] == "1"
        assert env["TRANSFORMERS_OFFLINE"] == "1"
        output = Path(command[command.index("--output") + 1])
        output.mkdir()
        save_audio(output / "generated.wav", fixture_audio(seconds=output_seconds, sample_rate=24000))
        (output / "runtime.json").write_text(json.dumps({
            "device": "cpu", "precision": "fp32", "sample_rate": 24000, "hop_size": 480,
            "random_seed": 42, "source_f0_frames": 150, "reference_f0_frames": 150,
            "python_version": "fixture", "torch_version": "fixture",
        }))

    monkeypatch.setattr("nsvp.adapters.soulx_singer.run_external", execute)
    converter = SoulXSingerSVCConverter(soulx_settings)
    source = fixture_audio()
    if expected:
        with pytest.raises(expected):
            converter.convert(source, source, 2, tmp_path / "work")
    else:
        output = converter.convert(source, source, 2, tmp_path / "work")
        assert output.samples == source.samples
        assert output.sample_rate == source.sample_rate
    command = captured[0]
    assert command[command.index("--pitch-shift") + 1] == "2"
    assert "--auto-shift" not in command
    assert command[command.index("--reference") + 1].endswith("reference.wav")
    assert (tmp_path / "work/output/generated.wav").is_file()


def test_soulx_missing_output_is_not_success(
    soulx_settings: SoulXSingerSettings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("nsvp.adapters.soulx_singer.run_external", lambda *a, **kw: None)
    with pytest.raises(NSVPError, match="generated.wav"):
        SoulXSingerSVCConverter(soulx_settings).convert(fixture_audio(), fixture_audio(), 0, tmp_path / "work")
