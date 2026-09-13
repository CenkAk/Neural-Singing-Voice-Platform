from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_dataset_pipeline import fixture_audio

from nsvp.adapters.external import EnvironmentProbe
from nsvp.audio.io import save_audio
from nsvp.components import ComponentFactory
from nsvp.config import AppConfig, ComponentSelection, SeedVCProviderConfig
from nsvp.contracts import BackendName, CompatibilityStatus, ConversionRequest, ModelMode
from nsvp.errors import BackendUnavailableError, ConfigurationError
from nsvp.registry import ModelRegistry
from nsvp.runtime import build_handlers
from nsvp.storage import LocalArtifactStore, sha256_file
from nsvp.testing_backends import DeterministicSeparator, IdentityVoiceConverter


def test_legacy_config_migrates_without_changing_defaults() -> None:
    with pytest.warns(DeprecationWarning):
        config = AppConfig(seed_vc_root=Path("seed"), seed_vc_checkpoint=Path("model.pt"))
    assert config.providers.seed_vc.repository_root == Path("seed")
    assert config.providers.seed_vc.checkpoint_path == Path("model.pt")
    assert config.components.voice_converter.provider == "seed_vc"
    assert config.providers.seed_vc.diffusion_steps == 30


def test_conflicting_legacy_config_is_rejected() -> None:
    with pytest.raises(ValidationError, match="conflicting"):
        AppConfig.model_validate({
            "seed_vc_root": "old", "providers": {"seed_vc": {"repository_root": "new"}},
        })


@pytest.mark.parametrize("data", [
    {"providers": {"unknown": {}}},
    {"providers": {"seed_vc": {"diffusion_step": 20}}},
    {"profiles": {"bad": {"provider": "seed_vc", "settings": {"diffusion_steps": 0}}}},
])
def test_malformed_provider_config_is_rejected(data: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate(data)


def test_named_profile_preserves_unmodified_provider_settings() -> None:
    config = AppConfig.model_validate({
        "providers": {"seed_vc": {"repository_root": "seed", "python_executable": sys.executable}},
        "profiles": {"fast": {"provider": "seed_vc", "settings": {"diffusion_steps": 8}}},
    })
    factory = ComponentFactory(config)
    selected = factory.settings(ComponentSelection(provider="seed_vc", profile="fast"), SeedVCProviderConfig)
    assert selected.diffusion_steps == 8
    assert selected.repository_root == Path("seed")
    assert config.providers.seed_vc.diffusion_steps == 30
    with pytest.raises(ConfigurationError, match="profile"):
        factory.settings(ComponentSelection(provider="seed_vc", profile="missing"), SeedVCProviderConfig)


def test_unsupported_and_unverified_are_distinct(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nsvp.components.probe_environment", lambda *args: EnvironmentProbe(
        python_version="3.10", torch_version="test", backends=[BackendName.CPU, BackendName.DIRECTML],
        modules={"torch": True}, device_names={"cpu": "test CPU"},
    ))
    factory = ComponentFactory(AppConfig())
    settings = SeedVCProviderConfig(backend=BackendName.CPU)
    execution = factory.execution(ComponentSelection(provider="seed_vc"), settings)
    assert execution.verification is CompatibilityStatus.NOT_VERIFIED
    assert execution.warnings
    with pytest.raises(BackendUnavailableError, match="cannot execute"):
        factory.execution(ComponentSelection(provider="seed_vc"), SeedVCProviderConfig(backend=BackendName.DIRECTML))


def test_unavailable_backend_respects_fallback_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("nsvp.components.probe_environment", lambda *args: EnvironmentProbe(
        python_version="3.10", torch_version="test", modules={"torch": True},
    ))
    config = AppConfig.model_validate({"device": {"allow_cpu_fallback": False}})
    with pytest.raises(BackendUnavailableError, match="unavailable"):
        ComponentFactory(config).execution(
            ComponentSelection(provider="seed_vc"), SeedVCProviderConfig(backend=BackendName.CUDA),
        )
    execution = ComponentFactory(AppConfig()).execution(
        ComponentSelection(provider="seed_vc"), SeedVCProviderConfig(backend=BackendName.CUDA),
    )
    assert execution.backend is BackendName.CPU
    assert any("fallback" in warning for warning in execution.warnings)


def test_runtime_uses_explicit_fake_factory(tmp_path: Path) -> None:
    config = AppConfig.model_validate({
        "artifact_root": str(tmp_path / "artifacts"),
        "components": {"separator": {"provider": "fake"}, "voice_converter": {"provider": "identity"}},
    })
    factory = ComponentFactory(config, separators={"fake": DeterministicSeparator}, converters={"identity": IdentityVoiceConverter})
    song = tmp_path / "input.wav"
    save_audio(song, fixture_audio())
    request = ConversionRequest(song_path=song, target_reference_path=song, output_name="test")
    stages: list[str] = []
    result = build_handlers(config, factory)["conversion"](request.model_dump(mode="json"), lambda _, stage: stages.append(stage))
    assert result["components"]["voice_converter"] == "identity-test-converter"
    assert "processing_vocal" in stages
    assert len(result["artifacts"]) == 7
    assert "evaluation_report.json" in result["artifacts"]


def test_registry_distinguishes_singer_checkpoint_and_zero_shot_profile(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    registry = ModelRegistry(store.root / "models", store)
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"synthetic checkpoint")
    tuned = registry.register(model_name="singer", version="v1", checkpoint=checkpoint,
        architecture="seed_vc", adapter="seed_vc", sample_rate=44100,
        dataset_version="dataset-1", smoke_test_passed=True)
    zero = registry.register_zero_shot(model_name="reference", version="v1", provider="soulx_singer", sample_rate=24000)
    assert tuned.checkpoint_sha256 == sha256_file(checkpoint)
    assert zero.mode is ModelMode.ZERO_SHOT_REFERENCE
    assert zero.checkpoint_artifact_id is None
    assert zero.dataset_version is None
    assert registry.get_profile("reference/v1") == zero
    with pytest.raises(ConfigurationError):
        registry.get_profile("../secret")
    with pytest.raises(ConfigurationError):
        registry.register_zero_shot(model_name="reference", version="v1", provider="soulx_singer", sample_rate=24000)


@pytest.mark.parametrize("metadata", [{"upstream_version": "a" * 40}, {"upstream_checkpoint_sha256": "b" * 64}])
def test_registered_upstream_identity_is_enforced_before_execution(tmp_path: Path, metadata: dict[str, str]) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    ModelRegistry(store.root / "models", store).register_zero_shot(
        model_name="pinned", version="v1", provider="seed_vc", sample_rate=44100, **metadata,
    )
    request = ConversionRequest(song_path=tmp_path / "input.wav", target_reference_path=tmp_path / "ref.wav",
        output_name="test", model_profile="pinned/v1")
    with pytest.raises(ConfigurationError, match="registered upstream"):
        ComponentFactory(AppConfig()).build_voice_converter(request, store)
