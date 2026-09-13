from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import BackendName
from .errors import ConfigurationError, DependencyUnavailableError


class DeviceConfig(BaseModel):
    backend: BackendName = BackendName.AUTO
    allow_cpu_fallback: bool = True
    precision: Literal["auto", "fp32", "fp16"] = "auto"


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ComponentSelection(StrictConfig):
    provider: str
    profile: str | None = None


class ComponentsConfig(StrictConfig):
    separator: ComponentSelection = Field(
        default_factory=lambda: ComponentSelection(provider="demucs")
    )
    vocal_preprocessor: ComponentSelection = Field(
        default_factory=lambda: ComponentSelection(provider="none")
    )
    voice_converter: ComponentSelection = Field(
        default_factory=lambda: ComponentSelection(provider="seed_vc")
    )
    pitch_extractor: ComponentSelection = Field(
        default_factory=lambda: ComponentSelection(provider="autocorrelation")
    )


class ExternalProviderConfig(StrictConfig):
    repository_root: Path | None = None
    python_executable: Path | None = None
    checkpoint_path: Path | None = None
    config_path: Path | None = None
    revision: str | None = None
    timeout_seconds: float = Field(default=1800, gt=0)
    backend: BackendName = BackendName.AUTO
    precision: Literal["auto", "fp32", "fp16"] = "auto"


class SeedVCProviderConfig(ExternalProviderConfig):
    diffusion_steps: int = Field(default=30, ge=1, le=1000)
    fp16: bool = False
    huggingface_cache: Path | None = None


class SoulXProviderConfig(ExternalProviderConfig):
    use_fp16: bool = False
    rmvpe_checkpoint_path: Path | None = None
    huggingface_cache: Path | None = None
    whisper_revision: str | None = None
    n_steps: int = Field(default=32, ge=1, le=1000)
    cfg: float = Field(default=3.0, ge=0, le=20)


class DemucsProviderConfig(StrictConfig):
    python_executable: Path | None = None
    model: str = "htdemucs"
    model_repository: Path | None = None
    backend: BackendName = BackendName.AUTO
    precision: Literal["auto", "fp32"] = "auto"
    timeout_seconds: float = Field(default=1800, gt=0)


class PitchProviderConfig(StrictConfig):
    fmin: float = Field(default=50, gt=0)
    fmax: float = Field(default=1100, gt=0)

    @model_validator(mode="after")
    def valid_range(self) -> PitchProviderConfig:
        if self.fmax <= self.fmin:
            raise ValueError("fmax must exceed fmin")
        return self


class ProvidersConfig(StrictConfig):
    seed_vc: SeedVCProviderConfig = Field(default_factory=SeedVCProviderConfig)
    soulx_singer: SoulXProviderConfig = Field(default_factory=SoulXProviderConfig)
    demucs: DemucsProviderConfig = Field(default_factory=DemucsProviderConfig)
    autocorrelation: PitchProviderConfig = Field(default_factory=PitchProviderConfig)


class ComponentProfile(StrictConfig):
    provider: str
    settings: dict[str, Any] = Field(default_factory=dict)


class VocalProcessingProfile(StrictConfig):
    stages: list[ComponentSelection] = Field(default_factory=list)


class DatasetAuditConfig(StrictConfig):
    enabled: bool = True
    pitch_percentiles: list[float] = Field(default_factory=lambda: [5.0, 25.0, 50.0, 75.0, 95.0])
    midi_bin_edges: list[float] = Field(default_factory=lambda: [24.0, 36.0, 48.0, 60.0, 72.0, 84.0, 96.0, 108.0])
    narrow_range_semitones: float = Field(default=12, gt=0)
    silence_threshold_db: float = -45

    @model_validator(mode="after")
    def valid_thresholds(self) -> DatasetAuditConfig:
        if not self.pitch_percentiles or any(not 0 <= p <= 100 for p in self.pitch_percentiles):
            raise ValueError("pitch percentiles must be between 0 and 100")
        if len(self.midi_bin_edges) < 2 or any(
            b <= a for a, b in zip(self.midi_bin_edges, self.midi_bin_edges[1:])
        ):
            raise ValueError("MIDI bin edges must be strictly increasing")
        return self


class AudioConfig(BaseModel):
    training_sample_rate: int = 44_100
    minimum_segment_seconds: float = 2.0
    target_maximum_segment_seconds: float = 15.0
    absolute_maximum_segment_seconds: float = 30.0
    minimum_silence_seconds: float = 0.25
    silence_threshold_db: float = -45.0
    clipping_threshold: float = 0.999


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    components: ComponentsConfig = Field(default_factory=ComponentsConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    profiles: dict[str, ComponentProfile] = Field(default_factory=dict)
    vocal_processing_profiles: dict[str, VocalProcessingProfile] = Field(default_factory=dict)
    dataset_audit: DatasetAuditConfig = Field(default_factory=DatasetAuditConfig)
    mlflow_tracking_uri: str | None = None
    artifact_root: Path = Path("artifacts")
    database_path: Path = Path("artifacts/nsvp.sqlite3")
    model_cache: Path = Path("artifacts/model-cache")
    seed_vc_root: Path | None = None
    seed_vc_checkpoint: Path | None = None
    seed_vc_config: Path | None = None

    @model_validator(mode="before")
    @classmethod
    def migrate_seed_vc(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        mapping = {
            "seed_vc_root": "repository_root",
            "seed_vc_checkpoint": "checkpoint_path",
            "seed_vc_config": "config_path",
        }
        legacy = {key: data[key] for key in mapping if data.get(key) is not None}
        if not legacy:
            return data
        providers = data.get("providers", {})
        if isinstance(providers, ProvidersConfig):
            providers = providers.model_dump()
        if not isinstance(providers, dict):
            raise ConfigurationError("providers must be a mapping")
        providers = dict(providers)
        seed = providers.get("seed_vc", {})
        if isinstance(seed, SeedVCProviderConfig):
            seed = seed.model_dump()
        if not isinstance(seed, dict):
            raise ConfigurationError("seed_vc provider must be a mapping")
        seed = dict(seed)
        for old, path in legacy.items():
            new = mapping[old]
            if seed.get(new) is not None and Path(seed[new]) != Path(path):
                raise ValueError(f"conflicting {old} and providers.seed_vc.{new}")
            seed[new] = path
        providers["seed_vc"] = seed
        data["providers"] = providers
        warnings.warn(
            "seed_vc_* settings are deprecated; use providers.seed_vc",
            DeprecationWarning,
            stacklevel=2,
        )
        return data

    @model_validator(mode="after")
    def validate_profiles(self) -> AppConfig:
        provider_types: dict[str, type[BaseModel]] = {
            "seed_vc": SeedVCProviderConfig,
            "soulx_singer": SoulXProviderConfig,
            "demucs": DemucsProviderConfig,
            "autocorrelation": PitchProviderConfig,
        }
        for name, profile in self.profiles.items():
            provider_type = provider_types.get(profile.provider)
            if provider_type is None:
                raise ValueError(f"unknown provider in profile {name}: {profile.provider}")
            provider_type.model_validate(profile.settings)
        return self


def load_config(path: Path | None = None) -> AppConfig:
    config_path = path or Path(os.getenv("NSVP_CONFIG", "configs/default.yaml"))
    data: dict[str, Any] = {}
    if config_path.exists():
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise DependencyUnavailableError("PyYAML is required to load YAML configuration") from exc
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if loaded:
            if not isinstance(loaded, dict):
                raise ConfigurationError("configuration root must be a mapping")
            data = loaded
    if backend := os.getenv("NSVP_DEVICE_BACKEND"):
        data.setdefault("device", {})["backend"] = backend
    if root := os.getenv("NSVP_ARTIFACT_ROOT"):
        data["artifact_root"] = root
    if database := os.getenv("NSVP_DATABASE_PATH"):
        data["database_path"] = database
    if tracking_uri := os.getenv("MLFLOW_TRACKING_URI"):
        data["mlflow_tracking_uri"] = tracking_uri
    try:
        return AppConfig.model_validate(data)
    except Exception as exc:
        raise ConfigurationError(str(exc)) from exc
