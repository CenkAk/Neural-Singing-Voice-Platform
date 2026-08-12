from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .contracts import BackendName
from .errors import ConfigurationError, DependencyUnavailableError


class DeviceConfig(BaseModel):
    backend: BackendName = BackendName.AUTO
    allow_cpu_fallback: bool = True
    precision: str = "auto"


class AudioConfig(BaseModel):
    training_sample_rate: int = 44_100
    minimum_segment_seconds: float = 2.0
    target_maximum_segment_seconds: float = 15.0
    absolute_maximum_segment_seconds: float = 30.0
    minimum_silence_seconds: float = 0.25
    silence_threshold_db: float = -45.0
    clipping_threshold: float = 0.999


class AppConfig(BaseModel):
    device: DeviceConfig = Field(default_factory=DeviceConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    artifact_root: Path = Path("artifacts")
    database_path: Path = Path("artifacts/nsvp.sqlite3")
    model_cache: Path = Path("artifacts/model-cache")
    seed_vc_root: Path | None = None
    seed_vc_checkpoint: Path | None = None
    seed_vc_config: Path | None = None


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
    try:
        return AppConfig.model_validate(data)
    except Exception as exc:
        raise ConfigurationError(str(exc)) from exc
