from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_validator

Array = npt.NDArray[Any]
FloatArray = npt.NDArray[np.float32]


class AudioBuffer(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    waveform: FloatArray
    sample_rate: int = Field(gt=0)

    @field_validator("waveform")
    @classmethod
    def validate_waveform(cls, value: Array) -> FloatArray:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 1:
            array = array[np.newaxis, :]
        if array.ndim != 2 or array.shape[1] == 0:
            raise ValueError("waveform must have shape [channels, samples] and be non-empty")
        if not np.isfinite(array).all():
            raise ValueError("waveform contains NaN or infinity")
        return np.ascontiguousarray(array)

    @property
    def channels(self) -> int:
        return int(self.waveform.shape[0])

    @property
    def samples(self) -> int:
        return int(self.waveform.shape[1])

    @property
    def duration_seconds(self) -> float:
        return self.samples / self.sample_rate


class StemSet(BaseModel):
    vocals: AudioBuffer
    instrumental: AudioBuffer
    extras: dict[str, AudioBuffer] = Field(default_factory=dict)


class PitchTrack(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    timestamps: Array
    f0_hz: Array
    voiced: Array
    confidence: Array | None = None
    extractor: str

    @field_validator("timestamps", "f0_hz", "voiced", "confidence")
    @classmethod
    def arrays_are_one_dimensional(cls, value: Array | None) -> Array | None:
        if value is None:
            return value
        array = np.asarray(value)
        if array.ndim != 1:
            raise ValueError("pitch arrays must be one-dimensional")
        return array


class FeatureSequence(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    frames: Array
    frame_rate: float = Field(gt=0)
    encoder: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class BackendName(str, Enum):
    AUTO = "auto"
    CUDA = "cuda"
    ROCM = "rocm"
    DIRECTML = "directml"
    MPS = "mps"
    CPU = "cpu"


class BackendCapabilities(BaseModel):
    backend: BackendName
    device_name: str
    precision: str
    supported_dtypes: list[str]
    component_compatibility: dict[str, bool] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class ConversionRequest(BaseModel):
    song_path: Path
    target_reference_path: Path
    output_name: str
    model_name: str = "seed-vc-v1"
    model_version: str = "upstream"
    transpose_semitones: int = Field(default=0, ge=-12, le=12)
    backend: BackendName = BackendName.AUTO
    keep_intermediates: bool = True


class ConversionResult(BaseModel):
    conversion_id: str
    artifacts: dict[str, str]
    warnings: list[str]
    processing_time_seconds: float
    components: dict[str, str]


class SegmentRecord(BaseModel):
    segment_id: str
    source_file: str
    source_sha256: str
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    sample_rate: int = Field(gt=0)
    split: str
    artifact_id: str | None = None
    quality: dict[str, float | int | bool] = Field(default_factory=dict)


class DatasetManifest(BaseModel):
    dataset_id: str
    version: str
    singer_name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    segments: list[SegmentRecord]
    source_files: list[str]
    config: dict[str, Any]
    analysis: dict[str, Any] = Field(default_factory=dict)


class SingerModelManifest(BaseModel):
    model_name: str
    version: str
    architecture: str
    adapter: str
    sample_rate: int
    dataset_version: str
    checkpoint_sha256: str
    checkpoint_artifact_id: str
    observed_pitch_range_hz: tuple[float, float] | None = None
    evaluation_status: str = "not_measured"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class JobState(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class JobRecord(BaseModel):
    id: str
    kind: str
    state: JobState
    payload: dict[str, Any]
    progress: float = Field(default=0, ge=0, le=1)
    stage: str = "queued"
    error: dict[str, Any] | None = None
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None


class EvaluationReport(BaseModel):
    source_duration_seconds: float
    output_duration_seconds: float
    peak: float
    clipping_samples: int
    f0_cents_rmse: float | None = None
    f0_correlation: float | None = None
    voicing_error: float | None = None
    raw_pitch_accuracy: float | None = None
    raw_chroma_accuracy: float | None = None
    singer_similarity: float | None = None
    limitations: list[str] = Field(default_factory=list)
