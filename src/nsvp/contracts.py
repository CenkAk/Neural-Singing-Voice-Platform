from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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

    @model_validator(mode="after")
    def consistent_frames(self) -> PitchTrack:
        count = self.timestamps.size
        if self.f0_hz.size != count or self.voiced.size != count:
            raise ValueError("pitch timestamps, F0 and voicing must have equal lengths")
        if self.confidence is not None and self.confidence.size != count:
            raise ValueError("pitch confidence must have the same number of frames")
        if not np.isfinite(self.timestamps).all() or not np.isfinite(self.f0_hz).all():
            raise ValueError("pitch values must be finite")
        if np.any(self.f0_hz < 0) or np.any(self.timestamps < 0):
            raise ValueError("pitch timestamps and frequencies must be nonnegative")
        if np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("pitch timestamps must be strictly increasing")
        return self


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


class TaskType(str, Enum):
    SVC = "svc"
    SVS = "svs"
    SEPARATION = "separation"
    PREPROCESSING = "preprocessing"
    PITCH = "pitch"
    MULTI_SINGER_SEPARATION = "multi_singer_separation"


class ModelMode(str, Enum):
    FINETUNED = "finetuned"
    ZERO_SHOT_REFERENCE = "zero_shot_reference"


class InputCondition(str, Enum):
    CLEAN_LEAD = "clean_lead"
    MIXED_VOCAL = "mixed_vocal"
    SEPARATED_LEAD = "separated_lead"
    UNKNOWN = "unknown"


class CompatibilityStatus(str, Enum):
    VERIFIED = "verified"
    NOT_VERIFIED = "not_verified"
    UNSUPPORTED = "unsupported"


class ComponentCapabilities(BaseModel):
    name: str
    version: str = "unknown"
    task: TaskType
    stability: Literal["stable", "experimental", "research_reference", "test_only"]
    sample_rates: list[int] = Field(default_factory=list)
    zero_shot: bool = False
    supports_training: bool = False
    f0_conditioning: bool = False
    midi_conditioning: bool = False
    reference_audio_conditioning: bool = False
    backends: dict[BackendName, CompatibilityStatus] = Field(default_factory=dict)
    precisions: list[str] = Field(default_factory=lambda: ["fp32"])
    maximum_tested_input_seconds: float | None = None
    external_environment: bool = False
    installed: bool = False
    configured: bool = False
    checkpoint_sha256: str | None = None
    license_name: str | None = None
    license_url: str | None = None
    warnings: list[str] = Field(default_factory=list)


class ComponentExecution(BaseModel):
    provider: str
    profile: str | None = None
    version: str = "unknown"
    checkpoint_sha256: str | None = None
    config_sha256: str | None = None
    backend: BackendName
    device_name: str
    precision: str
    python_version: str | None = None
    torch_version: str | None = None
    verification: CompatibilityStatus = CompatibilityStatus.NOT_VERIFIED
    warnings: list[str] = Field(default_factory=list)


class VocalSource(BaseModel):
    source_id: str
    audio: AudioBuffer
    label: str | None = None


class VocalProcessingResult(BaseModel):
    selected: AudioBuffer
    intermediates: dict[str, AudioBuffer] = Field(default_factory=dict)
    sources: list[VocalSource] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class EvaluatorMetadata(BaseModel):
    library_version: str | None = None
    name: str
    version: str
    checkpoint_sha256: str | None = None
    domain: str
    limitations: list[str] = Field(default_factory=list)
    model: str | None = None
    language: str | None = None
    reference_kind: str | None = None
    normalization_version: str | None = None


class MetricResult(BaseModel):
    value: float | None = None
    status: Literal["measured", "not_measured", "unsupported", "failed"] = "not_measured"
    unit: str | None = None
    reason: str | None = None
    evaluator: EvaluatorMetadata | None = None

    @model_validator(mode="after")
    def valid_measurement(self) -> MetricResult:
        if self.status == "measured":
            if self.value is None or not np.isfinite(self.value):
                raise ValueError("measured metrics require a finite value")
        elif self.value is not None:
            raise ValueError("unmeasured metrics must have a null value")
        return self


class ConversionRequest(BaseModel):
    song_path: Path
    target_reference_path: Path
    output_name: str
    model_name: str = "seed-vc-v1"
    model_version: str = "upstream"
    transpose_semitones: int = Field(default=0, ge=-12, le=12)
    backend: BackendName = BackendName.AUTO
    keep_intermediates: bool = True
    separator: str | None = None
    voice_converter: str | None = None
    converter_profile: str | None = None
    separator_profile: str | None = None
    vocal_processing_profile: str | None = None
    model_profile: str | None = None
    precision: Literal["auto", "fp32", "fp16"] = "auto"
    random_seed: int = Field(default=42, ge=0, le=2**32 - 1)
    input_kind: Literal["song", "vocal"] = "song"
    input_condition: InputCondition = InputCondition.UNKNOWN
    instrumental_path: Path | None = None
    preparation_manifest_artifact_id: str | None = None
    selected_source_id: str | None = None
    language: Literal["tr", "en"] | None = None
    reference_text: str | None = Field(default=None, max_length=20000)


class ProcessTelemetry(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    elapsed_seconds: float = Field(ge=0)
    sampled_peak_rss_bytes: int | None = Field(default=None, ge=0)
    samples: int = Field(ge=0)
    missed_samples: int = Field(ge=0)
    sampling_interval_seconds: float = Field(default=0.1, gt=0)
    scope: Literal["provider_process_tree_rss_sum"] = "provider_process_tree_rss_sum"
    returncode: int | None = None


class ConversionResult(BaseModel):
    conversion_id: str
    artifacts: dict[str, str]
    warnings: list[str]
    processing_time_seconds: float
    components: dict[str, str]
    executions: dict[str, ComponentExecution] = Field(default_factory=dict)
    input_condition: InputCondition = InputCondition.UNKNOWN
    provider_process: ProcessTelemetry | None = None


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
    mode: ModelMode = ModelMode.FINETUNED
    task: TaskType = TaskType.SVC
    provider: str | None = None
    provider_profile: str | None = None
    upstream_version: str | None = None
    upstream_checkpoint_sha256: str | None = None
    dataset_version: str | None = None
    checkpoint_sha256: str | None = None
    checkpoint_artifact_id: str | None = None
    observed_pitch_range_hz: tuple[float, float] | None = None
    evaluation_status: str = "not_measured"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_identity(self) -> SingerModelManifest:
        if self.mode is ModelMode.FINETUNED:
            if not self.dataset_version or not self.checkpoint_sha256 or not self.checkpoint_artifact_id:
                raise ValueError("fine-tuned models require dataset and singer checkpoint metadata")
        elif self.checkpoint_artifact_id or self.checkpoint_sha256 or self.dataset_version:
            raise ValueError("zero-shot profiles must not contain singer-specific checkpoint metadata")
        if self.mode is ModelMode.ZERO_SHOT_REFERENCE and not self.provider:
            raise ValueError("zero-shot profiles require a provider")
        return self


class JobState(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
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
    attempt_count: int = 0
    worker_id: str | None = None
    heartbeat_at: datetime | None = None


class PitchPreview(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    source: list[tuple[float, float | None]] = Field(max_length=1000)
    output: list[tuple[float, float | None]] = Field(max_length=1000)
    source_extractor: str
    output_extractor: str
    source_transpose_semitones: int
    sampling: str = "At most 1000 evenly selected frames per track; unvoiced frames are null; no interpolation"


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
    schema_version: str = "0.2"
    families: dict[str, dict[str, MetricResult]] = Field(default_factory=dict)
    input_condition: InputCondition = InputCondition.UNKNOWN
    pitch_preview: PitchPreview | None = None
