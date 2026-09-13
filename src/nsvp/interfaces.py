from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from .contracts import (
    AudioBuffer,
    DatasetManifest,
    EvaluatorMetadata,
    FeatureSequence,
    MetricResult,
    PitchTrack,
    StemSet,
    VocalSource,
)


class SourceSeparator(Protocol):
    name: str
    def separate(self, song: AudioBuffer, work_dir: Path) -> StemSet: ...


class PitchExtractor(Protocol):
    name: str
    def extract(self, audio: AudioBuffer) -> PitchTrack: ...


class ContentEncoder(Protocol):
    name: str
    def encode(self, audio: AudioBuffer) -> FeatureSequence: ...


class VoiceConverter(Protocol):
    name: str
    def convert(self, source_vocal: AudioBuffer, target_reference: AudioBuffer, semitones: int, work_dir: Path) -> AudioBuffer: ...


@runtime_checkable
class ConversionDiagnostics(Protocol):
    diagnostic_artifacts: dict[str, Path]


class AudioMixer(Protocol):
    name: str
    def mix(self, vocal: AudioBuffer, instrumental: AudioBuffer) -> AudioBuffer: ...


class VocalPreprocessor(Protocol):
    name: str
    def process(self, vocal: AudioBuffer, work_dir: Path) -> AudioBuffer: ...


class MultiSingerSeparator(Protocol):
    name: str
    def separate_singers(self, vocal: AudioBuffer, work_dir: Path) -> list[VocalSource]: ...


class ContentEvaluator(Protocol):
    metadata: EvaluatorMetadata
    def evaluate(self, source: AudioBuffer, output: AudioBuffer) -> dict[str, MetricResult]: ...


class SingerSimilarityEvaluator(Protocol):
    metadata: EvaluatorMetadata
    def evaluate(self, reference: AudioBuffer, output: AudioBuffer) -> MetricResult: ...


class TrainingBridge(Protocol):
    name: str
    def export_dataset(self, manifest: DatasetManifest, output: Path) -> Path: ...
    def run(self, config_path: Path, run_name: str, resume: Path | None = None) -> None: ...
