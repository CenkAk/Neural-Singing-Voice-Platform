from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .contracts import AudioBuffer, FeatureSequence, PitchTrack, StemSet


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


class AudioMixer(Protocol):
    name: str
    def mix(self, vocal: AudioBuffer, instrumental: AudioBuffer) -> AudioBuffer: ...

