from __future__ import annotations

from pathlib import Path

import numpy as np

from .contracts import AudioBuffer, StemSet


class DeterministicSeparator:
    """Test-only linear separator used to validate orchestration without pretrained weights."""

    name = "deterministic-test-separator"

    def separate(self, song: AudioBuffer, work_dir: Path) -> StemSet:
        vocals = AudioBuffer(waveform=(song.waveform * 0.6).astype(np.float32), sample_rate=song.sample_rate)
        instrumental = AudioBuffer(waveform=(song.waveform * 0.4).astype(np.float32), sample_rate=song.sample_rate)
        return StemSet(vocals=vocals, instrumental=instrumental)


class IdentityVoiceConverter:
    """Test-only converter; outputs must never be presented as a trained singer model."""

    name = "identity-test-converter"

    def convert(self, source_vocal: AudioBuffer, target_reference: AudioBuffer, semitones: int, work_dir: Path) -> AudioBuffer:
        if semitones != 0:
            raise ValueError("test converter does not implement transposition")
        return source_vocal

