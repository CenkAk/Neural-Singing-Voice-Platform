from __future__ import annotations

from pathlib import Path

import numpy as np

from .contracts import AudioBuffer, EvaluatorMetadata, MetricResult, StemSet, VocalSource


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


class DeterministicMultiSingerSeparator:
    """Synthetic sources for source-selection tests, not actual singer separation."""

    name = "deterministic-test-multi-singer"

    def separate_singers(self, audio: AudioBuffer, work_dir: Path) -> list[VocalSource]:
        return [
            VocalSource(source_id="source-a", audio=audio, label="Synthetic source A"),
            VocalSource(source_id="source-b", audio=AudioBuffer(
                waveform=(audio.waveform * 0.5).astype(np.float32), sample_rate=audio.sample_rate,
            ), label="Synthetic source B"),
        ]


class UnmeasuredContentEvaluator:
    """Test double for metadata propagation without invented linguistic scores."""

    metadata = EvaluatorMetadata(name="test-content", version="1", domain="synthetic", limitations=["No ASR model is executed."])

    def evaluate(self, source: AudioBuffer, output: AudioBuffer) -> dict[str, MetricResult]:
        return {"wer": MetricResult(reason="Test evaluator does not measure linguistic content.")}


class UnmeasuredSingerEvaluator:
    """Test double for optional singer evaluation without synthetic identity claims."""

    metadata = EvaluatorMetadata(name="test-singer", version="1", domain="synthetic", limitations=["No singer model is executed."])

    def evaluate(self, reference: AudioBuffer, output: AudioBuffer) -> MetricResult:
        return MetricResult(reason="Test evaluator does not measure singer identity.")
