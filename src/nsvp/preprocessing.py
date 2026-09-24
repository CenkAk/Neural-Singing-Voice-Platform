from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .audio.processing import apply_gain, loudness_metrics, remove_dc
from .config import StudioCleanConfig
from .contracts import AudioBuffer, VocalProcessingResult, VocalSource
from .errors import ConfigurationError
from .interfaces import VocalPreprocessor


class NoOpVocalPreprocessor:
    name = "none"

    def process(self, vocal: AudioBuffer, work_dir: Path) -> AudioBuffer:
        return vocal


class StudioCleanVocalPreprocessor:
    name = "studio_clean"

    def __init__(self, settings: StudioCleanConfig) -> None:
        self.settings = settings

    def process(self, vocal: AudioBuffer, work_dir: Path) -> AudioBuffer:
        centered = remove_dc(vocal)
        peak = float(np.max(np.abs(centered.waveform)))
        if peak == 0:
            return centered
        loudness = loudness_metrics(centered)["integrated_loudness_lufs"]
        if loudness.value is None:
            raise ConfigurationError(loudness.reason or "Studio-clean requires a finite loudness measurement")
        gain = min(self.settings.maximum_gain_db, self.settings.target_lufs - loudness.value)
        output = apply_gain(centered, gain)
        peak = float(np.max(np.abs(output.waveform)))
        if peak > self.settings.peak_ceiling:
            output = AudioBuffer(waveform=output.waveform * (self.settings.peak_ceiling / peak), sample_rate=output.sample_rate)
        return output


def process_vocal(
    vocal: AudioBuffer, stages: Sequence[VocalPreprocessor], work_dir: Path,
) -> VocalProcessingResult:
    current = vocal
    intermediates: dict[str, AudioBuffer] = {}
    for index, stage in enumerate(stages):
        current = stage.process(current, work_dir / f"stage-{index:02d}")
        if stage.name != "none":
            intermediates[f"vocal_stage_{index:02d}.wav"] = current
    return VocalProcessingResult(selected=current, intermediates=intermediates)


def select_source(sources: Sequence[VocalSource], source_id: str) -> VocalSource:
    identifiers = [source.source_id for source in sources]
    if len(set(identifiers)) != len(identifiers):
        raise ConfigurationError("multi-singer separator returned duplicate source identifiers")
    for source in sources:
        if source.source_id == source_id:
            return source
    raise ConfigurationError(f"unknown vocal source: {source_id}")
