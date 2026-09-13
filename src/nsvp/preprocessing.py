from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .contracts import AudioBuffer, VocalProcessingResult, VocalSource
from .errors import ConfigurationError
from .interfaces import VocalPreprocessor


class NoOpVocalPreprocessor:
    name = "none"

    def process(self, vocal: AudioBuffer, work_dir: Path) -> AudioBuffer:
        return vocal


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
