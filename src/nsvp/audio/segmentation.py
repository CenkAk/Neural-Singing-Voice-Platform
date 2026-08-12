from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from ..contracts import AudioBuffer


@dataclass(frozen=True)
class SegmentBoundary:
    start_sample: int
    end_sample: int

    def duration(self, sample_rate: int) -> float:
        return (self.end_sample - self.start_sample) / sample_rate


def find_segments(
    audio: AudioBuffer,
    minimum_seconds: float = 2.0,
    target_maximum_seconds: float = 15.0,
    absolute_maximum_seconds: float = 30.0,
    minimum_silence_seconds: float = 0.25,
    silence_threshold_db: float = -45.0,
) -> list[SegmentBoundary]:
    if not 0 < minimum_seconds <= target_maximum_seconds <= absolute_maximum_seconds:
        raise ValueError("segment durations must satisfy 0 < minimum <= target maximum <= absolute maximum")
    mono = audio.waveform.mean(axis=0)
    hop = max(1, round(audio.sample_rate * 0.02))
    usable = mono[: (mono.size // hop) * hop]
    if not usable.size:
        return []
    rms = np.sqrt(np.mean(usable.reshape(-1, hop) ** 2, axis=1) + 1e-12)
    silent = rms < 10.0 ** (silence_threshold_db / 20.0)
    minimum_silent_frames = max(1, round(minimum_silence_seconds * audio.sample_rate / hop))
    cut_candidates: list[int] = []
    run_start: int | None = None
    for index, is_silent in enumerate(silent):
        if is_silent and run_start is None:
            run_start = index
        if (not is_silent or index == len(silent) - 1) and run_start is not None:
            run_end = index if not is_silent else index + 1
            if run_end - run_start >= minimum_silent_frames:
                cut_candidates.append(((run_start + run_end) // 2) * hop)
            run_start = None

    cuts = [0]
    cursor = 0
    while audio.samples - cursor > round(target_maximum_seconds * audio.sample_rate):
        target = cursor + round(target_maximum_seconds * audio.sample_rate)
        absolute = min(audio.samples, cursor + round(absolute_maximum_seconds * audio.sample_rate))
        valid = [candidate for candidate in cut_candidates if cursor + round(minimum_seconds * audio.sample_rate) <= candidate <= absolute]
        cut = min(valid, key=lambda value: abs(value - target)) if valid else target
        if cut <= cursor:
            break
        cuts.append(cut)
        cursor = cut
    cuts.append(audio.samples)

    boundaries: list[SegmentBoundary] = []
    for start, end in pairwise(cuts):
        if (end - start) / audio.sample_rate < minimum_seconds and boundaries:
            previous = boundaries.pop()
            boundaries.append(SegmentBoundary(previous.start_sample, end))
        else:
            boundaries.append(SegmentBoundary(start, end))
    return [item for item in boundaries if item.end_sample > item.start_sample]
