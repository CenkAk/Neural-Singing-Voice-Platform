from __future__ import annotations

import numpy as np

from .audio.processing import analyze_audio
from .contracts import AudioBuffer, EvaluationReport, PitchTrack


def evaluate_pitch(source: PitchTrack, converted: PitchTrack, tolerance_cents: float = 50.0) -> dict[str, float | None]:
    count = min(source.f0_hz.size, converted.f0_hz.size)
    if count == 0:
        return {"f0_cents_rmse": None, "f0_correlation": None, "voicing_error": None, "raw_pitch_accuracy": None, "raw_chroma_accuracy": None}
    source_voiced = source.voiced[:count].astype(bool)
    converted_voiced = converted.voiced[:count].astype(bool)
    voicing_error = float(np.mean(source_voiced != converted_voiced))
    both = source_voiced & converted_voiced & (source.f0_hz[:count] > 0) & (converted.f0_hz[:count] > 0)
    if not np.any(both):
        return {"f0_cents_rmse": None, "f0_correlation": None, "voicing_error": voicing_error, "raw_pitch_accuracy": 0.0, "raw_chroma_accuracy": 0.0}
    source_hz = source.f0_hz[:count][both]
    converted_hz = converted.f0_hz[:count][both]
    cents = 1200.0 * np.log2(converted_hz / source_hz)
    rmse = float(np.sqrt(np.mean(cents**2)))
    correlation = float(np.corrcoef(source_hz, converted_hz)[0, 1]) if source_hz.size > 1 and np.std(source_hz) > 0 and np.std(converted_hz) > 0 else None
    pitch_accuracy = float(np.mean(np.abs(cents) <= tolerance_cents))
    chroma_error = np.abs(((cents + 600.0) % 1200.0) - 600.0)
    return {
        "f0_cents_rmse": rmse,
        "f0_correlation": correlation,
        "voicing_error": voicing_error,
        "raw_pitch_accuracy": pitch_accuracy,
        "raw_chroma_accuracy": float(np.mean(chroma_error <= tolerance_cents)),
    }


def evaluate_audio(source: AudioBuffer, output: AudioBuffer, source_pitch: PitchTrack | None = None, output_pitch: PitchTrack | None = None) -> EvaluationReport:
    quality = analyze_audio(output)
    metrics = evaluate_pitch(source_pitch, output_pitch) if source_pitch is not None and output_pitch is not None else {}
    return EvaluationReport(
        source_duration_seconds=source.duration_seconds,
        output_duration_seconds=output.duration_seconds,
        peak=quality.peak,
        clipping_samples=quality.clipping_samples,
        limitations=["Singer similarity is not measured without an explicitly configured embedding model."],
        **metrics,
    )

