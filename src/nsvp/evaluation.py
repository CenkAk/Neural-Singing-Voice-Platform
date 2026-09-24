from __future__ import annotations

import logging

import numpy as np

from .audio.processing import analyze_audio
from .contracts import (
    AudioBuffer,
    EvaluationReport,
    InputCondition,
    MetricResult,
    PitchPreview,
    PitchTrack,
    ProcessTelemetry,
)
from .fidelity import evaluate_fidelity, evaluate_separation, measured
from .interfaces import ContentEvaluator, SingerSimilarityEvaluator

logger = logging.getLogger(__name__)


def preview_frames(track: PitchTrack) -> list[tuple[float, float | None]]:
    indices = np.linspace(0, track.timestamps.size - 1, min(1000, track.timestamps.size), dtype=int)
    return [(float(track.timestamps[index]), float(track.f0_hz[index])
             if track.voiced[index] and track.f0_hz[index] > 0 else None) for index in indices]


def align_pitch(source: PitchTrack, output: PitchTrack) -> tuple[PitchTrack, PitchTrack]:
    indices = np.array([], dtype=np.int64)
    nearest = np.array([], dtype=np.int64)
    if source.timestamps.size and output.timestamps.size:
        indices = np.flatnonzero((source.timestamps >= output.timestamps[0]) & (source.timestamps <= output.timestamps[-1]))
        times = source.timestamps[indices]
        right = np.searchsorted(output.timestamps, times).clip(0, output.timestamps.size - 1)
        left = np.maximum(right - 1, 0)
        nearest = np.where(np.abs(output.timestamps[left] - times) <= np.abs(output.timestamps[right] - times), left, right)
        if output.timestamps.size > 1:
            maximum_gap = float(np.median(np.diff(output.timestamps))) / 2 + 1e-7
            valid = np.abs(output.timestamps[nearest] - times) <= maximum_gap
            indices, nearest = indices[valid], nearest[valid]
    times = source.timestamps[indices]
    return (
        PitchTrack(timestamps=times, f0_hz=source.f0_hz[indices], voiced=source.voiced[indices], extractor=source.extractor),
        PitchTrack(timestamps=times, f0_hz=output.f0_hz[nearest], voiced=output.voiced[nearest], extractor=output.extractor),
    )


def evaluate_pitch(source: PitchTrack, converted: PitchTrack, tolerance_cents: float = 50.0) -> dict[str, float | None]:
    source, converted = align_pitch(source, converted)
    count = min(source.f0_hz.size, converted.f0_hz.size)
    if count == 0:
        return {"f0_cents_rmse": None, "f0_correlation": None, "voicing_error": None, "raw_pitch_accuracy": None, "raw_chroma_accuracy": None}
    source_voiced = source.voiced[:count].astype(bool)
    converted_voiced = converted.voiced[:count].astype(bool)
    voicing_error = float(np.mean(source_voiced != converted_voiced))
    both = source_voiced & converted_voiced & (source.f0_hz[:count] > 0) & (converted.f0_hz[:count] > 0)
    if not np.any(both):
        return {"f0_cents_rmse": None, "f0_correlation": None, "voicing_error": voicing_error, "raw_pitch_accuracy": None, "raw_chroma_accuracy": None}
    source_hz = source.f0_hz[:count][both].astype(np.float64)
    converted_hz = converted.f0_hz[:count][both].astype(np.float64)
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


def evaluate_audio(
    source: AudioBuffer, output: AudioBuffer, source_pitch: PitchTrack | None = None,
    output_pitch: PitchTrack | None = None, *, target_reference: AudioBuffer | None = None,
    content_evaluator: ContentEvaluator | None = None,
    singer_evaluator: SingerSimilarityEvaluator | None = None,
    ground_truth_stem: AudioBuffer | None = None, separated_stem: AudioBuffer | None = None,
    input_condition: InputCondition = InputCondition.UNKNOWN, transpose_semitones: int = 0,
    pipeline_seconds: float | None = None,
    provider_process: ProcessTelemetry | None = None,
) -> EvaluationReport:
    quality = analyze_audio(output)
    if source_pitch is not None and transpose_semitones:
        source_pitch = source_pitch.model_copy(update={"f0_hz": source_pitch.f0_hz * 2 ** (transpose_semitones / 12)})
    metrics = evaluate_pitch(source_pitch, output_pitch) if source_pitch is not None and output_pitch is not None else {}
    pitch_names = ("f0_cents_rmse", "f0_correlation", "voicing_error", "raw_pitch_accuracy", "raw_chroma_accuracy")
    families: dict[str, dict[str, MetricResult]] = {
        "pitch": {name: measured(metrics.get(name)) for name in pitch_names},
        "content": {name: MetricResult(reason="Configured content evaluator did not supply this metric."
            if content_evaluator is not None else "No content evaluator configured.")
            for name in ("wer", "cer", "phoneme_error_rate")},
        "timbre": {"singer_similarity": MetricResult(reason="No reviewed singing-domain evaluator configured.")},
        "fidelity": evaluate_fidelity(source, output),
        "separation": {"si_sdr": MetricResult(unit="dB", reason="Paired ground-truth and separated stems are required.")},
        "performance": {
            "pipeline_seconds": measured(pipeline_seconds, unit="seconds", reason="Pipeline duration excludes evaluation."),
            "pipeline_real_time_factor": measured(pipeline_seconds / source.duration_seconds if pipeline_seconds is not None else None),
            "peak_ram_bytes": MetricResult(unit="bytes", reason="Application-wide peak RAM was not measured; provider samples are reported separately when available."),
            "peak_vram_bytes": MetricResult(unit="bytes", reason="Device memory was not sampled."),
        },
    }
    families["fidelity"].update({"peak": measured(quality.peak), "clipping_samples": measured(float(quality.clipping_samples))})
    if provider_process is not None:
        families["performance"].update({
            "provider_process_seconds": measured(provider_process.elapsed_seconds, unit="seconds",
                reason="Subprocess lifetime includes imports, model loading, inference and output writing."),
            "provider_sampled_peak_rss_bytes": measured(provider_process.sampled_peak_rss_bytes, unit="bytes",
                reason=f"Process-tree RSS sum with target sampling interval {provider_process.sampling_interval_seconds:g}s; "
                f"{provider_process.samples} samples, {provider_process.missed_samples} missed. "
                "Shared pages may be counted twice, short peaks may be missed; excludes the core process and GPU memory."),
        })
    if source_pitch is not None and output_pitch is not None:
        aligned, _ = align_pitch(source_pitch, output_pitch)
        families["pitch"]["alignment_coverage"] = measured(aligned.timestamps.size / source_pitch.timestamps.size if source_pitch.timestamps.size else None)
    limitations = [
        "Pipeline timing includes loading, separation, conversion and artifact writing, and excludes evaluation; it is not model-only inference timing.",
        "Spectral differences include intended timbre changes; they do not establish perceptual quality.",
        "V1 RPA/RCA use mutually voiced frames; interpret them alongside voicing error.",
        "Timestamp alignment uses nearest frames without bridging missing frame intervals.",
    ]
    if transpose_semitones:
        limitations.append(f"Pitch preservation uses source F0 shifted by {transpose_semitones} semitones.")
    if content_evaluator is not None:
        try:
            for name, metric in content_evaluator.evaluate(source, output).items():
                families["content"][name] = metric.model_copy(update={"evaluator": content_evaluator.metadata})
            limitations.extend(content_evaluator.metadata.limitations)
        except InterruptedError:
            raise
        except Exception:
            logger.exception("Configured content evaluator failed")
            families["content"]["evaluation"] = MetricResult(status="failed", reason="Configured content evaluator failed.", evaluator=content_evaluator.metadata)
            for name in ("wer", "cer"):
                families["content"][name] = MetricResult(status="failed", reason="Configured content evaluator failed.", evaluator=content_evaluator.metadata)
        limitations.append("Conventional speech ASR can be unreliable for singing.")
    if singer_evaluator is not None and target_reference is not None:
        try:
            similarity = singer_evaluator.evaluate(target_reference, output)
            families["timbre"]["singer_similarity"] = similarity.model_copy(update={"evaluator": singer_evaluator.metadata})
            limitations.extend(singer_evaluator.metadata.limitations)
        except InterruptedError:
            raise
        except Exception:
            logger.exception("Configured singer evaluator failed")
            families["timbre"]["singer_similarity"] = MetricResult(status="failed", reason="Configured singer evaluator failed.", evaluator=singer_evaluator.metadata)
    elif singer_evaluator is not None:
        families["timbre"]["singer_similarity"] = MetricResult(reason="Target reference audio is required.", evaluator=singer_evaluator.metadata)
    if ground_truth_stem is not None and separated_stem is not None:
        families["separation"]["si_sdr"] = evaluate_separation(ground_truth_stem, separated_stem)
    return EvaluationReport(
        schema_version="0.3",
        pitch_preview=PitchPreview(source=preview_frames(source_pitch), output=preview_frames(output_pitch),
            source_extractor=source_pitch.extractor, output_extractor=output_pitch.extractor,
            source_transpose_semitones=transpose_semitones)
            if source_pitch is not None and output_pitch is not None else None,
        source_duration_seconds=source.duration_seconds,
        output_duration_seconds=output.duration_seconds,
        peak=quality.peak,
        clipping_samples=quality.clipping_samples,
        limitations=limitations, families=families, input_condition=input_condition,
        singer_similarity=families["timbre"]["singer_similarity"].value,
        f0_cents_rmse=metrics.get("f0_cents_rmse"),
        f0_correlation=metrics.get("f0_correlation"),
        voicing_error=metrics.get("voicing_error"),
        raw_pitch_accuracy=metrics.get("raw_pitch_accuracy"),
        raw_chroma_accuracy=metrics.get("raw_chroma_accuracy"),
    )
