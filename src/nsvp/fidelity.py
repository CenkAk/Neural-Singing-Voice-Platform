from __future__ import annotations

import importlib

import numpy as np
import numpy.typing as npt

from .audio.processing import resample_audio, to_mono
from .contracts import AudioBuffer, MetricResult

Signal = npt.NDArray[np.float64]


def measured(value: float | None, *, unit: str | None = None, reason: str | None = None) -> MetricResult:
    if value is None or not np.isfinite(value):
        return MetricResult(unit=unit, reason=reason or "Metric is undefined for these inputs.")
    return MetricResult(value=float(value), status="measured", unit=unit, reason=reason)


def paired_waveforms(source: AudioBuffer, output: AudioBuffer) -> tuple[Signal, Signal]:
    reference = to_mono(source).waveform[0].astype(np.float64)
    candidate = resample_audio(to_mono(output), source.sample_rate).waveform[0].astype(np.float64)
    count = min(reference.size, candidate.size)
    return reference[:count], candidate[:count]


def _magnitude(signal: Signal, frame: int, hop: int) -> Signal:
    padded = np.pad(signal, (0, max(0, frame - signal.size)))
    windows = np.lib.stride_tricks.sliding_window_view(padded, frame)[::hop]
    return np.asarray(np.abs(np.fft.rfft(windows * np.hanning(frame), axis=-1)), dtype=np.float64)


def evaluate_fidelity(source: AudioBuffer, output: AudioBuffer) -> dict[str, MetricResult]:
    reference, candidate = paired_waveforms(source, output)
    frame, hop = max(8, round(source.sample_rate * 0.04)), max(1, round(source.sample_rate * 0.01))
    source_spectrum, output_spectrum = _magnitude(reference, frame, hop), _magnitude(candidate, frame, hop)
    denominator = float(np.linalg.norm(source_spectrum))
    spectral = float(np.linalg.norm(output_spectrum - source_spectrum) / denominator) if denominator > 1e-12 else None
    envelope_frame = max(1, round(source.sample_rate * 0.05))
    count = reference.size // envelope_frame
    envelope_correlation = None
    if count > 1:
        reference_envelope = np.sqrt(np.mean(reference[:count * envelope_frame].reshape(count, -1)**2, axis=1))
        output_envelope = np.sqrt(np.mean(candidate[:count * envelope_frame].reshape(count, -1)**2, axis=1))
        if np.std(reference_envelope) > 1e-12 and np.std(output_envelope) > 1e-12:
            envelope_correlation = float(np.corrcoef(reference_envelope, output_envelope)[0, 1])
    result = {
        "spectral_convergence": measured(spectral, reason="Aligned common duration; spectral change is not a perceptual quality score."),
        "rms": measured(float(np.sqrt(np.mean(output.waveform.astype(np.float64)**2)))),
        "duration_ratio": measured(output.duration_seconds / source.duration_seconds),
        "energy_envelope_correlation": measured(envelope_correlation),
        "true_peak": MetricResult(reason="True-peak measurement is not implemented in v0.2."),
        "integrated_loudness": MetricResult(unit="LUFS", reason="Optional pyloudnorm evaluator is unavailable."),
    }
    try:
        loudness = importlib.import_module("pyloudnorm")
    except ImportError:
        return result
    try:
        value = float(loudness.Meter(output.sample_rate).integrated_loudness(output.waveform.T))
        result["integrated_loudness"] = measured(value, unit="LUFS")
    except ValueError:
        result["integrated_loudness"] = MetricResult(unit="LUFS", reason="Audio is too short or invalid for integrated loudness.")
    return result


def evaluate_separation(reference: AudioBuffer, separated: AudioBuffer) -> MetricResult:
    target, estimate = paired_waveforms(reference, separated)
    target, estimate = target - target.mean(), estimate - estimate.mean()
    energy = float(np.dot(target, target))
    if energy <= 1e-12 or float(np.dot(estimate, estimate)) <= 1e-12:
        return MetricResult(unit="dB", reason="SI-SDR is undefined for a silent reference or estimate.")
    projection = np.dot(estimate, target) / energy * target
    noise = estimate - projection
    numerator, denominator = float(np.dot(projection, projection)), float(np.dot(noise, noise))
    if numerator <= 1e-12 or denominator <= 1e-12:
        return MetricResult(unit="dB", reason="SI-SDR is infinite or undefined; no finite score is reported.")
    return measured(float(10 * np.log10(numerator / denominator)), unit="dB")
