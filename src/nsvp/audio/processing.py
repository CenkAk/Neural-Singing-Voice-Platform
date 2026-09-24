from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..contracts import AudioBuffer, EvaluatorMetadata, MetricResult
from ..errors import DependencyUnavailableError

RESAMPLING_VERSION = "polyphase-kaiser5-v1"


@dataclass(frozen=True)
class AudioQuality:
    duration_seconds: float
    sample_rate: int
    channels: int
    peak: float
    rms: float
    clipping_samples: int
    silence_ratio: float
    dc_offset: float
    finite: bool

    def as_dict(self) -> dict[str, float | int | bool]:
        return self.__dict__.copy()


def analyze_audio(audio: AudioBuffer, clipping_threshold: float = 0.999, silence_db: float = -60.0) -> AudioQuality:
    waveform = audio.waveform
    absolute = np.abs(waveform)
    peak = float(absolute.max(initial=0.0))
    rms = float(np.sqrt(np.mean(np.square(waveform), dtype=np.float64)))
    silence_threshold = 10.0 ** (silence_db / 20.0)
    frame = max(1, int(audio.sample_rate * 0.02))
    mono = waveform.mean(axis=0)
    usable = mono[: (mono.size // frame) * frame]
    if usable.size:
        frame_rms = np.sqrt(np.mean(usable.reshape(-1, frame) ** 2, axis=1))
        silence_ratio = float(np.mean(frame_rms < silence_threshold))
    else:
        silence_ratio = float(rms < silence_threshold)
    return AudioQuality(
        duration_seconds=audio.duration_seconds,
        sample_rate=audio.sample_rate,
        channels=audio.channels,
        peak=peak,
        rms=rms,
        clipping_samples=int(np.count_nonzero(absolute >= clipping_threshold)),
        silence_ratio=silence_ratio,
        dc_offset=float(np.mean(waveform, dtype=np.float64)),
        finite=bool(np.isfinite(waveform).all()),
    )


def to_mono(audio: AudioBuffer) -> AudioBuffer:
    if audio.channels == 1:
        return audio
    return AudioBuffer(waveform=audio.waveform.mean(axis=0, keepdims=True), sample_rate=audio.sample_rate)


def loudness_metrics(audio: AudioBuffer) -> dict[str, MetricResult]:
    quality = analyze_audio(audio)
    crest = 20 * math.log10(quality.peak / quality.rms) if quality.rms > 0 else None
    result = {"crest_factor_db": MetricResult(value=crest, status="measured", unit="dB") if crest is not None
        else MetricResult(unit="dB", reason="Silent audio has no finite crest factor.")}
    metadata = EvaluatorMetadata(name="pyloudnorm", version="BS.1770", domain="audio",
        limitations=["Integrated loudness is not a noise, reverberation or perceptual quality score."])
    if audio.duration_seconds < 0.4:
        loudness = MetricResult(unit="LUFS", reason="At least 400 ms of audio is required.", evaluator=metadata)
    elif audio.channels > 2:
        loudness = MetricResult(status="unsupported", unit="LUFS", reason="Channel layout is unknown for multichannel audio.", evaluator=metadata)
    else:
        try:
            meter = importlib.import_module("pyloudnorm").Meter(audio.sample_rate)
            value = float(meter.integrated_loudness(audio.waveform.T))
            loudness = MetricResult(value=value, status="measured", unit="LUFS", evaluator=metadata) if math.isfinite(value) else MetricResult(
                unit="LUFS", reason="No audio above the loudness gate.", evaluator=metadata)
        except ImportError:
            loudness = MetricResult(unit="LUFS", reason="pyloudnorm is not installed.", evaluator=metadata)
        except (ValueError, RuntimeError):
            loudness = MetricResult(status="failed", unit="LUFS", reason="Loudness measurement failed.", evaluator=metadata)
    result["integrated_loudness_lufs"] = loudness
    return result


def remove_dc(audio: AudioBuffer) -> AudioBuffer:
    centered = audio.waveform - audio.waveform.mean(axis=1, keepdims=True)
    return AudioBuffer(waveform=centered.astype(np.float32), sample_rate=audio.sample_rate)


def resample_audio(audio: AudioBuffer, target_sample_rate: int, *, method: Literal["polyphase", "legacy_linear"] = "polyphase") -> AudioBuffer:
    if target_sample_rate <= 0:
        raise ValueError("target sample rate must be positive")
    if audio.sample_rate == target_sample_rate:
        return audio
    target_samples = max(1, round(audio.samples * target_sample_rate / audio.sample_rate))
    if method == "polyphase":
        try:
            signal = importlib.import_module("scipy.signal")
        except ImportError as exc:
            raise DependencyUnavailableError("Polyphase resampling requires the audio extra (scipy)") from exc
        divisor = math.gcd(audio.sample_rate, target_sample_rate)
        waveform = signal.resample_poly(audio.waveform, target_sample_rate // divisor,
            audio.sample_rate // divisor, axis=1, window=("kaiser", 5.0))
        result = AudioBuffer(waveform=np.asarray(waveform, dtype=np.float32), sample_rate=target_sample_rate)
        return match_length(result, target_samples)
    if method != "legacy_linear":
        raise ValueError("unknown resampling method")
    old_positions = np.linspace(0.0, 1.0, audio.samples, endpoint=False)
    new_positions = np.linspace(0.0, 1.0, target_samples, endpoint=False)
    channels = [np.interp(new_positions, old_positions, channel) for channel in audio.waveform]
    return AudioBuffer(waveform=np.asarray(channels, dtype=np.float32), sample_rate=target_sample_rate)


def preprocess_audio(audio: AudioBuffer, target_sample_rate: int, mono: bool = True) -> AudioBuffer:
    result = to_mono(audio) if mono else audio
    result = remove_dc(result)
    return resample_audio(result, target_sample_rate)


def match_length(audio: AudioBuffer, samples: int) -> AudioBuffer:
    if audio.samples == samples:
        return audio
    if audio.samples > samples:
        waveform = audio.waveform[:, :samples]
    else:
        waveform = np.pad(audio.waveform, ((0, 0), (0, samples - audio.samples)))
    return AudioBuffer(waveform=waveform, sample_rate=audio.sample_rate)


def apply_gain(audio: AudioBuffer, gain_db: float) -> AudioBuffer:
    gain = 10.0 ** (gain_db / 20.0)
    return AudioBuffer(waveform=(audio.waveform * gain).astype(np.float32), sample_rate=audio.sample_rate)


def mix_audio(
    vocal: AudioBuffer,
    instrumental: AudioBuffer,
    vocal_gain_db: float = 0.0,
    instrumental_gain_db: float = 0.0,
    peak_ceiling: float = 0.98,
) -> tuple[AudioBuffer, float]:
    vocal = resample_audio(vocal, instrumental.sample_rate)
    vocal = match_length(vocal, instrumental.samples)
    if instrumental.channels > 1 and vocal.channels == 1:
        vocal = AudioBuffer(waveform=np.repeat(vocal.waveform, instrumental.channels, axis=0), sample_rate=vocal.sample_rate)
    if vocal.channels != instrumental.channels:
        raise ValueError("vocal and instrumental channel counts are incompatible")
    mixed = apply_gain(vocal, vocal_gain_db).waveform + apply_gain(instrumental, instrumental_gain_db).waveform
    peak = float(np.max(np.abs(mixed), initial=0.0))
    headroom_db = 0.0
    if peak > peak_ceiling:
        scale = peak_ceiling / peak
        mixed *= scale
        headroom_db = 20.0 * math.log10(scale)
    return AudioBuffer(waveform=mixed.astype(np.float32), sample_rate=instrumental.sample_rate), headroom_db
