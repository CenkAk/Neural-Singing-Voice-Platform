from __future__ import annotations

from typing import Any

import numpy as np

from .audio.processing import resample_audio, to_mono
from .contracts import AudioBuffer, PitchTrack
from .errors import DependencyUnavailableError


class AutocorrelationPitchExtractor:
    """Dependency-light analysis fallback; not the production SVC F0 estimator."""

    name = "autocorrelation"

    def __init__(self, frame_seconds: float = 0.04, hop_seconds: float = 0.01, fmin: float = 50.0, fmax: float = 1_100.0) -> None:
        self.frame_seconds = frame_seconds
        self.hop_seconds = hop_seconds
        self.fmin = fmin
        self.fmax = fmax

    def extract(self, audio: AudioBuffer) -> PitchTrack:
        analysis = resample_audio(to_mono(audio), min(audio.sample_rate, 8_000))
        mono = analysis.waveform[0]
        frame = max(8, round(self.frame_seconds * analysis.sample_rate))
        hop = max(1, round(self.hop_seconds * analysis.sample_rate))
        lag_min = max(1, int(analysis.sample_rate / self.fmax))
        lag_max = min(frame - 2, int(analysis.sample_rate / self.fmin))
        starts = range(0, max(1, mono.size - frame + 1), hop)
        f0: list[float] = []
        confidence: list[float] = []
        for start in starts:
            window = mono[start : start + frame]
            if window.size < frame:
                window = np.pad(window, (0, frame - window.size))
            window = (window - window.mean()) * np.hanning(frame)
            energy = float(np.dot(window, window))
            if energy < 1e-8:
                f0.append(0.0)
                confidence.append(0.0)
                continue
            fft_size = 1 << (2 * frame - 1).bit_length()
            spectrum = np.fft.rfft(window, n=fft_size)
            correlation = np.fft.irfft(spectrum * np.conj(spectrum), n=fft_size)[:frame]
            region = correlation[lag_min : lag_max + 1]
            lag = int(np.argmax(region)) + lag_min
            score = float(correlation[lag] / max(correlation[0], 1e-12))
            f0.append(analysis.sample_rate / lag if score >= 0.25 else 0.0)
            confidence.append(max(0.0, min(1.0, score)))
        values = np.asarray(f0, dtype=np.float32)
        times = np.arange(values.size, dtype=np.float32) * self.hop_seconds
        return PitchTrack(
            timestamps=times,
            f0_hz=values,
            voiced=values > 0,
            confidence=np.asarray(confidence, dtype=np.float32),
            extractor=self.name,
        )


class TorchCrepePitchExtractor:
    name = "torchcrepe"

    def __init__(self, device: Any = "cpu", model: str = "full", confidence_threshold: float = 0.3) -> None:
        self.device = device
        self.model = model
        self.confidence_threshold = confidence_threshold

    def extract(self, audio: AudioBuffer) -> PitchTrack:
        try:
            import torch
            import torchcrepe
        except ImportError as exc:
            raise DependencyUnavailableError("TorchCREPE requires the 'ml' dependency group") from exc
        prepared = resample_audio(to_mono(audio), 16_000)
        tensor = torch.from_numpy(prepared.waveform).to(self.device)
        hop = 160
        pitch, periodicity = torchcrepe.predict(
            tensor, 16_000, hop, 50.0, 1_100.0, self.model, batch_size=1024, device=self.device, return_periodicity=True
        )
        f0 = pitch.squeeze(0).detach().cpu().numpy().astype(np.float32)
        confidence = periodicity.squeeze(0).detach().cpu().numpy().astype(np.float32)
        voiced = confidence >= self.confidence_threshold
        f0[~voiced] = 0.0
        return PitchTrack(
            timestamps=np.arange(f0.size, dtype=np.float32) * hop / 16_000,
            f0_hz=f0,
            voiced=voiced,
            confidence=confidence,
            extractor=self.name,
        )


class PyWorldPitchExtractor:
    name = "pyworld"

    def extract(self, audio: AudioBuffer) -> PitchTrack:
        try:
            import pyworld
        except ImportError as exc:
            raise DependencyUnavailableError("PyWORLD requires the 'ml' dependency group") from exc
        mono = to_mono(audio).waveform[0].astype(np.float64)
        f0, times = pyworld.dio(mono, audio.sample_rate, f0_floor=50.0, f0_ceil=1_100.0)
        f0 = pyworld.stonemask(mono, f0, times, audio.sample_rate).astype(np.float32)
        return PitchTrack(timestamps=times.astype(np.float32), f0_hz=f0, voiced=f0 > 0, extractor=self.name)
