from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from ..contracts import AudioBuffer
from ..errors import AudioValidationError, DependencyUnavailableError

SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}


def load_audio(path: Path) -> AudioBuffer:
    path = path.resolve()
    if not path.is_file():
        raise AudioValidationError(f"audio file does not exist: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise AudioValidationError(f"unsupported audio extension: {path.suffix}")
    try:
        import soundfile as sf  # type: ignore[import-untyped]
    except ImportError:
        if path.suffix.lower() != ".wav":
            raise DependencyUnavailableError("soundfile is required for non-WAV audio")
        return _load_wave(path)
    try:
        waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except Exception as exc:
        raise AudioValidationError(f"could not decode audio: {path.name}: {exc}") from exc
    return AudioBuffer(waveform=waveform.T, sample_rate=int(sample_rate))


def save_audio(path: Path, audio: AudioBuffer, subtype: str = "PCM_16") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf
    except ImportError:
        if path.suffix.lower() != ".wav" or subtype != "PCM_16":
            raise DependencyUnavailableError("soundfile is required for this output format")
        _save_wave(path, audio)
        return
    sf.write(path, audio.waveform.T, audio.sample_rate, subtype=subtype)


def _load_wave(path: Path) -> AudioBuffer:
    try:
        with wave.open(str(path), "rb") as stream:
            channels = stream.getnchannels()
            sample_rate = stream.getframerate()
            sample_width = stream.getsampwidth()
            frames = stream.readframes(stream.getnframes())
    except (wave.Error, EOFError) as exc:
        raise AudioValidationError(f"corrupt WAV file: {path.name}: {exc}") from exc
    if sample_width not in (1, 2, 3, 4):
        raise AudioValidationError(f"unsupported WAV sample width: {sample_width}")
    if sample_width == 1:
        values = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        values = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        values = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        signed = raw[:, 0].astype(np.int32) | (raw[:, 1].astype(np.int32) << 8) | (raw[:, 2].astype(np.int32) << 16)
        signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
        values = signed.astype(np.float32) / 8388608.0
    return AudioBuffer(waveform=values.reshape(-1, channels).T, sample_rate=sample_rate)


def _save_wave(path: Path, audio: AudioBuffer) -> None:
    pcm = (np.clip(audio.waveform.T, -1.0, 1.0) * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(audio.channels)
        stream.setsampwidth(2)
        stream.setframerate(audio.sample_rate)
        stream.writeframes(pcm.tobytes())
