from pathlib import Path

import numpy as np

from nsvp.audio.io import load_audio, save_audio
from nsvp.audio.processing import resample_audio
from nsvp.contracts import AudioBuffer, ConversionRequest
from nsvp.pipeline import ConversionPipeline
from nsvp.storage import LocalArtifactStore
from nsvp.testing_backends import IdentityVoiceConverter


def test_polyphase_rejects_aliasing_and_preserves_shape() -> None:
    time = np.arange(48000) / 48000
    waveform = np.sin(2 * np.pi * 12000 * time).astype(np.float32)
    audio = AudioBuffer(waveform=np.stack([waveform, waveform * 0.5]), sample_rate=48000)
    filtered = resample_audio(audio, 16000)
    legacy = resample_audio(audio, 16000, method="legacy_linear")
    assert filtered.waveform.shape == (2, 16000)
    assert np.sqrt(np.mean(filtered.waveform[:, 100:-100] ** 2)) < 0.001
    assert np.sqrt(np.mean(legacy.waveform ** 2)) > 0.3
    low = AudioBuffer(waveform=np.sin(2 * np.pi * 440 * time).astype(np.float32)[None, :], sample_rate=48000)
    assert abs(float(np.sqrt(np.mean(resample_audio(low, 16000).waveform ** 2))) - 2**-0.5) < 0.005


def test_conversion_raw_float_artifact_does_not_hide_clipping(tmp_path: Path) -> None:
    audio = AudioBuffer(waveform=np.full((1, 16000), 1.25, dtype=np.float32), sample_rate=16000)
    source = tmp_path / "source.wav"
    save_audio(source, audio, subtype="FLOAT")
    store = LocalArtifactStore(tmp_path / "artifacts")
    pipeline = ConversionPipeline(None, IdentityVoiceConverter(), store)
    result = pipeline.run(ConversionRequest(song_path=source, target_reference_path=source,
        output_name="float", input_kind="vocal"))
    raw = load_audio(store.resolve(result.artifacts["converted_vocal_raw.wav"]))
    assert float(raw.waveform.max()) == 1.25
