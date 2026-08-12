from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from nsvp.audio.io import load_audio, save_audio
from nsvp.audio.processing import analyze_audio, mix_audio, preprocess_audio
from nsvp.audio.segmentation import find_segments
from nsvp.contracts import AudioBuffer


def sine(seconds: float = 1.0, sample_rate: int = 16_000, frequency: float = 220.0, channels: int = 1) -> AudioBuffer:
    time = np.arange(round(seconds * sample_rate), dtype=np.float32) / sample_rate
    signal = (0.4 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)
    return AudioBuffer(waveform=np.repeat(signal[np.newaxis, :], channels, axis=0), sample_rate=sample_rate)


class AudioTests(unittest.TestCase):
    def test_wave_round_trip_and_quality(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tone.wav"
            save_audio(path, sine(channels=2))
            loaded = load_audio(path)
        self.assertEqual(loaded.channels, 2)
        self.assertEqual(loaded.sample_rate, 16_000)
        self.assertAlmostEqual(loaded.duration_seconds, 1.0, places=3)
        self.assertEqual(analyze_audio(loaded).clipping_samples, 0)

    def test_preprocess_resamples_mono_and_removes_dc(self) -> None:
        audio = sine(channels=2)
        audio = AudioBuffer(waveform=audio.waveform + 0.1, sample_rate=audio.sample_rate)
        processed = preprocess_audio(audio, 8_000)
        self.assertEqual(processed.channels, 1)
        self.assertEqual(processed.sample_rate, 8_000)
        self.assertAlmostEqual(float(processed.waveform.mean()), 0.0, places=5)

    def test_mix_prevents_clipping_and_matches_length(self) -> None:
        loud = AudioBuffer(waveform=np.ones((1, 1_000), dtype=np.float32) * 0.9, sample_rate=1_000)
        mixed, headroom = mix_audio(loud, loud)
        self.assertLess(headroom, 0)
        self.assertLessEqual(float(np.abs(mixed.waveform).max()), 0.98001)
        self.assertEqual(mixed.samples, loud.samples)

    def test_segmentation_prefers_silence(self) -> None:
        tone = sine(seconds=4.0, sample_rate=1_000).waveform
        silence = np.zeros((1, 1_000), dtype=np.float32)
        audio = AudioBuffer(waveform=np.concatenate([tone, silence, tone], axis=1), sample_rate=1_000)
        segments = find_segments(audio, minimum_seconds=2, target_maximum_seconds=5, absolute_maximum_seconds=6, minimum_silence_seconds=0.2)
        self.assertEqual(len(segments), 2)
        self.assertTrue(4.0 <= segments[0].duration(audio.sample_rate) <= 5.0)


if __name__ == "__main__":
    unittest.main()
