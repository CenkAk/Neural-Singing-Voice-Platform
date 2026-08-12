from __future__ import annotations

import unittest

import numpy as np

from nsvp.contracts import AudioBuffer, PitchTrack
from nsvp.evaluation import evaluate_pitch
from nsvp.pitch import AutocorrelationPitchExtractor


class PitchTests(unittest.TestCase):
    def test_autocorrelation_finds_synthetic_pitch(self) -> None:
        sample_rate = 8_000
        time = np.arange(sample_rate, dtype=np.float32) / sample_rate
        audio = AudioBuffer(waveform=(0.5 * np.sin(2 * np.pi * 200 * time))[None, :], sample_rate=sample_rate)
        track = AutocorrelationPitchExtractor().extract(audio)
        median = float(np.median(track.f0_hz[track.voiced]))
        self.assertAlmostEqual(median, 200.0, delta=8.0)

    def test_pitch_metrics_identical_tracks(self) -> None:
        f0 = np.array([0, 220, 221, 0, 330], dtype=np.float32)
        track = PitchTrack(timestamps=np.arange(5), f0_hz=f0, voiced=f0 > 0, extractor="fixture")
        metrics = evaluate_pitch(track, track)
        self.assertEqual(metrics["f0_cents_rmse"], 0.0)
        self.assertEqual(metrics["raw_pitch_accuracy"], 1.0)
        self.assertEqual(metrics["voicing_error"], 0.0)


if __name__ == "__main__":
    unittest.main()

