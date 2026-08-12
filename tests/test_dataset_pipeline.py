from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from nsvp.audio.io import load_audio, save_audio
from nsvp.config import AudioConfig
from nsvp.contracts import AudioBuffer, ConversionRequest
from nsvp.datasets import DatasetManager, render_dataset_report
from nsvp.pipeline import ConversionPipeline
from nsvp.storage import LocalArtifactStore
from nsvp.testing_backends import DeterministicSeparator, IdentityVoiceConverter


def fixture_audio(seconds: float = 3.0, sample_rate: int = 8_000) -> AudioBuffer:
    time = np.arange(round(seconds * sample_rate), dtype=np.float32) / sample_rate
    return AudioBuffer(waveform=(0.3 * np.sin(2 * np.pi * 220 * time))[None, :], sample_rate=sample_rate)


class DatasetPipelineTests(unittest.TestCase):
    def test_dataset_version_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            save_audio(raw / "recording.wav", fixture_audio())
            store = LocalArtifactStore(root / "artifacts")
            config = AudioConfig(training_sample_rate=8_000, minimum_segment_seconds=1, target_maximum_segment_seconds=2, absolute_maximum_segment_seconds=3)
            manager = DatasetManager(store, config)
            first = manager.prepare(raw, "authorized-singer")
            second = manager.prepare(raw, "authorized-singer")
            report = root / "report.html"
            render_dataset_report(first, report)
            self.assertEqual(first.version, second.version)
            self.assertTrue(report.is_file())
            self.assertTrue(any(item.split == "train" for item in first.segments))

    def test_synthetic_vertical_slice_produces_all_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            song = root / "song.wav"
            reference = root / "reference.wav"
            save_audio(song, fixture_audio())
            save_audio(reference, fixture_audio())
            store = LocalArtifactStore(root / "artifacts")
            pipeline = ConversionPipeline(DeterministicSeparator(), IdentityVoiceConverter(), store)
            result = pipeline.run(ConversionRequest(song_path=song, target_reference_path=reference, output_name="fixture"))
            expected = {"source_vocal.wav", "instrumental.wav", "converted_vocal_raw.wav", "converted_vocal_processed.wav", "final_mix.wav", "conversion_report.json"}
            self.assertEqual(set(result.artifacts), expected)
            for artifact_id in result.artifacts.values():
                self.assertTrue(store.resolve(artifact_id).is_file())
            final_mix = load_audio(store.resolve(result.artifacts["final_mix.wav"]))
            self.assertAlmostEqual(final_mix.duration_seconds, 3.0, places=2)


if __name__ == "__main__":
    unittest.main()
