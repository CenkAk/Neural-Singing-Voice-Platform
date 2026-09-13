from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from test_dataset_pipeline import fixture_audio

from nsvp.audio.io import save_audio
from nsvp.config import AudioConfig, DatasetAuditConfig
from nsvp.contracts import AudioBuffer, ConversionRequest, VocalSource
from nsvp.datasets import DatasetManager
from nsvp.errors import ConfigurationError
from nsvp.pipeline import ConversionPipeline
from nsvp.preprocessing import NoOpVocalPreprocessor, process_vocal, select_source
from nsvp.storage import LocalArtifactStore
from nsvp.testing_backends import DeterministicSeparator, IdentityVoiceConverter


def test_noop_preserves_buffer_and_source_selection_is_explicit(tmp_path: Path) -> None:
    audio = fixture_audio()
    assert NoOpVocalPreprocessor().process(audio, tmp_path) is audio
    sources = [VocalSource(source_id="source-1", audio=audio), VocalSource(source_id="source-2", audio=audio)]
    assert select_source(sources, "source-2").source_id == "source-2"
    with pytest.raises(ConfigurationError):
        select_source(sources, "lead")


def test_preprocessors_execute_in_order_and_preserve_original(tmp_path: Path) -> None:
    audio = fixture_audio()

    class Gain:
        name = "gain-fixture"

        def process(self, vocal: AudioBuffer, work_dir: Path) -> AudioBuffer:
            return AudioBuffer(waveform=vocal.waveform * 2, sample_rate=vocal.sample_rate)

    class Offset:
        name = "offset-fixture"

        def process(self, vocal: AudioBuffer, work_dir: Path) -> AudioBuffer:
            return AudioBuffer(waveform=vocal.waveform + 0.1, sample_rate=vocal.sample_rate)

    result = process_vocal(audio, [Gain(), Offset()], tmp_path)
    np.testing.assert_allclose(result.selected.waveform, audio.waveform * 2 + 0.1)
    assert len(result.intermediates) == 2


def test_pipeline_cleanup_on_failure(tmp_path: Path) -> None:
    class FailedConverter:
        name = "failed-fixture"
        def __init__(self) -> None:
            self.diagnostic_artifacts: dict[str, Path] = {}

        def convert(self, source_vocal: AudioBuffer, target_reference: AudioBuffer, semitones: int, work_dir: Path) -> AudioBuffer:
            work_dir.mkdir(parents=True, exist_ok=True)
            raw = work_dir / "raw.wav"
            save_audio(raw, source_vocal)
            self.diagnostic_artifacts = {"model_output_original.wav": raw}
            raise RuntimeError("fixture failure")

    audio = tmp_path / "source.wav"
    save_audio(audio, fixture_audio())
    store = LocalArtifactStore(tmp_path / "artifacts")
    pipeline = ConversionPipeline(DeterministicSeparator(), FailedConverter(), store)
    with pytest.raises(RuntimeError):
        pipeline.run(ConversionRequest(song_path=audio, target_reference_path=audio, output_name="test", keep_intermediates=False))
    assert not list((store.root / "work").iterdir())
    assert len(list(store.root.glob("conversion-*/*/model_output_original.wav"))) == 1


def test_dataset_audit_preserves_dataset_version_and_source_splits(tmp_path: Path) -> None:
    source_root = tmp_path / "input"
    source_root.mkdir()
    save_audio(source_root / "one.wav", fixture_audio())
    save_audio(source_root / "two.wav", fixture_audio(seconds=4))
    store = LocalArtifactStore(tmp_path / "artifacts")
    audio = AudioConfig(training_sample_rate=8000)
    audited = DatasetManager(store, audio).prepare(source_root, "singer")
    original = DatasetManager(store, audio, audit_config=DatasetAuditConfig(enabled=False)).prepare(source_root, "singer")
    assert audited.version == original.version
    assert [(s.segment_id, s.split) for s in audited.segments] == [(s.segment_id, s.split) for s in original.segments]
    audit = audited.analysis["audit"]
    assert audit["total_usable_duration_seconds"] == pytest.approx(sum(s.duration_seconds for s in audited.segments))
    assert audit["voiced_duration_seconds"] > 0
    assert audit["warnings"]
    for source in audited.source_files:
        assert len({s.split for s in audited.segments if s.source_sha256 == source}) == 1


def test_direct_vocal_input_bypasses_separator(tmp_path: Path) -> None:
    audio = tmp_path / "source.wav"
    save_audio(audio, fixture_audio())
    result = ConversionPipeline(None, IdentityVoiceConverter(), LocalArtifactStore(tmp_path / "artifacts")).run(
        ConversionRequest(song_path=audio, target_reference_path=audio, output_name="test", input_kind="vocal"),
    )
    assert result.components["separator"] == "none"
