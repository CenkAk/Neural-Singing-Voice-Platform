import shutil
from pathlib import Path

import numpy as np
import pytest
from test_dataset_pipeline import fixture_audio

from nsvp.audio.io import save_audio
from nsvp.audio.processing import loudness_metrics
from nsvp.config import AudioConfig, StudioCleanConfig
from nsvp.contracts import AudioBuffer
from nsvp.datasets import DatasetManager, render_dataset_report
from nsvp.errors import ConfigurationError
from nsvp.preprocessing import StudioCleanVocalPreprocessor
from nsvp.storage import LocalArtifactStore


def test_original_format_loudness_and_duplicates_survive_processing(tmp_path: Path) -> None:
    inputs = tmp_path / "input"
    inputs.mkdir()
    mono = fixture_audio(sample_rate=48000)
    stereo = AudioBuffer(waveform=np.repeat(mono.waveform, 2, axis=0) + 0.1, sample_rate=48000)
    save_audio(inputs / "original.wav", stereo, subtype="FLOAT")
    shutil.copyfile(inputs / "original.wav", inputs / "copy.wav")
    save_audio(inputs / "other.wav", fixture_audio())
    manifest = DatasetManager(LocalArtifactStore(tmp_path / "artifacts"), AudioConfig(training_sample_rate=16000)).prepare(inputs, "singer")
    assert all(segment.sample_rate == 16000 for segment in manifest.segments)
    original = next(item for item in manifest.analysis["original_files"] if item["file_name"] == "original.wav")
    assert original["original"]["sample_rate"] == 48000
    assert original["original"]["channels"] == 2
    assert original["original"]["dc_offset"] == pytest.approx(0.1, abs=1e-5)
    assert original["loudness"]["integrated_loudness_lufs"]["status"] == "measured"
    assert len(manifest.analysis["exact_duplicates"]) == 1
    assert len(manifest.segments) == len({item.segment_id for item in manifest.segments})
    duplicate_hash = manifest.analysis["exact_duplicates"][0]["sha256"]
    copies = [item for item in manifest.segments if item.source_sha256 == duplicate_hash]
    assert {item.source_file for item in copies} == {"original.wav", "copy.wav"}
    assert len({item.split for item in copies}) == 1
    assert manifest.analysis["source_warnings"]
    report = tmp_path / "report.html"
    render_dataset_report(manifest, report)
    assert "Original files, before processing" in report.read_text()


def test_silence_short_audio_and_gain_have_honest_loudness() -> None:
    silence = AudioBuffer(waveform=np.zeros((1, 16000), dtype=np.float32), sample_rate=16000)
    metrics = loudness_metrics(silence)
    assert all(item.value is None and item.status == "not_measured" for item in metrics.values())
    assert loudness_metrics(fixture_audio(seconds=0.1))["integrated_loudness_lufs"].status == "not_measured"
    audio = fixture_audio(sample_rate=16000)
    louder = audio.model_copy(update={"waveform": audio.waveform * 2})
    low, high = loudness_metrics(audio), loudness_metrics(louder)
    assert high["integrated_loudness_lufs"].value is not None and low["integrated_loudness_lufs"].value is not None
    assert high["integrated_loudness_lufs"].value - low["integrated_loudness_lufs"].value == pytest.approx(6.0206, abs=0.001)
    assert high["crest_factor_db"].value == pytest.approx(low["crest_factor_db"].value)


def test_studio_clean_reaches_target_and_bounds_gain_and_peak(tmp_path: Path) -> None:
    audio = fixture_audio(sample_rate=16000)
    offset = audio.model_copy(update={"waveform": audio.waveform + 0.1})
    stage = StudioCleanVocalPreprocessor(StudioCleanConfig())
    cleaned = stage.process(offset, tmp_path)
    assert abs(float(cleaned.waveform.mean())) < 1e-6
    assert cleaned.samples == audio.samples and cleaned.sample_rate == audio.sample_rate
    assert loudness_metrics(cleaned)["integrated_loudness_lufs"].value == pytest.approx(-23, abs=0.01)
    quiet = audio.model_copy(update={"waveform": audio.waveform * 0.01})
    bounded = stage.process(quiet, tmp_path)
    assert float(np.max(np.abs(bounded.waveform))) <= float(np.max(np.abs(quiet.waveform))) * 10**(12 / 20) * 1.001
    with pytest.raises(ConfigurationError, match="loudness gate"):
        stage.process(audio.model_copy(update={"waveform": audio.waveform * 0.0001}), tmp_path)
    peaky = audio.model_copy(update={"waveform": audio.waveform.copy()})
    peaky.waveform[0, 100] = 10
    limited = stage.process(peaky, tmp_path)
    assert float(np.max(np.abs(limited.waveform))) <= 0.980001
