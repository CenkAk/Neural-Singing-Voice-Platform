from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError
from test_dataset_pipeline import fixture_audio

from nsvp.contracts import AudioBuffer, EvaluatorMetadata, MetricResult, PitchTrack
from nsvp.evaluation import evaluate_audio, evaluate_pitch
from nsvp.fidelity import evaluate_fidelity, evaluate_separation


def test_timestamp_alignment_does_not_truncate_by_array_index() -> None:
    source = PitchTrack(timestamps=np.array([0.0, 0.02, 0.04]), f0_hz=np.array([100, 200, 300]), voiced=np.ones(3, dtype=bool), extractor="fixture")
    output = PitchTrack(timestamps=np.array([0.0, 0.01, 0.02, 0.03, 0.04]), f0_hz=np.array([100, 150, 200, 250, 300]), voiced=np.ones(5, dtype=bool), extractor="fixture")
    assert evaluate_pitch(source, output)["f0_cents_rmse"] == 0


def test_nonoverlapping_pitch_is_unmeasured() -> None:
    source = PitchTrack(timestamps=np.array([0, 1]), f0_hz=np.array([100, 200]), voiced=np.ones(2, dtype=bool), extractor="fixture")
    output = PitchTrack(timestamps=np.array([2, 3]), f0_hz=np.array([100, 200]), voiced=np.ones(2, dtype=bool), extractor="fixture")
    assert all(value is None for value in evaluate_pitch(source, output).values())


def test_invalid_pitch_array_lengths_are_rejected() -> None:
    with pytest.raises(ValidationError, match="equal lengths"):
        PitchTrack(timestamps=np.array([0, 1]), f0_hz=np.array([100]), voiced=np.ones(2, dtype=bool), extractor="fixture")


def test_spectral_convergence_identity_and_gain() -> None:
    audio = fixture_audio()
    assert evaluate_fidelity(audio, audio)["spectral_convergence"].value == 0
    doubled = AudioBuffer(waveform=audio.waveform * 2, sample_rate=audio.sample_rate)
    assert evaluate_fidelity(audio, doubled)["spectral_convergence"].value == pytest.approx(1)


def test_si_sdr_matches_orthogonal_synthetic_noise() -> None:
    times = np.arange(8000) / 8000
    source = np.sin(2 * np.pi * 200 * times)
    noise = np.sin(2 * np.pi * 400 * times)
    reference = AudioBuffer(waveform=source.astype(np.float32), sample_rate=8000)
    estimate = AudioBuffer(waveform=(source + 0.1 * noise).astype(np.float32), sample_rate=8000)
    assert evaluate_separation(reference, estimate).value == pytest.approx(20, abs=0.001)
    silent = AudioBuffer(waveform=np.zeros((1, 8000), dtype=np.float32), sample_rate=8000)
    assert evaluate_separation(silent, estimate).status == "not_measured"


def test_metric_states_do_not_allow_fabricated_values() -> None:
    with pytest.raises(ValidationError):
        MetricResult(status="not_measured", value=0)
    with pytest.raises(ValidationError):
        MetricResult(status="measured", value=float("nan"))
    report = evaluate_audio(fixture_audio(), fixture_audio())
    assert set(report.families) == {"pitch", "content", "timbre", "fidelity", "separation"}
    assert report.families["timbre"]["singer_similarity"].value is None
    assert report.families["content"]["wer"].status == "not_measured"


def test_optional_evaluator_failure_is_explicit() -> None:
    class FailedEvaluator:
        metadata = EvaluatorMetadata(name="failed-fixture", version="1", domain="synthetic")

        def evaluate(self, source: AudioBuffer, output: AudioBuffer) -> dict[str, MetricResult]:
            raise RuntimeError("fixture failure")

    report = evaluate_audio(fixture_audio(), fixture_audio(), content_evaluator=FailedEvaluator())
    assert report.families["content"]["evaluation"].status == "failed"
    assert report.families["content"]["evaluation"].evaluator is not None
