from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_dataset_pipeline import fixture_audio

from nsvp.adapters import evaluators
from nsvp.adapters.evaluators import (
    EvaluatorOutput,
    WhisperContentEvaluator,
    error_rate,
    normalize_text,
)
from nsvp.api import create_app
from nsvp.audio.io import save_audio
from nsvp.benchmarking import BenchmarkRunner, BenchmarkSpec, public_config_snapshot
from nsvp.components import ComponentFactory
from nsvp.config import AppConfig, LocalEvaluatorConfig
from nsvp.contracts import AudioBuffer, ConversionRequest
from nsvp.errors import ConfigurationError
from nsvp.evaluation import evaluate_audio
from nsvp.jobs import JobStore, Worker
from nsvp.provenance import RunManifest, reproduce_request
from nsvp.runtime import build_handlers
from nsvp.storage import LocalArtifactStore
from nsvp.testing_backends import IdentityVoiceConverter


@pytest.mark.parametrize("kind", ["content", "singer"])
def test_reproduction_checks_evaluator_assets_before_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    model = tmp_path / "model"
    model.mkdir()
    weights = model / "weights.bin"
    weights.write_bytes(b"original fixture")
    settings = LocalEvaluatorConfig(enabled=True, model_directory=model, model_name="fixture/model", revision="a" * 40)
    config = AppConfig(artifact_root=tmp_path / "artifacts")
    setattr(config.evaluation, kind, settings)

    def run(settings: LocalEvaluatorConfig, *args: object) -> tuple[EvaluatorOutput, str]:
        return EvaluatorOutput(source_text="same", output_text="same", similarity=0.5,
            library_version="fixture"), evaluators.model_directory_digest(model)

    monkeypatch.setattr(evaluators, "evaluate_external", run)
    source = tmp_path / "source.wav"
    save_audio(source, fixture_audio())
    factory = ComponentFactory(config, converters={"identity": IdentityVoiceConverter})
    request = ConversionRequest(song_path=source, target_reference_path=source, output_name="test",
        input_kind="vocal", voice_converter="identity")
    result = build_handlers(config, factory)["conversion"](request.model_dump(mode="json"), lambda *_: None)
    store = LocalArtifactStore(config.artifact_root)
    manifest = RunManifest.model_validate_json(store.resolve(result["artifacts"]["run_manifest.json"]).read_text())
    assert reproduce_request(manifest, store, config).voice_converter == "identity"
    weights.write_bytes(b"changed fixture")
    with pytest.raises(ConfigurationError, match="evaluator asset checksum differs"):
        reproduce_request(manifest, store, config)
    weights.write_bytes(b"original fixture")
    assert manifest.evaluation is not None
    store.resolve(manifest.evaluation.artifact_id).write_bytes(b"tampered report")
    with pytest.raises(ConfigurationError, match="evaluation report checksum"):
        reproduce_request(manifest, store, config)


def test_text_scoring_and_reference_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    assert normalize_text("IŞIK, İÇİN!", "tr") == "ışık için"
    assert normalize_text("Hello,   WORLD!", "en") == "hello world"
    assert error_rate(["one", "two"], ["one", "three", "four"]) == 1
    assert error_rate([], ["word"]) is None
    assert error_rate(["word"], []) == 1

    def run(settings: LocalEvaluatorConfig, kind: str, source: AudioBuffer,
            output: AudioBuffer, language: str | None) -> tuple[EvaluatorOutput, str]:
        return EvaluatorOutput(source_text="wrong source", output_text="IŞIK İÇİN", library_version="fixture"), "a" * 64

    monkeypatch.setattr(evaluators, "evaluate_external", run)
    supervised = WhisperContentEvaluator(LocalEvaluatorConfig(), "tr", "ışık için")
    scores = supervised.evaluate(fixture_audio(), fixture_audio())
    assert scores["wer"].value == scores["cer"].value == 0
    assert supervised.metadata.reference_kind == "verified_lyrics"
    assert supervised.metadata.library_version == "fixture"
    assert supervised.metadata.checkpoint_sha256 == "a" * 64
    proxy = WhisperContentEvaluator(LocalEvaluatorConfig(), "tr")
    assert proxy.evaluate(fixture_audio(), fixture_audio())["wer"].value == 1
    assert proxy.metadata.reference_kind == "source_asr_proxy"
    empty = WhisperContentEvaluator(LocalEvaluatorConfig(), "tr", "  ")
    assert empty.evaluate(fixture_audio(), fixture_audio())["wer"].status == "not_measured"

    def cancel(*args: object) -> tuple[EvaluatorOutput, str]:
        raise InterruptedError("cancelled")

    monkeypatch.setattr(evaluators, "evaluate_external", cancel)
    with pytest.raises(InterruptedError):
        evaluate_audio(fixture_audio(), fixture_audio(), content_evaluator=supervised)


def test_configured_evaluators_flow_through_conversion_benchmark_and_api_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str | None]] = []

    def run(settings: LocalEvaluatorConfig, kind: str, source: AudioBuffer,
            output: AudioBuffer, language: str | None) -> tuple[EvaluatorOutput, str]:
        calls.append((kind, language))
        return EvaluatorOutput(source_text="hello world", output_text="hello world", similarity=0.75,
            library_version="synthetic-test"), "b" * 64

    monkeypatch.setattr(evaluators, "evaluate_external", run)
    config = AppConfig(artifact_root=tmp_path / "artifacts", database_path=tmp_path / "jobs.db")
    config.evaluation.content.enabled = config.evaluation.singer.enabled = True
    config.evaluation.content.model_directory = tmp_path / "private-model"
    assert "private-model" not in str(public_config_snapshot(config.model_dump(mode="json")))
    factory = ComponentFactory(config, converters={"identity": IdentityVoiceConverter})
    handlers = build_handlers(config, factory)
    source = tmp_path / "source.wav"
    save_audio(source, fixture_audio())
    store = LocalArtifactStore(config.artifact_root)
    artifact = store.put_file(source, "test", "source.wav")
    request = ConversionRequest(song_path=source, target_reference_path=source, output_name="test",
        input_kind="vocal", voice_converter="identity", language="en", reference_text="hello world")
    result = handlers["conversion"](request.model_dump(mode="json"), lambda *_: None)
    report = json.loads(store.resolve(result["artifacts"]["evaluation_report.json"]).read_text())
    assert report["families"]["content"]["wer"]["value"] == 0
    phonemes = report["families"]["content"]["phoneme_error_rate"]
    assert phonemes["status"] == "not_measured" and phonemes["value"] is None
    assert phonemes["reason"] == "Configured content evaluator did not supply this metric."
    assert report["families"]["timbre"]["singer_similarity"]["value"] == 0.75
    spec = BenchmarkSpec.model_validate({"name": "fixture", "cases": [{"case_id": "one",
        "source_artifact_id": artifact, "reference_artifact_id": artifact, "language": "en"}],
        "configurations": [{"configuration_id": "identity", "voice_converter": "identity"}]})
    benchmark = BenchmarkRunner(factory, store).run(spec)
    assert benchmark.results[0].status == "succeeded"
    assert benchmark.results[0].evaluation is not None
    assert benchmark.results[0].evaluation.families["content"]["wer"].value == 0
    client = TestClient(create_app(config, factory))
    response = client.post("/evaluation-jobs", json={"source_artifact_id": artifact,
        "output_artifact_id": artifact, "reference_artifact_id": artifact, "language": "en"})
    assert response.status_code == 202
    assert Worker(JobStore(config.database_path), handlers).run_once()
    completed = client.get("/jobs/" + response.json()["id"]).json()
    assert completed["state"] == "SUCCEEDED"
    assert completed["result"]["evaluation"]["families"]["timbre"]["singer_similarity"]["value"] == 0.75
    assert calls == [("content", "en"), ("singer", None)] * 3


def test_missing_local_model_is_explicit_without_download() -> None:
    evaluator = WhisperContentEvaluator(LocalEvaluatorConfig(enabled=True))
    report = evaluate_audio(fixture_audio(), fixture_audio(), content_evaluator=evaluator)
    assert report.families["content"]["wer"].status == "failed"
    assert report.families["content"]["wer"].value is None
    assert report.families["content"]["phoneme_error_rate"].status == "not_measured"


@pytest.mark.parametrize("executable", [None, Path(sys.executable)])
def test_evaluator_rejects_implicit_or_core_python_before_launch(
    executable: Path | None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_launch(*args: object, **kwargs: object) -> None:
        pytest.fail("Invalid environment must fail before launching a runner")

    monkeypatch.setattr(evaluators, "run_external", unexpected_launch)
    settings = LocalEvaluatorConfig(enabled=True, python_executable=executable)
    with pytest.raises(ConfigurationError, match="separate"):
        evaluators.evaluate_external(settings, "content", fixture_audio(), fixture_audio(), "en")
