from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_dataset_pipeline import fixture_audio

from nsvp.audio.io import save_audio
from nsvp.benchmarking import (
    BenchmarkCaseResult,
    BenchmarkRun,
    BenchmarkRunner,
    BenchmarkSpec,
    summarize_benchmark,
)
from nsvp.components import ComponentFactory
from nsvp.config import AppConfig
from nsvp.contracts import EvaluatorMetadata, InputCondition, MetricResult
from nsvp.evaluation import evaluate_audio
from nsvp.provenance import RunManifest
from nsvp.storage import LocalArtifactStore
from nsvp.testing_backends import IdentityVoiceConverter


def test_benchmark_compares_identical_inputs_and_persists_honest_results(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path / "artifacts")
    audio = tmp_path / "source.wav"
    save_audio(audio, fixture_audio(seconds=1))
    artifact_id = store.put_file(audio, "uploads")
    config = AppConfig(artifact_root=store.root)
    factory = ComponentFactory(config, converters={"fixture-a": IdentityVoiceConverter, "fixture-b": IdentityVoiceConverter})
    spec = BenchmarkSpec.model_validate({
        "name": "Synthetic <comparison>",
        "cases": [{"case_id": "clean", "source_artifact_id": artifact_id, "reference_artifact_id": artifact_id, "input_condition": "clean_lead"}],
        "configurations": [
            {"configuration_id": "a", "voice_converter": "fixture-a"},
            {"configuration_id": "b", "voice_converter": "fixture-b"},
            {"configuration_id": "missing", "voice_converter": "soulx_singer"},
        ],
    })
    run = BenchmarkRunner(factory, store).run(spec)
    assert [row.status for row in run.results[:2]] == ["succeeded", "succeeded"]
    assert run.results[0].source_sha256 == run.results[1].source_sha256
    assert run.results[0].random_seed == run.results[1].random_seed
    assert run.results[2].status == "not_tested"
    assert run.results[2].evaluation is None
    assert run.results[0].conversion is not None
    manifest = RunManifest.model_validate_json(store.resolve(run.results[0].conversion.artifacts["run_manifest.json"]).read_text())
    assert (manifest.benchmark_run_id, manifest.benchmark_case_id, manifest.benchmark_configuration_id) == (run.run_id, "clean", "a")
    data = json.loads(store.resolve(run.artifacts["benchmark.json"]).read_text())
    assert len(data["results"]) == 3
    assert "artifact_root" not in data["config_snapshot"]
    report = store.resolve(run.artifacts["benchmark.html"]).read_text(encoding="utf-8")
    assert "Synthetic &lt;comparison&gt;" in report
    assert "not_measured" in report
    assert data["summary"]
    performance = run.results[0].evaluation
    assert performance is not None
    assert performance.families["performance"]["pipeline_seconds"].value is not None
    assert performance.families["performance"]["peak_vram_bytes"].value is None


def test_summary_pairs_common_cases_and_separates_evaluator_versions() -> None:
    spec = BenchmarkSpec.model_validate({"name": "summary", "cases": [
        {"case_id": name, "source_artifact_id": "source", "reference_artifact_id": "reference"} for name in ("one", "two")],
        "configurations": [{"configuration_id": name} for name in ("a", "b", "c")]})
    run = BenchmarkRun(run_id="test", name="summary", spec=spec, config_snapshot={}, hardware={})
    for configuration, case, value, version in (("a", "one", 1, "1"), ("a", "two", 100, "1"),
            ("b", "one", 3, "1"), ("b", "two", None, "1"), ("c", "one", 9, "2")):
        metric = MetricResult(value=value, status="measured" if value is not None else "failed",
            evaluator=EvaluatorMetadata(name="fixture", version=version, domain="synthetic"))
        report = evaluate_audio(fixture_audio(), fixture_audio())
        report.families = {"content": {"wer": metric}}
        run.results.append(BenchmarkCaseResult(case_id=case, configuration_id=configuration, random_seed=42,
            input_condition=InputCondition.UNKNOWN, status="succeeded", source_sha256="source", reference_sha256="reference",
            processing_time_seconds=1, evaluation=report))
    summary = summarize_benchmark(run)
    assert len(summary) == 2
    original = next(item for item in summary if item["definition"]["evaluator"]["version"] == "1")
    assert original["configurations"][0]["mean"] == 50.5
    assert original["configurations"][1]["measured_count"] == 1
    assert original["configurations"][1]["observed_status_counts"]["failed"] == 1
    paired = original["paired"][0]
    assert paired["common_count"] == 1 and paired["left_mean"] == 1 and paired["right_mean"] == 3
    assert paired["mean_difference_left_minus_right"] == -2
    assert original["paired"][1]["common_count"] == 0
    assert original["paired"][1]["right_mean"] is None


def test_benchmark_rejects_duplicate_configuration_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        BenchmarkSpec.model_validate({
            "name": "test", "cases": [{"case_id": "one", "source_artifact_id": "x", "reference_artifact_id": "y"}],
            "configurations": [{"configuration_id": "same"}, {"configuration_id": "same"}],
        })
