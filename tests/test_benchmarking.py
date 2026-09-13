from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_dataset_pipeline import fixture_audio

from nsvp.audio.io import save_audio
from nsvp.benchmarking import BenchmarkRunner, BenchmarkSpec
from nsvp.components import ComponentFactory
from nsvp.config import AppConfig
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
    data = json.loads(store.resolve(run.artifacts["benchmark.json"]).read_text())
    assert len(data["results"]) == 3
    assert "artifact_root" not in data["config_snapshot"]
    report = store.resolve(run.artifacts["benchmark.html"]).read_text(encoding="utf-8")
    assert "Synthetic &lt;comparison&gt;" in report
    assert "not_measured" in report


def test_benchmark_rejects_duplicate_configuration_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        BenchmarkSpec.model_validate({
            "name": "test", "cases": [{"case_id": "one", "source_artifact_id": "x", "reference_artifact_id": "y"}],
            "configurations": [{"configuration_id": "same"}, {"configuration_id": "same"}],
        })
