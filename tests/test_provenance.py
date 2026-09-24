from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_dataset_pipeline import fixture_audio

from nsvp.api import create_app
from nsvp.audio.io import save_audio
from nsvp.compatibility import compatibility_report
from nsvp.components import ComponentFactory
from nsvp.config import AppConfig
from nsvp.contracts import ConversionRequest
from nsvp.errors import ConfigurationError
from nsvp.jobs import JobStore, Worker
from nsvp.provenance import (
    RunManifest,
    artifact_digest,
    reproduce_request,
    verify_execution_identity,
)
from nsvp.runtime import build_handlers
from nsvp.storage import LocalArtifactStore, sha256_file
from nsvp.testing_backends import IdentityVoiceConverter


def test_failed_compatibility_evidence_does_not_certify_backend(tmp_path: Path) -> None:
    config = AppConfig(artifact_root=tmp_path / "artifacts", database_path=tmp_path / "jobs.db")
    jobs = JobStore(config.database_path)
    job = jobs.enqueue("conversion", {"backend": "rocm", "voice_converter": "seed_vc"})
    client = TestClient(create_app(config))
    assert client.get("/compatibility-report", params={"failed_job_id": job.id}).status_code == 400
    assert jobs.claim_next() is not None
    jobs.fail(job.id, {"code": "external_execution_failed", "message": "private-recording-path"})
    response = client.get("/compatibility-report", params=[("failed_job_id", job.id), ("failed_job_id", job.id)])
    assert response.status_code == 200
    report = response.json()
    assert report["conversions"] == [] and len(report["failed_jobs"]) == 1
    failure = report["failed_jobs"][0]
    assert failure["status"] == "failed" and failure["requested_backend"] == "rocm"
    assert failure["error_code"] == "external_execution_failed"
    assert report["adapter_support"]["seed_vc"]["rocm"] == "not_tested"
    assert "private-recording-path" not in response.text
    assert client.get("/compatibility-report", params={"failed_job_id": "missing"}).status_code == 400


def test_manifest_and_reproduction_through_api_worker(tmp_path: Path) -> None:
    source = tmp_path / "private-source.wav"
    save_audio(source, fixture_audio())
    config = AppConfig(artifact_root=tmp_path / "artifacts", database_path=tmp_path / "jobs.db")
    factory = ComponentFactory(config, converters={"identity": IdentityVoiceConverter})
    handlers = build_handlers(config, factory)
    request = ConversionRequest(song_path=source, target_reference_path=source,
        output_name="test", input_kind="vocal", voice_converter="identity")
    result = handlers["conversion"](request.model_dump(mode="json"), lambda *_: None)
    store = LocalArtifactStore(config.artifact_root)
    identifier = result["artifacts"]["run_manifest.json"]
    raw = store.resolve(identifier).read_text(encoding="utf-8")
    assert str(tmp_path) not in raw and "private-source" not in raw
    manifest = RunManifest.model_validate_json(raw)
    assert manifest.schema_version == "0.3"
    assert manifest.inputs["source"].sha256 == sha256_file(source)
    assert manifest.evaluation is not None
    assert manifest.evaluation.artifact_id == result["artifacts"]["evaluation_report.json"]
    assert "run_manifest.json" not in manifest.outputs
    empty = compatibility_report(store, [])
    assert empty.adapter_support["seed_vc"]["rocm"] == "not_tested"
    assert empty.adapter_support["seed_vc"]["directml"] == "unsupported"
    with pytest.raises(ConfigurationError, match="real converter"):
        compatibility_report(store, [identifier])
    # Metadata-only fixture for report validation, never real inference evidence.
    external = manifest.model_copy(deep=True)
    external.executions["voice_converter"].provider = "seed_vc"
    external.executions["voice_converter"].version = "a" * 40
    external.executions["voice_converter"].checkpoint_sha256 = "b" * 64
    external_id = store.put_json(external.model_dump(mode="json"), "test-evidence", "manifest.json")
    with pytest.raises(ConfigurationError, match="execution identities differ"):
        compatibility_report(store, [external_id])
    fixture_report = json.loads(store.resolve(manifest.outputs["conversion_report.json"].artifact_id).read_text())
    fixture_report["executions"] = {name: item.model_dump(mode="json") for name, item in external.executions.items()}
    fixture_report_id = store.put_json(fixture_report, "test-evidence", "conversion_report.json")
    external.outputs["conversion_report.json"] = artifact_digest(store, fixture_report_id)
    external_id = store.put_json(external.model_dump(mode="json"), "test-evidence", "manifest.json")
    evidence = compatibility_report(store, [external_id, external_id])
    assert len(evidence.conversions) == 1
    assert evidence.conversions[0].output_duration_seconds == fixture_audio().duration_seconds
    output = store.resolve(manifest.outputs["converted_vocal_raw.wav"].artifact_id)
    original_audio = output.read_bytes()
    output.write_bytes(b"corrupted")
    with pytest.raises(ConfigurationError, match="checksum"):
        compatibility_report(store, [external_id])
    output.write_bytes(original_audio)
    client = TestClient(create_app(config, factory))
    assert client.get("/compatibility-report").json()["conversions"] == []
    assert client.get("/compatibility-report", params={"manifest_artifact_id": identifier}).status_code == 400
    assert client.get("/compatibility-report", params={"manifest_artifact_id": external_id}).json()["conversions"][0]["run_id"] == manifest.run_id
    response = client.post("/reproduce-jobs", json={"manifest_artifact_id": identifier})
    assert response.status_code == 202
    worker = Worker(JobStore(config.database_path), handlers)
    assert worker.run_once()
    reproduced = client.get("/jobs/" + response.json()["id"]).json()
    assert reproduced["state"] == "SUCCEEDED"
    assert reproduced["result"]["reproduced_from_run_id"] == result["conversion_id"]
    assert reproduced["result"]["conversion_id"] != result["conversion_id"]
    assert client.post("/reproduce-jobs", json={"manifest_artifact_id": result["artifacts"]["evaluation_report.json"]}).status_code == 400
    original = manifest.executions["voice_converter"]
    changed = original.model_copy(update={"checkpoint_sha256": "b" * 64})
    with pytest.raises(ConfigurationError, match="identity differs"):
        verify_execution_identity(manifest, {"voice_converter": changed})
    config.providers.seed_vc.diffusion_steps += 1
    with pytest.raises(ConfigurationError, match="configuration differs"):
        reproduce_request(manifest, store, config)
    config.providers.seed_vc.diffusion_steps -= 1
    store.resolve(manifest.inputs["source"].artifact_id).write_bytes(b"corrupted")
    with pytest.raises(ConfigurationError, match="checksum"):
        reproduce_request(manifest, store, config)
