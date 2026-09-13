from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from nsvp.api import create_app
from nsvp.audio.io import save_audio
from nsvp.components import ComponentFactory
from nsvp.config import AppConfig, DeviceConfig
from nsvp.contracts import AudioBuffer, BackendName
from nsvp.jobs import JobStore, Worker
from nsvp.runtime import build_handlers
from nsvp.testing_backends import DeterministicMultiSingerSeparator, IdentityVoiceConverter


def test_health_capabilities_upload_and_job_creation() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        config = AppConfig(
            artifact_root=root / "artifacts",
            database_path=root / "jobs.sqlite3",
            device=DeviceConfig(backend=BackendName.CPU),
        )
        client = TestClient(create_app(config))
        assert client.get("/health").json()["status"] == "ok"
        capabilities = client.get("/capabilities")
        assert capabilities.status_code == 200
        assert capabilities.json()["backend"] == "cpu"

        fixture = root / "fixture.wav"
        save_audio(fixture, AudioBuffer(waveform=np.zeros((1, 800), dtype=np.float32), sample_rate=8_000))
        with fixture.open("rb") as stream:
            uploaded = client.post("/uploads", files={"file": ("fixture.wav", stream, "audio/wav")})
        assert uploaded.status_code == 200
        artifact_id = uploaded.json()["artifact_id"]
        queued = client.post(
            "/conversion-jobs",
            json={"song_artifact_id": artifact_id, "reference_artifact_id": artifact_id, "output_name": "api-fixture"},
        )
        assert queued.status_code == 202
        job_id = queued.json()["id"]
        assert client.get(f"/jobs/{job_id}").json()["state"] == "QUEUED"
        assert client.post(f"/jobs/{job_id}/cancel").json()["state"] == "CANCELLED"


def test_v02_artifact_only_preparation_conversion_and_benchmark(tmp_path: Path) -> None:
    config = AppConfig(artifact_root=tmp_path / "artifacts", database_path=tmp_path / "jobs.db")
    factory = ComponentFactory(config, converters={"identity": IdentityVoiceConverter},
        multi_singer_separators={"fake-multi": DeterministicMultiSingerSeparator})
    client = TestClient(create_app(config, factory))
    worker = Worker(JobStore(config.database_path), build_handlers(config, factory))
    fixture = tmp_path / "source.wav"
    t = np.arange(8000, dtype=np.float32) / 8000
    save_audio(fixture, AudioBuffer(waveform=(0.2 * np.sin(2 * np.pi * 220 * t))[None, :], sample_rate=8000))
    with fixture.open("rb") as stream:
        artifact = client.post("/uploads", files={"file": ("source.wav", stream, "audio/wav")}).json()["artifact_id"]
    assert client.get("/providers").status_code == 200
    assert client.post("/datasets/analyze", json={"source_root": str(tmp_path), "singer_name": "test"}).status_code == 422
    assert client.post("/conversion-jobs", json={"song_artifact_id": "../source.wav", "reference_artifact_id": artifact, "output_name": "test"}).status_code == 400
    response = client.post("/vocal-preparation-jobs", json={"source_artifact_id": artifact, "multi_singer_separator": "fake-multi", "input_condition": "mixed_vocal"})
    assert response.status_code == 202
    assert worker.run_once()
    preparation = client.get("/jobs/" + response.json()["id"]).json()
    assert preparation["state"] == "SUCCEEDED"
    prepared = preparation["result"]
    assert len(prepared["sources"]) == 2
    payload = {"preparation_manifest_artifact_id": prepared["artifacts"]["vocal_preparation.json"],
        "reference_artifact_id": artifact, "output_name": "selected", "voice_converter": "identity", "keep_intermediates": False}
    assert client.post("/conversion-jobs", json=payload).status_code == 422
    assert client.post("/conversion-jobs", json={**payload, "selected_source_id": "not-a-source"}).status_code == 400
    queued = client.post("/conversion-jobs", json={**payload, "selected_source_id": "source-b"})
    assert queued.status_code == 202
    assert worker.run_once()
    conversion = client.get("/jobs/" + queued.json()["id"]).json()
    assert conversion["state"] == "SUCCEEDED"
    assert str(tmp_path) not in str(conversion)
    artifacts = conversion["result"]["artifacts"]
    evaluation = client.get("/artifacts/" + artifacts["evaluation_report.json"]).json()
    assert evaluation["input_condition"] == "mixed_vocal"
    assert evaluation["families"]["timbre"]["singer_similarity"]["value"] is None
    assert client.get("/artifacts/" + artifacts["converted_vocal_raw.wav"]).headers["content-type"].startswith("audio/")
    queued = client.post("/benchmark-runs", json={"name": "API synthetic comparison",
        "cases": [{"case_id": "lead", "source_artifact_id": artifact, "reference_artifact_id": artifact}],
        "configurations": [{"configuration_id": "identity", "voice_converter": "identity"}, {"configuration_id": "missing", "voice_converter": "seed_vc"}]})
    assert queued.status_code == 202
    assert worker.run_once()
    benchmark = client.get("/benchmark-runs/" + queued.json()["id"]).json()
    assert benchmark["state"] == "SUCCEEDED"
    assert [row["status"] for row in benchmark["result"]["results"]] == ["succeeded", "not_tested"]
    assert len(client.get("/benchmark-runs").json()) == 1
    assert client.get("/evaluations/" + queued.json()["id"]).status_code == 404
