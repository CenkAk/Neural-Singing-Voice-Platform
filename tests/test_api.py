from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from nsvp.api import create_app
from nsvp.audio.io import save_audio
from nsvp.config import AppConfig, DeviceConfig
from nsvp.contracts import AudioBuffer, BackendName


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
