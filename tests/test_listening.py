from pathlib import Path

from fastapi.testclient import TestClient
from test_dataset_pipeline import fixture_audio

from nsvp.api import create_app
from nsvp.audio.io import save_audio
from nsvp.components import ComponentFactory
from nsvp.config import AppConfig
from nsvp.contracts import ConversionRequest
from nsvp.provenance import RunManifest
from nsvp.runtime import build_handlers
from nsvp.storage import LocalArtifactStore
from nsvp.testing_backends import IdentityVoiceConverter


def test_blind_session_persistence_rating_and_media(tmp_path: Path) -> None:
    config = AppConfig(artifact_root=tmp_path / "artifacts", database_path=tmp_path / "artifacts/jobs.db")
    factory = ComponentFactory(config, converters={"private-provider": IdentityVoiceConverter})
    source = tmp_path / "secret-recording.wav"
    save_audio(source, fixture_audio())
    handler = build_handlers(config, factory)["conversion"]
    request = ConversionRequest(song_path=source, target_reference_path=source, output_name="private-output",
        input_kind="vocal", voice_converter="private-provider")
    results = [handler(request.model_dump(mode="json"), lambda *_: None) for _ in range(2)]
    first, second = [result["artifacts"]["run_manifest.json"] for result in results]
    listener = "a" * 32
    client = TestClient(create_app(config, factory))
    payload = {"listener_id": listener, "first_manifest_id": first, "second_manifest_id": second}
    response = client.post("/listening-sessions", json=payload)
    assert response.status_code == 201
    session = response.json()
    assert session["status"] == "pending" and "mapping" not in session and "rating" not in session
    for secret in ("private-provider", "secret-recording", first, second, "checkpoint", "executions"):
        assert secret not in response.text
    for label in ("A", "B", "source", "reference"):
        media = client.get(session["media"][label])
        assert media.status_code == 200 and media.headers["content-type"] == "audio/wav"
        assert "private" not in str(media.headers) and "secret" not in str(media.headers)
        assert media.headers["cache-control"] == "no-store"
    url = f"/listening-sessions/{session['id']}"
    assert client.get(url, params={"listener_id": "b" * 32}).status_code == 404
    assert client.get(url + "/audio/A", params={"listener_id": "b" * 32}).status_code == 404
    assert client.get("/artifacts/jobs.db").status_code == 404
    # A fresh application reads the same pending session and keeps its assignment.
    client = TestClient(create_app(config, factory))
    assert client.get(url, params={"listener_id": listener}).json() == session
    scores = {"naturalness": 4, "singer_similarity": 3, "content_preservation": 5, "artifact_severity": 2}
    rating = {"a": scores, "b": scores, "preference": "tie"}
    rate_url = url + "/ratings?listener_id=" + listener
    assert client.post(rate_url, json={"preference": "A"}).status_code == 422
    invalid = {**rating, "a": {**scores, "naturalness": True}}
    assert client.post(rate_url, json=invalid).status_code == 422
    rated = client.post(rate_url, json=rating)
    assert rated.status_code == 200 and rated.json()["status"] == "rated"
    assert {entry["run_id"] for entry in rated.json()["mapping"].values()} == {result["conversion_id"] for result in results}
    assert client.post(rate_url, json=rating).json() == rated.json()
    assert client.post(rate_url, json={**rating, "preference": "A"}).status_code == 409
    history = client.get("/listening-sessions", params={"listener_id": listener}).json()
    assert len(history) == 1 and history[0] == rated.json()
    summary = client.get("/listening-summary", params={"listener_id": listener}).json()
    assert summary["rated_session_count"] == 1 and len(summary["runs"]) == 2
    assert all(run["rating_count"] == 1 and run["means"] == scores for run in summary["runs"])
    assert len(summary["groups"]) == 1
    assert summary["groups"][0]["provider"] == "private-provider"
    assert summary["groups"][0]["rating_count"] == 2
    assert summary["groups"][0]["preference_percent"] == 0
    store = LocalArtifactStore(config.artifact_root)
    benchmark_ids = []
    for result, configuration_id in zip(results, ("steps-20", "steps-30")):
        manifest = RunManifest.model_validate_json(store.resolve(result["artifacts"]["run_manifest.json"]).read_text())
        manifest.benchmark_run_id = "paired-benchmark"
        manifest.benchmark_case_id = "mango"
        manifest.benchmark_configuration_id = configuration_id
        benchmark_ids.append(store.put_json(manifest.model_dump(mode="json"), "test-listening", configuration_id + ".json"))
    other_listener = "b" * 32
    paired = client.post("/listening-sessions", json={"listener_id": other_listener,
        "first_manifest_id": benchmark_ids[0], "second_manifest_id": benchmark_ids[1]}).json()
    assert "mapping" not in paired
    rated_pair = client.post(f"/listening-sessions/{paired['id']}/ratings?listener_id={other_listener}",
        json={"a": scores, "b": {**scores, "naturalness": 2}, "preference": "A"}).json()
    grouped = client.get("/listening-summary", params={"listener_id": other_listener}).json()["groups"]
    assert {group["configuration_id"] for group in grouped} == {"steps-20", "steps-30"}
    assert all(group["provider"] == "private-provider" and group["case_id"] == "mango"
        and group["benchmark_run_id"] == "paired-benchmark" and group["rating_count"] == 1 for group in grouped)
    preferred = rated_pair["mapping"]["A"]["benchmark_configuration_id"]
    assert next(group for group in grouped if group["configuration_id"] == preferred)["preference_percent"] == 100
    assert next(group for group in grouped if group["configuration_id"] != preferred)["preference_percent"] == 0
    assert next(group for group in grouped if group["configuration_id"] != preferred)["means"]["naturalness"] == 2
    assert client.post("/listening-sessions", json={**payload, "second_manifest_id": first}).status_code == 400
    store.resolve(results[0]["artifacts"]["converted_vocal_raw.wav"]).write_bytes(b"corrupt")
    assert client.post("/listening-sessions", json=payload).status_code == 400
