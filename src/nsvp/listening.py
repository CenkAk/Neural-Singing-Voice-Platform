from __future__ import annotations

import json
import secrets
import tempfile
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .audio.io import load_audio, save_audio
from .jobs import JobStore
from .provenance import RunManifest
from .storage import LocalArtifactStore, sha256_file


class ListeningRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    listener_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    first_manifest_id: str
    second_manifest_id: str


class ListeningScores(BaseModel):
    model_config = ConfigDict(extra="forbid")
    naturalness: int = Field(ge=1, le=5, strict=True)
    singer_similarity: int = Field(ge=1, le=5, strict=True)
    content_preservation: int = Field(ge=1, le=5, strict=True)
    artifact_severity: int = Field(ge=1, le=5, strict=True)


class ListeningRating(BaseModel):
    model_config = ConfigDict(extra="forbid")
    a: ListeningScores
    b: ListeningScores
    preference: Literal["A", "B", "tie"]


class ListeningStore:
    def __init__(self, jobs: JobStore, artifacts: LocalArtifactStore) -> None:
        self.jobs, self.artifacts = jobs, artifacts
        with closing(jobs.connect()) as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS listening_sessions (
                id TEXT PRIMARY KEY, listener_id TEXT NOT NULL, created_at TEXT NOT NULL,
                mapping TEXT NOT NULL, media TEXT NOT NULL, rating TEXT, rated_at TEXT
            )""")
            connection.execute("CREATE INDEX IF NOT EXISTS listening_listener ON listening_sessions(listener_id,created_at)")

    def create(self, request: ListeningRequest) -> dict[str, Any]:
        manifests = [RunManifest.model_validate_json(self.artifacts.resolve(identifier).read_text(encoding="utf-8"))
            for identifier in (request.first_manifest_id, request.second_manifest_id)]
        first, second = manifests
        manifest_ids = {first.run_id: request.first_manifest_id, second.run_id: request.second_manifest_id}
        if first.run_id == second.run_id:
            raise ValueError("Choose two distinct conversion runs")
        for name in ("source", "reference"):
            if name not in first.inputs or name not in second.inputs or first.inputs[name].sha256 != second.inputs[name].sha256:
                raise ValueError("Listening comparisons require the same source and reference")
        if first.request.get("transpose_semitones", 0) != second.request.get("transpose_semitones", 0):
            raise ValueError("Listening comparisons require the same transposition")
        if secrets.randbelow(2):
            manifests.reverse()
        session_id = uuid.uuid4().hex
        mapping, media = {}, {}
        # Re-encoding removes filenames and embedded tags from the listening audio.
        with tempfile.TemporaryDirectory(prefix="nsvp-listening-") as temporary:
            for label, manifest in zip(("A", "B"), manifests):
                mapping[label] = {"run_id": manifest.run_id,
                    "manifest_artifact_id": manifest_ids[manifest.run_id],
                    "executions": {name: execution.model_dump(mode="json") for name, execution in manifest.executions.items()}}
                if "converted_vocal_raw.wav" not in manifest.outputs:
                    raise ValueError("Conversion manifest has no raw vocal output")
                digest = manifest.outputs["converted_vocal_raw.wav"]
                path = self.artifacts.resolve(digest.artifact_id)
                if sha256_file(path) != digest.sha256:
                    raise ValueError("Listening audio checksum mismatch")
                audio = load_audio(path)
                neutral = Path(temporary) / f"{label}.wav"
                save_audio(neutral, audio, subtype="FLOAT")
                media[label] = self.artifacts.put_file(neutral, f"listening-{session_id}", neutral.name)
            for label, name in (("source", "source"), ("reference", "reference")):
                digest = first.inputs[name]
                path = self.artifacts.resolve(digest.artifact_id)
                if sha256_file(path) != digest.sha256:
                    raise ValueError("Listening context checksum mismatch")
                neutral = Path(temporary) / f"{label}.wav"
                save_audio(neutral, load_audio(path), subtype="FLOAT")
                media[label] = self.artifacts.put_file(neutral, f"listening-{session_id}", neutral.name)
        with closing(self.jobs.connect()) as connection:
            connection.execute("INSERT INTO listening_sessions(id,listener_id,created_at,mapping,media) VALUES(?,?,?,?,?)",
                (session_id, request.listener_id, datetime.now(timezone.utc).isoformat(), json.dumps(mapping), json.dumps(media)))
        return self.get(session_id, request.listener_id)

    def get(self, session_id: str, listener_id: str) -> dict[str, Any]:
        with closing(self.jobs.connect()) as connection:
            row = connection.execute("SELECT * FROM listening_sessions WHERE id=? AND listener_id=?",
                (session_id, listener_id)).fetchone()
        if row is None:
            raise KeyError(session_id)
        result: dict[str, Any] = {"id": row["id"], "created_at": row["created_at"],
            "status": "rated" if row["rating"] else "pending",
            "media": {label: f"/listening-sessions/{session_id}/audio/{label}?listener_id={listener_id}"
                for label in ("A", "B", "source", "reference")},
            "scale": "1 to 5. Higher is better except artifact severity, where higher means worse."}
        if row["rating"]:
            result.update({"rating": json.loads(row["rating"]), "mapping": json.loads(row["mapping"]), "rated_at": row["rated_at"]})
        return result

    def rate(self, session_id: str, listener_id: str, rating: ListeningRating) -> dict[str, Any]:
        raw = rating.model_dump_json()
        with closing(self.jobs.connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT rating FROM listening_sessions WHERE id=? AND listener_id=?",
                (session_id, listener_id)).fetchone()
            if row is None:
                raise KeyError(session_id)
            if row["rating"] is not None and row["rating"] != raw:
                raise ValueError("This session already has an immutable rating")
            connection.execute("UPDATE listening_sessions SET rating=?,rated_at=? WHERE id=? AND rating IS NULL",
                (raw, datetime.now(timezone.utc).isoformat(), session_id))
            connection.execute("COMMIT")
        return self.get(session_id, listener_id)

    def audio(self, session_id: str, listener_id: str, label: str) -> Path:
        if label not in {"A", "B", "source", "reference"}:
            raise KeyError(label)
        with closing(self.jobs.connect()) as connection:
            row = connection.execute("SELECT media FROM listening_sessions WHERE id=? AND listener_id=?",
                (session_id, listener_id)).fetchone()
        if row is None:
            raise KeyError(session_id)
        return self.artifacts.resolve(str(json.loads(row["media"])[label]))

    def history(self, listener_id: str, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("Invalid pagination")
        with closing(self.jobs.connect()) as connection:
            rows = connection.execute("SELECT id FROM listening_sessions WHERE listener_id=? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (listener_id, limit, offset)).fetchall()
        return [self.get(row["id"], listener_id) for row in rows]

    def summary(self, listener_id: str) -> dict[str, Any]:
        with closing(self.jobs.connect()) as connection:
            rows = connection.execute("SELECT mapping,rating FROM listening_sessions WHERE listener_id=? AND rating IS NOT NULL",
                (listener_id,)).fetchall()
        runs: dict[str, dict[str, Any]] = {}
        for row in rows:
            mapping, rating = json.loads(row["mapping"]), ListeningRating.model_validate_json(row["rating"])
            for label, scores in (("A", rating.a), ("B", rating.b)):
                run_id = str(mapping[label]["run_id"])
                result = runs.setdefault(run_id, {"run_id": run_id, "rating_count": 0,
                    "preferred_count": 0, "tie_count": 0, "means": dict.fromkeys(ListeningScores.model_fields, 0.0)})
                result["rating_count"] += 1
                result["preferred_count"] += rating.preference == label
                result["tie_count"] += rating.preference == "tie"
                for name, score in scores.model_dump().items():
                    result["means"][name] += score
        for result in runs.values():
            result["means"] = {name: total / result["rating_count"] for name, total in result["means"].items()}
        return {"rated_session_count": len(rows), "runs": list(runs.values()),
            "limitations": ["Descriptive ratings from this local listener only; no statistical significance claim.",
                "Artifact severity uses the opposite direction: higher means worse."]}
