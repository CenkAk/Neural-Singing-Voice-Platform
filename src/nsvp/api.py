import importlib.util
import uuid
from contextlib import closing
from pathlib import Path
from typing import Any

from .config import AppConfig, load_config
from .device import DeviceManager
from .errors import ArtifactNotFoundError
from .jobs import JobStore
from .registry import ModelRegistry
from .storage import LocalArtifactStore

MAX_UPLOAD_BYTES = 1024 * 1024 * 1024
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}


def create_app(config: AppConfig | None = None) -> Any:
    if importlib.util.find_spec("fastapi") is None:
        raise RuntimeError("FastAPI is not installed; install the 'api' dependency group")
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import FileResponse, PlainTextResponse
    from pydantic import BaseModel

    active = config or load_config()
    store = LocalArtifactStore(active.artifact_root)
    jobs = JobStore(active.database_path)
    registry = ModelRegistry(active.artifact_root / "models", store)
    app = FastAPI(title="Neural Singing Voice Platform", version="0.1.0")

    class DatasetJobRequest(BaseModel):
        source_root: Path
        singer_name: str

    class ConversionJobRequest(BaseModel):
        song_artifact_id: str
        reference_artifact_id: str
        output_name: str
        transpose_semitones: int = 0

    class SeparationJobRequest(BaseModel):
        song_artifact_id: str

    class TrainingJobRequest(BaseModel):
        manifest_artifact_id: str
        training_config_artifact_id: str
        run_name: str
        resume_artifact_id: str | None = None

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": "0.1.0"}

    @app.get("/capabilities")
    def capabilities() -> dict[str, Any]:
        manager = DeviceManager()
        return (
            manager.report_without_torch().model_dump(mode="json")
            if importlib.util.find_spec("torch") is None
            else manager.detect(active.device.backend).capabilities.model_dump(mode="json")
        )

    @app.get("/models")
    def models() -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in registry.list()]

    @app.get("/models/{name}/{version}")
    def model(name: str, version: str) -> dict[str, Any]:
        for item in registry.list():
            if item.model_name == name and item.version == version:
                return item.model_dump(mode="json")
        raise HTTPException(status_code=404, detail="model not found")

    @app.get("/models/{name}")
    def latest_model(name: str) -> dict[str, Any]:
        matches = [item for item in registry.list() if item.model_name == name]
        if not matches:
            raise HTTPException(status_code=404, detail="model not found")
        return matches[-1].model_dump(mode="json")

    @app.post("/uploads")
    async def upload_audio(file: UploadFile = File(...)) -> dict[str, str]:  # noqa: B008
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_AUDIO_EXTENSIONS:
            raise HTTPException(status_code=415, detail="unsupported audio format")
        upload_dir = active.artifact_root / "uploads-staging"
        upload_dir.mkdir(parents=True, exist_ok=True)
        temporary = upload_dir / f"{uuid.uuid4().hex}{suffix}"
        size = 0
        with temporary.open("wb") as stream:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    temporary.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="upload exceeds 1 GiB limit")
                stream.write(chunk)
        artifact_id = store.put_file(temporary, "uploads", temporary.name)
        temporary.unlink(missing_ok=True)
        return {"artifact_id": artifact_id}

    @app.post("/datasets/analyze", status_code=202)
    def analyze_dataset(request: DatasetJobRequest) -> dict[str, Any]:
        if not request.source_root.resolve().is_dir():
            raise HTTPException(status_code=400, detail="dataset directory does not exist")
        return jobs.enqueue("dataset_analyze", request.model_dump(mode="json")).model_dump(mode="json")

    app.add_api_route("/datasets", analyze_dataset, methods=["POST"], status_code=202)

    @app.post("/conversion-jobs", status_code=202)
    def conversion_job(request: ConversionJobRequest) -> dict[str, Any]:
        song = store.resolve(request.song_artifact_id)
        reference = store.resolve(request.reference_artifact_id)
        payload = {
            "song_path": str(song), "target_reference_path": str(reference), "output_name": request.output_name,
            "transpose_semitones": request.transpose_semitones, "model_name": "seed-vc-v1", "model_version": "configured",
            "backend": active.device.backend.value, "keep_intermediates": True,
        }
        return jobs.enqueue("conversion", payload).model_dump(mode="json")

    app.add_api_route("/convert", conversion_job, methods=["POST"], status_code=202)

    @app.post("/separation-jobs", status_code=202)
    def separation_job(request: SeparationJobRequest) -> dict[str, Any]:
        store.resolve(request.song_artifact_id)
        payload = {"song_artifact_id": request.song_artifact_id, "job_namespace": uuid.uuid4().hex}
        return jobs.enqueue("separation", payload).model_dump(mode="json")

    app.add_api_route("/separate", separation_job, methods=["POST"], status_code=202)

    @app.post("/training-jobs", status_code=202)
    def training_job(request: TrainingJobRequest) -> dict[str, Any]:
        store.resolve(request.manifest_artifact_id)
        store.resolve(request.training_config_artifact_id)
        if request.resume_artifact_id:
            store.resolve(request.resume_artifact_id)
        return jobs.enqueue("training", request.model_dump(mode="json")).model_dump(mode="json")

    app.add_api_route("/models/train", training_job, methods=["POST"], status_code=202)

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        try:
            job = jobs.get(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="job not found")
        response = job.model_dump(mode="json")
        response["result"] = jobs.result(job_id)
        return response

    @app.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, Any]:
        try:
            return jobs.cancel(job_id).model_dump(mode="json")
        except KeyError:
            raise HTTPException(status_code=404, detail="job not found")

    @app.get("/jobs/{job_id}/artifacts")
    def job_artifacts(job_id: str) -> dict[str, Any]:
        try:
            result = jobs.result(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="job not found")
        return {"artifacts": (result or {}).get("artifacts", {})}

    @app.get("/jobs/{job_id}/report")
    def job_report(job_id: str) -> dict[str, Any]:
        try:
            return {"job": jobs.get(job_id).model_dump(mode="json"), "result": jobs.result(job_id)}
        except KeyError:
            raise HTTPException(status_code=404, detail="job not found")

    @app.get("/artifacts/{artifact_id:path}")
    def artifact(artifact_id: str) -> FileResponse:
        try:
            path = store.resolve(artifact_id)
        except (ArtifactNotFoundError, ValueError):
            raise HTTPException(status_code=404, detail="artifact not found")
        return FileResponse(path)

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        with closing(jobs.connect()) as connection:
            rows = connection.execute("SELECT state,COUNT(*) count FROM jobs GROUP BY state").fetchall()
        lines = ["# HELP nsvp_jobs_total Jobs by current state", "# TYPE nsvp_jobs_total gauge"]
        lines.extend(f'nsvp_jobs_total{{state="{row["state"]}"}} {row["count"]}' for row in rows)
        return "\n".join(lines) + "\n"

    return app


app = create_app()
