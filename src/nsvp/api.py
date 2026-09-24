import importlib.util
import re
import uuid
from contextlib import closing
from pathlib import Path
from typing import Annotated, Any

from .api_contracts import (
    ConversionJobRequest,
    DatasetJobRequest,
    EvaluationJobRequest,
    ReproduceJobRequest,
    SeparationJobRequest,
    TrainingJobRequest,
    VocalPreparationRequest,
)
from .benchmarking import BenchmarkSpec, public_config_snapshot
from .compatibility import compatibility_report
from .components import ComponentFactory
from .config import AppConfig, load_config
from .contracts import JobRecord
from .device import DeviceManager
from .errors import ArtifactNotFoundError, ConfigurationError, JobStateError, NSVPError
from .jobs import JobStore
from .listening import ListeningRating, ListeningRequest, ListeningStore
from .preparation import resolve_conversion_request
from .registry import ModelRegistry
from .storage import LocalArtifactStore

MAX_UPLOAD_BYTES = 1024 * 1024 * 1024
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus"}


def create_app(config: AppConfig | None = None, factory: ComponentFactory | None = None) -> Any:
    if importlib.util.find_spec("fastapi") is None:
        raise RuntimeError("FastAPI is not installed; install the 'api' dependency group")
    from fastapi import FastAPI, File, HTTPException, Query, UploadFile
    from fastapi.responses import FileResponse, PlainTextResponse

    active = config or load_config()
    store = LocalArtifactStore(active.artifact_root)
    jobs = JobStore(active.database_path)
    listening = ListeningStore(jobs, store)
    registry = ModelRegistry(active.artifact_root / "models", store)
    components = factory or ComponentFactory(active)
    app = FastAPI(title="Neural Singing Voice Platform", version="0.2.0")

    def public_job(job: JobRecord, include_result: bool = False) -> dict[str, Any]:
        response = job.model_dump(mode="json")
        response["payload"] = public_config_snapshot(job.payload)
        if job.error:
            response["error"] = {
                "code": job.error.get("code", "job_failed"),
                "message": "The job failed. Check the selected providers and worker logs.",
            }
        if include_result:
            response["result"] = public_config_snapshot(jobs.result(job.id) or {})
        return response

    def require_artifact(artifact_id: str) -> Path:
        try:
            return store.resolve(artifact_id)
        except (ArtifactNotFoundError, ValueError):
            raise HTTPException(status_code=404, detail="Audio or report artifact not found") from None

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": "0.2.0"}

    @app.get("/capabilities")
    def capabilities(probe: bool = False) -> dict[str, Any]:
        manager = DeviceManager()
        core = (
            manager.report_without_torch().model_dump(mode="json")
            if importlib.util.find_spec("torch") is None
            else manager.detect(active.device.backend).capabilities.model_dump(mode="json")
        )
        return {**core, "components": [item.model_dump(mode="json") for item in components.capabilities(probe)]}

    @app.get("/components")
    def available_components(probe: bool = False) -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in components.capabilities(probe)]

    @app.get("/compatibility-report")
    def conversion_compatibility(manifest_artifact_id: Annotated[list[str] | None, Query(max_length=20)] = None,
                               failed_job_id: Annotated[list[str] | None, Query(max_length=20)] = None) -> dict[str, Any]:
        try:
            return compatibility_report(store, manifest_artifact_id or [], [jobs.get(identifier) for identifier in failed_job_id or []]).model_dump(mode="json")
        except (NSVPError, ValueError, KeyError):
            raise HTTPException(status_code=400, detail="Valid conversion evidence or failed job IDs are required") from None

    @app.get("/providers")
    def configured_providers() -> dict[str, Any]:
        return {
            "defaults": active.components.model_dump(mode="json"),
            "profiles": {name: {"provider": profile.provider} for name, profile in active.profiles.items()},
            "vocal_processing_profiles": sorted(active.vocal_processing_profiles),
            "providers": [item.model_dump(mode="json") for item in components.capabilities()],
        }

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
        try:
            with temporary.open("wb") as stream:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail="upload exceeds 1 GiB limit")
                    stream.write(chunk)
            artifact_id = store.put_file(temporary, "uploads", temporary.name)
        finally:
            temporary.unlink(missing_ok=True)
        return {"artifact_id": artifact_id}

    @app.post("/datasets/analyze", status_code=202)
    def analyze_dataset(request: DatasetJobRequest) -> dict[str, Any]:
        for artifact_id in request.source_artifact_ids:
            require_artifact(artifact_id)
        return public_job(jobs.enqueue("dataset_analyze", request.model_dump(mode="json")))

    app.add_api_route("/datasets", analyze_dataset, methods=["POST"], status_code=202)

    @app.post("/conversion-jobs", status_code=202)
    def conversion_job(request: ConversionJobRequest) -> dict[str, Any]:
        payload = request.model_dump(mode="json")
        try:
            resolved = resolve_conversion_request(payload, store)
            components.validate_conversion_selection(resolved)
        except (ArtifactNotFoundError, ValueError, ConfigurationError):
            raise HTTPException(status_code=400, detail="Invalid source, reference, model, or processing selection") from None
        return public_job(jobs.enqueue("conversion", payload))

    app.add_api_route("/convert", conversion_job, methods=["POST"], status_code=202)

    @app.post("/separation-jobs", status_code=202)
    def separation_job(request: SeparationJobRequest) -> dict[str, Any]:
        require_artifact(request.song_artifact_id)
        payload = {**request.model_dump(mode="json"), "job_namespace": uuid.uuid4().hex}
        return public_job(jobs.enqueue("separation", payload))

    app.add_api_route("/separate", separation_job, methods=["POST"], status_code=202)

    @app.post("/training-jobs", status_code=202)
    def training_job(request: TrainingJobRequest) -> dict[str, Any]:
        require_artifact(request.manifest_artifact_id)
        require_artifact(request.training_config_artifact_id)
        if request.resume_artifact_id:
            require_artifact(request.resume_artifact_id)
        if request.provider != "seed_vc":
            raise HTTPException(status_code=400, detail="This provider does not support training in v0.2")
        return public_job(jobs.enqueue("training", request.model_dump(mode="json")))

    app.add_api_route("/models/train", training_job, methods=["POST"], status_code=202)

    @app.post("/vocal-preparation-jobs", status_code=202)
    def prepare_vocal_job(request: VocalPreparationRequest) -> dict[str, Any]:
        require_artifact(request.source_artifact_id)
        try:
            components.build_vocal_preprocessors(request.vocal_processing_profile)
            if request.multi_singer_separator:
                components.build_multi_singer_separator(request.multi_singer_separator)
        except ConfigurationError:
            raise HTTPException(status_code=400, detail="Vocal processing provider is not available") from None
        return public_job(jobs.enqueue("vocal_preparation", request.model_dump(mode="json")))

    @app.post("/evaluation-jobs", status_code=202)
    def evaluation_job(request: EvaluationJobRequest) -> dict[str, Any]:
        for artifact_id in (request.source_artifact_id, request.output_artifact_id, request.reference_artifact_id,
            request.ground_truth_stem_artifact_id, request.separated_stem_artifact_id):
            if artifact_id:
                require_artifact(artifact_id)
        return public_job(jobs.enqueue("evaluation", request.model_dump(mode="json")))

    @app.post("/reproduce-jobs", status_code=202)
    def reproduce_job(request: ReproduceJobRequest) -> dict[str, Any]:
        from .provenance import RunManifest

        path = require_artifact(request.manifest_artifact_id)
        try:
            RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError):
            raise HTTPException(status_code=400, detail="A v0.3 run manifest is required") from None
        return public_job(jobs.enqueue("reproduce", request.model_dump(mode="json")))

    @app.post("/benchmark-runs", status_code=202)
    def benchmark_run(request: BenchmarkSpec) -> dict[str, Any]:
        for case in request.cases:
            for artifact_id in (case.source_artifact_id, case.reference_artifact_id, case.ground_truth_stem_artifact_id, case.instrumental_artifact_id):
                if artifact_id:
                    require_artifact(artifact_id)
        return public_job(jobs.enqueue("benchmark", request.model_dump(mode="json")))

    def list_kind(kind: str, limit: int, offset: int) -> list[dict[str, Any]]:
        try:
            return [public_job(job, include_result=True) for job in jobs.list(kind, limit, offset)]
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid pagination") from None

    @app.get("/benchmark-runs")
    def benchmark_runs(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        return list_kind("benchmark", limit, offset)

    @app.get("/evaluations")
    def evaluations(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        return list_kind("evaluation", limit, offset)

    @app.get("/conversion-runs")
    def conversion_runs(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        return list_kind("conversion", limit, offset)

    def get_kind(job_id: str, kind: str) -> dict[str, Any]:
        try:
            job = jobs.get(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found") from None
        if job.kind != kind:
            raise HTTPException(status_code=404, detail="Run not found")
        return public_job(job, include_result=True)

    @app.get("/benchmark-runs/{job_id}")
    def benchmark_detail(job_id: str) -> dict[str, Any]:
        return get_kind(job_id, "benchmark")

    @app.get("/evaluations/{job_id}")
    def evaluation_detail(job_id: str) -> dict[str, Any]:
        return get_kind(job_id, "evaluation")

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        try:
            job = jobs.get(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="job not found")
        return public_job(job, include_result=True)

    @app.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, Any]:
        try:
            return public_job(jobs.cancel(job_id))
        except KeyError:
            raise HTTPException(status_code=404, detail="job not found")
        except JobStateError:
            raise HTTPException(status_code=409, detail="Completed jobs cannot be cancelled") from None

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
            return {"job": public_job(jobs.get(job_id)), "result": public_config_snapshot(jobs.result(job_id) or {})}
        except KeyError:
            raise HTTPException(status_code=404, detail="job not found")

    @app.get("/artifacts/{artifact_id:path}")
    def artifact(artifact_id: str) -> FileResponse:
        if re.fullmatch(r"[^/\\]+/[a-f0-9]{16}/[^/\\]+", artifact_id) is None:
            raise HTTPException(status_code=404, detail="artifact not found")
        try:
            path = store.resolve(artifact_id)
        except (ArtifactNotFoundError, ValueError):
            raise HTTPException(status_code=404, detail="artifact not found")
        return FileResponse(path)

    @app.post("/listening-sessions", status_code=201)
    def create_listening_session(request: ListeningRequest) -> dict[str, Any]:
        try:
            return listening.create(request)
        except (ValueError, ArtifactNotFoundError):
            raise HTTPException(status_code=400, detail="Choose two valid conversion manifests with matching source, reference and transposition") from None

    @app.get("/listening-sessions")
    def listening_history(listener_id: str, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        try:
            return listening.history(listener_id, limit, offset)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid pagination") from None

    @app.get("/listening-summary")
    def listening_summary(listener_id: str) -> dict[str, Any]:
        return listening.summary(listener_id)

    @app.get("/listening-sessions/{session_id}")
    def listening_session(session_id: str, listener_id: str) -> dict[str, Any]:
        try:
            return listening.get(session_id, listener_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Listening session not found") from None

    @app.post("/listening-sessions/{session_id}/ratings")
    def listening_rating(session_id: str, listener_id: str, request: ListeningRating) -> dict[str, Any]:
        try:
            return listening.rate(session_id, listener_id, request)
        except KeyError:
            raise HTTPException(status_code=404, detail="Listening session not found") from None
        except ValueError:
            raise HTTPException(status_code=409, detail="This session already has a rating") from None

    @app.get("/listening-sessions/{session_id}/audio/{label}")
    def listening_audio(session_id: str, label: str, listener_id: str) -> FileResponse:
        try:
            path = listening.audio(session_id, listener_id, label)
        except (KeyError, ArtifactNotFoundError):
            raise HTTPException(status_code=404, detail="Listening audio not found") from None
        return FileResponse(path, media_type="audio/wav", filename=f"{label}.wav",
            content_disposition_type="inline", headers={"Cache-Control": "no-store"})

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        with closing(jobs.connect()) as connection:
            rows = connection.execute("SELECT state,COUNT(*) count FROM jobs GROUP BY state").fetchall()
        lines = ["# HELP nsvp_jobs_total Jobs by current state", "# TYPE nsvp_jobs_total gauge"]
        lines.extend(f'nsvp_jobs_total{{state="{row["state"]}"}} {row["count"]}' for row in rows)
        return "\n".join(lines) + "\n"

    return app


app = create_app()
