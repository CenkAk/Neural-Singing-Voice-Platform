from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .config import AppConfig
from .contracts import ComponentExecution, ConversionRequest, ConversionResult, EvaluationReport
from .errors import ConfigurationError
from .storage import LocalArtifactStore, sha256_file


class ArtifactDigest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    artifact_id: str
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["0.3"] = "0.3"
    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    inputs: dict[str, ArtifactDigest]
    outputs: dict[str, ArtifactDigest]
    request: dict[str, Any]
    configuration: dict[str, Any] | None = None
    configuration_sha256: str | None = None
    executions: dict[str, ComponentExecution]
    git_commit: str | None = None
    source_fingerprint: str
    system: str
    architecture: str
    python_version: str
    benchmark_run_id: str | None = None
    benchmark_case_id: str | None = None
    benchmark_configuration_id: str | None = None
    dataset_id: str | None = None
    dataset_version: str | None = None
    evaluation: ArtifactDigest | None = None
    warnings: list[str] = Field(default_factory=lambda: [
        "Matching seeds and assets do not guarantee deterministic third-party inference.",
    ])


def source_fingerprint() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def configuration_snapshot(config: AppConfig) -> dict[str, Any]:
    from .benchmarking import public_config_snapshot

    return public_config_snapshot(config.model_dump(mode="json"))


def snapshot_hash(snapshot: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def artifact_digest(store: LocalArtifactStore, artifact_id: str) -> ArtifactDigest:
    return ArtifactDigest(artifact_id=artifact_id, sha256=sha256_file(store.resolve(artifact_id)))


def record_conversion(
    request: ConversionRequest, result: ConversionResult, store: LocalArtifactStore,
    config: AppConfig | None,
) -> None:
    from .benchmarking import git_commit

    inputs: dict[str, ArtifactDigest] = {}
    for name, path in (("source", request.song_path), ("reference", request.target_reference_path),
                       ("instrumental", request.instrumental_path)):
        if path is not None:
            identifier = store.put_file(path, "run-inputs", name + path.suffix.lower())
            inputs[name] = artifact_digest(store, identifier)
    if request.preparation_manifest_artifact_id:
        inputs["preparation"] = artifact_digest(store, request.preparation_manifest_artifact_id)
    snapshot = configuration_snapshot(config) if config is not None else None
    manifest = RunManifest(
        run_id=result.conversion_id, inputs=inputs,
        outputs={name: artifact_digest(store, identifier) for name, identifier in result.artifacts.items()},
        request=request.model_dump(mode="json", exclude={"song_path", "target_reference_path", "instrumental_path"}),
        configuration=snapshot, configuration_sha256=snapshot_hash(snapshot) if snapshot else None,
        executions=result.executions, git_commit=git_commit(), source_fingerprint=source_fingerprint(),
        system=platform.system(), architecture=platform.machine(), python_version=platform.python_version(),
    )
    result.artifacts["run_manifest.json"] = store.put_json(
        manifest.model_dump(mode="json"), f"conversion-{result.conversion_id}", "run_manifest.json",
    )


def finalize_manifest(
    result: ConversionResult, store: LocalArtifactStore, *, benchmark_run_id: str | None = None,
    benchmark_case_id: str | None = None, benchmark_configuration_id: str | None = None,
    dataset_id: str | None = None, dataset_version: str | None = None,
) -> None:
    manifest = RunManifest.model_validate_json(store.resolve(result.artifacts["run_manifest.json"]).read_text(encoding="utf-8"))
    manifest.outputs = {name: artifact_digest(store, identifier) for name, identifier in result.artifacts.items() if name != "run_manifest.json"}
    manifest.evaluation = manifest.outputs.get("evaluation_report.json")
    manifest.benchmark_run_id = benchmark_run_id
    manifest.benchmark_case_id = benchmark_case_id
    manifest.benchmark_configuration_id = benchmark_configuration_id
    manifest.dataset_id, manifest.dataset_version = dataset_id, dataset_version
    result.artifacts["run_manifest.json"] = store.put_json(
        manifest.model_dump(mode="json"), f"conversion-{result.conversion_id}", "run_manifest.json",
    )


def reproduce_request(manifest: RunManifest, store: LocalArtifactStore, config: AppConfig) -> ConversionRequest:
    if manifest.configuration is None or manifest.configuration_sha256 is None:
        raise ConfigurationError("Run does not include a reproducible configuration")
    if snapshot_hash(manifest.configuration) != manifest.configuration_sha256:
        raise ConfigurationError("Run configuration checksum does not match")
    if configuration_snapshot(config) != manifest.configuration:
        raise ConfigurationError("Local configuration differs from the recorded run")
    if source_fingerprint() != manifest.source_fingerprint:
        raise ConfigurationError("NSVP source differs from the recorded run")
    paths: dict[str, Path] = {}
    for name, artifact in manifest.inputs.items():
        path = store.resolve(artifact.artifact_id)
        if sha256_file(path) != artifact.sha256:
            raise ConfigurationError(f"Recorded {name} artifact checksum does not match")
        paths[name] = path
    if "source" not in paths or "reference" not in paths:
        raise ConfigurationError("Run requires recorded source and reference artifacts")
    verify_evaluator_assets(manifest, store, config)
    return ConversionRequest.model_validate({
        **manifest.request, "song_path": paths["source"], "target_reference_path": paths["reference"],
        "instrumental_path": paths.get("instrumental"),
    })


def verify_evaluator_assets(manifest: RunManifest, store: LocalArtifactStore, config: AppConfig) -> None:
    from .adapters.evaluators import model_directory_digest

    if not (config.evaluation.content.enabled or config.evaluation.singer.enabled):
        return
    artifact = manifest.evaluation
    if artifact is None or artifact != manifest.outputs.get("evaluation_report.json"):
        raise ConfigurationError("Strict reproduction requires a recorded evaluation report")
    path = store.resolve(artifact.artifact_id)
    if sha256_file(path) != artifact.sha256:
        raise ConfigurationError("Recorded evaluation report checksum does not match")
    report = EvaluationReport.model_validate_json(path.read_text(encoding="utf-8"))
    for family, metric, name, settings in (
        ("content", "wer", "faster-whisper", config.evaluation.content),
        ("timbre", "singer_similarity", "speechbrain-ecapa", config.evaluation.singer),
    ):
        if not settings.enabled:
            continue
        result = report.families.get(family, {}).get(metric)
        metadata = result.evaluator if result else None
        if (metadata is None or not metadata.checkpoint_sha256 or settings.model_directory is None
                or metadata.name != name or metadata.version != settings.revision or metadata.model != settings.model_name):
            raise ConfigurationError(f"Recorded {family} evaluator identity is missing or differs")
        if model_directory_digest(settings.model_directory) != metadata.checkpoint_sha256:
            raise ConfigurationError(f"Local {family} evaluator asset checksum differs from the recorded run")


def verify_execution_identity(manifest: RunManifest, executions: dict[str, ComponentExecution]) -> None:
    if executions.keys() != manifest.executions.keys():
        raise ConfigurationError("Resolved components differ from the recorded run")
    fields = ("provider", "profile", "version", "checkpoint_sha256", "config_sha256", "backend", "precision", "python_version", "torch_version")
    for name, execution in executions.items():
        previous = manifest.executions[name]
        if any(getattr(previous, field) != getattr(execution, field) for field in fields):
            raise ConfigurationError(f"Resolved {name} identity differs from the recorded run")
        if previous.version == "unknown":
            raise ConfigurationError("A provider with unknown revision cannot be reproduced strictly")
