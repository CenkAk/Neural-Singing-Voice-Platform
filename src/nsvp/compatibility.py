from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from .audio.io import load_audio
from .audio.processing import analyze_audio
from .components import component_catalog
from .contracts import BackendName, CompatibilityStatus, ComponentExecution, JobRecord, JobState
from .errors import ConfigurationError
from .provenance import RunManifest
from .storage import LocalArtifactStore, sha256_file


class ConversionEvidence(BaseModel):
    run_id: str
    manifest_sha256: str
    status: Literal["verified"] = "verified"
    scope: str = "Completed conversion with intact artifacts, not perceptual quality or general backend certification"
    system: str
    architecture: str
    execution: ComponentExecution
    pipeline_seconds: float = Field(ge=0, allow_inf_nan=False)
    output_duration_seconds: float = Field(gt=0, allow_inf_nan=False)
    benchmark_run_id: str | None


class FailedJobEvidence(BaseModel):
    job_id: str
    kind: str
    created_at: datetime
    status: Literal["failed"] = "failed"
    scope: str = "Recorded job failure; does not establish hardware incompatibility or a resolved execution device"
    requested_backend: BackendName | None
    requested_provider: str | None
    error_code: str


class CompatibilityReport(BaseModel):
    schema_version: Literal["0.3"] = "0.3"
    adapter_support: dict[str, dict[str, str]]
    conversions: list[ConversionEvidence]
    failed_jobs: list[FailedJobEvidence] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=lambda: [
        "Adapter support is separate from the exact executed configurations below.",
        "Execution precision is the requested mode; internal modules may use mixed dtypes.",
        "No peak RAM or VRAM measurement is inferred from a completed conversion.",
        "This report covers only supplied manifests and failed job IDs, not all attempted configurations.",
    ])


def compatibility_report(store: LocalArtifactStore, manifest_ids: list[str], failed_jobs: Sequence[JobRecord] = ()) -> CompatibilityReport:
    support = {
        name: {backend.value: "unsupported" if status is CompatibilityStatus.UNSUPPORTED else "not_tested"
               for backend, status in capabilities.backends.items()}
        for name, capabilities in component_catalog().items()
        if capabilities.external_environment
    }
    records: dict[str, ConversionEvidence] = {}
    for identifier in manifest_ids:
        path = store.resolve(identifier)
        manifest = RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
        if not {"source", "reference"}.issubset(manifest.inputs):
            raise ConfigurationError("Compatibility evidence requires source and reference digests")
        execution = manifest.executions.get("voice_converter")
        if (execution is None or execution.provider not in support or execution.version in {"synthetic", "unknown"}
                or not execution.checkpoint_sha256 or execution.backend is BackendName.AUTO):
            raise ConfigurationError("Compatibility evidence requires a real converter with recorded revision, checkpoint and backend")
        for artifact in [*manifest.inputs.values(), *manifest.outputs.values()]:
            if sha256_file(store.resolve(artifact.artifact_id)) != artifact.sha256:
                raise ConfigurationError("Compatibility evidence artifact checksum does not match")
        if "converted_vocal_raw.wav" not in manifest.outputs or "conversion_report.json" not in manifest.outputs:
            raise ConfigurationError("Compatibility evidence requires conversion output and report")
        audio = load_audio(store.resolve(manifest.outputs["converted_vocal_raw.wav"].artifact_id))
        quality = analyze_audio(audio)
        if not quality.finite or audio.samples == 0:
            raise ConfigurationError("Compatibility evidence contains invalid audio")
        report = json.loads(store.resolve(manifest.outputs["conversion_report.json"].artifact_id).read_text(encoding="utf-8"))
        if not isinstance(report, dict) or report.get("conversion_id") != manifest.run_id:
            raise ConfigurationError("Conversion report and manifest run IDs differ")
        reported_executions = report.get("executions")
        if not isinstance(reported_executions, dict) or reported_executions != {
            name: component.model_dump(mode="json") for name, component in manifest.executions.items()
        }:
            raise ConfigurationError("Conversion report and manifest execution identities differ")
        seconds = report.get("processing_time_seconds")
        if not isinstance(seconds, (float, int)) or isinstance(seconds, bool):
            raise ConfigurationError("Conversion report requires numeric pipeline duration")
        records[manifest.run_id] = ConversionEvidence(
            run_id=manifest.run_id, manifest_sha256=sha256_file(path), system=manifest.system,
            architecture=manifest.architecture, execution=execution,
            pipeline_seconds=seconds, output_duration_seconds=audio.duration_seconds,
            benchmark_run_id=manifest.benchmark_run_id,
        )
    failures: dict[str, FailedJobEvidence] = {}
    for job in failed_jobs:
        if job.state is not JobState.FAILED or job.kind not in {"conversion", "reproduce", "benchmark"}:
            raise ConfigurationError("Failure evidence requires a failed conversion, reproduction or benchmark job")
        backend = job.payload.get("backend")
        requested_backend = BackendName(backend) if isinstance(backend, str) and backend in {item.value for item in BackendName} else None
        provider = job.payload.get("voice_converter")
        code = (job.error or {}).get("code")
        failures[job.id] = FailedJobEvidence(job_id=job.id, kind=job.kind, created_at=job.created_at,
            requested_backend=requested_backend, requested_provider=provider if isinstance(provider, str) and provider in support else None,
            error_code=code if isinstance(code, str) and re.fullmatch(r"[a-z_]{1,80}", code) else "unknown")
    return CompatibilityReport(adapter_support=support, conversions=list(records.values()), failed_jobs=list(failures.values()))
