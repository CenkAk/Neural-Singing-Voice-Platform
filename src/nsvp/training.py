from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .adapters.external import python_executable, run_external
from .contracts import DatasetManifest
from .errors import ConfigurationError, DependencyUnavailableError
from .storage import LocalArtifactStore

logger = logging.getLogger(__name__)


class SeedVCTrainingBridge:
    """Deterministically exports NSVP data and launches a pinned Seed-VC training checkout."""

    name = "seed_vc"

    def __init__(self, seed_vc_root: Path, store: LocalArtifactStore, python: Path | None = None) -> None:
        self.seed_vc_root = seed_vc_root
        self.store = store
        self.python = python

    def export_dataset(self, manifest: DatasetManifest, output: Path) -> Path:
        output.mkdir(parents=True, exist_ok=True)
        entries = []
        for segment in sorted(manifest.segments, key=lambda item: item.segment_id):
            if segment.artifact_id is None:
                continue
            entries.append({"audio": str(self.store.resolve(segment.artifact_id)), "speaker": manifest.singer_name, "split": segment.split})
        path = output / "nsvp-seed-vc-dataset.json"
        path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        return path

    def run(self, config_path: Path, run_name: str, resume: Path | None = None) -> None:
        train_script = self.seed_vc_root.resolve() / "train.py"
        if not train_script.is_file():
            raise DependencyUnavailableError(f"Seed-VC training entrypoint not found: {train_script}")
        command = [str(python_executable(self.python)), str(train_script), "--config", str(config_path.resolve()), "--run-name", run_name]
        if resume is not None:
            command.extend(["--resume", str(resume.resolve())])
        run_external(command, cwd=self.seed_vc_root, timeout=86400, label="Seed-VC training")


class ExperimentTracker:
    def __init__(self, tracking_uri: str, artifact_root: Path | None = None) -> None:
        if not tracking_uri.startswith("sqlite:///") or "?" in tracking_uri or "#" in tracking_uri:
            raise ConfigurationError("v0.2 MLflow tracking accepts local SQLite destinations only")
        database = Path(tracking_uri[len("sqlite:///"):]).resolve()
        root = artifact_root.resolve() if artifact_root else database.parent
        if root not in database.parents:
            raise ConfigurationError("MLflow database must be inside the configured artifact root")
        self.tracking_uri = "sqlite:///" + database.as_posix()
        self.artifact_directory = root / "mlflow-artifacts"
        try:
            import mlflow
        except ImportError as exc:
            raise DependencyUnavailableError("MLflow requires the 'training' dependency group") from exc
        self.mlflow = mlflow
        self.mlflow.set_tracking_uri(self.tracking_uri)

    def log_run_metadata(self, params: dict[str, object], dataset_name: str, dataset_version: str) -> None:
        self.mlflow.log_params({**params, "dataset_name": dataset_name, "dataset_version": dataset_version})

    def record_run(
        self, run_name: str, params: dict[str, object], metrics: dict[str, float],
        documents: dict[str, dict[str, Any]], *, failed: bool = False,
    ) -> str:
        client = self.mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri)
        location = self.artifact_directory.as_uri()
        experiment = client.get_experiment_by_name("nsvp-local-v02")
        if experiment is None:
            experiment_id = client.create_experiment("nsvp-local-v02", artifact_location=location)
        else:
            if experiment.artifact_location != location:
                raise ConfigurationError("MLflow experiment artifacts must use the configured local directory")
            experiment_id = experiment.experiment_id
        run = client.create_run(experiment_id, tags={"mlflow.runName": run_name})
        run_id = str(run.info.run_id)
        if not str(run.info.artifact_uri).startswith(location + "/"):
            raise ConfigurationError("MLflow run resolved to a non-local artifact location")
        status = "FAILED"
        try:
            for key, value in params.items():
                if value is not None:
                    client.log_param(run_id, key, value)
            for key, value in metrics.items():
                client.log_metric(run_id, key, value)
            for name, document in documents.items():
                client.log_dict(run_id, document, name)
            status = "FAILED" if failed else "FINISHED"
            return run_id
        finally:
            client.set_terminated(run_id, status=status)


def track_optional_run(
    tracking_uri: str | None, artifact_root: Path, run_name: str, params: dict[str, object],
    metrics: dict[str, float], documents: dict[str, dict[str, Any]], *, failed: bool = False,
) -> tuple[str | None, list[str]]:
    if not tracking_uri:
        return None, []
    try:
        tracker = ExperimentTracker(tracking_uri, artifact_root)
        return tracker.record_run(run_name, params, metrics, documents, failed=failed), []
    except Exception:
        logger.exception("Optional local MLflow tracking failed")
        return None, ["Local MLflow tracking failed; local NSVP results remain available."]
