from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from .contracts import DatasetManifest
from .errors import DependencyUnavailableError, NSVPError
from .storage import LocalArtifactStore


class SeedVCTrainingBridge:
    """Deterministically exports NSVP data and launches a pinned Seed-VC training checkout."""

    def __init__(self, seed_vc_root: Path, store: LocalArtifactStore) -> None:
        self.seed_vc_root = seed_vc_root
        self.store = store

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
        train_script = self.seed_vc_root / "train.py"
        if not train_script.is_file():
            raise DependencyUnavailableError(f"Seed-VC training entrypoint not found: {train_script}")
        command = [sys.executable, str(train_script), "--config", str(config_path), "--run-name", run_name]
        if resume is not None:
            command.extend(["--resume", str(resume)])
        completed = subprocess.run(command, cwd=self.seed_vc_root, capture_output=True, text=True, check=False)
        if completed.returncode:
            raise NSVPError(f"Seed-VC training failed: {completed.stderr[-3000:]}")


class ExperimentTracker:
    def __init__(self, tracking_uri: str) -> None:
        try:
            import mlflow
        except ImportError as exc:
            raise DependencyUnavailableError("MLflow requires the 'training' dependency group") from exc
        self.mlflow = mlflow
        self.mlflow.set_tracking_uri(tracking_uri)

    def log_run_metadata(self, params: dict[str, object], dataset_name: str, dataset_version: str) -> None:
        self.mlflow.log_params({**params, "dataset_name": dataset_name, "dataset_version": dataset_version})
