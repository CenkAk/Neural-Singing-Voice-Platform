from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .adapters import DemucsSeparator, SeedVCConverter, SeedVCSettings
from .audio.io import load_audio, save_audio
from .config import AppConfig
from .contracts import ConversionRequest, DatasetManifest
from .datasets import DatasetManager, render_dataset_report
from .pipeline import ConversionPipeline
from .storage import LocalArtifactStore
from .training import SeedVCTrainingBridge


def build_handlers(config: AppConfig) -> dict[str, Callable[[dict[str, Any], Callable[[float, str], None]], dict[str, Any]]]:
    store = LocalArtifactStore(config.artifact_root)

    def dataset_analyze(payload: dict[str, Any], progress: Callable[[float, str], None]) -> dict[str, Any]:
        progress(0.1, "validating_audio")
        manifest = DatasetManager(store, config.audio).prepare(Path(payload["source_root"]), payload["singer_name"])
        progress(0.85, "rendering_report")
        report = config.artifact_root / "reports" / f"{manifest.dataset_id}.html"
        render_dataset_report(manifest, report)
        manifest_id = store.put_json(manifest.model_dump(mode="json"), "dataset-manifests", f"{manifest.dataset_id}.json")
        report_id = store.put_file(report, "dataset-reports", report.name)
        return {"dataset_id": manifest.dataset_id, "manifest_artifact_id": manifest_id, "report_artifact_id": report_id}

    def convert(payload: dict[str, Any], progress: Callable[[float, str], None]) -> dict[str, Any]:
        if config.seed_vc_root is None or config.seed_vc_checkpoint is None or config.seed_vc_config is None:
            raise ValueError("Seed-VC root, checkpoint, and config must be configured for conversion jobs")
        progress(0.05, "loading_audio")
        converter = SeedVCConverter(SeedVCSettings(
            repository_root=config.seed_vc_root,
            checkpoint_path=config.seed_vc_checkpoint,
            config_path=config.seed_vc_config,
        ))
        pipeline = ConversionPipeline(DemucsSeparator(), converter, store)
        progress(0.1, "separating_and_converting")
        result = pipeline.run(ConversionRequest.model_validate(payload))
        progress(0.95, "storing_artifacts")
        return result.model_dump(mode="json")

    def separate(payload: dict[str, Any], progress: Callable[[float, str], None]) -> dict[str, Any]:
        progress(0.1, "loading_audio")
        song = load_audio(store.resolve(payload["song_artifact_id"]))
        progress(0.2, "separating_stems")
        stems = DemucsSeparator().separate(song, config.artifact_root / "work" / payload["job_namespace"])
        output = config.artifact_root / "separations" / payload["job_namespace"]
        output.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, str] = {}
        for name, audio in (("vocals.wav", stems.vocals), ("instrumental.wav", stems.instrumental)):
            path = output / name
            save_audio(path, audio)
            artifacts[name] = store.put_file(path, f"separation-{payload['job_namespace']}", name)
        return {"artifacts": artifacts, "separator": "demucs-htdemucs"}

    def train(payload: dict[str, Any], progress: Callable[[float, str], None]) -> dict[str, Any]:
        if config.seed_vc_root is None:
            raise ValueError("seed_vc_root must be configured for training jobs")
        manifest_path = store.resolve(payload["manifest_artifact_id"])
        manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        upstream_config = store.resolve(payload["training_config_artifact_id"])
        bridge = SeedVCTrainingBridge(config.seed_vc_root, store)
        progress(0.1, "exporting_training_dataset")
        exported = bridge.export_dataset(manifest, config.artifact_root / "training" / payload["run_name"])
        resume = store.resolve(payload["resume_artifact_id"]) if payload.get("resume_artifact_id") else None
        progress(0.2, "training")
        bridge.run(upstream_config, payload["run_name"], resume)
        return {"run_name": payload["run_name"], "dataset_export": str(exported), "status": "completed", "checkpoint": "Not registered; run checkpoint smoke validation first"}

    return {"dataset_analyze": dataset_analyze, "conversion": convert, "separation": separate, "training": train}
