from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .api_contracts import EvaluationJobRequest, VocalPreparationRequest
from .audio.io import load_audio, save_audio
from .components import ComponentFactory
from .config import AppConfig
from .contracts import ConversionRequest, DatasetManifest
from .datasets import DatasetManager, render_dataset_report
from .evaluation import evaluate_audio
from .pipeline import ConversionPipeline
from .preparation import prepare_vocal, resolve_conversion_request
from .provenance import RunManifest, finalize_manifest, reproduce_request, verify_execution_identity
from .storage import LocalArtifactStore
from .training import track_optional_run


def build_handlers(config: AppConfig, factory: ComponentFactory | None = None) -> dict[str, Callable[[dict[str, Any], Callable[[float, str], None]], dict[str, Any]]]:
    store = LocalArtifactStore(config.artifact_root)
    components = factory or ComponentFactory(config)

    def dataset_analyze(payload: dict[str, Any], progress: Callable[[float, str], None]) -> dict[str, Any]:
        progress(0.1, "validating_audio")
        manager = DatasetManager(store, config.audio, components.build_pitch_extractor(), config.dataset_audit)
        if "source_artifact_ids" in payload:
            manifest = manager.prepare_artifacts(payload["source_artifact_ids"], payload["singer_name"])
        else:
            manifest = manager.prepare(Path(payload["source_root"]), payload["singer_name"])
        progress(0.85, "rendering_report")
        report = config.artifact_root / "reports" / f"{manifest.dataset_id}.html"
        render_dataset_report(manifest, report)
        manifest_id = store.put_json(manifest.model_dump(mode="json"), "dataset-manifests", f"{manifest.dataset_id}.json")
        report_id = store.put_file(report, "dataset-reports", report.name)
        return {"dataset_id": manifest.dataset_id, "manifest_artifact_id": manifest_id, "report_artifact_id": report_id}

    def convert(payload: dict[str, Any], progress: Callable[[float, str], None], replay: RunManifest | None = None) -> dict[str, Any]:
        request = resolve_conversion_request(payload, store)
        progress(0.05, "resolving_components")
        pipeline = build_conversion_pipeline(components, request, store)
        if replay is not None:
            verify_execution_identity(replay, pipeline.executions)
        result = pipeline.run(request, progress)
        progress(0.9, "evaluating")
        source = load_audio(store.resolve(result.artifacts.get("selected_vocal.wav", result.artifacts["source_vocal.wav"])))
        output = load_audio(store.resolve(result.artifacts["converted_vocal_raw.wav"]))
        pitch = components.build_pitch_extractor()
        report = evaluate_audio(
            source, output, pitch.extract(source), pitch.extract(output),
            target_reference=load_audio(request.target_reference_path),
            input_condition=request.input_condition, transpose_semitones=request.transpose_semitones,
            content_evaluator=components.build_content_evaluator(request.language, request.reference_text),
            singer_evaluator=components.build_singer_evaluator(),
            pipeline_seconds=result.processing_time_seconds,
            provider_process=result.provider_process,
        )
        result.artifacts["evaluation_report.json"] = store.put_json(
            report.model_dump(mode="json"), f"conversion-{result.conversion_id}", "evaluation_report.json",
        )
        finalize_manifest(result, store)
        progress(0.95, "storing_artifacts")
        return result.model_dump(mode="json")

    def reproduce(payload: dict[str, Any], progress: Callable[[float, str], None]) -> dict[str, Any]:
        manifest = RunManifest.model_validate_json(store.resolve(payload["manifest_artifact_id"]).read_text(encoding="utf-8"))
        request = reproduce_request(manifest, store, config)
        result = convert(request.model_dump(mode="json"), progress, replay=manifest)
        result["reproduced_from_run_id"] = manifest.run_id
        return result

    def separate(payload: dict[str, Any], progress: Callable[[float, str], None]) -> dict[str, Any]:
        progress(0.1, "loading_audio")
        song = load_audio(store.resolve(payload["song_artifact_id"]))
        progress(0.2, "separating_stems")
        selection = ConversionRequest.model_validate({
            "song_path": store.resolve(payload["song_artifact_id"]),
            "target_reference_path": store.resolve(payload["song_artifact_id"]),
            "output_name": payload["job_namespace"],
            **{key: payload[key] for key in ("separator", "separator_profile", "backend") if key in payload},
        })
        separator, execution = components.build_separator(selection)
        stems = separator.separate(song, config.artifact_root / "work" / payload["job_namespace"])
        output = config.artifact_root / "separations" / payload["job_namespace"]
        output.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, str] = {}
        for name, audio in (("vocals.wav", stems.vocals), ("instrumental.wav", stems.instrumental)):
            path = output / name
            save_audio(path, audio)
            artifacts[name] = store.put_file(path, f"separation-{payload['job_namespace']}", name)
        return {"artifacts": artifacts, "separator": separator.name, "execution": execution.model_dump(mode="json")}

    def train(payload: dict[str, Any], progress: Callable[[float, str], None]) -> dict[str, Any]:
        manifest_path = store.resolve(payload["manifest_artifact_id"])
        manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        upstream_config = store.resolve(payload["training_config_artifact_id"])
        bridge = components.build_training_bridge(store, payload.get("provider", "seed_vc"))
        progress(0.1, "exporting_training_dataset")
        exported = bridge.export_dataset(manifest, config.artifact_root / "training" / payload["run_name"])
        resume = store.resolve(payload["resume_artifact_id"]) if payload.get("resume_artifact_id") else None
        progress(0.2, "training")
        from .benchmarking import git_commit, public_config_snapshot
        from .storage import sha256_file

        started = time.perf_counter()
        failed = True
        try:
            bridge.run(upstream_config, payload["run_name"], resume)
            failed = False
        finally:
            tracking_id, tracking_warnings = track_optional_run(
                config.mlflow_tracking_uri, store.root, payload["run_name"], {
                    "git_commit": git_commit(), "dataset_id": manifest.dataset_id,
                    "dataset_version": manifest.version, "provider": bridge.name,
                    "training_config_sha256": sha256_file(upstream_config),
                    "sample_rate": config.audio.training_sample_rate,
                }, {"runtime_seconds": time.perf_counter() - started},
                {"config.json": public_config_snapshot(config.model_dump(mode="json"))}, failed=failed,
            )
        return {"run_name": payload["run_name"], "dataset_export_created": exported.is_file(), "status": "completed",
            "checkpoint": "Not registered; run checkpoint smoke validation first", "mlflow_run_id": tracking_id, "warnings": tracking_warnings}

    def benchmark(payload: dict[str, Any], progress: Callable[[float, str], None]) -> dict[str, Any]:
        from .benchmarking import BenchmarkRunner, BenchmarkSpec

        spec = BenchmarkSpec.model_validate(payload)
        return BenchmarkRunner(components, store).run(spec, progress).model_dump(mode="json")

    def evaluate(payload: dict[str, Any], progress: Callable[[float, str], None]) -> dict[str, Any]:
        request = EvaluationJobRequest.model_validate(payload)
        progress(0.1, "loading_audio")
        source = load_audio(store.resolve(request.source_artifact_id))
        output = load_audio(store.resolve(request.output_artifact_id))
        pitch = components.build_pitch_extractor()
        progress(0.3, "evaluating")
        report = evaluate_audio(
            source, output, pitch.extract(source), pitch.extract(output),
            target_reference=load_audio(store.resolve(request.reference_artifact_id)) if request.reference_artifact_id else None,
            ground_truth_stem=load_audio(store.resolve(request.ground_truth_stem_artifact_id)) if request.ground_truth_stem_artifact_id else None,
            separated_stem=load_audio(store.resolve(request.separated_stem_artifact_id)) if request.separated_stem_artifact_id else None,
            input_condition=request.input_condition, transpose_semitones=request.transpose_semitones,
            content_evaluator=components.build_content_evaluator(request.language, request.reference_text),
            singer_evaluator=components.build_singer_evaluator(),
        )
        artifact = store.put_json(report.model_dump(mode="json"), "evaluations", "evaluation_report.json")
        return {"evaluation": report.model_dump(mode="json"), "artifacts": {"evaluation_report.json": artifact}}

    def vocal_preparation(payload: dict[str, Any], progress: Callable[[float, str], None]) -> dict[str, Any]:
        request = VocalPreparationRequest.model_validate(payload)
        return prepare_vocal(request, components, store, progress).model_dump(mode="json")

    return {"dataset_analyze": dataset_analyze, "conversion": convert, "reproduce": reproduce, "separation": separate,
        "training": train, "benchmark": benchmark, "evaluation": evaluate, "vocal_preparation": vocal_preparation}


def build_conversion_pipeline(
    factory: ComponentFactory, request: ConversionRequest, store: LocalArtifactStore,
) -> ConversionPipeline:
    converter, execution = factory.build_voice_converter(request, store)
    executions = {"voice_converter": execution}
    separator = None
    if request.input_kind == "song":
        separator, separation_execution = factory.build_separator(request)
        executions["separator"] = separation_execution
    return ConversionPipeline(
        separator, converter, store,
        preprocessors=[] if request.preparation_manifest_artifact_id else factory.build_vocal_preprocessors(request.vocal_processing_profile),
        executions=executions,
        configuration=factory.config,
    )
