from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

from .audio.io import load_audio, save_audio
from .benchmarking import BenchmarkRunner, BenchmarkSpec
from .components import ComponentFactory
from .config import load_config
from .contracts import BackendName, ConversionRequest
from .datasets import DatasetManager, render_dataset_report
from .device import DeviceManager
from .evaluation import evaluate_audio
from .jobs import JobStore, Worker
from .logging import configure_logging
from .registry import ModelRegistry
from .runtime import build_handlers
from .storage import LocalArtifactStore


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="nsvp")
    root.add_argument("--config", type=Path)
    commands = root.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--backend", choices=[item.value for item in BackendName], default="auto")
    dataset = commands.add_parser("dataset")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    prepare = dataset_commands.add_parser("prepare")
    prepare.add_argument("source", type=Path)
    prepare.add_argument("--singer", required=True)
    prepare.add_argument("--report", type=Path, default=Path("artifacts/reports/training_data_report.html"))
    separate = commands.add_parser("separate")
    separate.add_argument("song", type=Path)
    separate.add_argument("--output", type=Path, default=Path("outputs/separation"))
    convert = commands.add_parser("convert")
    convert.add_argument("song", type=Path)
    convert.add_argument("--reference", type=Path, required=True)
    convert.add_argument("--output-name", required=True)
    convert.add_argument("--transpose", type=int, default=0)
    convert.add_argument("--voice-converter")
    convert.add_argument("--converter-profile")
    convert.add_argument("--separator")
    convert.add_argument("--separator-profile")
    convert.add_argument("--vocal-processing-profile")
    convert.add_argument("--model-profile")
    convert.add_argument("--backend", choices=[item.value for item in BackendName], default="auto")
    convert.add_argument("--precision", choices=["auto", "fp32", "fp16"], default="auto")
    convert.add_argument("--seed", type=int, default=42)
    convert.add_argument("--input-kind", choices=["song", "vocal"], default="song")
    convert.add_argument("--discard-work", action="store_true")
    convert.add_argument("--language", choices=["tr", "en"])
    convert.add_argument("--reference-text", help="Verified source lyrics for supervised WER/CER")
    commands.add_parser("smoke", parents=[convert], add_help=False, help="Run one explicit model conversion and persist smoke evidence")
    components = commands.add_parser("components")
    components.add_argument("--probe", action="store_true")
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("spec", type=Path)
    reproduce = commands.add_parser("reproduce")
    reproduce.add_argument("manifest_artifact_id")
    compatibility = commands.add_parser("compatibility-report")
    compatibility.add_argument("manifest_artifact_ids", nargs="*")
    compatibility.add_argument("--failed-job", action="append", default=[], help="Include an explicit failed job ID")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("source", type=Path)
    evaluate.add_argument("output", type=Path)
    evaluate.add_argument("--reference", type=Path)
    evaluate.add_argument("--language", choices=["tr", "en"])
    evaluate.add_argument("--reference-text", help="Verified source lyrics for supervised WER/CER")
    worker = commands.add_parser("worker")
    worker.add_argument("--kinds", nargs="+", help="Only claim these job kinds")
    models = commands.add_parser("models")
    model_commands = models.add_subparsers(dest="model_command", required=True)
    model_commands.add_parser("list")
    download = model_commands.add_parser("download-seed-vc")
    download.add_argument("destination", type=Path)
    download.add_argument("--commit", required=True, help="Reviewed upstream commit hash to pin")
    return root


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    configure_logging()
    config = load_config(args.config)
    store = LocalArtifactStore(config.artifact_root)
    factory = ComponentFactory(config)
    if args.command == "doctor":
        manager = DeviceManager()
        if importlib.util.find_spec("torch") is None:
            backend_report = manager.report_without_torch()
        else:
            backend_report = manager.detect(BackendName(args.backend)).capabilities
        print(backend_report.model_dump_json(indent=2))
    elif args.command == "dataset":
        manifest = DatasetManager(store, config.audio, factory.build_pitch_extractor(), config.dataset_audit).prepare(args.source, args.singer)
        render_dataset_report(manifest, args.report)
        print(manifest.model_dump_json(indent=2))
    elif args.command == "separate":
        separator, _ = factory.build_separator()
        stems = separator.separate(load_audio(args.song), args.output / "work")
        save_audio(args.output / "vocals.wav", stems.vocals)
        save_audio(args.output / "instrumental.wav", stems.instrumental)
        print(json.dumps({"vocals": str(args.output / 'vocals.wav'), "instrumental": str(args.output / 'instrumental.wav')}, indent=2))
    elif args.command in ("convert", "smoke"):
        request = ConversionRequest(
            song_path=args.song, target_reference_path=args.reference, output_name=args.output_name,
            transpose_semitones=args.transpose, voice_converter=args.voice_converter,
            converter_profile=args.converter_profile, separator=args.separator,
            separator_profile=args.separator_profile, model_profile=args.model_profile,
            vocal_processing_profile=args.vocal_processing_profile, backend=BackendName(args.backend),
            precision=args.precision, random_seed=args.seed, input_kind=args.input_kind,
            keep_intermediates=not args.discard_work,
            language=args.language, reference_text=args.reference_text,
        )
        result = build_handlers(config, factory)["conversion"](request.model_dump(mode="json"), lambda value, stage: None)
        if args.command == "smoke":
            result["artifacts"]["smoke_report.json"] = store.put_json({
                "status": "passed", "scope": "One explicit conversion, not model quality or general hardware certification",
                "result": result,
            }, f"smoke-{result['conversion_id']}", "smoke_report.json")
        print(json.dumps(result, indent=2))
    elif args.command == "components":
        print(json.dumps([entry.model_dump(mode="json") for entry in factory.capabilities(args.probe)], indent=2))
    elif args.command == "benchmark":
        spec = BenchmarkSpec.model_validate_json(args.spec.read_text(encoding="utf-8"))
        print(BenchmarkRunner(factory, store).run(spec).model_dump_json(indent=2))
    elif args.command == "reproduce":
        result = build_handlers(config, factory)["reproduce"](
            {"manifest_artifact_id": args.manifest_artifact_id}, lambda value, stage: None,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "compatibility-report":
        from .compatibility import compatibility_report

        jobs = JobStore(config.database_path) if args.failed_job else None
        failures = [jobs.get(identifier) for identifier in args.failed_job] if jobs else []
        print(compatibility_report(store, args.manifest_artifact_ids, failures).model_dump_json(indent=2))
    elif args.command == "evaluate":
        source, output = load_audio(args.source), load_audio(args.output)
        pitch = factory.build_pitch_extractor()
        evaluation_report = evaluate_audio(source, output, pitch.extract(source), pitch.extract(output),
            target_reference=load_audio(args.reference) if args.reference else None,
            content_evaluator=factory.build_content_evaluator(args.language, args.reference_text),
            singer_evaluator=factory.build_singer_evaluator())
        print(evaluation_report.model_dump_json(indent=2))
    elif args.command == "worker":
        Worker(JobStore(config.database_path), build_handlers(config), kinds=tuple(args.kinds or ())).run_forever()
    elif args.command == "models" and args.model_command == "list":
        registry = ModelRegistry(config.artifact_root / "models", store)
        print(json.dumps([item.model_dump(mode="json") for item in registry.list()], indent=2, default=str))
    elif args.command == "models" and args.model_command == "download-seed-vc":
        if args.destination.exists():
            raise SystemExit(f"destination already exists: {args.destination}")
        subprocess.run(["git", "clone", "https://github.com/Plachtaa/seed-vc.git", str(args.destination)], check=True)
        subprocess.run(["git", "checkout", args.commit], cwd=args.destination, check=True)
        print("Seed-VC source pinned. Download the reviewed checkpoint separately and configure its paths; inference will not auto-select weights.")


if __name__ == "__main__":
    main()
