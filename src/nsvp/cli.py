from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path

from .adapters import DemucsSeparator, SeedVCConverter, SeedVCSettings
from .audio.io import load_audio, save_audio
from .config import load_config
from .contracts import BackendName, ConversionRequest
from .datasets import DatasetManager, render_dataset_report
from .device import DeviceManager
from .evaluation import evaluate_audio
from .jobs import JobStore, Worker
from .logging import configure_logging
from .pipeline import ConversionPipeline
from .pitch import AutocorrelationPitchExtractor
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
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("source", type=Path)
    evaluate.add_argument("output", type=Path)
    commands.add_parser("worker")
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
    if args.command == "doctor":
        manager = DeviceManager()
        if importlib.util.find_spec("torch") is None:
            backend_report = manager.report_without_torch()
        else:
            backend_report = manager.detect(BackendName(args.backend)).capabilities
        print(backend_report.model_dump_json(indent=2))
    elif args.command == "dataset":
        manifest = DatasetManager(store, config.audio).prepare(args.source, args.singer)
        render_dataset_report(manifest, args.report)
        print(manifest.model_dump_json(indent=2))
    elif args.command == "separate":
        stems = DemucsSeparator().separate(load_audio(args.song), args.output / "work")
        save_audio(args.output / "vocals.wav", stems.vocals)
        save_audio(args.output / "instrumental.wav", stems.instrumental)
        print(json.dumps({"vocals": str(args.output / 'vocals.wav'), "instrumental": str(args.output / 'instrumental.wav')}, indent=2))
    elif args.command == "convert":
        if config.seed_vc_root is None or config.seed_vc_checkpoint is None or config.seed_vc_config is None:
            raise SystemExit("configure seed_vc_root, seed_vc_checkpoint, and seed_vc_config first")
        converter = SeedVCConverter(SeedVCSettings(repository_root=config.seed_vc_root, checkpoint_path=config.seed_vc_checkpoint, config_path=config.seed_vc_config))
        request = ConversionRequest(song_path=args.song, target_reference_path=args.reference, output_name=args.output_name, transpose_semitones=args.transpose)
        print(ConversionPipeline(DemucsSeparator(), converter, store).run(request).model_dump_json(indent=2))
    elif args.command == "evaluate":
        source, output = load_audio(args.source), load_audio(args.output)
        pitch = AutocorrelationPitchExtractor()
        evaluation_report = evaluate_audio(source, output, pitch.extract(source), pitch.extract(output))
        print(evaluation_report.model_dump_json(indent=2))
    elif args.command == "worker":
        Worker(JobStore(config.database_path), build_handlers(config)).run_forever()
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
