from __future__ import annotations

import html
import json
import logging
import platform
import statistics
import subprocess
import time
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .audio.io import load_audio
from .components import ComponentFactory
from .contracts import (
    BackendName,
    ConversionRequest,
    ConversionResult,
    EvaluationReport,
    InputCondition,
)
from .errors import BackendUnavailableError, ConfigurationError
from .evaluation import evaluate_audio
from .provenance import finalize_manifest
from .storage import LocalArtifactStore, sha256_file
from .training import track_optional_run

logger = logging.getLogger(__name__)


class BenchmarkCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    source_artifact_id: str
    reference_artifact_id: str
    ground_truth_stem_artifact_id: str | None = None
    instrumental_artifact_id: str | None = None
    input_kind: Literal["song", "vocal"] = "vocal"
    input_condition: InputCondition = InputCondition.UNKNOWN
    dataset_id: str | None = None
    dataset_version: str | None = None
    tags: list[str] = Field(default_factory=list, max_length=32)
    language: Literal["tr", "en"] | None = None
    reference_text: str | None = Field(default=None, max_length=20000)


class BenchmarkConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    configuration_id: str
    voice_converter: str | None = None
    converter_profile: str | None = None
    separator: str | None = None
    separator_profile: str | None = None
    vocal_processing_profile: str | None = None
    model_profile: str | None = None
    backend: BackendName = BackendName.AUTO
    precision: Literal["auto", "fp32", "fp16"] = "auto"
    transpose_semitones: int = Field(default=0, ge=-12, le=12)


class BenchmarkSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    cases: list[BenchmarkCase] = Field(min_length=1)
    configurations: list[BenchmarkConfiguration] = Field(min_length=1)
    seeds: list[int] = Field(default_factory=lambda: [42], min_length=1)

    @model_validator(mode="after")
    def unique_identifiers(self) -> BenchmarkSpec:
        for identifiers in ([case.case_id for case in self.cases], [item.configuration_id for item in self.configurations]):
            if len(set(identifiers)) != len(identifiers):
                raise ValueError("benchmark case and configuration identifiers must be unique")
        if any(seed < 0 or seed > 2**32 - 1 for seed in self.seeds) or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("benchmark seeds must be unique uint32 values")
        return self


class BenchmarkCaseResult(BaseModel):
    case_id: str
    configuration_id: str
    random_seed: int
    input_condition: InputCondition
    status: Literal["succeeded", "not_tested", "unsupported", "failed"]
    source_sha256: str
    reference_sha256: str
    ground_truth_sha256: str | None = None
    processing_time_seconds: float
    conversion: ConversionResult | None = None
    evaluation: EvaluationReport | None = None
    warnings: list[str] = Field(default_factory=list)
    mlflow_run_id: str | None = None
    source_sample_rate: int | None = None


class BenchmarkRun(BaseModel):
    schema_version: Literal["0.2", "0.3"] = "0.3"
    run_id: str
    name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    spec: BenchmarkSpec
    config_snapshot: dict[str, Any]
    git_commit: str | None = None
    hardware: dict[str, str]
    results: list[BenchmarkCaseResult] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(default_factory=dict)
    summary: list[dict[str, Any]] = Field(default_factory=list)


def public_config_snapshot(value: dict[str, Any]) -> dict[str, Any]:
    """Record reproducible settings without publishing machine-local paths or tracking credentials."""
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key.endswith(("_path", "_root", "_cache", "_executable", "_uri", "_directory")) or key in {
            "model_repository", "seed_vc_config", "seed_vc_checkpoint", "dataset_export",
        }:
            if item is not None:
                result[key + "_configured"] = True
            continue
        if isinstance(item, dict):
            result[key] = public_config_snapshot(item)
        elif isinstance(item, list):
            result[key] = [public_config_snapshot(entry) if isinstance(entry, dict) else entry for entry in item]
        else:
            result[key] = item
    return result


def git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


class BenchmarkRunner:
    def __init__(self, factory: ComponentFactory, store: LocalArtifactStore) -> None:
        self.factory, self.store = factory, store

    def run(self, spec: BenchmarkSpec, progress: Callable[[float, str], None] | None = None) -> BenchmarkRun:
        from .runtime import build_conversion_pipeline

        run = BenchmarkRun(
            run_id=uuid.uuid4().hex, name=spec.name, spec=spec,
            config_snapshot=public_config_snapshot(self.factory.config.model_dump(mode="json")),
            git_commit=git_commit(), hardware={"system": platform.system(), "machine": platform.machine(), "processor": platform.processor(), "python": platform.python_version()},
        )
        total = len(spec.cases) * len(spec.configurations) * len(spec.seeds)
        for case in spec.cases:
            source = self.store.resolve(case.source_artifact_id)
            reference = self.store.resolve(case.reference_artifact_id)
            ground_truth = self.store.resolve(case.ground_truth_stem_artifact_id) if case.ground_truth_stem_artifact_id else None
            instrumental = self.store.resolve(case.instrumental_artifact_id) if case.instrumental_artifact_id else None
            for configuration in spec.configurations:
                for seed in spec.seeds:
                    started = time.perf_counter()
                    index = len(run.results)

                    def notify(
                        fraction: float, stage: str, item_index: int = index,
                        case_id: str = case.case_id, configuration_id: str = configuration.configuration_id,
                    ) -> None:
                        if progress is not None:
                            progress((item_index + fraction) / total, f"benchmark:{case_id}:{configuration_id}:{stage}")

                    notify(0, "starting")
                    result = BenchmarkCaseResult(
                        case_id=case.case_id, configuration_id=configuration.configuration_id,
                        random_seed=seed, input_condition=case.input_condition, status="failed",
                        source_sha256=sha256_file(source), reference_sha256=sha256_file(reference),
                        ground_truth_sha256=sha256_file(ground_truth) if ground_truth else None,
                        processing_time_seconds=0,
                    )
                    try:
                        choices = configuration.model_dump(exclude={"configuration_id"})
                        request = ConversionRequest.model_validate({
                            **choices, "song_path": source, "target_reference_path": reference,
                            "output_name": f"benchmark-{run.run_id}", "random_seed": seed,
                            "input_kind": case.input_kind, "input_condition": case.input_condition,
                            "instrumental_path": instrumental, "keep_intermediates": True,
                            "language": case.language, "reference_text": case.reference_text,
                        })
                        pipeline = build_conversion_pipeline(self.factory, request, self.store)
                        conversion = pipeline.run(request, lambda value, stage: notify(value * 0.85, stage))
                        original = load_audio(self.store.resolve(conversion.artifacts["source_vocal.wav"]))
                        selected = load_audio(self.store.resolve(conversion.artifacts.get("selected_vocal.wav", conversion.artifacts["source_vocal.wav"])))
                        result.source_sample_rate = selected.sample_rate
                        output = load_audio(self.store.resolve(conversion.artifacts["converted_vocal_raw.wav"]))
                        pitch = self.factory.build_pitch_extractor()
                        evaluation = evaluate_audio(
                            selected, output, pitch.extract(selected), pitch.extract(output),
                            target_reference=load_audio(reference), input_condition=case.input_condition,
                            transpose_semitones=request.transpose_semitones,
                            ground_truth_stem=load_audio(ground_truth) if ground_truth else None,
                            separated_stem=original if ground_truth else None,
                            content_evaluator=self.factory.build_content_evaluator(case.language, case.reference_text),
                            singer_evaluator=self.factory.build_singer_evaluator(),
                            pipeline_seconds=conversion.processing_time_seconds,
                            provider_process=conversion.provider_process,
                        )
                        evaluation_id = self.store.put_json(evaluation.model_dump(mode="json"), f"benchmark-{run.run_id}", f"evaluation-{index}.json")
                        conversion.artifacts["evaluation_report.json"] = evaluation_id
                        finalize_manifest(conversion, self.store, benchmark_run_id=run.run_id,
                            benchmark_case_id=case.case_id, benchmark_configuration_id=configuration.configuration_id,
                            dataset_id=case.dataset_id, dataset_version=case.dataset_version)
                        result.conversion, result.evaluation = conversion, evaluation
                        result.status = "succeeded"
                        result.warnings.extend(conversion.warnings)
                    except InterruptedError:
                        self._save(run)
                        raise
                    except BackendUnavailableError as exc:
                        result.status = "unsupported"
                        result.warnings.append(exc.code)
                    except ConfigurationError as exc:
                        result.status = "not_tested"
                        result.warnings.append(exc.code)
                    except Exception:
                        logger.exception("Benchmark case failed: %s / %s", case.case_id, configuration.configuration_id)
                        result.status = "failed"
                        result.warnings.append("Case execution failed; inspect worker logs.")
                    result.processing_time_seconds = time.perf_counter() - started
                    if result.conversion is not None:
                        converter = result.conversion.executions.get("voice_converter")
                        params: dict[str, object] = {
                            "git_commit": run.git_commit, "case_id": case.case_id,
                            "dataset_id": case.dataset_id, "dataset_version": case.dataset_version,
                            "input_condition": case.input_condition.value, "random_seed": seed,
                            "provider": converter.provider if converter else configuration.voice_converter,
                            "model_version": converter.version if converter else None,
                            "checkpoint_sha256": converter.checkpoint_sha256 if converter else None,
                            "backend": converter.backend.value if converter else None,
                            "device": converter.device_name if converter else None,
                            "precision": converter.precision if converter else None,
                            "separator": result.conversion.components.get("separator"),
                            "vocal_processing_profile": configuration.vocal_processing_profile,
                            "pitch_extractor": self.factory.config.components.pitch_extractor.provider,
                            "sample_rate": result.source_sample_rate,
                        }
                        tracked_metrics = {"runtime_seconds": result.processing_time_seconds}
                        if result.evaluation:
                            tracked_metrics.update({f"{family}.{name}": metric.value
                                for family, metrics in result.evaluation.families.items()
                                for name, metric in metrics.items() if metric.status == "measured" and metric.value is not None})
                        result.mlflow_run_id, tracking_warnings = track_optional_run(
                            self.factory.config.mlflow_tracking_uri, self.store.root,
                            f"{spec.name}-{case.case_id}-{configuration.configuration_id}-{seed}", params, tracked_metrics,
                            {"config.json": run.config_snapshot, "result.json": result.model_dump(mode="json")},
                        )
                        result.warnings.extend(tracking_warnings)
                    run.results.append(result)
                    self._save(run)
                    notify(1, "complete")
        return run

    def _save(self, run: BenchmarkRun) -> None:
        run.summary = summarize_benchmark(run)
        namespace = f"benchmark-{run.run_id}"
        json_id = self.store.put_json(run.model_dump(mode="json", exclude={"artifacts"}), namespace, "benchmark.json")
        report_dir = self.store.root / "benchmark-reports" / run.run_id
        report_dir.mkdir(parents=True, exist_ok=True)
        report = report_dir / "benchmark.html"
        report.write_text(render_benchmark_report(run), encoding="utf-8")
        run.artifacts = {"benchmark.json": json_id, "benchmark.html": self.store.put_file(report, namespace, report.name)}


def summarize_benchmark(run: BenchmarkRun) -> list[dict[str, Any]]:
    configurations = [item.configuration_id for item in run.spec.configurations]
    groups: dict[str, dict[str, dict[tuple[str, int, str, str], float]]] = {}
    counts: dict[str, dict[str, dict[str, int]]] = {}
    for row in run.results:
        if row.evaluation is None:
            continue
        for family, metrics in row.evaluation.families.items():
            for name, metric in metrics.items():
                definition = json.dumps({"metric": f"{family}.{name}", "unit": metric.unit,
                    "input_condition": row.input_condition.value,
                    "evaluator": metric.evaluator.model_dump(mode="json") if metric.evaluator else None}, sort_keys=True)
                group = groups.setdefault(definition, {})
                statuses = counts.setdefault(definition, {}).setdefault(row.configuration_id, {})
                statuses[metric.status] = statuses.get(metric.status, 0) + 1
                if row.status == "succeeded" and metric.status == "measured" and metric.value is not None:
                    group.setdefault(row.configuration_id, {})[(row.case_id, row.random_seed, row.source_sha256, row.reference_sha256)] = metric.value
    summaries = []
    for definition, group in sorted(groups.items()):
        aggregates, paired = [], []
        for configuration in configurations:
            values = list(group.get(configuration, {}).values())
            aggregates.append({"configuration_id": configuration, "measured_count": len(values),
                "mean": statistics.mean(values) if values else None,
                "median": statistics.median(values) if values else None,
                "sample_standard_deviation": statistics.stdev(values) if len(values) > 1 else None,
                "observed_status_counts": counts[definition].get(configuration, {})})
        for left, right in combinations(configurations, 2):
            a, b = group.get(left, {}), group.get(right, {})
            common = sorted(a.keys() & b.keys())
            paired.append({"left": left, "right": right, "common_count": len(common),
                "case_seed_pairs": [{"case_id": key[0], "seed": key[1]} for key in common],
                "left_mean": statistics.mean(a[key] for key in common) if common else None,
                "right_mean": statistics.mean(b[key] for key in common) if common else None,
                "mean_difference_left_minus_right": statistics.mean(a[key] - b[key] for key in common) if common else None})
        summaries.append({"definition": json.loads(definition), "configurations": aggregates, "paired": paired,
            "limitation": "Descriptive case-seed averages. Seeds are not independent recordings; no significance claim."})
    return summaries


def render_benchmark_report(run: BenchmarkRun) -> str:
    rows = []
    for result in run.results:
        values = []
        if result.evaluation:
            for family, metrics in result.evaluation.families.items():
                for name, metric in metrics.items():
                    value = f"{metric.value:.6g}" if metric.value is not None else metric.status
                    values.append(f"<li>{html.escape(family + '.' + name)}: {html.escape(value)}</li>")
        rows.append(
            f"<tr><td>{html.escape(result.case_id)}</td><td>{html.escape(result.configuration_id)}</td>"
            f"<td>{html.escape(result.input_condition.value)}</td><td>{result.random_seed}</td>"
            f"<td>{result.status}</td><td>{result.processing_time_seconds:.3f}</td>"
            f"<td><ul>{''.join(values) or '<li>not_measured</li>'}</ul></td></tr>"
        )
    return (
        "<!doctype html><html lang='en'><meta charset='utf-8'><title>NSVP benchmark</title>"
        "<style>body{font-family:system-ui;margin:2rem}table{border-collapse:collapse;width:100%}"
        "td,th{padding:.6rem;border:1px solid #ccc;text-align:left}td{vertical-align:top}</style>"
        f"<h1>{html.escape(run.name)}</h1><p>Run {run.run_id}</p>"
        "<p>Synthetic backends test orchestration only. Missing measurements are not quality scores. "
        "Spectral differences are not a complete perceptual quality measure.</p>"
        "<table><thead><tr><th>Case</th><th>Configuration</th><th>Input condition</th><th>Seed</th>"
        "<th>Status</th><th>Seconds</th><th>Measurements</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        f"<h2>Common-case summaries</h2><pre>{html.escape(json.dumps(run.summary, indent=2))}</pre></html>"
    )
