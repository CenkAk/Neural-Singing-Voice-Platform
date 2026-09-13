from __future__ import annotations

import json
import shutil
import time
import uuid
from collections.abc import Callable, Sequence

import numpy as np

from .audio.io import load_audio, save_audio
from .audio.processing import analyze_audio, mix_audio, remove_dc
from .contracts import AudioBuffer, ComponentExecution, ConversionRequest, ConversionResult, StemSet
from .interfaces import ConversionDiagnostics, SourceSeparator, VocalPreprocessor, VoiceConverter
from .preprocessing import process_vocal
from .storage import LocalArtifactStore


class ConversionPipeline:
    def __init__(
        self, separator: SourceSeparator | None, converter: VoiceConverter, store: LocalArtifactStore,
        *, preprocessors: Sequence[VocalPreprocessor] = (),
        executions: dict[str, ComponentExecution] | None = None,
    ) -> None:
        self.separator = separator
        self.converter = converter
        self.store = store
        self.preprocessors = preprocessors
        self.executions = executions or {}

    def run(
        self, request: ConversionRequest, progress: Callable[[float, str], None] | None = None,
    ) -> ConversionResult:
        started = time.perf_counter()
        conversion_id = uuid.uuid4().hex
        work_dir = self.store.root / "work" / conversion_id
        output_dir = self.store.root / "conversions" / conversion_id
        work_dir.mkdir(parents=True, exist_ok=False)
        output_dir.mkdir(parents=True, exist_ok=False)
        warnings = [warning for execution in self.executions.values() for warning in execution.warnings]

        def notify(value: float, stage: str) -> None:
            if progress is not None:
                progress(value, stage)

        try:
            notify(0.1, "loading_audio")
            song = load_audio(request.song_path)
            target = load_audio(request.target_reference_path)
            notify(0.2, "separating_stems")
            if request.input_kind == "vocal":
                instrumental = load_audio(request.instrumental_path) if request.instrumental_path else AudioBuffer(
                    waveform=np.zeros_like(song.waveform), sample_rate=song.sample_rate,
                )
                stems = StemSet(vocals=song, instrumental=instrumental)
            else:
                if self.separator is None:
                    raise ValueError("song conversion requires a separator")
                stems = self.separator.separate(song, work_dir / "separation")
            notify(0.4, "processing_vocal")
            processed = process_vocal(stems.vocals, self.preprocessors, work_dir / "preprocessing")
            warnings.extend(processed.warnings)
            notify(0.5, "converting_voice")
            converted_raw = self.converter.convert(processed.selected, target, request.transpose_semitones, work_dir / "conversion")
            notify(0.8, "postprocessing_and_mixing")
            converted_processed = remove_dc(converted_raw)
            final_mix, headroom_db = mix_audio(converted_processed, stems.instrumental)
            if headroom_db < 0:
                warnings.append(f"Applied {headroom_db:.2f} dB common headroom to prevent clipping")
            named_audio = {
                "source_vocal.wav": stems.vocals,
                "instrumental.wav": stems.instrumental,
                "converted_vocal_raw.wav": converted_raw,
                "converted_vocal_processed.wav": converted_processed,
                "final_mix.wav": final_mix,
            }
            if request.keep_intermediates:
                named_audio.update(processed.intermediates)
            if processed.intermediates:
                # The selected converter input is required for reproducible evaluation.
                named_audio["selected_vocal.wav"] = processed.selected
            artifacts: dict[str, str] = {}
            if isinstance(self.converter, ConversionDiagnostics):
                for name, path in self.converter.diagnostic_artifacts.items():
                    if path.is_file():
                        artifacts[name] = self.store.put_file(path, f"conversion-{conversion_id}", name)
            for name, audio in named_audio.items():
                path = output_dir / name
                save_audio(path, audio)
                artifacts[name] = self.store.put_file(path, f"conversion-{conversion_id}", name)
            elapsed = time.perf_counter() - started
            report = {
                "conversion_id": conversion_id,
                "input_duration_seconds": song.duration_seconds,
                "target_voice": request.model_profile or request.model_name,
                "model_profile": request.model_profile,
                "model_version": request.model_version,
                "separator": self.separator.name if self.separator else "none",
                "voice_converter": self.converter.name,
                "vocal_preprocessors": [stage.name for stage in self.preprocessors],
                "input_condition": request.input_condition.value,
                "preparation_manifest_artifact_id": request.preparation_manifest_artifact_id,
                "selected_source_id": request.selected_source_id,
                "random_seed": request.random_seed,
                "executions": {key: value.model_dump(mode="json") for key, value in self.executions.items()},
                "transpose_semitones": request.transpose_semitones,
                "processing_time_seconds": elapsed,
                "final_quality": analyze_audio(final_mix).as_dict(),
                "warnings": warnings,
            }
            report_path = output_dir / "conversion_report.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            artifacts[report_path.name] = self.store.put_file(report_path, f"conversion-{conversion_id}", report_path.name)
            return ConversionResult(
                conversion_id=conversion_id,
                artifacts=artifacts,
                warnings=warnings,
                processing_time_seconds=elapsed,
                components={"separator": self.separator.name if self.separator else "none", "voice_converter": self.converter.name},
                executions=self.executions,
                input_condition=request.input_condition,
            )
        finally:
            # Preserve failed model output for diagnosis before deleting disposable work.
            if isinstance(self.converter, ConversionDiagnostics):
                for name, path in self.converter.diagnostic_artifacts.items():
                    if path.is_file():
                        self.store.put_file(path, f"conversion-{conversion_id}", name)
            if not request.keep_intermediates:
                shutil.rmtree(work_dir, ignore_errors=True)
