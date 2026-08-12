from __future__ import annotations

import json
import shutil
import time
import uuid

from .audio.io import load_audio, save_audio
from .audio.processing import analyze_audio, mix_audio, remove_dc
from .contracts import ConversionRequest, ConversionResult
from .interfaces import SourceSeparator, VoiceConverter
from .storage import LocalArtifactStore


class ConversionPipeline:
    def __init__(self, separator: SourceSeparator, converter: VoiceConverter, store: LocalArtifactStore) -> None:
        self.separator = separator
        self.converter = converter
        self.store = store

    def run(self, request: ConversionRequest) -> ConversionResult:
        started = time.perf_counter()
        conversion_id = uuid.uuid4().hex
        work_dir = self.store.root / "work" / conversion_id
        output_dir = self.store.root / "conversions" / conversion_id
        work_dir.mkdir(parents=True, exist_ok=False)
        output_dir.mkdir(parents=True, exist_ok=False)
        warnings: list[str] = []
        try:
            song = load_audio(request.song_path)
            target = load_audio(request.target_reference_path)
            stems = self.separator.separate(song, work_dir / "separation")
            converted_raw = self.converter.convert(stems.vocals, target, request.transpose_semitones, work_dir / "conversion")
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
            artifacts: dict[str, str] = {}
            for name, audio in named_audio.items():
                path = output_dir / name
                save_audio(path, audio)
                artifacts[name] = self.store.put_file(path, f"conversion-{conversion_id}", name)
            elapsed = time.perf_counter() - started
            report = {
                "conversion_id": conversion_id,
                "input_duration_seconds": song.duration_seconds,
                "target_voice": request.model_name,
                "model_version": request.model_version,
                "separator": self.separator.name,
                "voice_converter": self.converter.name,
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
                components={"separator": self.separator.name, "voice_converter": self.converter.name},
            )
        finally:
            if not request.keep_intermediates:
                shutil.rmtree(work_dir, ignore_errors=True)
