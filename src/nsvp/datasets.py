from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .audio.io import SUPPORTED_EXTENSIONS, load_audio, save_audio
from .audio.processing import analyze_audio, preprocess_audio
from .audio.segmentation import find_segments
from .config import AudioConfig
from .contracts import DatasetManifest, SegmentRecord
from .interfaces import PitchExtractor
from .pitch import AutocorrelationPitchExtractor
from .storage import LocalArtifactStore, sha256_file


def discover_audio(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)


def deterministic_split(checksum: str) -> str:
    bucket = int(checksum[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "evaluation"


class DatasetManager:
    def __init__(
        self,
        store: LocalArtifactStore,
        audio_config: AudioConfig,
        pitch_extractor: PitchExtractor | None = None,
    ) -> None:
        self.store = store
        self.config = audio_config
        self.pitch_extractor = pitch_extractor or AutocorrelationPitchExtractor()

    def prepare(self, source_root: Path, singer_name: str) -> DatasetManifest:
        files = discover_audio(source_root)
        if not files:
            raise ValueError(f"no supported audio found under {source_root}")
        records: list[SegmentRecord] = []
        sources: list[str] = []
        observed_f0: list[float] = []
        checksums = {source: sha256_file(source) for source in files}
        splits = {checksum: deterministic_split(checksum) for checksum in checksums.values()}
        if "train" not in splits.values():
            splits[checksums[files[0]]] = "train"
        for source in files:
            checksum = checksums[source]
            sources.append(checksum)
            audio = preprocess_audio(load_audio(source), self.config.training_sample_rate, mono=True)
            split = splits[checksum]
            boundaries = find_segments(
                audio,
                self.config.minimum_segment_seconds,
                self.config.target_maximum_segment_seconds,
                self.config.absolute_maximum_segment_seconds,
                self.config.minimum_silence_seconds,
                self.config.silence_threshold_db,
            )
            for index, boundary in enumerate(boundaries):
                segment = type(audio)(waveform=audio.waveform[:, boundary.start_sample : boundary.end_sample], sample_rate=audio.sample_rate)
                segment_id = f"{singer_name}-{checksum[:10]}-{index:04d}"
                temporary = self.store.root / "_staging" / f"{segment_id}.wav"
                save_audio(temporary, segment)
                artifact_id = self.store.put_file(temporary, f"datasets-{singer_name}", f"{segment_id}.wav")
                temporary.unlink(missing_ok=True)
                quality = analyze_audio(segment, self.config.clipping_threshold).as_dict()
                pitch = self.pitch_extractor.extract(segment)
                voiced_f0 = pitch.f0_hz[pitch.voiced]
                if voiced_f0.size:
                    observed_f0.extend(float(value) for value in voiced_f0)
                    quality.update(
                        {
                            "f0_min_hz": float(np.min(voiced_f0)),
                            "f0_median_hz": float(np.median(voiced_f0)),
                            "f0_max_hz": float(np.max(voiced_f0)),
                            "voiced_ratio": float(np.mean(pitch.voiced)),
                        }
                    )
                records.append(
                    SegmentRecord(
                        segment_id=segment_id,
                        source_file=source.name,
                        source_sha256=checksum,
                        start_seconds=boundary.start_sample / audio.sample_rate,
                        end_seconds=boundary.end_sample / audio.sample_rate,
                        duration_seconds=segment.duration_seconds,
                        sample_rate=segment.sample_rate,
                        split=split,
                        artifact_id=artifact_id,
                        quality=quality,
                    )
                )
        config = self.config.model_dump(mode="json")
        version_payload = json.dumps({"sources": sorted(sources), "config": config}, sort_keys=True).encode()
        version = f"dataset-{hashlib.sha256(version_payload).hexdigest()[:12]}"
        analysis: dict[str, object] = {"pitch_extractor": self.pitch_extractor.name}
        if observed_f0:
            values = np.asarray(observed_f0, dtype=np.float64)
            lower, upper = float(values.min()), float(values.max())
            edges = np.geomspace(max(lower, 1.0), max(upper, lower + 1.0), num=25)
            counts, edges = np.histogram(values, bins=edges)
            analysis.update(
                {
                    "observed_pitch_range_hz": [lower, upper],
                    "median_f0_hz": float(np.median(values)),
                    "voiced_frame_count": int(values.size),
                    "pitch_histogram": {"bin_edges_hz": edges.tolist(), "counts": counts.tolist()},
                }
            )
        return DatasetManifest(
            dataset_id=f"{singer_name}-{version}",
            version=version,
            singer_name=singer_name,
            segments=records,
            source_files=sorted(sources),
            config=config,
            analysis=analysis,
        )


def render_dataset_report(manifest: DatasetManifest, output: Path) -> None:
    durations = [segment.duration_seconds for segment in manifest.segments]
    clipping = sum(int(segment.quality.get("clipping_samples", 0)) for segment in manifest.segments)
    total = sum(durations)
    pitch_range = manifest.analysis.get("observed_pitch_range_hz")
    pitch_text = "Not measured"
    if isinstance(pitch_range, list) and len(pitch_range) == 2:
        pitch_text = f"{float(pitch_range[0]):.1f}–{float(pitch_range[1]):.1f} Hz"
    histogram = manifest.analysis.get("pitch_histogram", {})
    histogram_html = "<p>Pitch distribution not measured.</p>"
    if isinstance(histogram, dict):
        counts = histogram.get("counts", [])
        if isinstance(counts, list) and counts:
            maximum = max(int(value) for value in counts) or 1
            bars = "".join(
                f"<span title='{count} frames' style='height:{max(2, int(int(count) / maximum * 120))}px'></span>"
                for count in counts
            )
            histogram_html = f"<div class='histogram'>{bars}</div>"
    rows = "".join(
        f"<tr><td>{segment.segment_id}</td><td>{segment.split}</td><td>{segment.duration_seconds:.2f}</td><td>{int(segment.quality.get('clipping_samples', 0))}</td></tr>"
        for segment in manifest.segments
    )
    html = f"""<!doctype html><html><head><meta charset='utf-8'><title>Training Data Report</title>
<style>body{{font-family:system-ui;max-width:1000px;margin:40px auto;padding:0 20px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}.metric{{display:inline-block;margin:8px;padding:16px;background:#f4f4f4;border-radius:8px}}.histogram{{height:130px;display:flex;align-items:end;gap:3px;border-bottom:1px solid #777}}.histogram span{{flex:1;background:#5a8f42;min-width:2px}}</style></head>
<body><h1>Training Data Report: {manifest.singer_name}</h1><p>Dataset version: <code>{manifest.version}</code></p>
<div class='metric'>Usable duration<br><strong>{total / 60:.2f} min</strong></div>
<div class='metric'>Segments<br><strong>{len(manifest.segments)}</strong></div>
<div class='metric'>Clipping samples<br><strong>{clipping}</strong></div>
<div class='metric'>Observed pitch range<br><strong>{pitch_text}</strong></div>
<h2>Pitch distribution</h2>{histogram_html}
<h2>Segments</h2><table><thead><tr><th>ID</th><th>Split</th><th>Duration (s)</th><th>Clipping</th></tr></thead><tbody>{rows}</tbody></table>
<p>Observed range must not be interpreted as comfortable vocal range. Extractor: {manifest.analysis.get('pitch_extractor', 'Not measured')}.</p></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
