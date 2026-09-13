from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import numpy as np

from .audio.io import SUPPORTED_EXTENSIONS, load_audio, save_audio
from .audio.processing import analyze_audio, preprocess_audio
from .audio.segmentation import find_segments
from .config import AudioConfig, DatasetAuditConfig
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
        audit_config: DatasetAuditConfig | None = None,
    ) -> None:
        self.store = store
        self.config = audio_config
        self.pitch_extractor = pitch_extractor or AutocorrelationPitchExtractor()
        self.audit = audit_config or DatasetAuditConfig()

    def prepare(self, source_root: Path, singer_name: str) -> DatasetManifest:
        files = discover_audio(source_root)
        if not files:
            raise ValueError(f"no supported audio found under {source_root}")
        return self._prepare_files(files, singer_name)

    def prepare_artifacts(self, artifact_ids: list[str], singer_name: str) -> DatasetManifest:
        if not artifact_ids:
            raise ValueError("dataset requires at least one audio artifact")
        files = sorted({self.store.resolve(artifact_id) for artifact_id in artifact_ids})
        return self._prepare_files(files, singer_name)

    def _prepare_files(self, files: list[Path], singer_name: str) -> DatasetManifest:
        if not singer_name or singer_name in {".", ".."} or any(c in singer_name for c in "/\\:\0"):
            raise ValueError("singer name must be a safe identifier")
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
                if self.audit.enabled:
                    times = pitch.timestamps.astype(np.float64)
                    if times.size:
                        ends = np.append(times[1:], segment.duration_seconds)
                        spans = np.maximum(0, np.minimum(ends, segment.duration_seconds) - times)
                        quality["voiced_duration_seconds"] = float(np.sum(spans[pitch.voiced.astype(bool)]))
                    else:
                        quality["voiced_duration_seconds"] = 0.0
                    quality["audit_silence_ratio"] = analyze_audio(segment, self.config.clipping_threshold, self.audit.silence_threshold_db).silence_ratio
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
        if self.audit.enabled:
            analysis["audit"] = dataset_audit(records, observed_f0, self.audit)
        return DatasetManifest(
            dataset_id=f"{singer_name}-{version}",
            version=version,
            singer_name=singer_name,
            segments=records,
            source_files=sorted(sources),
            config=config,
            analysis=analysis,
        )


def dataset_audit(records: list[SegmentRecord], observed_f0: list[float], config: DatasetAuditConfig) -> dict[str, object]:
    total = sum(record.duration_seconds for record in records)
    durations = np.asarray([record.duration_seconds for record in records], dtype=np.float64)
    per_split: dict[str, float] = {"train": 0, "validation": 0, "evaluation": 0}
    source_contribution: dict[str, float] = {}
    for record in records:
        per_split[record.split] = per_split.get(record.split, 0) + record.duration_seconds
        source_contribution[record.source_sha256] = source_contribution.get(record.source_sha256, 0) + record.duration_seconds
    result: dict[str, object] = {
        "config": config.model_dump(), "total_usable_duration_seconds": total,
        "duration_per_split_seconds": per_split, "source_contribution_seconds": source_contribution,
        "voiced_duration_seconds": sum(float(record.quality.get("voiced_duration_seconds", 0)) for record in records),
        "clipping_samples": sum(int(record.quality.get("clipping_samples", 0)) for record in records),
        "clipped_segments": sum(int(record.quality.get("clipping_samples", 0)) > 0 for record in records),
        "silence_ratio": sum(record.duration_seconds * float(record.quality.get("audit_silence_ratio", 0)) for record in records) / total if total else None,
        "segment_duration_percentiles_seconds": np.percentile(durations, [0, 25, 50, 75, 100]).tolist() if durations.size else [],
        "pitch_percentiles_hz": None, "register_coverage": None,
    }
    warnings: list[str] = []
    values = np.asarray(observed_f0, dtype=np.float64)
    values = values[values > 0]
    if values.size:
        result["pitch_percentiles_hz"] = dict(zip([str(p) for p in config.pitch_percentiles], np.percentile(values, config.pitch_percentiles).tolist()))
        midi = 69 + 12 * np.log2(values / 440)
        counts, _ = np.histogram(midi, bins=config.midi_bin_edges)
        result["register_coverage"] = {
            "midi_bin_edges": config.midi_bin_edges, "voiced_frame_counts": counts.tolist(),
            "below_range_frames": int(np.count_nonzero(midi < config.midi_bin_edges[0])),
            "above_range_frames": int(np.count_nonzero(midi > config.midi_bin_edges[-1])),
            "limitation": "Pitch-bin coverage is not physiological vocal-register classification.",
        }
        width = float(np.percentile(midi, 95) - np.percentile(midi, 5))
        result["central_pitch_span_semitones"] = width
        if width < config.narrow_range_semitones:
            warnings.append("Most voiced frames occupy a narrow pitch range; inspect coverage before training.")
    else:
        warnings.append("No voiced pitch frames were detected.")
    result["warnings"] = warnings
    return result


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
        f"<tr><td>{html.escape(segment.segment_id)}</td><td>{html.escape(segment.split)}</td><td>{segment.duration_seconds:.2f}</td><td>{int(segment.quality.get('clipping_samples', 0))}</td></tr>"
        for segment in manifest.segments
    )
    document = f"""<!doctype html><html><head><meta charset='utf-8'><title>Training Data Report</title>
<style>body{{font-family:system-ui;max-width:1000px;margin:40px auto;padding:0 20px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}.metric{{display:inline-block;margin:8px;padding:16px;background:#f4f4f4;border-radius:8px}}.histogram{{height:130px;display:flex;align-items:end;gap:3px;border-bottom:1px solid #777}}.histogram span{{flex:1;background:#5a8f42;min-width:2px}}</style></head>
<body><h1>Training Data Report: {html.escape(manifest.singer_name)}</h1><p>Dataset version: <code>{html.escape(manifest.version)}</code></p>
<div class='metric'>Usable duration<br><strong>{total / 60:.2f} min</strong></div>
<div class='metric'>Segments<br><strong>{len(manifest.segments)}</strong></div>
<div class='metric'>Clipping samples<br><strong>{clipping}</strong></div>
<div class='metric'>Observed pitch range<br><strong>{pitch_text}</strong></div>
<h2>Pitch distribution</h2>{histogram_html}
<h2>Segments</h2><table><thead><tr><th>ID</th><th>Split</th><th>Duration (s)</th><th>Clipping</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Dataset audit</h2><pre>{html.escape(json.dumps(manifest.analysis.get('audit', {'status': 'not_measured'}), indent=2))}</pre>
<p>Observed range must not be interpreted as comfortable vocal range. Extractor: {html.escape(str(manifest.analysis.get('pitch_extractor', 'Not measured')))}.</p></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
