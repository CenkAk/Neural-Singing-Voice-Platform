from __future__ import annotations

import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .api_contracts import (
    ConversionJobRequest,
    SourceArtifact,
    VocalPreparationRequest,
    VocalPreparationResult,
)
from .audio.io import load_audio, save_audio
from .components import ComponentFactory
from .contracts import AudioBuffer, ConversionRequest, VocalSource
from .errors import ConfigurationError
from .preprocessing import process_vocal
from .storage import LocalArtifactStore


def prepare_vocal(
    request: VocalPreparationRequest, components: ComponentFactory, store: LocalArtifactStore,
    progress: Callable[[float, str], None],
) -> VocalPreparationResult:
    preparation_id = uuid.uuid4().hex
    namespace = f"vocal-preparation-{preparation_id}"
    original = store.resolve(request.source_artifact_id)
    with tempfile.TemporaryDirectory(prefix="nsvp-vocal-") as directory:
        work = Path(directory)
        progress(0.1, "loading_audio")
        audio = load_audio(original)
        artifacts: dict[str, str] = {}
        used: dict[str, str] = {}

        def persist(name: str, value: AudioBuffer) -> str:
            path = work / name
            save_audio(path, value)
            artifact_id = store.put_file(path, namespace, name)
            artifacts[name] = artifact_id
            return artifact_id

        instrumental_id = None
        if request.input_kind == "song":
            selection = ConversionRequest(
                song_path=original, target_reference_path=original, output_name=preparation_id,
                separator=request.separator, separator_profile=request.separator_profile, backend=request.backend,
            )
            separator, _ = components.build_separator(selection)
            stems = separator.separate(audio, work / "separation")
            audio = stems.vocals
            instrumental_id = persist("instrumental.wav", stems.instrumental)
            used["separator"] = separator.name
        persist("source_vocal.wav", audio)
        progress(0.4, "processing_vocal")
        stages = components.build_vocal_preprocessors(request.vocal_processing_profile)
        processed = process_vocal(audio, stages, work / "processing")
        for name, intermediate in processed.intermediates.items():
            persist(name, intermediate)
        used["vocal_preprocessors"] = ",".join(stage.name for stage in stages)
        sources = [VocalSource(source_id="source-1", audio=processed.selected)]
        if request.multi_singer_separator:
            separator_multi = components.build_multi_singer_separator(request.multi_singer_separator)
            progress(0.6, "separating_singers")
            sources = separator_multi.separate_singers(processed.selected, work / "singers")
            used["multi_singer_separator"] = separator_multi.name
        if not sources or len({source.source_id for source in sources}) != len(sources):
            raise ConfigurationError("multi-singer output requires distinct source identifiers")
        saved_sources = [SourceArtifact(source_id=source.source_id, label=source.label,
            artifact_id=persist(f"source_{index:02d}.wav", source.audio)) for index, source in enumerate(sources)]
        result = VocalPreparationResult(
            preparation_id=preparation_id, input_condition=request.input_condition,
            original_source_artifact_id=request.source_artifact_id,
            instrumental_artifact_id=instrumental_id, sources=saved_sources, artifacts=artifacts, components=used,
        )
        manifest_id = store.put_json(result.model_dump(mode="json"), namespace, "vocal_preparation.json")
        result.artifacts["vocal_preparation.json"] = manifest_id
        return result


def resolve_conversion_request(payload: dict[str, Any], store: LocalArtifactStore) -> ConversionRequest:
    if "song_path" in payload:
        # Previously queued local jobs and CLI payloads may use paths internally.
        return ConversionRequest.model_validate(payload)
    request = ConversionJobRequest.model_validate(payload)
    instrumental = store.resolve(request.instrumental_artifact_id) if request.instrumental_artifact_id else None
    prepared = None
    if request.preparation_manifest_artifact_id:
        manifest = store.resolve(request.preparation_manifest_artifact_id)
        prepared = VocalPreparationResult.model_validate_json(manifest.read_text(encoding="utf-8"))
        selected = next((source for source in prepared.sources if source.source_id == request.selected_source_id), None)
        if selected is None:
            raise ConfigurationError("selected source does not belong to the preparation")
        song = store.resolve(selected.artifact_id)
        instrumental = store.resolve(prepared.instrumental_artifact_id) if prepared.instrumental_artifact_id else None
    elif request.song_artifact_id:
        song = store.resolve(request.song_artifact_id)
    else:
        raise ConfigurationError("source artifact is missing")
    choices = request.model_dump(exclude={"song_artifact_id", "reference_artifact_id", "instrumental_artifact_id", "preparation_manifest_artifact_id", "selected_source_id"})
    if prepared is not None:
        choices.update({"input_kind": "vocal", "input_condition": prepared.input_condition,
            "preparation_manifest_artifact_id": request.preparation_manifest_artifact_id,
            "selected_source_id": request.selected_source_id})
    return ConversionRequest.model_validate({**choices, "song_path": song,
        "target_reference_path": store.resolve(request.reference_artifact_id), "instrumental_path": instrumental})
