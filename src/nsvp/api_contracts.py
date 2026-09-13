from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import BackendName, InputCondition


class APIRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DatasetJobRequest(APIRequest):
    source_artifact_ids: list[str] = Field(min_length=1)
    singer_name: str = Field(min_length=1, pattern=r"^[^/\\:\x00]+$")


class ConversionJobRequest(APIRequest):
    song_artifact_id: str | None = None
    reference_artifact_id: str
    output_name: str = Field(min_length=1)
    transpose_semitones: int = Field(default=0, ge=-12, le=12)
    separator: str | None = None
    separator_profile: str | None = None
    voice_converter: str | None = None
    converter_profile: str | None = None
    vocal_processing_profile: str | None = None
    model_profile: str | None = None
    backend: BackendName = BackendName.AUTO
    precision: Literal["auto", "fp32", "fp16"] = "auto"
    keep_intermediates: bool = True
    random_seed: int = Field(default=42, ge=0, le=2**32 - 1)
    input_kind: Literal["song", "vocal"] = "song"
    input_condition: InputCondition = InputCondition.UNKNOWN
    instrumental_artifact_id: str | None = None
    preparation_manifest_artifact_id: str | None = None
    selected_source_id: str | None = None

    @model_validator(mode="after")
    def source_selection(self) -> ConversionJobRequest:
        if self.preparation_manifest_artifact_id:
            if self.song_artifact_id or not self.selected_source_id:
                raise ValueError("prepared conversion requires selected_source_id and no song_artifact_id")
            if self.instrumental_artifact_id:
                raise ValueError("prepared conversion uses the preparation's instrumental")
            if self.separator or self.separator_profile or self.vocal_processing_profile:
                raise ValueError("prepared conversion already includes separation and vocal preprocessing")
        elif not self.song_artifact_id or self.selected_source_id:
            raise ValueError("conversion requires a source artifact or an explicit prepared-source selection")
        return self


class SeparationJobRequest(APIRequest):
    song_artifact_id: str
    separator: str | None = None
    separator_profile: str | None = None
    backend: BackendName = BackendName.AUTO


class VocalPreparationRequest(APIRequest):
    source_artifact_id: str
    input_kind: Literal["song", "vocal"] = "vocal"
    input_condition: InputCondition = InputCondition.UNKNOWN
    separator: str | None = None
    separator_profile: str | None = None
    vocal_processing_profile: str | None = None
    multi_singer_separator: str | None = None
    backend: BackendName = BackendName.AUTO


class SourceArtifact(BaseModel):
    source_id: str
    artifact_id: str
    label: str | None = None


class VocalPreparationResult(BaseModel):
    preparation_id: str
    input_condition: InputCondition
    original_source_artifact_id: str
    instrumental_artifact_id: str | None = None
    sources: list[SourceArtifact] = Field(min_length=1)
    artifacts: dict[str, str]
    components: dict[str, str]

    @model_validator(mode="after")
    def distinct_sources(self) -> VocalPreparationResult:
        if len({source.source_id for source in self.sources}) != len(self.sources):
            raise ValueError("prepared sources must have unique identifiers")
        return self


class TrainingJobRequest(APIRequest):
    manifest_artifact_id: str
    training_config_artifact_id: str
    run_name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    resume_artifact_id: str | None = None
    provider: str = "seed_vc"


class EvaluationJobRequest(APIRequest):
    source_artifact_id: str
    output_artifact_id: str
    reference_artifact_id: str | None = None
    ground_truth_stem_artifact_id: str | None = None
    separated_stem_artifact_id: str | None = None
    input_condition: InputCondition = InputCondition.UNKNOWN
    transpose_semitones: int = Field(default=0, ge=-12, le=12)
