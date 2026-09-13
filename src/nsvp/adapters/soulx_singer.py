from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from ..audio.io import load_audio, save_audio
from ..audio.processing import match_length, resample_audio, to_mono
from ..config import SoulXProviderConfig
from ..contracts import AudioBuffer
from ..errors import AudioValidationError, ConfigurationError, NSVPError
from .external import offline_environment, python_executable, run_external, validate_revision


class SoulXSingerSettings(SoulXProviderConfig):
    repository_root: Path = Field()
    checkpoint_path: Path = Field()
    config_path: Path = Field()
    rmvpe_checkpoint_path: Path = Field()
    huggingface_cache: Path = Field()
    revision: str = Field(pattern=r"^[a-fA-F0-9]{40}$")
    whisper_revision: str = Field(pattern=r"^[a-fA-F0-9]{40}$")
    device: str = "cpu"
    random_seed: int = Field(default=42, ge=0, le=2**32 - 1)


class SoulXRuntimeMetadata(BaseModel):
    device: str
    precision: str
    sample_rate: int = Field(gt=0)
    hop_size: int = Field(gt=0)
    random_seed: int
    source_f0_frames: int = Field(gt=0)
    reference_f0_frames: int = Field(gt=0)
    python_version: str
    torch_version: str
    parameter_dtypes: list[str] = Field(default_factory=list)


class SoulXSingerSVCConverter:
    name = "soulx-singer-svc"

    def __init__(self, settings: SoulXSingerSettings) -> None:
        self.settings = settings
        self.runtime_metadata: SoulXRuntimeMetadata | None = None
        self.diagnostic_artifacts: dict[str, Path] = {}

    def validate_installation(self) -> None:
        settings = self.settings
        python_executable(settings.python_executable)
        validate_revision(settings.repository_root, settings.revision)
        for path in (
            settings.repository_root / "cli/inference_svc.py",
            settings.repository_root / "preprocess/tools/f0_extraction.py",
            settings.checkpoint_path, settings.config_path, settings.rmvpe_checkpoint_path,
        ):
            if not path.is_file():
                raise ConfigurationError(f"SoulX required file is missing: {path}")
        whisper_root = settings.huggingface_cache / "hub/models--openai--whisper-base"
        revision_file = whisper_root / "refs/main"
        if not revision_file.is_file() or revision_file.read_text().strip() != settings.whisper_revision:
            raise ConfigurationError("local Whisper main reference must match the configured whisper_revision")
        snapshot = whisper_root / "snapshots" / settings.whisper_revision
        if not all((snapshot / name).is_file() for name in ("config.json", "preprocessor_config.json")):
            raise ConfigurationError("local Whisper snapshot is missing configuration files")
        if not any((snapshot / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
            raise ConfigurationError("local Whisper snapshot is missing model weights")

    def convert(
        self, source_vocal: AudioBuffer, target_reference: AudioBuffer,
        semitones: int, work_dir: Path,
    ) -> AudioBuffer:
        self.validate_installation()
        work_dir = work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        source_path, reference_path = work_dir / "source.wav", work_dir / "reference.wav"
        save_audio(source_path, to_mono(source_vocal))
        save_audio(reference_path, to_mono(target_reference))
        output_dir = work_dir / "output"
        if output_dir.exists():
            raise ConfigurationError("SoulX output directory must be new to prevent stale output reuse")
        settings = self.settings
        command = [
            str(python_executable(settings.python_executable)),
            str(Path(__file__).parent / "runners/soulx_singer.py"),
            "--source", str(source_path), "--reference", str(reference_path),
            "--output", str(output_dir), "--checkpoint", str(settings.checkpoint_path.resolve()),
            "--config", str(settings.config_path.resolve()),
            "--rmvpe-checkpoint", str(settings.rmvpe_checkpoint_path.resolve()),
            "--device", settings.device, "--seed", str(settings.random_seed),
            "--pitch-shift", str(semitones), "--n-steps", str(settings.n_steps),
            "--cfg", str(settings.cfg),
        ]
        if settings.use_fp16:
            command.append("--fp16")
        env = offline_environment(settings.huggingface_cache)
        env["HF_HUB_CACHE"] = str(settings.huggingface_cache.resolve() / "hub")
        run_external(command, cwd=settings.repository_root, env=env,
            timeout=settings.timeout_seconds, label="SoulX-Singer-SVC")
        output_path = output_dir / "generated.wav"
        metadata_path = output_dir / "runtime.json"
        self.diagnostic_artifacts = {
            "model_output_original.wav": output_path, "model_runtime.json": metadata_path,
            "model_alignment.json": work_dir / "alignment.json",
        }
        if not output_path.is_file():
            raise NSVPError("SoulX completed without the expected generated.wav")
        if not metadata_path.is_file():
            raise NSVPError("SoulX completed without runtime metadata")
        self.runtime_metadata = SoulXRuntimeMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
        expected_precision = "fp16" if settings.use_fp16 else "fp32"
        if self.runtime_metadata.precision != expected_precision or self.runtime_metadata.device.split(":")[0] != settings.device.split(":")[0]:
            raise NSVPError("SoulX runtime device or precision differs from the requested configuration")
        converted = load_audio(output_path)
        if converted.sample_rate != self.runtime_metadata.sample_rate:
            raise AudioValidationError("SoulX output sample rate differs from runtime metadata")
        difference = converted.duration_seconds - source_vocal.duration_seconds
        tolerance = self.runtime_metadata.hop_size / self.runtime_metadata.sample_rate
        if abs(difference) > tolerance + 1 / source_vocal.sample_rate:
            raise AudioValidationError("SoulX output duration differs by more than one model frame; raw output retained")
        (work_dir / "alignment.json").write_text(json.dumps({
            "raw_duration_seconds": converted.duration_seconds,
            "source_duration_seconds": source_vocal.duration_seconds,
            "adjustment_seconds": difference,
        }, indent=2), encoding="utf-8")
        return match_length(resample_audio(converted, source_vocal.sample_rate), source_vocal.samples)
