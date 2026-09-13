from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field

from ..audio.io import load_audio, save_audio
from ..audio.processing import match_length, preprocess_audio, resample_audio
from ..contracts import AudioBuffer
from ..errors import ConfigurationError, DependencyUnavailableError, NSVPError
from .external import offline_environment, python_executable, run_external, validate_revision


class SeedVCSettings(BaseModel):
    repository_root: Path
    checkpoint_path: Path
    config_path: Path
    diffusion_steps: int = 30
    fp16: bool = False
    python_executable: Path | None = None
    device: str = "auto"
    random_seed: int = Field(default=42, ge=0, le=2**32 - 1)
    timeout_seconds: float = Field(default=1800, gt=0)
    revision: str | None = None
    huggingface_cache: Path | None = None


class SeedVCConverter:
    """Pinned external Seed-VC adapter; it never permits implicit model selection/download."""

    name = "seed-vc-v1-svc"
    sample_rate = 44_100

    def __init__(self, settings: SeedVCSettings) -> None:
        self.settings = settings
        self.diagnostic_artifacts: dict[str, Path] = {}

    def validate_installation(self) -> None:
        python_executable(self.settings.python_executable)
        root = self.settings.repository_root.resolve()
        validate_revision(root, self.settings.revision)
        if not (root / "inference.py").is_file():
            raise DependencyUnavailableError(f"Seed-VC inference.py not found under {root}")
        for label, path in (("checkpoint", self.settings.checkpoint_path), ("config", self.settings.config_path)):
            if not path.resolve().is_file():
                raise ConfigurationError(f"Seed-VC {label} does not exist: {path}")

    def convert(self, source_vocal: AudioBuffer, target_reference: AudioBuffer, semitones: int, work_dir: Path) -> AudioBuffer:
        self.validate_installation()
        work_dir = work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        source_path = work_dir / "seed-source.wav"
        target_path = work_dir / "seed-target.wav"
        output_dir = work_dir / "seed-output"
        output_dir.mkdir(exist_ok=True)
        prepared_source = preprocess_audio(source_vocal, self.sample_rate, mono=True)
        prepared_target = preprocess_audio(target_reference, self.sample_rate, mono=True)
        save_audio(source_path, prepared_source)
        save_audio(target_path, prepared_target)
        command = [
            str(python_executable(self.settings.python_executable)),
            str(Path(__file__).parent / "runners" / "seed_vc.py"),
            "--source",
            str(source_path),
            "--target",
            str(target_path),
            "--output",
            str(output_dir),
            "--diffusion-steps",
            str(self.settings.diffusion_steps),
            "--f0-condition",
            "True",
            "--auto-f0-adjust",
            "False",
            "--semi-tone-shift",
            str(semitones),
            "--checkpoint",
            str(self.settings.checkpoint_path.resolve()),
            "--config",
            str(self.settings.config_path.resolve()),
            "--fp16",
            str(self.settings.fp16),
            "--device", self.settings.device,
            "--seed", str(self.settings.random_seed),
        ]
        run_external(
            command,
            cwd=self.settings.repository_root,
            timeout=self.settings.timeout_seconds,
            env=offline_environment(self.settings.huggingface_cache),
            label="Seed-VC conversion",
        )
        outputs = sorted(output_dir.glob("*.wav"), key=lambda path: path.stat().st_mtime)
        if not outputs:
            raise NSVPError("Seed-VC completed without a WAV output")
        converted = load_audio(outputs[-1])
        alignment = work_dir / "alignment.json"
        alignment.write_text(json.dumps({
            "raw_duration_seconds": converted.duration_seconds,
            "source_duration_seconds": source_vocal.duration_seconds,
            "adjustment_seconds": converted.duration_seconds - source_vocal.duration_seconds,
            "policy": "V1 length matching after resampling; original model output preserved separately",
        }, indent=2), encoding="utf-8")
        self.diagnostic_artifacts = {"model_output_original.wav": outputs[-1], "model_alignment.json": alignment}
        converted = resample_audio(converted, source_vocal.sample_rate)
        return match_length(converted, source_vocal.samples)
