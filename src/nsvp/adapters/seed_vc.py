from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel

from ..audio.io import load_audio, save_audio
from ..audio.processing import match_length, preprocess_audio, resample_audio
from ..contracts import AudioBuffer
from ..errors import ConfigurationError, DependencyUnavailableError, NSVPError


class SeedVCSettings(BaseModel):
    repository_root: Path
    checkpoint_path: Path
    config_path: Path
    diffusion_steps: int = 30
    fp16: bool = False


class SeedVCConverter:
    """Pinned external Seed-VC adapter; it never permits implicit model selection/download."""

    name = "seed-vc-v1-svc"
    sample_rate = 44_100

    def __init__(self, settings: SeedVCSettings) -> None:
        self.settings = settings

    def validate_installation(self) -> None:
        root = self.settings.repository_root.resolve()
        if not (root / "inference.py").is_file():
            raise DependencyUnavailableError(f"Seed-VC inference.py not found under {root}")
        for label, path in (("checkpoint", self.settings.checkpoint_path), ("config", self.settings.config_path)):
            if not path.resolve().is_file():
                raise ConfigurationError(f"Seed-VC {label} does not exist: {path}")

    def convert(self, source_vocal: AudioBuffer, target_reference: AudioBuffer, semitones: int, work_dir: Path) -> AudioBuffer:
        self.validate_installation()
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
            sys.executable,
            "inference.py",
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
        ]
        completed = subprocess.run(
            command,
            cwd=self.settings.repository_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            raise NSVPError(f"Seed-VC conversion failed: {completed.stderr[-3000:]}")
        outputs = sorted(output_dir.glob("*.wav"), key=lambda path: path.stat().st_mtime)
        if not outputs:
            raise NSVPError("Seed-VC completed without a WAV output")
        converted = load_audio(outputs[-1])
        converted = resample_audio(converted, source_vocal.sample_rate)
        return match_length(converted, source_vocal.samples)

