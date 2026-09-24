from __future__ import annotations

from pathlib import Path

from ..audio.io import load_audio, save_audio
from ..audio.processing import match_length, resample_audio
from ..contracts import AudioBuffer, StemSet
from ..errors import ConfigurationError, NSVPError
from .external import python_executable, run_external


class DemucsSeparator:
    name = "demucs-htdemucs"

    def __init__(
        self, model: str = "htdemucs", device: str = "cpu", *,
        python: Path | None = None, model_repository: Path | None = None,
        timeout_seconds: float = 1800,
    ) -> None:
        self.model = model
        self.device = device
        self.python = python
        self.model_repository = model_repository
        self.timeout_seconds = timeout_seconds
        self.name = f"demucs-{model}"

    def separate(self, song: AudioBuffer, work_dir: Path) -> StemSet:
        if self.model_repository is None or not self.model_repository.is_dir():
            raise ConfigurationError("Demucs requires an explicit local model_repository; downloads are disabled")
        work_dir = work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        input_path = work_dir / "demucs-input.wav"
        output_root = work_dir / "demucs-output"
        save_audio(input_path, song)
        command = [
            str(python_executable(self.python)),
            "-m",
            "demucs.separate",
            "--two-stems=vocals",
            "-n",
            self.model,
            "--repo",
            str(self.model_repository.resolve()),
            "-d",
            self.device,
            "-o",
            str(output_root),
            str(input_path),
        ]
        run_external(command, cwd=work_dir, timeout=self.timeout_seconds, label="Demucs separation", device=self.device)
        stem_dir = output_root / self.model / input_path.stem
        vocals_path = stem_dir / "vocals.wav"
        instrumental_path = stem_dir / "no_vocals.wav"
        if not vocals_path.is_file() or not instrumental_path.is_file():
            raise NSVPError("Demucs completed without the expected vocals/no_vocals artifacts")
        vocals = self._align(load_audio(vocals_path), song)
        instrumental = self._align(load_audio(instrumental_path), song)
        return StemSet(vocals=vocals, instrumental=instrumental)

    @staticmethod
    def _align(stem: AudioBuffer, song: AudioBuffer) -> AudioBuffer:
        stem = resample_audio(stem, song.sample_rate)
        return match_length(stem, song.samples)
