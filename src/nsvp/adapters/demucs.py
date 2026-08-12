from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

from ..audio.io import load_audio, save_audio
from ..audio.processing import match_length, resample_audio
from ..contracts import AudioBuffer, StemSet
from ..errors import DependencyUnavailableError, NSVPError


class DemucsSeparator:
    name = "demucs-htdemucs"

    def __init__(self, model: str = "htdemucs", device: str = "cpu") -> None:
        self.model = model
        self.device = device

    def separate(self, song: AudioBuffer, work_dir: Path) -> StemSet:
        if importlib.util.find_spec("demucs") is None:
            raise DependencyUnavailableError("Demucs is unavailable; install the 'separation' dependency group")
        work_dir.mkdir(parents=True, exist_ok=True)
        input_path = work_dir / "demucs-input.wav"
        output_root = work_dir / "demucs-output"
        save_audio(input_path, song)
        command = [
            sys.executable,
            "-m",
            "demucs.separate",
            "--two-stems=vocals",
            "-n",
            self.model,
            "-d",
            self.device,
            "-o",
            str(output_root),
            str(input_path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode:
            raise NSVPError(f"Demucs separation failed: {completed.stderr[-2000:]}")
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

