from __future__ import annotations

import hashlib
import sys
import tempfile
import unicodedata
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ..audio.io import save_audio
from ..config import LocalEvaluatorConfig
from ..contracts import AudioBuffer, EvaluatorMetadata, MetricResult
from ..errors import ConfigurationError
from ..execution import check_cancelled
from ..storage import sha256_file
from .external import offline_environment, python_executable, run_external


def normalize_text(text: str, language: str | None) -> str:
    text = unicodedata.normalize("NFC", text)
    if language == "tr":
        text = text.translate(str.maketrans("Iİ", "ıi"))
    text = text.lower()
    return " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in text).split())


def error_rate(reference: list[str], hypothesis: list[str]) -> float | None:
    if not reference:
        return None
    previous = list(range(len(hypothesis) + 1))
    for i, expected in enumerate(reference, 1):
        check_cancelled()
        current = [i]
        for j, actual in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (expected != actual)))
        previous = current
    return previous[-1] / len(reference)


class EvaluatorOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_text: str | None = Field(default=None, max_length=20000)
    output_text: str | None = Field(default=None, max_length=20000)
    similarity: float | None = Field(default=None, ge=-1, le=1, allow_inf_nan=False)
    library_version: str


def model_directory_digest(directory: Path) -> str:
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    if not files:
        raise ConfigurationError("Evaluator model directory is empty")
    digest = hashlib.sha256()
    for path in files:
        check_cancelled()
        digest.update(path.relative_to(directory).as_posix().encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def evaluate_external(settings: LocalEvaluatorConfig, kind: str, source: AudioBuffer, output: AudioBuffer,
                      language: str | None) -> tuple[EvaluatorOutput, str]:
    if settings.python_executable is None:
        raise ConfigurationError("Evaluator requires an explicit Python executable in a separate environment")
    executable = python_executable(settings.python_executable)
    if executable == python_executable(Path(sys.executable)):
        raise ConfigurationError("Evaluator Python must be separate from the core environment")
    directory = settings.model_directory
    if directory is None or not directory.is_dir() or not settings.revision or not settings.model_name:
        raise ConfigurationError("Evaluator requires reviewed local model files, model name and revision")
    digest = model_directory_digest(directory)
    with tempfile.TemporaryDirectory(prefix="nsvp-evaluator-") as temporary:
        root = Path(temporary)
        save_audio(root / "source.wav", source, subtype="FLOAT")
        save_audio(root / "output.wav", output, subtype="FLOAT")
        command = [str(executable),
            str(Path(__file__).parent / "runners/evaluate.py"), "--kind", kind,
            "--model", str(directory.resolve()), "--source", str(root / "source.wav"),
            "--output", str(root / "output.wav"), "--result", str(root / "result.json")]
        if language:
            command.extend(["--language", language])
        run_external(command, cwd=root, timeout=settings.timeout_seconds,
            env=offline_environment(), label=f"Local {kind} evaluator", device="cpu")
        result = EvaluatorOutput.model_validate_json((root / "result.json").read_text(encoding="utf-8"))
    return result, digest


class WhisperContentEvaluator:
    def __init__(self, settings: LocalEvaluatorConfig, language: str | None = None, reference_text: str | None = None) -> None:
        self.settings, self.language, self.reference_text = settings, language, reference_text
        self.metadata = EvaluatorMetadata(name="faster-whisper", version=settings.revision or "unknown",
            domain="speech", model=settings.model_name, language=language,
            reference_kind="verified_lyrics" if reference_text is not None else "source_asr_proxy",
            normalization_version="nsvp-text-1",
            limitations=["Speech ASR can misrecognize singing; WER/CER do not establish perceptual quality."])
        if reference_text is None:
            self.metadata.limitations.append("Source-ASR comparison measures transcript agreement, not independently verified lyrics.")

    def evaluate(self, source: AudioBuffer, output: AudioBuffer) -> dict[str, MetricResult]:
        result, digest = evaluate_external(self.settings, "content", source, output, self.language)
        self.metadata.checkpoint_sha256 = digest
        self.metadata.library_version = result.library_version
        if result.output_text is None or (self.reference_text is None and result.source_text is None):
            raise ConfigurationError("Content evaluator omitted its transcription")
        reference = normalize_text(self.reference_text if self.reference_text is not None else result.source_text or "", self.language)
        candidate = normalize_text(result.output_text, self.language)
        scores = {"wer": error_rate(reference.split(), candidate.split()),
            "cer": error_rate(list(reference.replace(" ", "")), list(candidate.replace(" ", "")))}
        return {name: MetricResult(value=value, status="measured", evaluator=self.metadata) if value is not None
            else MetricResult(reason="Reference transcription is empty.", evaluator=self.metadata)
            for name, value in scores.items()}


class EcapaSingerEvaluator:
    def __init__(self, settings: LocalEvaluatorConfig) -> None:
        self.settings = settings
        self.metadata = EvaluatorMetadata(name="speechbrain-ecapa", version=settings.revision or "unknown",
            model=settings.model_name, domain="speech",
            limitations=["VoxCeleb speech embeddings are not validated as singing identity ground truth.",
                "Cosine similarity is not a calibrated probability or identity verification decision."])

    def evaluate(self, reference: AudioBuffer, output: AudioBuffer) -> MetricResult:
        result, digest = evaluate_external(self.settings, "singer", reference, output, None)
        self.metadata.checkpoint_sha256 = digest
        self.metadata.library_version = result.library_version
        if result.similarity is None:
            raise ConfigurationError("Singer evaluator omitted its similarity")
        return MetricResult(value=result.similarity, status="measured", evaluator=self.metadata)
