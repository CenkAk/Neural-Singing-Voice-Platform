"""Local evaluator entrypoint, executed by its own configured Python environment."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["content", "singer"], required=True)
    for name in ("model", "source", "output", "result"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--language", choices=["tr", "en"])
    args = parser.parse_args()
    result: dict[str, str | float]
    if args.kind == "content":
        whisper = importlib.import_module("faster_whisper")
        model = whisper.WhisperModel(args.model, device="cpu", compute_type="int8", local_files_only=True)
        transcripts = []
        for audio in (args.source, args.output):
            segments, _ = model.transcribe(audio, language=args.language, beam_size=5, temperature=0,
                vad_filter=False, condition_on_previous_text=False)
            transcripts.append(" ".join(segment.text for segment in segments))
        result = {"source_text": transcripts[0], "output_text": transcripts[1],
            "library_version": importlib.metadata.version("faster-whisper")}
    else:
        import numpy as np
        import torch

        sf = importlib.import_module("soundfile")
        signal = importlib.import_module("scipy.signal")
        speaker = importlib.import_module("speechbrain.inference.speaker")
        model = speaker.EncoderClassifier.from_hparams(source=args.model, savedir=str(Path(args.result).parent / "loaded"),
            overrides={"pretrained_path": str(Path(args.model).resolve())}, run_opts={"device": "cpu"})
        embeddings = []
        for audio in (args.source, args.output):
            waveform, rate = sf.read(audio, dtype="float32", always_2d=True)
            mono = waveform.mean(axis=1)
            divisor = math.gcd(rate, 16000)
            mono = signal.resample_poly(mono, 16000 // divisor, rate // divisor)
            if len(mono) < 16000 or float(np.mean(mono**2)) < 1e-10:
                raise ValueError("Speaker evaluation requires at least one second of non-silent audio")
            with torch.inference_mode():
                embedding = model.encode_batch(torch.from_numpy(mono).unsqueeze(0)).flatten()
            embeddings.append(embedding)
        similarity = float(torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0))
        result = {"similarity": similarity, "library_version": importlib.metadata.version("speechbrain")}
    Path(args.result).write_text(json.dumps(result, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
