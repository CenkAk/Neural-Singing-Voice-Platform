"""SoulX bridge executed exclusively inside the configured upstream environment."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import platform
import random
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("source", "reference", "output", "checkpoint", "config", "rmvpe-checkpoint"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pitch-shift", type=int, default=0)
    parser.add_argument("--n-steps", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=3)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    sys.path.insert(0, str(root))
    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.fp16 and not args.device.startswith("cuda"):
        raise ValueError("SoulX fp16 requires a CUDA-style device")
    # Load only the upstream F0 module, avoiding optional transcription/separation imports.
    spec = importlib.util.spec_from_file_location("nsvp_soulx_f0", root / "preprocess/tools/f0_extraction.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load upstream F0 extractor")
    f0_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = f0_module
    spec.loader.exec_module(f0_module)
    inference = importlib.import_module("cli.inference_svc")
    files = importlib.import_module("soulxsinger.utils.file_utils")
    config = files.load_config(args.config)
    sample_rate, hop_size = int(config.audio.sample_rate), int(config.audio.hop_size)
    audio_utils = importlib.import_module("soulxsinger.utils.audio_utils")
    source = audio_utils.load_wav(args.source, sample_rate)
    reference = audio_utils.load_wav(args.reference, sample_rate)
    max_duration = max(source.shape[-1], reference.shape[-1]) / sample_rate + hop_size / sample_rate
    extractor = f0_module.F0Extractor(
        model_path=args.rmvpe_checkpoint, device=args.device,
        target_sr=sample_rate, hop_size=hop_size, max_duration=max_duration,
    )
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=False)
    source_f0 = extractor.process(args.source, f0_path=str(output_dir / "source_f0.npy"))
    reference_f0 = extractor.process(args.reference, f0_path=str(output_dir / "reference_f0.npy"))
    for track in (source_f0, reference_f0):
        if track.ndim != 1 or not track.size or not np.isfinite(track).all() or np.any(track < 0):
            raise ValueError("Upstream F0 extractor returned an invalid track")
    model = inference.build_model(args.checkpoint, config, device=args.device, use_fp16=args.fp16)
    inference_args = argparse.Namespace(
        prompt_wav_path=args.reference, target_wav_path=args.source,
        prompt_f0_path=str(output_dir / "reference_f0.npy"),
        target_f0_path=str(output_dir / "source_f0.npy"),
        save_dir=str(output_dir), device=args.device, auto_shift=False,
        pitch_shift=args.pitch_shift, n_steps=args.n_steps, cfg=args.cfg, use_fp16=args.fp16,
    )
    inference.process(inference_args, config, model)
    (output_dir / "runtime.json").write_text(json.dumps({
        "device": str(next(model.parameters()).device), "precision": "fp16" if args.fp16 else "fp32",
        "sample_rate": sample_rate, "hop_size": hop_size, "random_seed": args.seed,
        "source_f0_frames": int(source_f0.size), "reference_f0_frames": int(reference_f0.size),
        "python_version": platform.python_version(), "torch_version": str(torch.__version__),
        "parameter_dtypes": sorted({str(parameter.dtype) for parameter in model.parameters()}),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
