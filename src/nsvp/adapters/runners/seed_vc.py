"""Executed by the configured Seed-VC Python, without importing NSVP."""
from __future__ import annotations

import argparse
import importlib
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def save_wave(path: str, waveform: torch.Tensor, sample_rate: int) -> None:
    import soundfile as sf  # type: ignore[import-untyped]

    # Preserve model samples, including peaks above unity, without TorchCodec.
    sf.write(path, waveform.detach().cpu().numpy().T, sample_rate, subtype="FLOAT")


def load_local_model(repo_id: str, model_filename: str = "pytorch_model.bin",
                     config_filename: str | None = None) -> str | tuple[str, str]:
    hf_hub_download = importlib.import_module("huggingface_hub").hf_hub_download

    model_path = hf_hub_download(repo_id=repo_id, filename=model_filename,
        cache_dir=os.environ.get("HF_HUB_CACHE"), local_files_only=True)
    if config_filename is None:
        return str(model_path)
    config_path = hf_hub_download(repo_id=repo_id, filename=config_filename,
        cache_dir=os.environ.get("HF_HUB_CACHE"), local_files_only=True)
    return str(model_path), str(config_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("source", "target", "output", "checkpoint", "config"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--diffusion-steps", type=int, default=30)
    parser.add_argument("--f0-condition", choices=["True"], default="True")
    parser.add_argument("--auto-f0-adjust", choices=["False"], default="False")
    parser.add_argument("--semi-tone-shift", type=int, default=0)
    parser.add_argument("--fp16", choices=["True", "False"], default="False")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sys.path.insert(0, str(Path.cwd()))
    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Seed-VC overwrites HF_HUB_CACHE during import. Load Hub constants first so
    # Transformers and BigVGAN resolve the same configured offline cache.
    hub_constants = importlib.import_module("huggingface_hub.constants")
    cache = os.environ.get("HF_HUB_CACHE", hub_constants.HF_HUB_CACHE)
    upstream = importlib.import_module("inference")
    os.environ["HF_HUB_CACHE"] = cache
    upstream.__dict__["load_custom_model_from_hf"] = load_local_model
    upstream.__dict__["torchaudio"] = SimpleNamespace(**{**vars(upstream.torchaudio), "save": save_wave})
    if args.device != "auto":
        upstream.__dict__["device"] = torch.device(args.device)
    args.fp16 = args.fp16 == "True"
    args.f0_condition = True
    args.auto_f0_adjust = False
    args.length_adjust = 1.0
    args.inference_cfg_rate = 0.7
    upstream.main(args)


if __name__ == "__main__":
    main()
