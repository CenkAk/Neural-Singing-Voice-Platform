"""Executed by the configured Seed-VC Python, without importing NSVP."""
from __future__ import annotations

import argparse
import importlib
import random
import sys
from pathlib import Path


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
    upstream = importlib.import_module("inference")
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
