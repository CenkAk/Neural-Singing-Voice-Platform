"""Run in a provider environment. Kernel checks do not establish model support."""
from __future__ import annotations

import argparse
import importlib
import json
import platform
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result: dict[str, object] = {
        "scope": "matmul and conv1d numerical smoke checks, not model compatibility",
        "status": "failed", "requested_device": args.device,
        "python": platform.python_version(), "platform": platform.platform(),
    }
    try:
        torch = importlib.import_module("torch")
        result.update(torch_version=torch.__version__, hip_version=torch.version.hip,
                      cuda_version=torch.version.cuda)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested GPU is unavailable to Torch")
        result["device_name"] = torch.cuda.get_device_name(0) if args.device == "cuda" else platform.processor()
        torch.manual_seed(42)
        left, right = torch.randn(64, 64), torch.randn(64, 64)
        signal, kernel = torch.randn(1, 4, 256), torch.randn(8, 4, 5)
        expected_matmul = left @ right
        expected_conv = torch.nn.functional.conv1d(signal, kernel)
        checks = []
        for dtype_name in ("float32", "float16"):
            dtype = getattr(torch, dtype_name)
            try:
                actual_matmul = left.to(args.device, dtype) @ right.to(args.device, dtype)
                actual_conv = torch.nn.functional.conv1d(signal.to(args.device, dtype), kernel.to(args.device, dtype))
                tolerance = 0.02 if dtype_name == "float16" else 0.001
                torch.testing.assert_close(actual_matmul.float().cpu(), expected_matmul, rtol=tolerance, atol=tolerance)
                torch.testing.assert_close(actual_conv.float().cpu(), expected_conv, rtol=tolerance, atol=tolerance)
                checks.append({"dtype": dtype_name, "status": "verified", "atol": tolerance, "rtol": tolerance})
            except (RuntimeError, AssertionError) as exc:
                checks.append({"dtype": dtype_name, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        result["checks"] = checks
        result["status"] = "verified" if all(check["status"] == "verified" for check in checks) else "failed"
    except (ImportError, OSError, RuntimeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2)
    args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if result["status"] == "verified" else 1


if __name__ == "__main__":
    raise SystemExit(main())
