from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import DependencyUnavailableError


def _to_cpu(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def save_portable_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    try:
        import torch
    except ImportError as exc:
        raise DependencyUnavailableError("PyTorch is required for checkpoint operations") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_to_cpu(payload), path)


def load_portable_checkpoint(path: Path, device: Any = "cpu") -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise DependencyUnavailableError("PyTorch is required for checkpoint operations") from exc
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint root must be a mapping")
    return payload
