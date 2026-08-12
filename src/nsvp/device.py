from __future__ import annotations

import importlib.util
import platform
from dataclasses import dataclass
from typing import Any

from .contracts import BackendCapabilities, BackendName
from .errors import BackendUnavailableError, DependencyUnavailableError


@dataclass(frozen=True)
class DeviceSelection:
    capabilities: BackendCapabilities
    torch_device: Any


class DeviceManager:
    """Central backend selection without importing torch at module import time."""

    def detect(self, requested: BackendName = BackendName.AUTO) -> DeviceSelection:
        torch = self._torch()
        available = self.available_backends(torch)
        if requested is BackendName.AUTO:
            selected = next(
                (item for item in (BackendName.CUDA, BackendName.ROCM, BackendName.DIRECTML, BackendName.MPS) if item in available),
                BackendName.CPU,
            )
        else:
            selected = requested
            if selected not in available:
                raise BackendUnavailableError(
                    f"requested backend '{selected.value}' is unavailable; available: "
                    + ", ".join(item.value for item in available)
                )

        warnings: list[str] = []
        if selected is BackendName.DIRECTML:
            import torch_directml

            device = torch_directml.device()
            name = f"DirectML device on {platform.system()}"
            precision = "fp32"
            warnings.append("DirectML component compatibility must be validated per model")
        elif selected in (BackendName.CUDA, BackendName.ROCM):
            device = torch.device("cuda")
            name = torch.cuda.get_device_name(0)
            precision = "fp16"
        elif selected is BackendName.MPS:
            device = torch.device("mps")
            name = "Apple Metal Performance Shaders"
            precision = "fp32"
        else:
            device = torch.device("cpu")
            name = platform.processor() or "CPU"
            precision = "fp32"

        return DeviceSelection(
            capabilities=BackendCapabilities(
                backend=selected,
                device_name=name,
                precision=precision,
                supported_dtypes=["float32"] + (["float16"] if precision == "fp16" else []),
                warnings=warnings,
            ),
            torch_device=device,
        )

    def available_backends(self, torch: Any | None = None) -> list[BackendName]:
        if torch is None:
            if importlib.util.find_spec("torch") is None:
                return [BackendName.CPU]
            torch = self._torch()
        result = [BackendName.CPU]
        if torch.cuda.is_available():
            result.insert(0, BackendName.ROCM if getattr(torch.version, "hip", None) else BackendName.CUDA)
        if importlib.util.find_spec("torch_directml") is not None:
            result.append(BackendName.DIRECTML)
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            result.append(BackendName.MPS)
        return result

    @staticmethod
    def _torch() -> Any:
        try:
            import torch
        except ImportError as exc:
            raise DependencyUnavailableError(
                "PyTorch is not installed; install an OS/backend-specific PyTorch build"
            ) from exc
        return torch

    def report_without_torch(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend=BackendName.CPU,
            device_name=platform.processor() or "CPU",
            precision="fp32",
            supported_dtypes=["float32"],
            warnings=["PyTorch is not installed; ML components are unavailable"],
        )
