from __future__ import annotations

from dataclasses import dataclass
import os

import torch


@dataclass(frozen=True)
class TorchRuntime:
    device: torch.device
    cuda_available: bool
    accelerator_name: str
    message: str


def resolve_torch_runtime() -> TorchRuntime:
    """Resolve the best available Torch device without failing on CUDA probe issues."""
    if str(os.environ.get("OPTIONS_STACK_FORCE_CPU", "")).strip().lower() in {"1", "true", "yes", "on"}:
        return TorchRuntime(
            device=torch.device("cpu"),
            cuda_available=False,
            accelerator_name="CPU",
            message="CPU detected and using.",
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) == "":
        return TorchRuntime(
            device=torch.device("cpu"),
            cuda_available=False,
            accelerator_name="CPU",
            message="CPU detected and using.",
        )

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False

    if cuda_available:
        try:
            accelerator_name = str(torch.cuda.get_device_name(0))
        except Exception:
            accelerator_name = "CUDA device"
        return TorchRuntime(
            device=torch.device("cuda"),
            cuda_available=True,
            accelerator_name=accelerator_name,
            message=f"GPU detected: {accelerator_name}. Using CUDA.",
        )

    return TorchRuntime(
        device=torch.device("cpu"),
        cuda_available=False,
        accelerator_name="CPU",
        message="CPU detected and using.",
    )
