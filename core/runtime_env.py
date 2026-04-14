from __future__ import annotations

import os
from pathlib import Path


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def host_has_nvidia_device() -> bool:
    direct_device_paths = (
        Path("/dev/nvidia0"),
        Path("/dev/nvidiactl"),
        Path("/dev/nvidia-uvm"),
    )
    if any(path.exists() for path in direct_device_paths):
        return True

    gpu_dir = Path("/proc/driver/nvidia/gpus")
    if gpu_dir.exists():
        try:
            return any(gpu_dir.iterdir())
        except OSError:
            return True
    return False


def apply_accelerator_policy(environment: dict[str, str] | None = None) -> tuple[dict[str, str], str]:
    env = dict(environment or os.environ)
    force_cpu = _truthy(env.get("OPTIONS_STACK_FORCE_CPU"))

    if not force_cpu and not host_has_nvidia_device():
        force_cpu = True

    env.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    if force_cpu:
        env["OPTIONS_STACK_FORCE_CPU"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = ""
        return env, "CPU detected and using."

    return env, "GPU auto-detect enabled."
