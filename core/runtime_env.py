from __future__ import annotations

import os
from pathlib import Path


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _read_path_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip().lower()
    except OSError:
        return ""


def _nvidia_proc_gpu_present() -> bool:
    gpu_dir = Path("/proc/driver/nvidia/gpus")
    if not gpu_dir.exists():
        return False
    try:
        return any(gpu_dir.iterdir())
    except OSError:
        return False


def _nvidia_pci_gpu_present() -> bool:
    pci_root = Path("/sys/bus/pci/devices")
    if not pci_root.exists():
        return False

    try:
        device_paths = tuple(pci_root.iterdir())
    except OSError:
        return False

    for device_path in device_paths:
        vendor = _read_path_text(device_path / "vendor")
        class_code = _read_path_text(device_path / "class")
        if vendor != "0x10de":
            continue
        if class_code.startswith("0x03") or class_code.startswith("0x12"):
            return True
    return False


def host_has_nvidia_device() -> bool:
    return _nvidia_proc_gpu_present() or _nvidia_pci_gpu_present()


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
