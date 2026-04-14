"""Utility script to report local PyTorch/GPU availability."""

if __name__ != "__main__":
    import pytest

    pytest.importorskip("torch")

import torch

from core.torch_device import resolve_torch_runtime


def main() -> None:
    runtime = resolve_torch_runtime()
    print(f"PyTorch Version: {torch.__version__}")
    print(runtime.message)
    if runtime.cuda_available:
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Memory Available: {total_memory_gb:.2f} GB")


if __name__ == "__main__":
    main()
