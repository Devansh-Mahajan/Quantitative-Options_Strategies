"""Utility script to report local PyTorch/GPU availability."""

if __name__ != "__main__":
    import pytest

    pytest.importorskip("torch")

import torch


def main() -> None:
    print(f"PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Memory Available: {total_memory_gb:.2f} GB")
    else:
        print("❌ NO GPU DETECTED. PyTorch is running on CPU.")


if __name__ == "__main__":
    main()
