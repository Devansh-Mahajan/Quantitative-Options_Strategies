import torch

print(f"PyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")
    print(f"Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("❌ NO GPU DETECTED. PyTorch is running on CPU.")