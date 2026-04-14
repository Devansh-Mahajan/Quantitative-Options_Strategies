import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.runtime_env import apply_accelerator_policy

os.environ.update(apply_accelerator_policy(os.environ.copy())[0])

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging

from core.neural_brain import StrategySelectorNet
from core.torch_device import resolve_torch_runtime

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("gpu_training")

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'brain_dataset.pt')
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'brain_weights.pth')

def train():
    # 1. Hardware Initialization
    runtime = resolve_torch_runtime()
    device = runtime.device
    logger.info(runtime.message)
    logger.info(f"🚀 Igniting Deep Learning Sequence on: {device.type.upper()}")

    # 2. Load the Tensors
    logger.info(f"📂 Loading Deep Macro Matrix from {DATA_PATH}...")
    dataset = torch.load(DATA_PATH, map_location="cpu", weights_only=False)
    X = dataset['X']
    y = dataset['y']

    # 3. Time-Series Split (80% Train, 20% Test)
    # CRITICAL: We do NOT shuffle time-series data, otherwise the bot looks into the future to predict the past.
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    # Create PyTorch DataLoaders to feed the GPU in chunks
    batch_size = 64
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    # 4. Instantiate The Brain
    # Input: 11 features | Hidden: 128 neurons | Layers: 2 deep | Classes: 3 (Theta, Vega, Hedge)
    input_size = X.shape[2]
    model = StrategySelectorNet(input_size=input_size, hidden_size=128, num_layers=2, num_classes=3).to(device)

    # 5. Optimizer and Loss Function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 6. The Training Loop
    epochs = 500  # We are running 500 full passes over 15 years of data
    logger.info(f"🔥 Feeding data to the RTX 5090. Training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            # Move the data off the slow RAM and onto the GPU VRAM
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Print progress every 50 epochs
        if (epoch + 1) % 50 == 0:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    val_loss += criterion(outputs, batch_y).item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                    
            acc = 100 * correct / total
            logger.info(f"Epoch [{epoch+1:3d}/{epochs}] | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(test_loader):.4f} | Out-of-Sample Accuracy: {acc:.2f}%")

    # 7. Save the Neural Weights
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logger.info(f"✅ Neural Brain weights saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
