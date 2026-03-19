import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
import time

# Ensure core is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.mega_neural_brain import MegaStrategyNet

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("Mega_Trainer")

# --- CONFIG ---
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'mega_universe_dataset.pt')
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'trading_model.pth')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 1024
EPOCHS = 100
LEARNING_RATE = 0.001

def train():
    logger.info(f"🚀 Igniting Institutional Training on: {DEVICE}")
    
    # 1. Load Data
    checkpoint = torch.load(DATA_PATH, weights_only=False)
    X, y = checkpoint['X'], checkpoint['y']
    
    # Split 80/20
    split = int(len(X) * 0.8)
    train_ds = TensorDataset(X[:split], y[:split])
    test_ds = TensorDataset(X[split:], y[split:])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Initialize the CORRECT Architecture (Mixture of Experts)
    input_dim = X.shape[2]
    model = MegaStrategyNet(input_size=input_dim, hidden_size=256, num_layers=3, num_classes=4).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # 3. Training Loop with Early Stopping
    best_val_loss = float('inf')
    patience, counter = 7, 0

    for epoch in range(EPOCHS):
        model.train()
        t_loss = 0
        for b_x, b_y in train_loader:
            b_x, b_y = b_x.to(DEVICE), b_y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(b_x), b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        v_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for b_x, b_y in test_loader:
                b_x, b_y = b_x.to(DEVICE), b_y.to(DEVICE)
                out = model(b_x)
                v_loss += criterion(out, b_y).item()
                correct += (out.argmax(1) == b_y).sum().item()
                total += b_y.size(0)

        avg_v_loss = v_loss / len(test_loader)
        acc = (correct / total) * 100
        logger.info(f"Epoch {epoch+1} | Val Loss: {avg_v_loss:.4f} | Acc: {acc:.2f}%")

        # Checkpoint / Early Stopping
        if avg_v_loss < best_val_loss:
            best_val_loss = avg_v_loss
            counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_size': input_dim,
                'hidden_size': 256,
                'num_layers': 3,
                'scaler': checkpoint['scaler'],
                'features_list': checkpoint['features_list']
            }, MODEL_SAVE_PATH)
        else:
            counter += 1
            if counter >= patience:
                logger.warning("🛑 Early stopping triggered.")
                break

    logger.info(f"✅ Training Complete. Mega Brain saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()