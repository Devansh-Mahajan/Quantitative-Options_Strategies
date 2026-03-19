import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.mega_neural_brain import MegaStrategyNet

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger("mega_gpu_training")

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'mega_universe_dataset.pt')
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'mega_brain_weights.pth')

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 Igniting Institutional Deep Learning Sequence on: {device.type.upper()}")

    dataset = torch.load(DATA_PATH, weights_only=False)
    X = dataset['X']
    y = dataset['y']

    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    batch_size = 1024 
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    input_size = X.shape[2]
    num_classes = 4 
    model = MegaStrategyNet(input_size=input_size, hidden_size=256, num_layers=3, num_classes=num_classes).to(device)

    # Increased weight_decay to heavily penalize memorization
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3) 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    epochs = 100 
    logger.info(f"🔥 Feeding {len(X_train)} matrices to the GPU. Training for {epochs} epochs...")

    start_time = time.time()
    
    # EARLY STOPPING TRACKERS
    best_val_loss = float('inf')
    patience = 5  # If Val Loss doesn't improve for 5 checks, we stop.
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            
        scheduler.step()
            
        # Check validation every single epoch now so we can catch overfitting instantly
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
        elapsed = (time.time() - start_time) / 60
        
        avg_train_loss = train_loss/len(train_loader)
        avg_val_loss = val_loss/len(test_loader)
        
        logger.info(f"Epoch [{epoch+1:3d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Accuracy: {acc:.2f}%")

        # --- EARLY STOPPING LOGIC ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the "best" model, not the overfitted one
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.warning(f"🛑 EARLY STOPPING TRIGGERED! Validation loss stopped improving. Reverting to best weights.")
                break

    logger.info(f"✅ Training complete. Best Mega Brain weights safely stored at {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()