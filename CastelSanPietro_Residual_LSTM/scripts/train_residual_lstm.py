import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("🚀 Training Residual Correction LSTM for CastelSanPietro...\n")

# ====================== DEVICE ======================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# ====================== LOAD DATA ======================
X_train = np.load('results/X_train_res.npy')
y_train = np.load('results/y_train_res.npy')
X_val   = np.load('results/X_val_res.npy')
y_val   = np.load('results/y_val_res.npy')
X_test  = np.load('results/X_test_res.npy')
y_test  = np.load('results/y_test_res.npy')

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val   = torch.tensor(X_val,   dtype=torch.float32)
y_val   = torch.tensor(y_val,   dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_test  = torch.tensor(y_test,  dtype=torch.float32)

print(f"Training samples: {X_train.shape[0]} | Time steps: {X_train.shape[1]}")

scaler_y = joblib.load('models/scaler_y_res.pkl')

# Data Loaders
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, y_val),     batch_size=64, shuffle=False)

# ====================== MODEL ======================
class ResidualLSTM(nn.Module):
    def __init__(self, input_size=14, hidden_size=64, num_layers=2, dropout=0.25):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = ResidualLSTM().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)   # L2 Regularization

# ====================== TRAINING ======================
epochs = 100
patience = 12
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        output = model(Xb)
        loss = criterion(output, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            output = model(Xb)
            val_loss += criterion(output, yb).item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'models/best_residual_lstm.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

print("\n✅ Training Completed!")

# Load best model
model.load_state_dict(torch.load('models/best_residual_lstm.pth'))
model.eval()

# ====================== EVALUATION ======================
def evaluate(X, y, name):
    model.eval()
    with torch.no_grad():
        pred = model(X.to(device)).cpu().numpy()
    true = scaler_y.inverse_transform(y.numpy())
    pred = scaler_y.inverse_transform(pred)
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    print(f"{name:12} → MAE: {mae:6.2f} | RMSE: {rmse:6.2f}")

print("\n" + "="*65)
print("RESIDUAL LSTM PERFORMANCE")
print("="*65)
evaluate(X_train, y_train, "Train")
evaluate(X_val,   y_val,   "Validation")
evaluate(X_test,  y_test,  "Test")

torch.save(model.state_dict(), 'models/final_residual_lstm.pth')
print("\nModel saved successfully!")