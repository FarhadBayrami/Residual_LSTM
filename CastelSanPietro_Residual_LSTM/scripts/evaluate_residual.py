import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("🚀 Evaluating Residual LSTM for CastelSanPietro...\n")

# ====================== LOAD RAW DATA ======================
df = pd.read_csv('data/CastelSanPietro.ts', 
                 sep=r'\s+', 
                 skiprows=1, 
                 header=None,
                 names=['YYYY','MM','DD','HH','mm','QM','Q','Rain','Prec','Evap','Snow',
                        'Temp','Etp','Soil','SoilSat','Perco','Surf','YSnow','EnSnow',
                        'SWE','Deep','DeepSat','Inf2Surf'])

df['datetime'] = pd.to_datetime({
    'year': df['YYYY'],
    'month': df['MM'],
    'day': df['DD'],
    'hour': df['HH']
})
df = df.set_index('datetime')
df = df.drop(columns=['Deep','DeepSat','mm'], errors='ignore')

# ====================== LOAD TEST DATA ======================
X_test = np.load('results/X_test_res.npy')
y_test = np.load('results/y_test_res.npy')

scaler_y = joblib.load('models/scaler_y_res.pkl')

# ====================== MODEL ======================
class ResidualLSTM(torch.nn.Module):
    def __init__(self, input_size=14, hidden_size=64, num_layers=2, dropout=0.25):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, 
                                  batch_first=True, dropout=dropout)
        self.fc1 = torch.nn.Linear(hidden_size, 32)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(32, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = ResidualLSTM()
model.load_state_dict(torch.load('models/best_residual_lstm.pth', weights_only=True))
model.eval()

# ====================== PREDICTIONS ======================
device = torch.device('cpu')
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

with torch.no_grad():
    residual_scaled = model(X_test_tensor).cpu().numpy()

residual_pred = scaler_y.inverse_transform(residual_scaled).flatten()

# Final Prediction
test_index = df.index[-len(residual_pred):]
Q_test = df.loc[test_index, 'Q'].values
final_pred = Q_test + residual_pred
y_true = df.loc[test_index, 'QM'].values

# ====================== SAVE RESULTS ======================
comparison_df = pd.DataFrame({
    'DateTime': test_index,
    'Observed_QM': y_true,
    'TOPKAPI_Q': Q_test,
    'Residual_LSTM_Prediction': final_pred,
    'Residual': residual_pred
})

comparison_df.to_csv('results/final_residual_comparison.csv', index=False)
print(f"✅ Results saved to: results/final_residual_comparison.csv")

# ====================== METRICS ======================
def print_metrics(name, true, pred):
    mae = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    print(f"{name:20} → MAE: {mae:6.2f} | RMSE: {rmse:6.2f}")

print("\n" + "="*80)
print("FINAL COMPARISON - CastelSanPietro")
print("="*80)
print_metrics("Residual LSTM", y_true, final_pred)
print_metrics("TOPKAPI", y_true, Q_test)

# ====================== PLOTS ======================
plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.plot(y_true[-2000:], label='Observed QM', linewidth=2)
plt.plot(final_pred[-2000:], label='Residual LSTM', linewidth=1.8)
plt.plot(Q_test[-2000:], label='TOPKAPI', linewidth=1.8, alpha=0.85)
plt.title('Last 2000 Hours - Residual LSTM vs TOPKAPI')
plt.ylabel('Discharge (m³/s)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 2)
plt.plot(y_true - final_pred, label='Residual LSTM Error', alpha=0.8)
plt.plot(y_true - Q_test, label='TOPKAPI Error', alpha=0.8)
plt.title('Error Comparison')
plt.ylabel('Error (m³/s)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/residual_comparison_plot.png', dpi=200, bbox_inches='tight')
plt.show()

print("\n✅ Evaluation completed successfully!")