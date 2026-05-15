import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

print("🚀 Preparing data for CastelSanPietro Residual LSTM...\n")

# Load new dataset
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

print(f"Data loaded! Shape: {df.shape}")

# ====================== FEATURES ======================
features = ['Q', 'Prec', 'Rain', 'Evap', 'Snow', 'Temp', 'Soil', 'SoilSat', 
            'Perco', 'Surf', 'YSnow', 'EnSnow', 'SWE', 'Inf2Surf']

df['Residual'] = df['QM'] - df['Q']

X = df[features].values
y = df['Residual'].values.reshape(-1, 1)

print(f"Using {len(features)} features")

# Scaling
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

os.makedirs('models', exist_ok=True)
joblib.dump(scaler_X, 'models/scaler_X_res.pkl')
joblib.dump(scaler_y, 'models/scaler_y_res.pkl')

# ====================== SEQUENCES ======================
def create_sequences(X, y, time_steps=48):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i+time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

TIME_STEPS = 48
X_seq, y_seq = create_sequences(X_scaled, y_scaled, TIME_STEPS)

train_size = int(len(X_seq) * 0.70)
val_size = int(len(X_seq) * 0.15)

np.save('results/X_train_res.npy', X_seq[:train_size])
np.save('results/y_train_res.npy', y_seq[:train_size])
np.save('results/X_val_res.npy', X_seq[train_size:train_size+val_size])
np.save('results/y_val_res.npy', y_seq[train_size:train_size+val_size])
np.save('results/X_test_res.npy', X_seq[train_size+val_size:])
np.save('results/y_test_res.npy', y_seq[train_size+val_size:])

print(f"\n✅ Data preparation completed for CastelSanPietro!")
print(f"Time steps: {TIME_STEPS} | Training samples: {X_seq[:train_size].shape[0]}")