import sys, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")

print(">>> ROOT:", ROOT)
print(">>> SRC PATH ADDED:", SRC)

sys.path.insert(0, ROOT)
sys.path.insert(0, SRC)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump

from tcn.prepare_sequences import create_sequences
from tcn.tcn_model import TCNRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../..", "data/processed/clean_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "../..", "models/tcn_pm25_v2.pt")
SCALER_PATH = os.path.join(BASE_DIR, "../..", "models/scaler_v2.pkl")

def train_tcn():

    # ---- LOAD DATA ----
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values("time").reset_index(drop=True)

    target = "pm2_5"
    drop_cols = ["pm10", "time"]
    
    features = df.drop(columns=drop_cols)
    y = df[target].values

    # ---- SCALE ----
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(features)

    # ---- CREATE SEQUENCES ----
    SEQ_LEN = 96 #horas
    X_seq, y_seq = create_sequences(X_scaled, y, SEQ_LEN)

    # ---- TRAIN/TEST SPLIT ----
    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # ---- TORCH DATASET ----
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=32, shuffle=True)

    # ---- MODEL ----
    model = TCNRegressor(num_features=X_train.shape[2], seq_length=SEQ_LEN)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # ---- TRAIN LOOP ----
    EPOCHS = 75

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{EPOCHS}  Loss={epoch_loss:.4f}")

    # ---- EVALUATION ----
    model.eval()
    preds = model(X_test_t).detach().numpy()

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\n---- METRICAS TCN (PyTorch) ----")
    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R2  :", r2)

    # ---- SAVE MODEL + SCALER ----
    torch.save(model.state_dict(), MODEL_PATH)
    dump(scaler, SCALER_PATH)

    return model, mae, rmse, r2


if __name__ == "__main__":
    train_tcn()
    print("\n=== Entrenamiento v2 finalizado. Ejecutando pruebas automáticas ===\n")

    import src.tcn.test_tcn_v2 as auto_test
    auto_test.run_tcn_v2_tests()
