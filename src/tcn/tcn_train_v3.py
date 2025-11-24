import sys, os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, ROOT)
sys.path.insert(0, SRC)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump

from src.tcn.tcn_model_v3 import TCNMultiTowers
from src.tcn.prepare_sequences_multi import create_sequences_multi
from src.config import CLEAN_DATA_PATH_V3

HORIZON_COLS = ["pm2_5_1h","pm2_5_12h","pm2_5_24h","pm2_5_72h","pm2_5_168h"]
SEQ_LEN = 72


def train_tcn_v3_multi_towers():

    print("\n=== Cargando dataset v3 MULTI COMPLETO ===")
    df = pd.read_csv(CLEAN_DATA_PATH_V3)

    Y = df[HORIZON_COLS].values

    drop_cols = ["time", "pm2_5", "pm10"] + HORIZON_COLS
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].values

    # Escalar
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Secuencias
    X_seq, Y_seq = create_sequences_multi(X_scaled, Y, SEQ_LEN)

    split = int(len(X_seq) * 0.8)

    X_train = torch.tensor(X_seq[:split], dtype=torch.float32)
    Y_train = torch.tensor(Y_seq[:split], dtype=torch.float32)

    X_test = torch.tensor(X_seq[split:], dtype=torch.float32)
    Y_test = torch.tensor(Y_seq[split:], dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train), batch_size=32, shuffle=True
    )

    model = TCNMultiTowers(num_inputs=X_train.shape[2])

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = nn.MSELoss()

    EPOCHS = 60

    print("\n=== Entrenando TCN Multi-Head con Torres ===\n")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb = xb.permute(0,2,1)

            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}  Loss={total_loss:.4f}")

    # Guardar modelo
    MODEL_PATH = "models/tcn_pm25_v3_multi_towers.pt"
    SCALER_PATH = "models/scaler_v3_multi_towers.pkl"

    torch.save(model.state_dict(), MODEL_PATH)
    dump(scaler, SCALER_PATH)

    print("\n=== Guardado ===")
    print(MODEL_PATH)
    print(SCALER_PATH)

    # === METRICAS ===
    print("\n=== MÉTRICAS POR HORIZONTE ===")

    model.eval()
    with torch.no_grad():
        preds = model(X_test.permute(0,2,1)).numpy()

    Y_true = Y_test.numpy()
    horizon_names = ["1h","12h","24h","72h","168h"]

    print("\nHorizonte | MAE | RMSE | R2\n-----------------------------------")
    for i, name in enumerate(horizon_names):
        mae = mean_absolute_error(Y_true[:,i], preds[:,i])
        rmse = np.sqrt(mean_squared_error(Y_true[:,i], preds[:,i]))
        r2 = r2_score(Y_true[:,i], preds[:,i])

        print(f"{name:<9} | {mae:5.3f} | {rmse:5.3f} | {r2:5.3f}")

    print("\n=== Entrenamiento COMPLETADO ===")


if __name__ == "__main__":
    train_tcn_v3_multi_towers()
