import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from joblib import dump
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.tcn.tcn_model_v3 import TCNMultiTowers
from src.tcn.prepare_sequences_multi import create_sequences_multi
from src.config import CLEAN_DATA_PATH_V3

# ======================================================
# CONFIG
# ======================================================
SEQ_LEN = 336  # 2 semanas de ventana
BATCH_SIZE = 64
EPOCHS = 60
LR = 3e-4

HORIZON_NAMES = ["pm2_5_1h","pm2_5_12h","pm2_5_24h","pm2_5_72h","pm2_5_168h"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Entrenando en: {DEVICE}")


def train_tcn_v3_multi():

    print("\n=== Cargando dataset v3 ===")
    df = pd.read_csv(CLEAN_DATA_PATH_V3)

    # Separar features y targets
    target_cols = HORIZON_NAMES
    feature_cols = [c for c in df.columns if c not in ["time", "pm2_5", "pm10"] + target_cols]

    X = df[feature_cols].values
    Y = df[target_cols].values

    # ----------------------------
    # Escalado
    # ----------------------------
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # ----------------------------
    # Crear secuencias multihorizon
    # ----------------------------
    X_seq, Y_seq = create_sequences_multi(X_scaled, Y, SEQ_LEN)

    X_t = torch.tensor(X_seq, dtype=torch.float32).permute(0, 2, 1)
    Y_t = torch.tensor(Y_seq, dtype=torch.float32)

    # Train/test split
    split = int(len(X_t) * 0.8)
    X_train, X_test = X_t[:split], X_t[split:]
    Y_train, Y_test = Y_t[:split], Y_t[split:]

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # ----------------------------
    # Modelo
    # ----------------------------
    model = TCNMultiTowers(num_inputs=X_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    best_loss = float("inf")
    patience_counter = 0
    EARLY_STOP = 10

    print("\n=== Entrenando modelo ===")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step(epoch_loss)

        print(f"Epoch {epoch+1}/{EPOCHS}  Loss={epoch_loss:.4f}")

        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP:
                print("Early stopping activado")
                break

    # ================================
    # Evaluación final
    # ================================
    model.eval()
    preds = model(X_test.to(DEVICE)).cpu().detach().numpy()

    print("\n=== MÉTRICAS POR HORIZONTE ===")
    print("\nHorizonte | MAE | RMSE | R2")
    print("-----------------------------------")

    for i, name in enumerate(HORIZON_NAMES):
        mae = mean_absolute_error(Y_test[:, i], preds[:, i])
        rmse = np.sqrt(((Y_test[:, i] - preds[:, i])**2).mean())
        r2 = r2_score(Y_test[:, i], preds[:, i])

        print(f"{name:<10} | {mae:.3f} | {rmse:.3f} | {r2:.3f}")

    # Guardar modelo y scaler
    torch.save(model.state_dict(), "models/tcn_pm25_v3_multi_towers.pt")
    dump(scaler, "models/scaler_v3_multi_towers.pkl")

    print("\n=== Entrenamiento COMPLETADO ===")


if __name__ == "__main__":
    train_tcn_v3_multi()
