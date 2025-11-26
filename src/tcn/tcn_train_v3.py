import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from joblib import dump
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score
)

from scipy.stats import pearsonr

from src.tcn.tcn_model_v3 import TCNMultiTowers
from src.tcn.prepare_sequences_multi import create_sequences_multi
from src.config import CLEAN_DATA_PATH_V3

# ======================================================
# CONFIG GLOBAL
# ======================================================
SEQ_LEN = 336
BATCH_SIZE = 64
EPOCHS = 80
LR = 2e-4
EARLY_STOP = 12

HORIZON_NAMES = ["pm2_5_1h","pm2_5_12h","pm2_5_24h","pm2_5_72h","pm2_5_168h"]

# ======================================================
# GPU CHECK
# ======================================================
if not torch.cuda.is_available():
    raise RuntimeError("GPU requerida para entrenar este modelo.")

DEVICE = torch.device("cuda")
print(f"Entrenando en {DEVICE}")

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True  # Optimización


# ======================================================
# MÉTRICAS EXTRA
# ======================================================
def mape(y, y_pred):
    y = np.array(y)
    y_pred = np.array(y_pred)
    eps = 1e-6
    return np.mean(np.abs((y - y_pred) / (y + eps)))


def skill_score(y, y_pred):
    """Compara contra modelo naive de persistencia."""
    naive = y[:-1]
    real = y[1:]
    model = y_pred[1:]
    return 1 - (np.mean((real - model)**2) / np.mean((real - naive)**2))


# ======================================================
# TRAINING PIPELINE
# ======================================================
def train_tcn_v3_multi():

    print("\n=== Cargando dataset v3 ===")
    df = pd.read_csv(CLEAN_DATA_PATH_V3)

    # --------------------------
    # Separar features y targets
    # --------------------------
    target_cols = HORIZON_NAMES
    feature_cols = [c for c in df.columns if c not in ["time", "pm2_5", "pm10"] + target_cols]

    X = df[feature_cols].values
    Y = df[target_cols].values

    # --------------------------
    # Escalado
    # --------------------------
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # --------------------------
    # Secuencias multi-horizon
    # --------------------------
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

    # --------------------------
    # Modelo
    # --------------------------
    model = TCNMultiTowers(num_inputs=X_train.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    best_loss = float("inf")
    patience_counter = 0

    print("\n=== Entrenando modelo ===")
    scaler_amp = torch.cuda.amp.GradScaler()

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()

            # AMP para mayor velocidad
            with torch.cuda.amp.autocast():
                pred = model(xb)
                loss = criterion(pred, yb)

            scaler_amp.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler_amp.step(optimizer)
            scaler_amp.update()

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
                print("Early stopping activado.")
                break

    # ======================================================
    # EVALUACIÓN FINAL
    # ======================================================
    print("\n=== Evaluación final ===")

    model.eval()
    preds = model(X_test.to(DEVICE)).cpu().detach().numpy()

    print("\n=== MÉTRICAS POR HORIZONTE ===\n")
    print("Horizonte | MAE | RMSE | R2 | MAPE | Corr | Skill")
    print("--------------------------------------------------------------")

    for i, name in enumerate(HORIZON_NAMES):
        y_true = Y_test[:, i]
        y_pred = preds[:, i]

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        r2 = r2_score(y_true, y_pred)
        mape_v = mape(y_true, y_pred)
        corr, _ = pearsonr(y_true, y_pred)
        skill = skill_score(y_true, y_pred)

        print(f"{name:<10} | {mae:.3f} | {rmse:.3f} | {r2:.3f} | {mape_v:.3f} | {corr:.3f} | {skill:.3f}")

    # -----------------------------
    # Guardado final
    # -----------------------------
    torch.save(model.state_dict(), "models/tcn_pm25_v3_multi_towers.pt")
    dump(scaler, "models/scaler_v3_multi_towers.pkl")

    print("\n=== Entrenamiento COMPLETADO ===")


# ======================================================
if __name__ == "__main__":
    train_tcn_v3_multi()
