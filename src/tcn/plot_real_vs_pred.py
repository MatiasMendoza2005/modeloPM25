import sys, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")

print(">>> ROOT:", ROOT)
print(">>> SRC PATH ADDED:", SRC)

sys.path.insert(0, ROOT)
sys.path.insert(0, SRC)

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

from tcn.tcn_model import TCNRegressor
from tcn.prepare_sequences import create_sequences

# ---- PATHS ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../..", "data/processed/clean_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "../..", "models/tcn_pm25.pt")
SCALER_PATH = os.path.join(BASE_DIR, "../..", "models/scaler.pkl")

# ---- LOAD DATA ----
df = pd.read_csv(DATA_PATH)
df = df.sort_values("time").reset_index(drop=True)

target = "pm2_5"
drop_cols = ["pm10", "time"]
features = df.drop(columns=drop_cols)
y = df[target].values

# ---- SCALE ----
scaler = load(SCALER_PATH)
X_scaled = scaler.transform(features)

# ---- CREATE SEQUENCES ----
SEQ_LEN = 48
X_seq, y_seq = create_sequences(X_scaled, y, SEQ_LEN)

# ---- TRAIN/TEST SPLIT ----
split = int(len(X_seq) * 0.8)
X_test, y_test = X_seq[split:], y_seq[split:]

X_test_t = torch.tensor(X_test, dtype=torch.float32)

# ---- LOAD MODEL ----
model = TCNRegressor(num_features=X_test.shape[2], seq_length=SEQ_LEN)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# ---- PREDICT ----
with torch.no_grad():
    preds = model(X_test_t).numpy()

# ---- PLOT ----
plt.figure(figsize=(14, 5))
plt.plot(y_test, label="Real PM2.5", alpha=0.7)
plt.plot(preds, label="Predicho PM2.5", alpha=0.7)
plt.title("Real vs Predicho PM2.5 (TCN)")
plt.xlabel("Índice de muestra (tiempo)")
plt.ylabel("PM2.5")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
