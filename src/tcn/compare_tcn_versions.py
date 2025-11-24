import sys, os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import load


# ---- PATH FIX ----
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, ROOT)
sys.path.insert(0, SRC)


from tcn.tcn_model import TCNRegressor
from tcn.prepare_sequences import create_sequences


# ---- PATHS ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../..", "data/processed/clean_dataset.csv")
MODEL_V1 = os.path.join(BASE_DIR, "../..", "models/tcn_pm25.pt")
SCALER_V1 = os.path.join(BASE_DIR, "../..", "models/scaler.pkl")
MODEL_V2 = os.path.join(BASE_DIR, "../..", "models/tcn_pm25_v2.pt")
SCALER_V2 = os.path.join(BASE_DIR, "../..", "models/scaler_v2.pkl")


# -------- VISUALIZATION --------
def compare_versions():
    print("\n=== Comparando TCN v1 y v2 ===\n")


df = pd.read_csv(DATA_PATH)
df = df.sort_values("time").reset_index(drop=True)


target = "pm2_5"
drop_cols = ["pm10", "time"]
features = df.drop(columns=drop_cols)
y = df[target].values


# ----- LOAD SCALERS -----
scaler1 = load(SCALER_V1)
scaler2 = load(SCALER_V2)


X_scaled_v1 = scaler1.transform(features)
X_scaled_v2 = scaler2.transform(features)


# SEQ_LEN FOR V1 = 48, V2 = 72
X_seq1, y_seq1 = create_sequences(X_scaled_v1, y, 48)
X_seq2, y_seq2 = create_sequences(X_scaled_v2, y, 72)


split1 = int(len(X_seq1) * 0.8)
split2 = int(len(X_seq2) * 0.8)


X_test1, y_test1 = X_seq1[split1:], y_seq1[split1:]
X_test2, y_test2 = X_seq2[split2:], y_seq2[split2:]


X_test1_t = torch.tensor(X_test1, dtype=torch.float32)
X_test2_t = torch.tensor(X_test2, dtype=torch.float32)


# ----- LOAD MODELS -----
model1 = TCNRegressor(num_features=X_test1.shape[2], seq_length=48)
model1.load_state_dict(torch.load(MODEL_V1))
model1.eval()


model2 = TCNRegressor(num_features=X_test2.shape[2], seq_length=72)
model2.load_state_dict(torch.load(MODEL_V2))
model2.eval()


with torch.no_grad():
    pred1 = model1(X_test1_t).numpy()
    pred2 = model2(X_test2_t).numpy()


# Make lengths equal for plotting
min_len = min(len(pred1), len(pred2), len(y_test1), len(y_test2))
plt.show()