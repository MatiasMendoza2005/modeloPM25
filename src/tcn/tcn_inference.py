import torch
import numpy as np
from joblib import load
import os

from tcn.tcn_model import TCNRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../..", "models/tcn_pm25.pt")
SCALER_PATH = os.path.join(BASE_DIR, "../..", "models/scaler.pkl")

def predict_next(sequence):
    scaler = load(SCALER_PATH)
    model = TCNRegressor(num_features=sequence.shape[1], seq_length=len(sequence))
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    seq_scaled = scaler.transform(sequence)
    seq_t = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0)

    pred = model(seq_t).item()
    return pred
