import numpy as np
import torch
from joblib import load

from ..tcn.tcn_model_v3 import TCNMultiTowers
from ..tcn.prepare_sequences_multi import create_sequences_multi

SEQ_LEN = 72
HORIZON_NAMES = ["pm25_1h","pm25_12h","pm25_24h","pm25_72h","pm25_168h"]

class PredictorTCNTowers:
    def __init__(self):
        self.scaler = load("models/scaler_v3_multi_towers.pkl")

        num_features = self.scaler.n_features_in_
        self.model = TCNMultiTowers(num_inputs=num_features)

        self.model.load_state_dict(
            torch.load("models/tcn_pm25_v3_multi_towers.pt", map_location="cpu")
        )
        self.model.eval()

    def predict(self, data_dict):
        X = np.column_stack(list(data_dict.values()))
        X_scaled = self.scaler.transform(X)

        seq, _ = create_sequences_multi(X_scaled, [[0]*5], SEQ_LEN)
        X_tensor = torch.tensor(seq, dtype=torch.float32).permute(0,2,1)

        with torch.no_grad():
            pred = self.model(X_tensor)[0].numpy()

        return {HORIZON_NAMES[i]: float(pred[i]) for i in range(5)}


predictor_multi_towers = PredictorTCNTowers()
