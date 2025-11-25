import numpy as np
import torch
from joblib import load
from ..tcn.create_single_sequence import create_single_sequence

# ---------------------------
# IMPORTS DE MODELOS
# ---------------------------
from ..tcn.tcn_model import TCNRegressor               # v2
from ..tcn.tcn_model_v3 import TCNMultiTowers           # v3 multi-head

from ..tcn.prepare_sequences import create_sequences               # v2
from ..tcn.prepare_sequences_multi import create_sequences_multi   # v3 multi-horizon


# ============================================================
#   CONFIGURACIÓN
# ============================================================

SEQ_LEN_V2 = 72
SEQ_LEN_V3 = 168

# Nombre de las salidas del modelo multi-head
HORIZON_NAMES = ["pm25_1h", "pm25_12h", "pm25_24h", "pm25_72h", "pm25_168h"]

FEATURES_V2 = [
    "nitrogen_dioxide",
    "ozone",
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "surface_pressure",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "wind_u",
    "wind_v",
    "is_rainy",
    "pm2_5"   # ← ESTA ES LA QUE FALTABA
]


# ============================================================
#   PREDICTOR TCN v2 (single-output)
# ============================================================

class PredictorTCNv2:
    def __init__(self):
        # Cargar scaler
        self.scaler = load("models/scaler_v2.pkl")

        num_features = self.scaler.n_features_in_

        # Crear modelo
        self.model = TCNRegressor(num_features, SEQ_LEN_V2)

        # Cargar pesos del modelo
        self.model.load_state_dict(
            torch.load("models/tcn_pm25_v2.pt", map_location="cpu")
        )
        self.model.eval()

    def predict(self, data_dict):
        # Convertir JSON → matriz
        X = np.column_stack([data_dict[col] for col in FEATURES_V2])

        # Normalizar
        X_scaled = self.scaler.transform(X)

        # Crear secuencia
        #seq, _ = create_sequences(X_scaled, [0], SEQ_LEN_V2)
        seq = create_single_sequence(X_scaled, SEQ_LEN_V2)
        X_tensor = torch.tensor(seq, dtype=torch.float32)

        with torch.no_grad():
            pred = self.model(X_tensor).item()

        return {"pm25_next": float(pred)}


# ============================================================
#   PREDICTOR TCN v3 MULTI-HEAD (5 horizontes)
# ============================================================

class PredictorTCNv3Multi:
    def __init__(self):
        # Cargar scaler
        self.scaler = load("models/scaler_v3_multi_towers.pkl")

        num_features = self.scaler.n_features_in_

        # Crear modelo multi-head
        self.model = TCNMultiTowers(num_inputs=num_features)

        # Cargar pesos del modelo
        self.model.load_state_dict(
            torch.load("models/tcn_pm25_v3_multi_towers.pt", map_location="cpu")
        )
        self.model.eval()

    def predict(self, data_dict):
        # Convertir JSON → matriz
        X = np.column_stack(list(data_dict.values()))

        # Normalizar
        X_scaled = self.scaler.transform(X)

        # Crear secuencia multi-horizonte
        #seq, _ = create_sequences_multi(X_scaled, [[0]*5], SEQ_LEN_V3)
        #seq = create_single_sequence(X_scaled, SEQ_LEN_V3)
        # Si hay EXACTAMENTE seq_len filas, crear 1 secuencia a mano
        if len(X_scaled) == SEQ_LEN_V3:
            seq = X_scaled.reshape(1, SEQ_LEN_V3, -1)
        else:
            seq, _ = create_sequences_multi(X_scaled, [[0]*5], SEQ_LEN_V3)

        X_tensor = torch.tensor(seq, dtype=torch.float32).permute(0, 2, 1)

        with torch.no_grad():
            pred = self.model(X_tensor)[0].numpy()

        # Convertir a JSON de salida
        return {HORIZON_NAMES[i]: float(pred[i]) for i in range(5)}




# ============================================================
#   INSTANCIAS GLOBALES — listas para usar en la API
# ============================================================

predictor_v2 = PredictorTCNv2()
predictor_v3 = PredictorTCNv3Multi()
