# test_tcn_v2.py
import sys, os
import traceback
import numpy as np
import pandas as pd
import torch
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===== FIX PARA IMPORTAR EL PAQUETE TCN =====
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src")

sys.path.insert(0, ROOT)
sys.path.insert(0, SRC)

from tcn.tcn_model import TCNRegressor
from tcn.prepare_sequences import create_sequences


# ===== PATHS =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../..", "data/processed/clean_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "../..", "models/tcn_pm25_v2.pt")
SCALER_PATH = os.path.join(BASE_DIR, "../..", "models/scaler_v2.pkl")


def run_tcn_v2_tests():
    print("====================================================")
    print("=== EJECUCIÓN AUTOMÁTICA DE PRUEBAS TCN v2 (RF08) ===")
    print("====================================================\n")

    try:
        # -------- 1) Verificar dataset  --------
        print("1) Verificando dataset...")
        df = pd.read_csv(DATA_PATH)
        print(f"✔ Dataset cargado correctamente: {len(df)} filas, {len(df.columns)} columnas\n")

        # -------- 2) Preparar secuencias --------
        print("2) Generando secuencias...")
        target = "pm2_5"
        drop_cols = ["pm10", "time"]
        features = df.drop(columns=drop_cols)
        y = df[target].values

        scaler = load(SCALER_PATH)
        X_scaled = scaler.transform(features)

        SEQ_LEN = 72
        X_seq, y_seq = create_sequences(X_scaled, y, SEQ_LEN)

        print(f"✔ Secuencias generadas: {X_seq.shape}\n")

        # -------- 3) Split --------
        split = int(len(X_seq) * 0.8)
        X_test, y_test = X_seq[split:], y_seq[split:]

        X_test_t = torch.tensor(X_test, dtype=torch.float32)

        # -------- 4) Cargar modelo --------
        print("3) Cargando modelo TCN v2...")
        model = TCNRegressor(num_features=X_test.shape[2], seq_length=SEQ_LEN)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        print("✔ Modelo cargado correctamente\n")

        # -------- 5) Predicción --------
        print("4) Generando predicción de prueba...")
        with torch.no_grad():
            preds = model(X_test_t).numpy()

        print(f"✔ Predicción muestra: {preds[0]}\n")

        # -------- 6) Métricas --------
        print("5) Validando métricas mínimas...")
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        print("---- MÉTRICAS TCN v2 ----")
        print("MAE :", mae)
        print("RMSE:", rmse)
        print("R2  :", r2, "\n")

        # Criterios mínimos
        assert r2 >= 0.80, "R2 no cumple el mínimo esperado (>=0.80)"
        assert mae <= 2.0, "MAE fuera del rango esperado (<=2.0)"
        assert rmse <= 3.0, "RMSE fuera del rango esperado (<=3.0)"

        print("✔ Todas las métricas cumplen los criterios mínimos\n")

        print("====================================================")
        print("=== TODAS LAS PRUEBAS AUTOMÁTICAS SUPERADAS ✔ ===")
        print("====================================================")

    except Exception as e:
        print("\n✘ ERROR DURANTE LAS PRUEBAS AUTOMÁTICAS:")
        print(str(e))
        traceback.print_exc()
        print("\n====================================================")
        print("=== PRUEBAS AUTOMÁTICAS FALLIDAS ✘ ===")
        print("====================================================")
