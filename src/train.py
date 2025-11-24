import sys, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")

print(">>> ROOT:", ROOT)
print(">>> SRC PATH ADDED:", SRC)

sys.path.insert(0, ROOT)
sys.path.insert(0, SRC)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from joblib import dump
from src.config import CLEAN_DATA_PATH, MODEL_PATH

def train_model():
    df = pd.read_csv(CLEAN_DATA_PATH)

    feature_cols = [
        "nitrogen_dioxide", "ozone",
        "temperature_2m", "relative_humidity_2m",
        "wind_speed_10m", "wind_direction_10m",
        "precipitation", "surface_pressure",
        "hour", "day_of_week", "month",
        "is_weekend", "wind_u", "wind_v", "is_rainy"
    ]

    X = df[feature_cols]
    y = df[["pm2_5", "pm10"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Normalización
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Guardar modelo y scaler
    dump({"model": model, "scaler": scaler, "features": feature_cols}, MODEL_PATH)

    return X_test_scaled, y_test, model

if __name__ == "__main__":
    # Ejecutar entrenamiento normal
    X_test_scaled, y_test, model = train_model()

    print("\n=== Entrenamiento finalizado. Ejecutando pruebas automáticas ===\n")

    # Llamar automáticamente al módulo de evidencia
    import src.test_train as auto_test
    auto_test.run_evidence_tests()
