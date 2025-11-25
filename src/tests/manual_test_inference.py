from src.api.inference import predictor_v2, predictor_v3
import pandas as pd

SEQ_LEN_V2 = 72
SEQ_LEN_V3 = 168   # multi-head requiere las últimas 168 horas


print("\n=== CARGANDO DATASET V3 ===")
df = pd.read_csv("data/processed/clean_dataset_v3.csv")
print("Filas totales:", len(df))


# ======================================================
#   VALIDACIÓN DE LONGITUD MÍNIMA PARA V3
# ======================================================
if len(df) < SEQ_LEN_V3 + 200:
    print("\n⚠️ ADVERTENCIA IMPORTANTE")
    print(f"El dataset tiene {len(df)} filas, pero V3 requiere mínimo {SEQ_LEN_V3} + 200 para ser estable.")
    print("La inferencia puede fallar o ser imprecisa.\n")


# ===============================
#   TOMAR LAS ÚLTIMAS SECUENCIAS
# ===============================

sample_v2 = df.tail(SEQ_LEN_V2)
sample_v3 = df.tail(SEQ_LEN_V3)

# ======================================================
#   ---- INPUT PARA V2 ----
# ======================================================

FEATURES_V2 = [
    "nitrogen_dioxide", "ozone", "temperature_2m",
    "relative_humidity_2m", "wind_speed_10m",
    "wind_direction_10m", "precipitation", "surface_pressure",
    "hour", "day_of_week", "month", "is_weekend",
    "wind_u", "wind_v", "is_rainy",
    "pm2_5"
]

print("\n=== PREPARANDO INPUT PARA V2 ===")

# Validar columnas
missing_v2 = [c for c in FEATURES_V2 if c not in df.columns]
if missing_v2:
    raise KeyError(f"❌ El dataset v3 no contiene las columnas necesarias para v2: {missing_v2}")

input_v2 = {col: sample_v2[col].tolist() for col in FEATURES_V2}

print("OK → FEATURES V2 preparados:", len(input_v2))


# ======================================================
#   ---- INPUT PARA V3 ----
# ======================================================

print("\n=== PREPARANDO INPUT PARA V3 ===")

cols_to_remove = [
    "time", "pm2_5", "pm10",
    "pm2_5_1h","pm2_5_12h","pm2_5_24h",
    "pm2_5_72h","pm2_5_168h"
]

feature_cols_v3 = [c for c in df.columns if c not in cols_to_remove]

# Validar número correcto (36 features)
if len(feature_cols_v3) != predictor_v3.scaler.n_features_in_:
    raise ValueError(f"""
❌ ERROR: El scaler de V3 espera {predictor_v3.scaler.n_features_in_} features,
pero el dataset tiene {len(feature_cols_v3)} features.

Features detectados:
{feature_cols_v3}
""")

input_v3 = {col: sample_v3[col].tolist() for col in feature_cols_v3}

print("OK → FEATURES V3 preparados:", len(input_v3))


# ======================================================
#   EJECUTAR PREDICCIONES
# ======================================================

# ---------- V2 ----------
print("\n=== PREDICCIÓN TCN V2 ===")
pred_v2 = predictor_v2.predict(input_v2)
print("Predicción:", pred_v2)

# Valor real
real_pm_now = df["pm2_5"].iloc[-1]
print("Valor real PM2.5 actual:", real_pm_now)


# ---------- V3 ----------
print("\n=== PREDICCIÓN TCN V3 (MULTI-HORIZON) ===")
pred_v3 = predictor_v3.predict(input_v3)
print(pred_v3)

# Valores reales
print("\n=== VALORES REALES ===")
real_dict = {
    "pm25_1h": df["pm2_5_1h"].iloc[-SEQ_LEN_V3],
    "pm25_12h": df["pm2_5_12h"].iloc[-SEQ_LEN_V3],
    "pm25_24h": df["pm2_5_24h"].iloc[-SEQ_LEN_V3],
    "pm25_72h": df["pm2_5_72h"].iloc[-SEQ_LEN_V3],
    "pm25_168h": df["pm2_5_168h"].iloc[-SEQ_LEN_V3],
}
print(real_dict)

print("\n=== TEST COMPLETADO ===")
