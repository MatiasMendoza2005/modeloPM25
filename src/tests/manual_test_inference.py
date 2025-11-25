from src.api.inference import predictor_v2, predictor_v3
import pandas as pd

# Cargar dataset v3 COMPLETO
df = pd.read_csv("data/processed/clean_dataset_v3.csv")

# Tomar las últimas 72 horas para testear
sample = df.tail(72)

# ---- INPUT V2 ----
FEATURES_V2 = [
    "nitrogen_dioxide", "ozone", "temperature_2m",
    "relative_humidity_2m", "wind_speed_10m",
    "wind_direction_10m", "precipitation", "surface_pressure",
    "hour", "day_of_week", "month", "is_weekend",
    "wind_u", "wind_v", "is_rainy",
    "pm2_5"
]

input_v2 = {col: sample[col].tolist() for col in FEATURES_V2}

print("\n=== INPUT V2 FINAL ===")
print(input_v2.keys())


# ---- INPUT V3 ----
cols_to_remove = [
    "time", "pm2_5", "pm10",
    "pm2_5_1h","pm2_5_12h","pm2_5_24h",
    "pm2_5_72h","pm2_5_168h"
]

input_v3 = {col: sample[col].tolist()
            for col in sample.columns if col not in cols_to_remove}

print("\n=== INPUT V3 FINAL ===")
print(len(input_v3), "features")  # debe ser 36

# =======================
#   PRUEBA V2
# =======================
print("\n=== PREDICCIÓN TCN V2 ===")
pred_v2 = predictor_v2.predict(input_v2)
print(pred_v2)

# Valor real (1 hora adelante)
real_v2 = df["pm2_5"].iloc[-1]
print("Valor real PM2.5 actual:", real_v2)


# =======================
#   PRUEBA V3
# =======================
print("\n=== PREDICCIÓN TCN V3 MULTI-HORIZON ===")
pred_v3 = predictor_v3.predict(input_v3)
print(pred_v3)

# Mostrar valores reales para comparar
print("\n=== VALORES REALES ===")
print({
    "pm25_1h": df["pm2_5_1h"].iloc[-1],
    "pm25_12h": df["pm2_5_12h"].iloc[-1],
    "pm25_24h": df["pm2_5_24h"].iloc[-1],
    "pm25_72h": df["pm2_5_72h"].iloc[-1],
    "pm25_168h": df["pm2_5_168h"].iloc[-1],
})
