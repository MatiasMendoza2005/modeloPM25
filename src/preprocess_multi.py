import pandas as pd
import numpy as np
from src.config import RAW_DATA_PATH, CLEAN_DATA_PATH_V3

# Horizontes multi-head
HORIZONS = {
    "pm2_5_1h": 1,
    "pm2_5_12h": 12,
    "pm2_5_24h": 24,
    "pm2_5_72h": 72,
    "pm2_5_168h": 168
}

# --------------------------------------
# 1. FEATURES AVANZADOS
# --------------------------------------
def add_advanced_features(df):

    # --- LAGS ---
    for lag in [1,3,6,12,24]:
        df[f"pm2_5_lag_{lag}"] = df["pm2_5"].shift(lag)

    # --- MEDIA MÓVIL ---
    df["pm2_5_ma_3"] = df["pm2_5"].rolling(3).mean()
    df["pm2_5_ma_6"] = df["pm2_5"].rolling(6).mean()
    df["pm2_5_ma_12"] = df["pm2_5"].rolling(12).mean()

    df["wind_speed_ma_6"] = df["wind_speed_10m"].rolling(6).mean()
    df["temperature_ma_6"] = df["temperature_2m"].rolling(6).mean()

    # --- VORTICIDAD ---
    df["vorticity"] = df["wind_speed_10m"] * df["surface_pressure"]

    # --- SECTORES DE VIENTO ---
    bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    labels = ["N","NE","E","SE","S","SW","W","NW"]
    df["wind_sector"] = pd.cut(df["wind_direction_10m"],
                               bins=bins, labels=labels, include_lowest=True)
    df = pd.get_dummies(df, columns=["wind_sector"], prefix="wind")

    # --- ESTACIONALIDAD ---
    df["day_of_year"] = df["time"].dt.dayofyear
    df["season"] = (df["time"].dt.month % 12 + 4) // 3

    return df


# --------------------------------------
# 2. MULTI-HORIZON TARGETS
# --------------------------------------
def add_multi_horizon_targets(df):
    for name, steps in HORIZONS.items():
        df[name] = df["pm2_5"].shift(-steps)
    return df


# --------------------------------------
# 3. PIPELINE COMPLETO
# --------------------------------------
def load_and_preprocess_v3_multi():

    print("=== Ejecutando preprocess v3 multi-horizon FULL ===")

    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["time"])

    # Features base
    base_features = [
        "nitrogen_dioxide", "ozone",
        "temperature_2m", "relative_humidity_2m",
        "wind_speed_10m", "wind_direction_10m",
        "precipitation", "surface_pressure",
        "hour", "day_of_week", "month",
        "is_weekend", "wind_u", "wind_v", "is_rainy"
    ]

    targets = ["pm2_5", "pm10"]

    df = df[["time"] + base_features + targets]

    # Añadir features avanzados TCN v3
    df = add_advanced_features(df)

    # Añadir horizons multi-head
    df = add_multi_horizon_targets(df)

    # Quitar cualquier NaN generado por shifts y rolling
    df = df.dropna().reset_index(drop=True)

    df.to_csv(CLEAN_DATA_PATH_V3, index=False)

    print("=== Columnas finales ===")
    print(df.columns.tolist())
    print("TOTAL:", len(df.columns))

    return df


if __name__ == "__main__":
    load_and_preprocess_v3_multi()
