import pandas as pd
import numpy as np
from src.config import RAW_DATA_PATH, CLEAN_DATA_PATH_V3


# --------------------------------------
# 1. AÑADIR FEATURES AVANZADOS (TCN v3)
# --------------------------------------
def add_advanced_features(df):

    # ----------------------------
    # 1. LAG FEATURES
    # ----------------------------
    lags = [1, 3, 6, 12, 24]
    for lag in lags:
        df[f"pm2_5_lag_{lag}"] = df["pm2_5"].shift(lag)

    # ----------------------------
    # 2. MEDIA MÓVIL (MA)
    # ----------------------------
    df["pm2_5_ma_3"] = df["pm2_5"].rolling(3).mean()
    df["pm2_5_ma_6"] = df["pm2_5"].rolling(6).mean()
    df["pm2_5_ma_12"] = df["pm2_5"].rolling(12).mean()

    df["wind_speed_ma_6"] = df["wind_speed_10m"].rolling(6).mean()
    df["temperature_ma_6"] = df["temperature_2m"].rolling(6).mean()

    # ----------------------------
    # 3. VORTICIDAD
    # ----------------------------
    df["vorticity"] = df["wind_speed_10m"] * df["surface_pressure"]

    # ----------------------------
    # 4. SECTORES DEL VIENTO
    # ----------------------------
    bins = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

    df["wind_sector"] = pd.cut(df["wind_direction_10m"],
                               bins=bins,
                               labels=labels,
                               include_lowest=True)

    df = pd.get_dummies(df, columns=["wind_sector"], prefix="wind")

    # ----------------------------
    # 5. VARIABLES ESTACIONALES
    # ----------------------------
    df["day_of_year"] = df["time"].dt.dayofyear
    df["season"] = (df["time"].dt.month % 12 + 4) // 3

    return df


# --------------------------------------
# PROCESAR DATASET COMPLETO (TCN v3)
# --------------------------------------
def load_and_preprocess():

    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["time"])

    # Features base
    feature_cols = [
        "nitrogen_dioxide", "ozone",
        "temperature_2m", "relative_humidity_2m",
        "wind_speed_10m", "wind_direction_10m",
        "precipitation", "surface_pressure",
        "hour", "day_of_week", "month",
        "is_weekend", "wind_u", "wind_v", "is_rainy"
    ]

    target_cols = ["pm2_5", "pm10"]

    print(df.columns.tolist())
    print(len(df.columns))

    df = df[["time"] + feature_cols + target_cols]

    # 🔥 NUEVOS FEATURES TCN v3
    df = add_advanced_features(df)

    # Quitar NaNs generados por rolling/lags
    df = df.dropna().reset_index(drop=True)

    df.to_csv(CLEAN_DATA_PATH_V3, index=False)
    return df

if __name__ == "__main__":
    print("\n=== Ejecutando preprocess TCN v3 ===")
    df = load_and_preprocess()
    print("\n=== Preprocess completado ===")
    print(df.columns.tolist())
    print("TOTAL:", len(df.columns))
