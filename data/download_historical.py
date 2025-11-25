import requests
import pandas as pd

AIR_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality?"
    "latitude=-17.7863&longitude=-63.1812&"
    "hourly=pm10,pm2_5,nitrogen_dioxide,ozone&"
    "start_date=2013-01-01&end_date=2025-06-30&format=json"
)

METEO_URL = (
    "https://archive-api.open-meteo.com/v1/archive?"
    "latitude=-17.7863&longitude=-63.1812&"
    "hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,"
    "wind_direction_10m,precipitation,surface_pressure&"
    "start_date=2013-01-01&end_date=2025-06-30&format=json"
)

def download_historical():
    print("=== Descargando datos de calidad del aire ===")
    air = requests.get(AIR_URL).json()

    print("=== Descargando datos meteorológicos ===")
    met = requests.get(METEO_URL).json()

    df_air = pd.DataFrame({**air["hourly"]})
    df_meteo = pd.DataFrame({**met["hourly"]})

    # Asegurar que ambas tengan columna "time"
    if "time" not in df_air or "time" not in df_meteo:
        raise RuntimeError("La API no devolvió la columna 'time'.")

    # Merge por tiempo (inner join)
    df = pd.merge(df_air, df_meteo, on="time", how="inner")

    df.to_csv("data/raw/dataset_pm_scz_2013_2025.csv", index=False)
    print("Guardado: data/raw/dataset_pm_scz_2013_2025.csv")

    return df

if __name__ == "__main__":
    download_historical()