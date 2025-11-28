import pandas as pd
import numpy as np
import torch
from joblib import load
from sklearn.preprocessing import MinMaxScaler

from src.tcn.tcn_model_v3 import TCNMultiTowers
from src.tcn.prepare_sequences_multi import create_sequences_multi
from src.config import CLEAN_DATA_PATH_V3

SEQ_LEN = 168
TARGET_COLS = ["pm2_5_1h","pm2_5_12h","pm2_5_24h","pm2_5_72h","pm2_5_168h"]


print("\n" + "="*80)
print("🔍 DEBUG TCN V3 — Verificación completa de datos, escalado y secuencias")
print("="*80)


# ----------------------------------------------------------------------
# 1. Carga del dataset
# ----------------------------------------------------------------------
print("\n📌 1. Cargando dataset...")
df = pd.read_csv(CLEAN_DATA_PATH_V3)
print(f"Dataset cargado: {len(df)} filas")

print("\nColumnas del dataset:")
print(df.columns.tolist())


# ----------------------------------------------------------------------
# 2. Verificación de targets multihorizon
# ----------------------------------------------------------------------
print("\n📌 2. Verificando columnas de targets multihorizon...")

missing = [col for col in TARGET_COLS if col not in df.columns]
if missing:
    print("❌ ERROR: Faltan columnas necesarias:", missing)
else:
    print("✅ Columnas de targets presentes:", TARGET_COLS)


# ----------------------------------------------------------------------
# 3. Verificación de valores Y
# ----------------------------------------------------------------------
print("\n📌 3. Inspeccionando valores originales de Y (primeras 10 filas):")
Y = df[TARGET_COLS].values
print(Y[:10])

print("\nRango de valores Y original:")
print("  min:", np.min(Y))
print("  max:", np.max(Y))


# ----------------------------------------------------------------------
# 4. Verificación de scalers (creados en memoria)
# ----------------------------------------------------------------------
print("\n📌 4. Escalando Y para ver si el rango es coherente...")

scaler_Y_test = MinMaxScaler()
Y_scaled_test = scaler_Y_test.fit_transform(Y)

print("Primeras 10 filas de Y escalado:")
print(Y_scaled_test[:10])

print("Rango de Y_scaled:")
print("  min:", Y_scaled_test.min())
print("  max:", Y_scaled_test.max())


# ----------------------------------------------------------------------
# 5. Cargar scalers del entrenamiento previo (si existen)
# ----------------------------------------------------------------------
print("\n📌 5. Intentando cargar scalers originales guardados...")

try:
    scaler_X = load("models/scaler_v3_multi_towers_X.pkl")
    scaler_Y = load("models/scaler_v3_multi_towers_Y.pkl")
    print("✅ Scalers cargados correctamente")
except:
    print("⚠️ No se pudieron cargar scalers desde models/")
    scaler_X = None
    scaler_Y = None


# ----------------------------------------------------------------------
# 6. Generar secuencias sin entrenar nada
# ----------------------------------------------------------------------
print("\n📌 6. Creando secuencias (sin entrenamiento)...")

feature_cols = [c for c in df.columns if c not in ["time", "pm2_5", "pm10"] + TARGET_COLS]
X = df[feature_cols].values

scaler_X_test = MinMaxScaler()
X_scaled_test = scaler_X_test.fit_transform(X)

X_seq, Y_seq = create_sequences_multi(X_scaled_test, Y_scaled_test, SEQ_LEN)

print("Shape de X_seq:", X_seq.shape)
print("Shape de Y_seq:", Y_seq.shape)

print("\nPrimer target Y_seq (escalado):")
print(Y_seq[0])


# ----------------------------------------------------------------------
# 7. Verificar forward-pass del modelo aleatorio
# ----------------------------------------------------------------------
print("\n📌 7. Probar una forward-pass del modelo sin entrenar...")

num_inputs = X_seq.shape[2]
model = TCNMultiTowers(num_inputs=num_inputs, use_global_pooling=True)

X_tensor = torch.tensor(X_seq[:1], dtype=torch.float32).permute(0,2,1)

pred = model(X_tensor)

print("\nSalida del modelo sin entrenar (debe ser forma (1,5)):")
print(pred)
print("Forma:", pred.shape)


# ----------------------------------------------------------------------
# 8. Verificación de consistencia scaler_Y
# ----------------------------------------------------------------------
if scaler_Y is not None:
    print("\n📌 8. Verificando coherencia entre scaler_Y guardado y los Y actuales...")

    # Escalar Y actual con scaler cargado
    try:
        Y_scaled_from_saved = scaler_Y.transform(Y)
        print("Primeras 10 filas Y escalado con scaler guardado:")
        print(Y_scaled_from_saved[:10])
    except:
        print("❌ ERROR: El scaler_Y guardado NO es compatible con el Y actual")

print("\n" + "="*80)
print("🔍 FIN DEL SCRIPT DE DIAGNÓSTICO")
print("="*80)
