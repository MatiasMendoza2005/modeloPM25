# test_train.py
import sys, os
import traceback
import pandas as pd
from src.train import train_model
from src.config import CLEAN_DATA_PATH

def run_evidence_tests():
    print("=== EJECUCIÓN AUTOMÁTICA DE PRUEBAS PARA EVIDENCIA ===")

    try:
        print("\n1) Verificando existencia del dataset...")
        df = pd.read_csv(CLEAN_DATA_PATH)
        print(f"✔ Dataset encontrado: {len(df)} filas, {len(df.columns)} columnas")

        print("\n2) Ejecutando entrenamiento automático...")
        X_test_scaled, y_test, model = train_model()
        print("✔ Entrenamiento completado correctamente.")

        print("\n3) Validando predicción...")
        sample = X_test_scaled[0].reshape(1, -1)
        pred = model.predict(sample)
        print(f"✔ Predicción generada: {pred}")

        print("\n=== TODAS LAS PRUEBAS SUPERADAS ===")

    except Exception as e:
        print("\n✘ ERROR DURANTE LAS PRUEBAS AUTOMÁTICAS:")
        print(str(e))
        traceback.print_exc()

if __name__ == "__main__":
    run_evidence_tests()