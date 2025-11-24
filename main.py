from src.preprocess import load_and_preprocess
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    print("▶️ 1. Preprocesando datos...")
    df = load_and_preprocess()
    print(f"Dataset final: {df.shape}")

    print("\n▶️ 2. Entrenando modelo...")
    X_test_scaled, y_test, model = train_model()

    print("\n▶️ 3. Evaluando modelo...")
    evaluate_model(X_test_scaled, y_test, model)

    print("\n✔ Pipeline completado correctamente.")

if __name__ == "__main__":
    main()
