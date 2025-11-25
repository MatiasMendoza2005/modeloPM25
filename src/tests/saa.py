from joblib import load
scaler = load("models/scaler_v2.pkl")
print("FEATURES V2 EXACTOS:")
print(scaler.feature_names_in_)
