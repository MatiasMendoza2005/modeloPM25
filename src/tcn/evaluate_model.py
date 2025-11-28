"""
Script para evaluar modelos TCN ya entrenados.
Permite calcular métricas completas para cualquier modelo guardado.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from joblib import load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

# Agregar paths
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.tcn.tcn_model import TCNRegressor
from src.tcn.tcn_model_v3 import TCNMultiTowers
from src.tcn.prepare_sequences import create_sequences
from src.tcn.prepare_sequences_multi import create_sequences_multi
from src.config import CLEAN_DATA_PATH, CLEAN_DATA_PATH_V3


# ======================================================
# FUNCIONES DE MÉTRICAS
# ======================================================

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    eps = 1e-6
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100


def skill_score(y_true, y_pred):
    """Compara contra modelo naive de persistencia."""
    if len(y_true) < 2:
        return np.nan
    
    naive = y_true[:-1]
    real = y_true[1:]
    model = y_pred[1:]
    
    mse_naive = np.mean((real - naive)**2)
    mse_model = np.mean((real - model)**2)
    
    if mse_naive == 0:
        return np.nan
    
    return 1 - (mse_model / mse_naive)


def calculate_metrics(y_true, y_pred, horizon_name=""):
    """Calcula todas las métricas para un horizonte"""
    
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    
    # Métricas básicas
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape_val = mape(y_true, y_pred)
    
    # Correlación de Pearson
    try:
        corr, p_value = pearsonr(y_true, y_pred)
    except:
        corr, p_value = np.nan, np.nan
    
    # Skill score
    skill = skill_score(y_true, y_pred)
    
    # Errores adicionales
    max_error = np.max(np.abs(y_true - y_pred))
    median_ae = np.median(np.abs(y_true - y_pred))
    
    metrics = {
        "horizonte": horizon_name,
        "MAE": mae,
        "RMSE": rmse,
        "MSE": mse,
        "R²": r2,
        "MAPE (%)": mape_val,
        "Correlación": corr,
        "Skill Score": skill,
        "Max Error": max_error,
        "Median AE": median_ae,
        "N samples": len(y_true)
    }
    
    return metrics


def print_metrics_table(metrics_list):
    """Imprime tabla de métricas formateada"""
    print("\n" + "="*100)
    print(f"{'Horizonte':<12} | {'MAE':>6} | {'RMSE':>6} | {'R²':>6} | {'MAPE%':>7} | {'Corr':>6} | {'Skill':>7} | {'MaxErr':>7}")
    print("="*100)
    
    for m in metrics_list:
        print(f"{m['horizonte']:<12} | "
              f"{m['MAE']:>6.2f} | "
              f"{m['RMSE']:>6.2f} | "
              f"{m['R²']:>6.3f} | "
              f"{m['MAPE (%)']:>7.1f} | "
              f"{m['Correlación']:>6.3f} | "
              f"{m['Skill Score']:>7.3f} | "
              f"{m['Max Error']:>7.2f}")
    
    print("="*100)


def print_detailed_metrics(metrics):
    """Imprime métricas detalladas de un horizonte"""
    print(f"\n=== {metrics['horizonte']} - Métricas Detalladas ===")
    print(f"  MAE:            {metrics['MAE']:.4f}")
    print(f"  RMSE:           {metrics['RMSE']:.4f}")
    print(f"  MSE:            {metrics['MSE']:.4f}")
    print(f"  R²:             {metrics['R²']:.4f}")
    print(f"  MAPE:           {metrics['MAPE (%)']:.2f}%")
    print(f"  Correlación:    {metrics['Correlación']:.4f}")
    print(f"  Skill Score:    {metrics['Skill Score']:.4f}")
    print(f"  Max Error:      {metrics['Max Error']:.4f}")
    print(f"  Median AE:      {metrics['Median AE']:.4f}")
    print(f"  N samples:      {metrics['N samples']}")


# ======================================================
# EVALUACIÓN TCN v2 (Single Output)
# ======================================================

def evaluate_tcn_v2(model_path, scaler_path, data_path, seq_len=96, device='cpu'):
    """Evalúa TCN v2 (single output)"""
    
    print("\n=== Evaluando TCN v2 (Single Output) ===")
    print(f"Modelo: {model_path}")
    print(f"Device: {device}")
    
    # Cargar datos
    df = pd.read_csv(data_path)
    df = df.sort_values("time").reset_index(drop=True)
    
    target = "pm2_5"
    drop_cols = ["pm10", "time"]
    
    features = df.drop(columns=drop_cols)
    y = df[target].values
    
    # Cargar scaler y escalar
    scaler = load(scaler_path)
    X_scaled = scaler.transform(features)
    
    # Crear secuencias
    X_seq, y_seq = create_sequences(X_scaled, y, seq_len)
    
    # Split (80/20)
    split = int(len(X_seq) * 0.8)
    X_test = X_seq[split:]
    y_test = y_seq[split:]
    
    # A tensor
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    # Cargar modelo
    num_features = X_test.shape[2]
    model = TCNRegressor(num_features=num_features, seq_length=seq_len)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Predicciones
    with torch.no_grad():
        preds = model(X_test_t).cpu().numpy().flatten()
    
    # Calcular métricas
    metrics = calculate_metrics(y_test, preds, horizon_name="pm2_5_1h")
    
    print_detailed_metrics(metrics)
    
    return metrics, y_test, preds


# ======================================================
# EVALUACIÓN TCN v3 (Multi-Horizon)
# ======================================================

def evaluate_tcn_v3(model_path, scaler_path, data_path, seq_len=336, device='cpu'):
    """Evalúa TCN v3 (multi-horizon con torres)"""
    
    print("\n=== Evaluando TCN v3 (Multi-Horizon) ===")
    print(f"Modelo: {model_path}")
    print(f"Device: {device}")
    
    HORIZON_NAMES = ["pm2_5_1h", "pm2_5_12h", "pm2_5_24h", "pm2_5_72h", "pm2_5_168h"]
    
    # Cargar datos
    df = pd.read_csv(data_path)
    
    target_cols = HORIZON_NAMES
    feature_cols = [c for c in df.columns if c not in ["time", "pm2_5", "pm10"] + target_cols]
    
    X = df[feature_cols].values
    Y = df[target_cols].values
    
    # Cargar scaler y escalar
    scaler = load(scaler_path)
    X_scaled = scaler.transform(X)
    
    # Crear secuencias
    X_seq, Y_seq = create_sequences_multi(X_scaled, Y, seq_len)
    
    # Split (85% para train+val, 15% test)
    val_split = int(len(X_seq) * 0.85)
    X_test = X_seq[val_split:]
    Y_test = Y_seq[val_split:]
    
    # A tensor
    X_test_t = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1).to(device)
    
    # Cargar modelo
    num_features = X_test.shape[2]
    model = TCNMultiTowers(num_inputs=num_features)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Predicciones
    with torch.no_grad():
        preds = model(X_test_t).cpu().numpy()
    
    # Calcular métricas por horizonte
    all_metrics = []
    
    for i, horizon_name in enumerate(HORIZON_NAMES):
        y_true = Y_test[:, i]
        y_pred = preds[:, i]
        
        metrics = calculate_metrics(y_true, y_pred, horizon_name=horizon_name)
        all_metrics.append(metrics)
    
    # Imprimir tabla resumen
    print_metrics_table(all_metrics)
    
    # Imprimir detalladas si se requiere
    print("\n=== Métricas Detalladas por Horizonte ===")
    for metrics in all_metrics:
        print_detailed_metrics(metrics)
    
    return all_metrics, Y_test, preds


# ======================================================
# COMPARAR MODELOS
# ======================================================

def compare_models(model_configs):
    """
    Compara múltiples modelos.
    
    model_configs: lista de dicts con:
        {
            'name': 'TCN v2',
            'type': 'v2' o 'v3',
            'model_path': 'path/to/model.pt',
            'scaler_path': 'path/to/scaler.pkl',
            'data_path': 'path/to/data.csv',
            'seq_len': 96 o 336,
            'device': 'cpu' o 'cuda'
        }
    """
    
    print("\n" + "="*100)
    print("COMPARACIÓN DE MODELOS")
    print("="*100)
    
    results = {}
    
    for config in model_configs:
        print(f"\n{'='*100}")
        print(f"Evaluando: {config['name']}")
        print(f"{'='*100}")
        
        if config['type'] == 'v2':
            metrics, y_true, y_pred = evaluate_tcn_v2(
                config['model_path'],
                config['scaler_path'],
                config['data_path'],
                config['seq_len'],
                config['device']
            )
            results[config['name']] = {'metrics': [metrics], 'y_true': y_true, 'y_pred': y_pred}
        
        elif config['type'] == 'v3':
            metrics_list, y_true, y_pred = evaluate_tcn_v3(
                config['model_path'],
                config['scaler_path'],
                config['data_path'],
                config['seq_len'],
                config['device']
            )
            results[config['name']] = {'metrics': metrics_list, 'y_true': y_true, 'y_pred': y_pred}
    
    return results


# ======================================================
# CLI
# ======================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluar modelos TCN entrenados')
    
    parser.add_argument('--model', type=str, required=True, help='Ruta al modelo (.pt)')
    parser.add_argument('--scaler', type=str, required=True, help='Ruta al scaler (.pkl)')
    parser.add_argument('--data', type=str, help='Ruta a los datos CSV (opcional, usa config por defecto)')
    parser.add_argument('--type', type=str, choices=['v2', 'v3'], required=True, help='Tipo de modelo (v2 o v3)')
    parser.add_argument('--seq-len', type=int, help='Longitud de secuencia (opcional, usa default)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device para inferencia')
    
    args = parser.parse_args()
    
    # Defaults
    if args.data is None:
        args.data = CLEAN_DATA_PATH_V3 if args.type == 'v3' else CLEAN_DATA_PATH
    
    if args.seq_len is None:
        args.seq_len = 336 if args.type == 'v3' else 96
    
    # Evaluar
    if args.type == 'v2':
        evaluate_tcn_v2(args.model, args.scaler, args.data, args.seq_len, args.device)
    else:
        evaluate_tcn_v3(args.model, args.scaler, args.data, args.seq_len, args.device)


if __name__ == "__main__":
    main()
