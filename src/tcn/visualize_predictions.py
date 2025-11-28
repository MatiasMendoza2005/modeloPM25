"""
Script para visualizar predicciones vs valores reales de modelos TCN.
Genera gráficos de análisis visual.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.tcn.evaluate_model import evaluate_tcn_v2, evaluate_tcn_v3

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)


def plot_predictions_vs_actual(y_true, y_pred, horizon_name, save_path=None):
    """Genera gráfico de predicciones vs valores reales"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Análisis de Predicciones - {horizon_name}', fontsize=16, fontweight='bold')
    
    # 1. Serie temporal
    ax1 = axes[0, 0]
    x = np.arange(len(y_true))
    ax1.plot(x, y_true, label='Real', color='blue', alpha=0.7, linewidth=1)
    ax1.plot(x, y_pred, label='Predicción', color='red', alpha=0.7, linewidth=1)
    ax1.set_xlabel('Muestras')
    ax1.set_ylabel('PM2.5 (µg/m³)')
    ax1.set_title('Serie Temporal: Real vs Predicción')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Scatter plot
    ax2 = axes[0, 1]
    ax2.scatter(y_true, y_pred, alpha=0.3, s=10)
    
    # Línea de identidad perfecta
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción Perfecta')
    
    ax2.set_xlabel('PM2.5 Real (µg/m³)')
    ax2.set_ylabel('PM2.5 Predicho (µg/m³)')
    ax2.set_title('Scatter Plot: Real vs Predicción')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Distribución de errores
    ax3 = axes[1, 0]
    errors = y_pred - y_true
    ax3.hist(errors, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Error = 0')
    ax3.axvline(x=np.mean(errors), color='green', linestyle='--', linewidth=2, label=f'Media = {np.mean(errors):.2f}')
    ax3.set_xlabel('Error de Predicción (µg/m³)')
    ax3.set_ylabel('Frecuencia')
    ax3.set_title('Distribución de Errores')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Residuos
    ax4 = axes[1, 1]
    ax4.scatter(y_pred, errors, alpha=0.3, s=10, color='orange')
    ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('PM2.5 Predicho (µg/m³)')
    ax4.set_ylabel('Residuos (µg/m³)')
    ax4.set_title('Análisis de Residuos')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_multi_horizon_comparison(y_true_dict, y_pred_dict, save_path=None):
    """Compara múltiples horizontes en un solo gráfico"""
    
    horizons = list(y_true_dict.keys())
    n_horizons = len(horizons)
    
    fig, axes = plt.subplots(n_horizons, 2, figsize=(16, 4*n_horizons))
    fig.suptitle('Comparación Multi-Horizon', fontsize=16, fontweight='bold')
    
    for i, horizon in enumerate(horizons):
        y_true = y_true_dict[horizon]
        y_pred = y_pred_dict[horizon]
        
        # Serie temporal (primeras 500 muestras)
        ax1 = axes[i, 0] if n_horizons > 1 else axes[0]
        n_samples = min(500, len(y_true))
        x = np.arange(n_samples)
        ax1.plot(x, y_true[:n_samples], label='Real', color='blue', alpha=0.7, linewidth=1.5)
        ax1.plot(x, y_pred[:n_samples], label='Predicción', color='red', alpha=0.7, linewidth=1.5)
        ax1.set_xlabel('Muestras')
        ax1.set_ylabel('PM2.5 (µg/m³)')
        ax1.set_title(f'{horizon} - Serie Temporal')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Scatter plot
        ax2 = axes[i, 1] if n_horizons > 1 else axes[1]
        ax2.scatter(y_true, y_pred, alpha=0.2, s=8)
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        # Calcular R²
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        
        ax2.set_xlabel('PM2.5 Real (µg/m³)')
        ax2.set_ylabel('PM2.5 Predicho (µg/m³)')
        ax2.set_title(f'{horizon} - Scatter (R²={r2:.3f})')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Gráfico guardado: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_tcn_v2(model_path, scaler_path, data_path, output_dir='outputs/plots'):
    """Visualiza resultados de TCN v2"""
    
    print("\n=== Generando visualizaciones para TCN v2 ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluar modelo
    metrics, y_true, y_pred = evaluate_tcn_v2(
        model_path=model_path,
        scaler_path=scaler_path,
        data_path=data_path,
        seq_len=96,
        device='cpu'
    )
    
    # Generar gráfico
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f'tcn_v2_analysis_{timestamp}.png')
    
    plot_predictions_vs_actual(y_true, y_pred, 'TCN v2 - pm2_5_1h', save_path)
    
    return metrics


def visualize_tcn_v3(model_path, scaler_path, data_path, output_dir='outputs/plots'):
    """Visualiza resultados de TCN v3"""
    
    print("\n=== Generando visualizaciones para TCN v3 ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluar modelo
    metrics_list, y_true, y_pred = evaluate_tcn_v3(
        model_path=model_path,
        scaler_path=scaler_path,
        data_path=data_path,
        seq_len=336,
        device='cpu'
    )
    
    HORIZON_NAMES = ["pm2_5_1h", "pm2_5_12h", "pm2_5_24h", "pm2_5_72h", "pm2_5_168h"]
    
    # Gráfico individual por horizonte
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for i, horizon_name in enumerate(HORIZON_NAMES):
        save_path = os.path.join(output_dir, f'tcn_v3_{horizon_name}_{timestamp}.png')
        plot_predictions_vs_actual(y_true[:, i], y_pred[:, i], f'TCN v3 - {horizon_name}', save_path)
    
    # Gráfico comparativo multi-horizon
    y_true_dict = {horizon: y_true[:, i] for i, horizon in enumerate(HORIZON_NAMES)}
    y_pred_dict = {horizon: y_pred[:, i] for i, horizon in enumerate(HORIZON_NAMES)}
    
    save_path_comp = os.path.join(output_dir, f'tcn_v3_comparison_{timestamp}.png')
    plot_multi_horizon_comparison(y_true_dict, y_pred_dict, save_path_comp)
    
    return metrics_list


def main():
    """Genera visualizaciones para todos los modelos disponibles"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualizar predicciones de modelos TCN')
    parser.add_argument('--model', type=str, required=True, help='Ruta al modelo')
    parser.add_argument('--scaler', type=str, required=True, help='Ruta al scaler')
    parser.add_argument('--data', type=str, help='Ruta a datos CSV')
    parser.add_argument('--type', type=str, choices=['v2', 'v3'], required=True, help='Tipo de modelo')
    parser.add_argument('--output', type=str, default='outputs/plots', help='Directorio de salida')
    
    args = parser.parse_args()
    
    # Defaults
    from src.config import CLEAN_DATA_PATH, CLEAN_DATA_PATH_V3
    if args.data is None:
        args.data = CLEAN_DATA_PATH_V3 if args.type == 'v3' else CLEAN_DATA_PATH
    
    # Visualizar
    if args.type == 'v2':
        visualize_tcn_v2(args.model, args.scaler, args.data, args.output)
    else:
        visualize_tcn_v3(args.model, args.scaler, args.data, args.output)
    
    print(f"\n✅ Visualizaciones completadas en: {args.output}")


if __name__ == "__main__":
    main()
