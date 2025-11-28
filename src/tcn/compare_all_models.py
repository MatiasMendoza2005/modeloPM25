"""
Script para comparar todos los modelos TCN entrenados.
Ejecuta evaluaciones completas y genera un reporte comparativo.
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from src.tcn.evaluate_model import compare_models
from src.config import CLEAN_DATA_PATH, CLEAN_DATA_PATH_V3


def main():
    """Compara todos los modelos disponibles"""
    
    base_models = os.path.join(ROOT, "models")
    
    # Configuración de modelos a comparar
    model_configs = []
    
    # TCN v2
    v2_model = os.path.join(base_models, "tcn_pm25_v2.pt")
    v2_scaler = os.path.join(base_models, "scaler_v2.pkl")
    
    if os.path.exists(v2_model) and os.path.exists(v2_scaler):
        model_configs.append({
            'name': 'TCN v2 (Single Output)',
            'type': 'v2',
            'model_path': v2_model,
            'scaler_path': v2_scaler,
            'data_path': CLEAN_DATA_PATH,
            'seq_len': 96,
            'device': 'cpu'
        })
    
    # TCN v3 - Mejor modelo guardado
    v3_best = os.path.join(base_models, "tcn_pm25_v3_multi_towers_best.pt")
    v3_scaler = os.path.join(base_models, "scaler_v3_multi_towers.pkl")
    
    if os.path.exists(v3_best) and os.path.exists(v3_scaler):
        model_configs.append({
            'name': 'TCN v3 Multi-Towers (BEST)',
            'type': 'v3',
            'model_path': v3_best,
            'scaler_path': v3_scaler,
            'data_path': CLEAN_DATA_PATH_V3,
            'seq_len': 336,
            'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
        })
    
    # TCN v3 - Último modelo
    v3_final = os.path.join(base_models, "tcn_pm25_v3_multi_towers.pt")
    
    if os.path.exists(v3_final) and os.path.exists(v3_scaler) and v3_final != v3_best:
        model_configs.append({
            'name': 'TCN v3 Multi-Towers (FINAL)',
            'type': 'v3',
            'model_path': v3_final,
            'scaler_path': v3_scaler,
            'data_path': CLEAN_DATA_PATH_V3,
            'seq_len': 336,
            'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
        })
    
    # Verificar que hay modelos para comparar
    if not model_configs:
        print("❌ No se encontraron modelos entrenados para comparar.")
        print(f"   Busca en: {base_models}")
        return
    
    print(f"\n✅ Se encontraron {len(model_configs)} modelos para comparar:")
    for config in model_configs:
        print(f"   - {config['name']}")
    
    # Ejecutar comparación
    results = compare_models(model_configs)
    
    # Resumen final
    print("\n" + "="*100)
    print("RESUMEN FINAL - COMPARACIÓN DE MODELOS")
    print("="*100)
    
    for model_name, data in results.items():
        print(f"\n{model_name}:")
        metrics_list = data['metrics']
        
        if len(metrics_list) == 1:
            # Single output (v2)
            m = metrics_list[0]
            print(f"  └─ {m['horizonte']}: MAE={m['MAE']:.2f}, RMSE={m['RMSE']:.2f}, R²={m['R²']:.3f}, Skill={m['Skill Score']:.3f}")
        else:
            # Multi-output (v3)
            for m in metrics_list:
                skill_indicator = "✅" if m['Skill Score'] > 0 else "❌"
                r2_indicator = "✅" if m['R²'] > 0.3 else "⚠️" if m['R²'] > 0.1 else "❌"
                print(f"  └─ {m['horizonte']}: MAE={m['MAE']:.2f}, R²={m['R²']:.3f} {r2_indicator}, Skill={m['Skill Score']:.3f} {skill_indicator}")
    
    print("\n" + "="*100)
    print("Leyenda:")
    print("  ✅ = Bueno (R²>0.3 o Skill>0)")
    print("  ⚠️  = Regular (0.1<R²<0.3)")
    print("  ❌ = Malo (R²<0.1 o Skill<0)")
    print("="*100)


if __name__ == "__main__":
    main()
