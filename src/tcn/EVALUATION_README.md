# Evaluación de Modelos TCN

Scripts para evaluar y comparar modelos TCN entrenados.

## 📊 Scripts Disponibles

### 1. `evaluate_model.py` - Evaluación Individual

Evalúa un modelo específico y muestra métricas detalladas.

**Uso:**

```bash
# TCN v2 (Single Output)
python -m src.tcn.evaluate_model \
    --model models/tcn_pm25_v2.pt \
    --scaler models/scaler_v2.pkl \
    --type v2 \
    --device cpu

# TCN v3 (Multi-Horizon)
python -m src.tcn.evaluate_model \
    --model models/tcn_pm25_v3_multi_towers_best.pt \
    --scaler models/scaler_v3_multi_towers.pkl \
    --type v3 \
    --device cuda
```

**Argumentos:**
- `--model`: Ruta al archivo del modelo (.pt)
- `--scaler`: Ruta al scaler (.pkl)
- `--type`: Tipo de modelo (`v2` o `v3`)
- `--device`: Device para inferencia (`cpu` o `cuda`)
- `--data`: (Opcional) Ruta a datos CSV
- `--seq-len`: (Opcional) Longitud de secuencia

### 2. `compare_all_models.py` - Comparación Automática

Compara todos los modelos entrenados disponibles en `models/`.

**Uso:**

```bash
python -m src.tcn.compare_all_models
```

Este script:
- ✅ Detecta automáticamente modelos disponibles
- ✅ Evalúa todos los modelos encontrados
- ✅ Genera reporte comparativo completo
- ✅ Indica con emojis qué modelos funcionan mejor

## 📈 Métricas Calculadas

Para cada horizonte de predicción:

| Métrica | Descripción | Rango | Ideal |
|---------|-------------|-------|-------|
| **MAE** | Error absoluto medio | [0, ∞) | Menor |
| **RMSE** | Raíz del error cuadrático medio | [0, ∞) | Menor |
| **R²** | Coeficiente de determinación | (-∞, 1] | > 0.5 |
| **MAPE** | Error porcentual absoluto medio | [0, ∞) | < 20% |
| **Correlación** | Correlación de Pearson | [-1, 1] | > 0.7 |
| **Skill Score** | Mejora vs modelo naive | (-∞, 1] | > 0 |
| **Max Error** | Error máximo observado | [0, ∞) | - |
| **Median AE** | Mediana del error absoluto | [0, ∞) | Menor |

### 🎯 Interpretación de Métricas

#### **R² (Coeficiente de Determinación)**
- **> 0.5**: ✅ Excelente
- **0.3 - 0.5**: ✅ Bueno
- **0.1 - 0.3**: ⚠️ Regular
- **< 0.1**: ❌ Malo

#### **Skill Score**
- **> 0.5**: ✅ Mucho mejor que persistencia
- **0.2 - 0.5**: ✅ Mejor que persistencia
- **0 - 0.2**: ⚠️ Ligeramente mejor
- **< 0**: ❌ Peor que persistencia (problema serio)

#### **MAPE (Mean Absolute Percentage Error)**
- **< 10%**: ✅ Excelente precisión
- **10-20%**: ✅ Buena precisión
- **20-50%**: ⚠️ Precisión aceptable
- **> 50%**: ❌ Baja precisión

## 🔍 Ejemplo de Salida

### TCN v2 (Single Output)
```
=== pm2_5_1h - Métricas Detalladas ===
  MAE:            2.6834
  RMSE:           3.3752
  R²:             0.4892
  MAPE:           74.13%
  Correlación:    0.7201
  Skill Score:    -2.9050
  Max Error:      28.4512
  Median AE:      2.1234
  N samples:      15234
```

### TCN v3 (Multi-Horizon)
```
====================================================================================================
Horizonte    |    MAE |   RMSE |     R² |  MAPE% |   Corr |  Skill | MaxErr
====================================================================================================
pm2_5_1h     |   2.68 |   3.38 |  0.489 |   74.1 |  0.720 |  0.250 |  28.45
pm2_5_12h    |   3.24 |   4.18 |  0.218 |   84.6 |  0.492 |  0.120 |  32.10
pm2_5_24h    |   3.41 |   4.38 |  0.139 |   87.4 |  0.428 | -0.050 |  35.67
pm2_5_72h    |   3.51 |   4.59 |  0.060 |   88.6 |  0.374 | -0.180 |  38.23
pm2_5_168h   |   3.56 |   4.68 |  0.030 |   91.3 |  0.349 | -0.220 |  40.12
====================================================================================================
```

## 🚀 Uso Programático

También puedes usar las funciones directamente en Python:

```python
from src.tcn.evaluate_model import evaluate_tcn_v3, compare_models

# Evaluar un modelo específico
metrics, y_true, y_pred = evaluate_tcn_v3(
    model_path="models/tcn_pm25_v3_multi_towers_best.pt",
    scaler_path="models/scaler_v3_multi_towers.pkl",
    data_path="data/processed/clean_dataset_v3.csv",
    seq_len=336,
    device='cuda'
)

# Comparar múltiples modelos
results = compare_models([
    {
        'name': 'TCN v3 BEST',
        'type': 'v3',
        'model_path': 'models/tcn_pm25_v3_multi_towers_best.pt',
        'scaler_path': 'models/scaler_v3_multi_towers.pkl',
        'data_path': 'data/processed/clean_dataset_v3.csv',
        'seq_len': 336,
        'device': 'cuda'
    }
])
```

## 📝 Notas

1. **shuffle=False aplicado**: Los modelos v3 nuevos entrenan con orden temporal preservado
2. **Validation Split**: Ahora usa 70% train, 15% validation, 15% test
3. **Pesos por Horizonte**: Loss ponderado (1h tiene 5x más peso que 168h)
4. **Early Stopping**: Basado en validation loss (paciencia de 12 épocas)
5. **Mejor Modelo Guardado**: Se guarda automáticamente el modelo con mejor val_loss

## 🔧 Troubleshooting

**Error: GPU no disponible**
```bash
# Usar CPU en lugar de CUDA
python -m src.tcn.evaluate_model ... --device cpu
```

**Error: Archivo no encontrado**
```bash
# Verificar que el modelo existe
ls models/

# Verificar que el scaler existe
ls models/*.pkl
```

**Error: Import modules**
```bash
# Asegurarte de ejecutar desde la raíz del proyecto
cd "C:\Users\mati9\OneDrive\Desktop\Uni\6to Semestre\ML"
python -m src.tcn.compare_all_models
```
