import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from joblib import dump
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score
)

from scipy.stats import pearsonr

from src.tcn.tcn_model_v3 import TCNMultiTowers
from src.tcn.prepare_sequences_multi import create_sequences_multi
from src.config import CLEAN_DATA_PATH_V3

# ======================================================
# CONFIG GLOBAL - OPTIMIZADO PARA EVITAR OVERFITTING
# ======================================================
SEQ_LEN = 168  # Reducido de 336 a 168 (7 días en lugar de 14)
BATCH_SIZE = 32  # Reducido de 64 a 32 para mejor generalización
EPOCHS = 100  # Aumentado pero con early stopping más agresivo
LR = 1e-4  # CORREGIDO: Reducido de 1e-3 para evitar exploding gradients
EARLY_STOP = 8  # Aumentado a 8 para dar más oportunidad al modelo
MIN_DELTA = 0.01  # CORREGIDO: Aumentado para detección más robusta de mejoras

HORIZON_NAMES = ["pm2_5_1h","pm2_5_12h","pm2_5_24h","pm2_5_72h","pm2_5_168h"]

# CORREGIDO: Pesos uniformes para evitar colapso de torres
HORIZON_WEIGHTS = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)

# ======================================================
# GPU CHECK
# ======================================================
if not torch.cuda.is_available():
    raise RuntimeError("GPU requerida para entrenar este modelo.")

DEVICE = torch.device("cuda")
print(f"Entrenando en {DEVICE}")

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True  # Optimización


# ======================================================
# MÉTRICAS EXTRA
# ======================================================
def mape(y, y_pred):
    y = np.array(y)
    y_pred = np.array(y_pred)
    eps = 1e-6
    return np.mean(np.abs((y - y_pred) / (y + eps)))


def skill_score(y, y_pred):
    """Compara contra modelo naive de persistencia."""
    naive = y[:-1]
    real = y[1:]
    model = y_pred[1:]
    return 1 - (np.mean((real - model)**2) / np.mean((real - naive)**2))


# ======================================================
# BASELINE DE PERSISTENCIA
# ======================================================
def calculate_baseline_metrics(Y_test):
    """Calcula métricas del baseline de persistencia para comparación"""
    print("\n=== CALCULANDO BASELINE DE PERSISTENCIA ===")
    baseline_metrics = {}
    
    for i, name in enumerate(HORIZON_NAMES):
        y_true = Y_test[:, i]
        # Baseline: último valor observado (persistencia)
        y_baseline = np.roll(y_true, 1)
        y_baseline[0] = y_true[0]  # Primer valor
        
        mae_baseline = mean_absolute_error(y_true, y_baseline)
        rmse_baseline = np.sqrt(mean_squared_error(y_true, y_baseline))
        r2_baseline = r2_score(y_true, y_baseline)
        
        baseline_metrics[name] = {
            'mae': mae_baseline,
            'rmse': rmse_baseline,
            'r2': r2_baseline
        }
        
        print(f"{name:<10} - MAE: {mae_baseline:.3f}, RMSE: {rmse_baseline:.3f}, R²: {r2_baseline:.3f}")
    
    return baseline_metrics


# ======================================================
# TRAINING PIPELINE
# ======================================================
def train_tcn_v3_multi():

    print("\n=== Cargando dataset v3 ===")
    df = pd.read_csv(CLEAN_DATA_PATH_V3)
    print(f"Dataset cargado: {len(df):,} filas")
    
    # Verificar cobertura temporal
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        print(f"Período: {df['time'].min()} a {df['time'].max()}")
        duration_years = (df['time'].max() - df['time'].min()).days / 365.25
        print(f"Duración: {duration_years:.2f} años")

    # --------------------------
    # Separar features y targets
    # --------------------------
    target_cols = HORIZON_NAMES
    feature_cols = [c for c in df.columns if c not in ["time", "pm2_5", "pm10"] + target_cols]

    X = df[feature_cols].values
    Y = df[target_cols].values

    # --------------------------
    # Escalado (SOLO en train para evitar leakage)
    # --------------------------
    print("\n=== Escalado de features ===")
    print(f"Features: {len(feature_cols)}")
    print(f"Targets: {len(target_cols)}")
    
    # CRÍTICO: Verificar que targets no estén en features
    feature_set = set(feature_cols)
    target_set = set(target_cols + ['pm2_5', 'pm10'])
    leakage = feature_set.intersection(target_set)
    if leakage:
        raise ValueError(f"❌ DATA LEAKAGE DETECTADO: {leakage} están en features")
    
    # Verificar que no haya lags de pm2_5 futuros
    future_leakage = [f for f in feature_cols if any(h in f for h in target_cols)]
    if future_leakage:
        print(f"⚠️  ADVERTENCIA: Features sospechosas de leakage: {future_leakage}")
    
    # === ESCALADO (solo una vez, ANTES DE dividir los sets) ===
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_Y = MinMaxScaler()
    Y_scaled = scaler_Y.fit_transform(Y)

    print("✅ Escalado completado (Features y Targets)")

    # Guardar scalers AHORA MISMO (antes de split)
    dump(scaler_X, "models/scaler_v3_multi_towers_X.pkl")
    dump(scaler_Y, "models/scaler_v3_multi_towers_Y.pkl")

    # --------------------------
    # Secuencias multi-horizon
    # --------------------------
    print(f"\n=== Creando secuencias (SEQ_LEN={SEQ_LEN}) ===")
    X_seq, Y_seq = create_sequences_multi(X_scaled, Y_scaled, SEQ_LEN)  # CORREGIDO: Usar Y_scaled
    print(f"Secuencias creadas: {len(X_seq):,}")
    print(f"Shape X: {X_seq.shape}, Shape Y: {Y_seq.shape}")

    X_t = torch.tensor(X_seq, dtype=torch.float32).permute(0, 2, 1)
    Y_t = torch.tensor(Y_seq, dtype=torch.float32)
    
    print(f"Tensor X: {X_t.shape}, Tensor Y: {Y_t.shape}")

    # Train/Validation/Test split (70/15/15) - TEMPORAL, NO ALEATORIO
    print("\n=== Split temporal (70/15/15) ===")
    train_split = int(len(X_t) * 0.70)
    val_split = int(len(X_t) * 0.85)
    
    X_train = X_t[:train_split]
    X_val = X_t[train_split:val_split]
    X_test = X_t[val_split:]
    
    Y_train = Y_t[:train_split]
    Y_val = Y_t[train_split:val_split]
    Y_test = Y_t[val_split:]
    
    print(f"Train: {len(X_train):,} secuencias")
    print(f"Val:   {len(X_val):,} secuencias")
    print(f"Test:  {len(X_test):,} secuencias")
    print(f"Ratio parámetros/datos: Se calculará después de crear el modelo")

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=BATCH_SIZE,
        shuffle=False,  # Preservar orden temporal
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        TensorDataset(X_val, Y_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # --------------------------
    # Modelo (SIMPLIFICADO)
    # --------------------------
    print("\n=== Inicializando modelo ===")
    model = TCNMultiTowers(num_inputs=X_train.shape[1]).to(DEVICE)
    
    # Contar parámetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parámetros totales: {total_params:,}")
    print(f"Parámetros entrenables: {trainable_params:,}")
    print(f"Ratio parámetros/muestras: {trainable_params/len(X_train):.2f}")
    
    if trainable_params / len(X_train) > 100:
        print("⚠️  ADVERTENCIA: Ratio alto, riesgo de overfitting")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)  # Agregar weight decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3, min_lr=1e-6  # Patience reducido
    )
    criterion = nn.MSELoss(reduction='none')  # No reducir para aplicar pesos
    horizon_weights = HORIZON_WEIGHTS.to(DEVICE)
    
    print(f"Learning rate: {LR}")
    print(f"Weight decay: 1e-4")
    print(f"Early stopping patience: {EARLY_STOP}")

    best_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    val_losses = []

    print("\n=== Entrenando modelo ===")
    print(f"Épocas máximas: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Batches por época: {len(train_loader)}")
    print("-" * 80)
    
    scaler_amp = torch.cuda.amp.GradScaler()

    for epoch in range(EPOCHS):
        # ===== TRAINING =====
        model.train()
        epoch_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()

            # AMP para mayor velocidad
            with torch.cuda.amp.autocast():
                pred = model(xb)
                # Loss ponderado por horizonte
                loss_per_horizon = criterion(pred, yb)  # (batch, 5)
                loss = (loss_per_horizon * horizon_weights).mean()

            scaler_amp.scale(loss).backward()

            # CORREGIDO: Gradient clipping más agresivo
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Reducido de 1.0 a 0.5

            scaler_amp.step(optimizer)
            scaler_amp.update()

            epoch_loss += loss.item()

        # ===== VALIDATION =====
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                loss_per_horizon = criterion(pred, yb)
                loss = (loss_per_horizon * horizon_weights).mean()
                val_loss += loss.item()
        
        scheduler.step(val_loss)
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        
        # Calcular mejora
        improvement = ((best_loss - val_loss) / best_loss * 100) if best_loss != float('inf') else 0

        print(f"Epoch {epoch+1}/{EPOCHS}  Train={epoch_loss:.4f}  Val={val_loss:.4f}  LR={optimizer.param_groups[0]['lr']:.6f}  Patience={patience_counter}/{EARLY_STOP}")

        # Early stopping basado en validation loss con MIN_DELTA
        if val_loss < best_loss - MIN_DELTA:
            best_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            # Guardar mejor modelo
            torch.save(model.state_dict(), "models/tcn_pm25_v3_multi_towers_best.pt")
            print(f"  ✅ Nuevo mejor modelo guardado (mejora: {improvement:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP:
                print(f"\n⏹️  Early stopping activado en época {epoch+1}")
                print(f"  Mejor modelo en época {best_epoch} con val_loss={best_loss:.4f}")
                break

    # ======================================================
    # EVALUACIÓN FINAL
    # ======================================================
    print("\n" + "="*80)
    print("  EVALUACIÓN FINAL")
    print("="*80)
    
    # Cargar mejor modelo
    print(f"\n📂 Cargando mejor modelo (época {best_epoch})...")
    model.load_state_dict(torch.load("models/tcn_pm25_v3_multi_towers_best.pt"))
    model.eval()
    
    # Calcular baseline primero (con datos desnormalizados)
    Y_test_original = scaler_Y.inverse_transform(Y_test.numpy())

    # Ahora sí: calcular baseline
    baseline_metrics = calculate_baseline_metrics(Y_test_original)      
    # Evaluar en validation
    preds_val_scaled = model(X_val.to(DEVICE)).cpu().detach().numpy()

    # 🔥 CLIP entre 0 y 1 ANTES de desnormalizar
    preds_val_scaled = np.clip(preds_val_scaled, 0, 1)

    # Desescalar
    preds_val = scaler_Y.inverse_transform(preds_val_scaled.reshape(-1, 5))
    Y_val_original = scaler_Y.inverse_transform(Y_val.cpu().numpy().reshape(-1, 5))

    # Verificación de shapes
    assert preds_val.shape == Y_val_original.shape, "Shape mismatch en VALIDATION"

    # Validación de shapes
    assert preds_val.shape == Y_val_original.shape, "Shape mismatch VAL!"

    print("\n" + "="*80)
    print("  MÉTRICAS EN VALIDATION SET")
    print("="*80)
    print(f"{'Horizonte':<12} | {'MAE':>6} | {'RMSE':>6} | {'R²':>6} | {'MAPE':>6} | {'Corr':>6} | {'Skill':>7} | {'vs Base':<8}")
    print("-"*80)
    
    for i, name in enumerate(HORIZON_NAMES):
        yt = np.asarray(Y_val_original[:, i], dtype="float64")  # CORREGIDO: Usar desnormalizado
        yp = np.asarray(preds_val[:, i], dtype="float64")

        mae = mean_absolute_error(yt, yp)
        rmse = np.sqrt(np.mean((yt - yp)**2))
        r2 = r2_score(yt, yp)
        mape_v = mape(yt, yp)

        try:
            corr, _ = pearsonr(yt, yp)
        except:
            corr = np.nan

        skill = skill_score(yt, yp)
        
        # Comparar con baseline
        base_mae = baseline_metrics[name]['mae']
        improvement = ((base_mae - mae) / base_mae * 100)
        better = "✅" if mae < base_mae else "❌"
        
        print(f"{name:<12} | {mae:>6.3f} | {rmse:>6.3f} | {r2:>6.3f} | {mape_v:>6.3f} | {corr:>6.3f} | {skill:>7.3f} | {better} {improvement:>+5.1f}%")
    
    # Evaluar en test
    preds_test_scaled = model(X_test.to(DEVICE)).cpu().detach().numpy()
    preds_test = scaler_Y.inverse_transform(preds_test_scaled.reshape(-1, 5))
    Y_test_original = scaler_Y.inverse_transform(Y_test.cpu().numpy().reshape(-1, 5))

    assert preds_test.shape == Y_test_original.shape, "Shape mismatch TEST!"

    print("\n" + "="*80)
    print("  MÉTRICAS EN TEST SET (DATOS NO VISTOS)")
    print("="*80)
    print(f"{'Horizonte':<12} | {'MAE':>6} | {'RMSE':>6} | {'R²':>6} | {'MAPE':>6} | {'Corr':>6} | {'Skill':>7} | {'vs Base':<8}")
    print("-"*80)

    for i, name in enumerate(HORIZON_NAMES):

        # --- asegurar arrays limpios ---
        yt = np.asarray(Y_test_original[:, i], dtype="float64")  # CORREGIDO: Usar desnormalizado
        yp = np.asarray(preds_test[:, i], dtype="float64")

        mae = mean_absolute_error(yt, yp)
        rmse = np.sqrt(np.mean((yt - yp)**2))
        r2 = r2_score(yt, yp)
        mape_v = mape(yt, yp)

        # Pearson puede fallar si la señal es constante → manejarlo
        try:
            corr, _ = pearsonr(yt, yp)
        except:
            corr = np.nan

        skill = skill_score(yt, yp)
        
        # Comparar con baseline
        base_mae = baseline_metrics[name]['mae']
        improvement = ((base_mae - mae) / base_mae * 100)
        better = "✅" if mae < base_mae else "❌"
        
        print(f"{name:<12} | {mae:>6.3f} | {rmse:>6.3f} | {r2:>6.3f} | {mape_v:>6.3f} | {corr:>6.3f} | {skill:>7.3f} | {better} {improvement:>+5.1f}%")

    # -----------------------------
    # Guardado final y resumen
    # -----------------------------
    torch.save(model.state_dict(), "models/tcn_pm25_v3_multi_towers.pt")
    dump(scaler_X, "models/scaler_v3_multi_towers_X.pkl")
    dump(scaler_Y, "models/scaler_v3_multi_towers_Y.pkl")  # NUEVO: Guardar scaler de targets
    
    print("\n" + "="*80)
    print("  RESUMEN FINAL")
    print("="*80)
    print(f"✅ Modelos guardados en models/")
    print(f"   - tcn_pm25_v3_multi_towers_best.pt (mejor modelo, época {best_epoch})")
    print(f"   - tcn_pm25_v3_multi_towers.pt (modelo final)")
    print(f"   - scaler_v3_multi_towers_X.pkl (scaler de features)")
    print(f"   - scaler_v3_multi_towers_Y.pkl (scaler de targets)")
    print(f"\n📊 Configuración usada:")
    print(f"   SEQ_LEN: {SEQ_LEN}")
    print(f"   Parámetros: {trainable_params:,}")
    print(f"   Épocas entrenadas: {best_epoch}/{EPOCHS}")
    print(f"   Mejor val_loss: {best_loss:.4f}")
    print("\n" + "="*80)
    print("  ENTRENAMIENTO COMPLETADO")
    print("="*80 + "\n")


# ======================================================
if __name__ == "__main__":
    train_tcn_v3_multi()
