import os
import sys
import argparse
import numpy as np
import pandas as pd
import logging
import time
import json
from joblib import Parallel, delayed

# Permitir importar modulos desde la raiz de optimization_lab
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.ui.market_loader import MarketLoader
from core.quant_engine.blender import ContinuousBlender
from core.quant_engine.physics import PhysicsDiscoverer
from core.quant_engine.nervous import RegimeShiftDetector
from core.ui.synthetic_generator import SyntheticMarketGenerator
from telemetry import setup_telemetry

logger = setup_telemetry()

# Silenciar logs internos para multiprocesamiento
logging.getLogger('core.quant_engine.blender').setLevel(logging.ERROR)
logging.getLogger('core.quant_engine.physics').setLevel(logging.ERROR)
logging.getLogger('core.ui.synthetic_generator').setLevel(logging.WARNING)

def load_best_parameters(mode: str, test_mode: bool) -> dict:
    """Carga los parámetros ganadores de parameter_tuner/results/<mode>/best_hyperparameters.csv (o _test.csv)."""
    filename = f"best_hyperparameters{'_test' if test_mode else ''}.csv"
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', mode, filename))
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo de parámetros: {path}. Por favor corre el entrenamiento en modo '{mode}' primero.")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"El archivo de parámetros está vacío: {path}")
    row = df.iloc[0]
    return {
        'spline_tol': float(row['spline_tol']),
        'cusum_h': float(row['cusum_h']),
        'drift_k': float(row['drift_k'])
    }

def calculate_oos_metrics(p_raw, p_val_raw, p_pred):
    """Calcula las métricas OOS y devuelve también la curva de retornos acumulada."""
    pred_dir = np.sign(np.diff(p_pred, prepend=p_raw[-1]))
    true_dir = np.sign(np.diff(p_val_raw, prepend=p_raw[-1]))
    
    true_returns = np.diff(p_val_raw, prepend=p_raw[-1]) / np.insert(p_val_raw[:-1], 0, p_raw[-1])
    strat_returns = pred_dir * true_returns
    
    # Hit Ratio
    aciertos_mask = (pred_dir == true_dir)
    hit_ratio = np.mean(aciertos_mask) if len(aciertos_mask) > 0 else 0.0
    
    # True Directional Alpha (TDA)
    tda = hit_ratio - 0.5
    
    # Cálculo de Tramos (Walk-Forward de 15 tramos)
    num_tranches = 15
    tranche_size = len(aciertos_mask) // num_tranches
    tranche_metrics = {}
    for i in range(num_tranches):
        start_idx = i * tranche_size
        end_idx = start_idx + tranche_size if i < num_tranches - 1 else len(aciertos_mask)
        
        # Hit Ratio por tramo
        tranche_mask = aciertos_mask[start_idx:end_idx]
        tranche_hit_ratio = np.mean(tranche_mask) if len(tranche_mask) > 0 else 0.0
        tranche_metrics[f'hit_tranche_{i+1}'] = float(tranche_hit_ratio)
        
        # MAPE por tramo
        p_val_tranche = p_val_raw[start_idx:end_idx]
        p_pred_tranche = p_pred[start_idx:end_idx]
        if len(p_val_tranche) > 0:
            tranche_mape = float(np.mean(np.abs((p_val_tranche - p_pred_tranche) / p_val_tranche)) * 100.0)
        else:
            tranche_mape = 0.0
        tranche_metrics[f'mape_tranche_{i+1}'] = tranche_mape
        
    # Curva Hit Acumulada
    hits = np.where(pred_dir == true_dir, 1, -1)
    hit_acumulado = np.cumsum(hits).tolist()
    
    # Kelly empírico (MCC)
    real_returns = np.abs(true_returns)
    W = np.mean(real_returns[aciertos_mask]) if np.any(aciertos_mask) else 0.0
    fallos_mask = (pred_dir != true_dir)
    L = np.mean(real_returns[fallos_mask]) if np.any(fallos_mask) else 1e-9
    L = L if L > 0 else 1e-9
    b = W / L
    kelly_f = hit_ratio - ((1.0 - hit_ratio) / b) if b > 0 else 0.0
    
    # Profit Factor (PF)
    gains = np.sum(strat_returns[strat_returns > 0])
    losses = np.sum(np.abs(strat_returns[strat_returns < 0]))
    profit_factor = float(gains / losses) if losses > 0 else 10.0
    
    # Max Drawdown (MDD)
    equity = np.cumprod(1.0 + strat_returns)
    peaks = np.maximum.accumulate(equity)
    drawdowns = (peaks - equity) / peaks
    max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
    
    # MAPE
    mape = float(np.mean(np.abs((p_val_raw - p_pred) / p_val_raw)) * 100.0)
    
    metrics = {
        'hit_ratio': float(hit_ratio),
        'tda': float(tda),
        'mcc_test': kelly_f,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'mape_test': mape,
        'hit_acumulado_curve': hit_acumulado,
        'equity_curve': equity.tolist()
    }
    
    # Agregar las métricas de los 15 tramos
    metrics.update(tranche_metrics)
    
    return metrics

def run_sindy_oos(df_train, df_val, spline_tol, cusum_h, drift_k):
    """Ejecuta el pipeline Blender -> CUSUM -> SINDy e intenta predecir OOS."""
    log_r, vol_z, p_raw, dt_val = MarketLoader.prepare_quant_input(df_train)
    t = np.arange(len(log_r), dtype=np.float64)
    
    blender = ContinuousBlender(tolerance=spline_tol)
    blender.fit(t, log_r, dominant_periods=np.array([]), feature_idx=0)
    blender.fit(t, vol_z, dominant_periods=np.array([]), feature_idx=1)
    smooth_r, r_dot, _ = blender.compute_continuous(0, t)
    v_smooth, v_dot, _ = blender.compute_continuous(1, t)
    
    detector = RegimeShiftDetector(threshold=cusum_h, drift=drift_k)
    cusum_report = detector.detect(log_r, smooth_r)
    shift_idx = cusum_report['shift_indices']
    num_regimes = len(shift_idx)
    
    start_idx = shift_idx[-1] if num_regimes > 0 else 0
    if len(t) - start_idx < 10:
        start_idx = 0
        
    x_matrix = np.column_stack((smooth_r, v_smooth))
    x_dot_matrix = np.column_stack((r_dot, v_dot))
    discoverer = PhysicsDiscoverer(poly_degree=1)
    
    _, _, p_val_raw, _ = MarketLoader.prepare_quant_input(df_val)
    horizon = len(df_val)
    
    p_rep = discoverer.extract_equations(
        t=t[start_idx:], x=x_matrix[start_idx:], x_dot=x_dot_matrix[start_idx:], 
        dt=dt_val, horizon_steps=horizon, sigma_res_r=0, sigma_res_v=0, 
        last_price=p_raw[-1], disable_norm=False, disable_returns=False
    )
    p_pred = np.array(p_rep['prediction']['det_price_path'])
    return p_raw, p_val_raw, p_pred

def run_classic_dynamic_benchmark(df_train, df_val):
    """Benchmark: Auto-Tuner clásico con sintonización in-sample dinámica de drift_k."""
    log_r, vol_z, p_raw, dt_val = MarketLoader.prepare_quant_input(df_train)
    t = np.arange(len(log_r), dtype=np.float64)
    
    # Parámetros fijos del autotuner clásico
    spline_tol = 0.0050
    cusum_h = 5.0
    
    blender = ContinuousBlender(tolerance=spline_tol)
    blender.fit(t, log_r, dominant_periods=np.array([]), feature_idx=0)
    blender.fit(t, vol_z, dominant_periods=np.array([]), feature_idx=1)
    smooth_r, r_dot, _ = blender.compute_continuous(0, t)
    v_smooth, v_dot, _ = blender.compute_continuous(1, t)
    
    x_matrix = np.column_stack((smooth_r, v_smooth))
    x_dot_matrix = np.column_stack((r_dot, v_dot))
    discoverer = PhysicsDiscoverer(poly_degree=1)
    
    best_drift = 0.1
    best_score = -float('inf')
    
    grid_k = np.arange(0.1, 5.1, 0.1)
    for k in grid_k:
        detector = RegimeShiftDetector(threshold=cusum_h, drift=k)
        cusum_report = detector.detect(log_r, smooth_r)
        shift_idx = cusum_report['shift_indices']
        num_regimes = len(shift_idx)
        
        boundaries = [0] + shift_idx + [len(t)]
        weighted_r2_sum = 0.0
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i+1]
            length = end - start
            if length < 15:
                continue
            
            try:
                p_rep = discoverer.extract_equations(
                    t=t[start:end], x=x_matrix[start:end], x_dot=x_dot_matrix[start:end], 
                    dt=dt_val, horizon_steps=1, sigma_res_r=0, sigma_res_v=0, 
                    last_price=p_raw[end-1], disable_norm=False, disable_returns=False
                )
                peso = length / float(len(t))
                r2 = max(0, p_rep.get('score', 0.0))
                weighted_r2_sum += r2 * peso
            except Exception:
                pass
            
        score = weighted_r2_sum / (1.0 + 0.1 * num_regimes)
        if score > best_score:
            best_score = score
            best_drift = k
            
    # Evaluar OOS con la mejor drift_k encontrada
    return run_sindy_oos(df_train, df_val, spline_tol, cusum_h, best_drift)

def process_universe_data(df_raw, universe_name, params_classic, params_syn, params_real):
    """Procesa un universo: Evalúa el benchmark dinámico y las 3 configuraciones estáticas."""
    try:
        train_len = 1850
        df_train = df_raw.iloc[:train_len]
        df_val = df_raw.iloc[train_len:]
        
        models_configs = {
            'Classic Static': params_classic,
            'Synthetic Static': params_syn,
            'Real Static': params_real
        }
        
        universe_summary = []
        universe_curves = {'ticker': universe_name, 'models': {}}
        
        # 1. Evaluar Benchmark (Classic Dynamic)
        try:
            p_raw, p_val_raw, p_pred_bench = run_classic_dynamic_benchmark(df_train, df_val)
            if len(p_pred_bench) == len(p_val_raw) and len(p_pred_bench) > 0:
                metrics = calculate_oos_metrics(p_raw, p_val_raw, p_pred_bench)
                curve_hit = metrics.pop('hit_acumulado_curve')
                curve_eq = metrics.pop('equity_curve')
                
                universe_summary.append({
                    'ticker': universe_name,
                    'model': 'Benchmark (Dynamic Classic)',
                    **metrics
                })
                
                universe_curves['models']['Benchmark (Dynamic Classic)'] = {
                    'hit_acumulado_curve': curve_hit,
                    'equity_curve': curve_eq
                }
        except Exception as e:
            logger.warning(f"Error evaluando Benchmark (Classic Dynamic) en {universe_name}: {e}")
            
        # 2. Evaluar las configuraciones estáticas optimizadas
        for model_name, p in models_configs.items():
            try:
                p_raw, p_val_raw, p_pred = run_sindy_oos(df_train, df_val, p['spline_tol'], p['cusum_h'], p['drift_k'])
                if len(p_pred) != len(p_val_raw) or len(p_pred) == 0:
                    continue
                metrics = calculate_oos_metrics(p_raw, p_val_raw, p_pred)
                curve_hit = metrics.pop('hit_acumulado_curve')
                curve_eq = metrics.pop('equity_curve')
                
                universe_summary.append({
                    'ticker': universe_name,
                    'model': model_name,
                    **metrics
                })
                
                universe_curves['models'][model_name] = {
                    'hit_acumulado_curve': curve_hit,
                    'equity_curve': curve_eq
                }
            except Exception:
                continue
                
        return universe_summary, universe_curves
        
    except Exception as e:
        logger.error(f"Error procesando {universe_name}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Suite de Validación de Parámetros en Universos Sintéticos")
    parser.add_argument('--test', action='store_true', help="Ejecutar prueba rápida de 3 universos.")
    args = parser.parse_args()
    
    num_universes = 3 if args.test else 100
    
    logger.info("==================================================")
    logger.info(f"Iniciando Suite de Validación de Parámetros en {num_universes} Universos Sintéticos...")
    logger.info("==================================================")
    
    try:
        params_classic = load_best_parameters('classic', args.test)
        params_syn = load_best_parameters('synthetic', args.test)
        params_real = load_best_parameters('real', args.test)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
        
    logger.info("Hiperparámetros cargados correctamente:")
    logger.info(f" - Classic Static: {params_classic}")
    logger.info(f" - Synthetic Static: {params_syn}")
    logger.info(f" - Real Static: {params_real}")
    
    logger.info("Generando universos sintéticos...")
    generator = SyntheticMarketGenerator(base_ticker='BTC-USD')
    
    universes = []
    t_start = time.time()
    for i in range(num_universes):
        df_univ = generator.generate_multiverse(num_candles=2000, mc_trajectories=300)
        universes.append((df_univ, f"Synthetic_Universe_{i+1}"))
        if (i+1) % 5 == 0 or args.test:
            logger.info(f"   [+] Generados {i+1}/{num_universes} universos en memoria.")
            
    logger.info(f"Generación completada en {time.time() - t_start:.2f} segundos.")
    
    logger.info(f"Evaluando {num_universes} universos en paralelo...")
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_universe_data)(df, name, params_classic, params_syn, params_real) 
        for df, name in universes
    )
    
    all_summaries = []
    all_curves = []
    
    for r in results:
        if r is not None:
            all_summaries.extend(r[0])
            all_curves.append(r[1])
            
    # Crear carpeta de resultados si no existe
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'synthetic'))
    os.makedirs(output_dir, exist_ok=True)
    
    # Exportar Resumen CSV
    df_summary = pd.DataFrame(all_summaries)
    summary_path = os.path.join(output_dir, f"validation_results{'_test' if args.test else ''}.csv")
    df_summary.to_csv(summary_path, index=False)
    logger.info(f"Guardado: {summary_path} con {len(df_summary)} filas.")
    
    # Exportar Curvas JSON
    curves_path = os.path.join(output_dir, f"validation_curves{'_test' if args.test else ''}.json")
    with open(curves_path, "w") as f:
        json.dump(all_curves, f)
    logger.info(f"Guardado: {curves_path}.")
    
    logger.info("¡Validación finalizada con éxito!")

if __name__ == "__main__":
    main()
