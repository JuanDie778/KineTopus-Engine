import argparse
import numpy as np
import pandas as pd
import logging
import time
import json
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

import os
import sys

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

def process_universe_data(df_raw, universe_name, grid_k, weights_syn, weights_real):
    """Procesa un universo sintético: Grid Search In-Sample + Evaluación OOS de ganadores."""
    try:
        train_len = 1850
        df_train = df_raw.iloc[:train_len]
        df_val = df_raw.iloc[train_len:]
        
        log_r, vol_z, p_raw, dt_val = MarketLoader.prepare_quant_input(df_train)
        t = np.arange(len(log_r), dtype=np.float64)
        
        blender = ContinuousBlender(tolerance=0.005)
        blender.fit(t, log_r, dominant_periods=np.array([]), feature_idx=0)
        blender.fit(t, vol_z, dominant_periods=np.array([]), feature_idx=1)
        smooth_r, r_dot, _ = blender.compute_continuous(0, t)
        v_smooth, v_dot, _ = blender.compute_continuous(1, t)
        
        _, _, p_val_raw, _ = MarketLoader.prepare_quant_input(df_val)
        horizon = len(df_val)
        
        results_is = []
        predictions_map = {} 
        
        # 1. Grid Search In-Sample
        for k_drift in grid_k:
            try:
                detector = RegimeShiftDetector(threshold=5.0, drift=k_drift)
                cusum_report = detector.detect(log_r, smooth_r)
                shift_idx = cusum_report['shift_indices']
                num_regimes = len(shift_idx)
                
                start_idx = shift_idx[-1] if num_regimes > 0 else 0
                if len(t) - start_idx < 10:
                    start_idx = 0
                    
                x_matrix = np.column_stack((smooth_r, v_smooth))
                x_dot_matrix = np.column_stack((r_dot, v_dot))
                discoverer = PhysicsDiscoverer(poly_degree=1)
                
                p_rep = discoverer.extract_equations(
                    t=t[start_idx:], x=x_matrix[start_idx:], x_dot=x_dot_matrix[start_idx:], 
                    dt=dt_val, horizon_steps=horizon, sigma_res_r=0, sigma_res_v=0, 
                    last_price=p_raw[-1], disable_norm=False, disable_returns=False
                )
                
                r2_train = p_rep['score']
                complexity = p_rep.get('complexity', 0)
                p_pred = np.array(p_rep['prediction']['det_price_path'])
                
                if len(p_pred) == 0 or len(p_pred) != len(p_val_raw):
                    continue
                
                score_classic = max(0, r2_train) / (1.0 + 0.1 * num_regimes)
                
                results_is.append({
                    'drift_k': k_drift,
                    'r2_train': r2_train,
                    'complexity': complexity,
                    'regimes': num_regimes,
                    'regimes_sq': num_regimes ** 2,
                    'score_classic': score_classic
                })
                predictions_map[k_drift] = p_pred
                
            except Exception:
                continue
                
        if not results_is:
            return None
            
        df_is = pd.DataFrame(results_is)
        
        # 2. Computar Scores Meta-Optimizados
        features = ['r2_train', 'complexity', 'regimes', 'regimes_sq']
        X = df_is[features].values
        if len(X) > 1:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = np.zeros_like(X)
            
        # Calcular Synthetic Score
        w_syn = np.array([weights_syn[f] for f in features])
        df_is['score_synthetic'] = np.dot(X_scaled, w_syn) + weights_syn.get('intercept', 0.0)
        
        # Calcular Real Score
        w_real = np.array([weights_real[f] for f in features])
        df_is['score_real'] = np.dot(X_scaled, w_real) + weights_real.get('intercept', 0.0)
        
        # 3. Elegir ganadores
        best_classic_idx = df_is['score_classic'].idxmax()
        best_syn_idx = df_is['score_synthetic'].idxmax()
        best_real_idx = df_is['score_real'].idxmax()
        
        winners = {
            'Classic': df_is.loc[best_classic_idx, 'drift_k'],
            'Synthetic': df_is.loc[best_syn_idx, 'drift_k'],
            'Real': df_is.loc[best_real_idx, 'drift_k']
        }
        
        # 4. Evaluación OOS de los ganadores
        universe_summary = []
        universe_curves = {'ticker': universe_name, 'models': {}}
        
        for model_name, drift_k in winners.items():
            p_pred = predictions_map[drift_k]
            metrics = calculate_oos_metrics(p_raw, p_val_raw, p_pred)
            
            curve_hit = metrics.pop('hit_acumulado_curve')
            curve_eq = metrics.pop('equity_curve')
            
            universe_summary.append({
                'ticker': universe_name,
                'model': model_name,
                'drift_k': drift_k,
                **metrics
            })
            
            universe_curves['models'][model_name] = {
                'drift_k': drift_k,
                'hit_acumulado_curve': curve_hit,
                'equity_curve': curve_eq
            }
            
        return universe_summary, universe_curves

    except Exception as e:
        logger.error(f"Error procesando {universe_name}: {e}")
        return None

def load_weights(path):
    try:
        df = pd.read_csv(path)
        return df.iloc[0].to_dict()
    except Exception as e:
        logger.error(f"Error cargando {path}: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(description="Suite de Validación en Universos Sintéticos")
    parser.add_argument('--test', action='store_true', help="Ejecutar una prueba rápida de solo 3 universos.")
    args = parser.parse_args()
    
    num_universes = 3 if args.test else 100
    
    logger.info("==================================================")
    logger.info(f"Iniciando Suite de Validación en {num_universes} Universos Sintéticos...")
    logger.info("==================================================")
    
    weights_syn_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'synthetic', 'fitness_equation_weights.csv')
    weights_real_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'real', 'real_fitness_equation_weights.csv')
    
    weights_syn = load_weights(weights_syn_path)
    weights_real = load_weights(weights_real_path)
    
    if not weights_syn or not weights_real:
        logger.error("Faltan archivos de pesos. Abortando.")
        return
        
    logger.info("Generando universos sintéticos a partir de BTC-USD semilla...")
    generator = SyntheticMarketGenerator(base_ticker='BTC-USD')
    
    universes = []
    t_start = time.time()
    for i in range(num_universes):
        # Generar universo de 2000 velas con 300 trayectorias Monte Carlo
        df_univ = generator.generate_multiverse(num_candles=2000, mc_trajectories=300)
        universes.append((df_univ, f"Synthetic_Universe_{i+1}"))
        if (i+1) % 5 == 0 or args.test:
            logger.info(f"   [+] Generados {i+1}/{num_universes} universos en memoria.")
            
    logger.info(f"Generación completada en {time.time() - t_start:.2f} segundos.")
    
    grid_k = np.linspace(0.1, 5.0, num=50)
    
    logger.info(f"Evaluando {num_universes} universos sintéticos en paralelo...")
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_universe_data)(df, name, grid_k, weights_syn, weights_real) for df, name in universes
    )
    
    all_summaries = []
    all_curves = []
    
    for r in results:
        if r is not None:
            all_summaries.extend(r[0])
            all_curves.append(r[1])
            
    # Exportar Resumen CSV
    summary_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'synthetic', 'synthetic_validation_results.csv')
    df_summary = pd.DataFrame(all_summaries)
    df_summary.to_csv(summary_path, index=False)
    logger.info(f"Guardado: {summary_path} con {len(df_summary)} filas.")
    
    # Exportar Curvas JSON
    curves_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'synthetic', 'synthetic_validation_curves.json')
    with open(curves_path, "w") as f:
        json.dump(all_curves, f)
    logger.info(f"Guardado: {curves_path}.")
    
    logger.info("¡Validación en Universos Sintéticos finalizada con éxito!")

if __name__ == "__main__":
    main()
