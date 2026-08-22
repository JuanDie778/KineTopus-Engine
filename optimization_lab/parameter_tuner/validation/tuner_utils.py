import numpy as np
import pandas as pd
import logging

import os
import sys

# Permitir que los modulos internos de optimization_lab se encuentren
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.ui.market_loader import MarketLoader
from core.quant_engine.blender import ContinuousBlender
from core.quant_engine.physics import PhysicsDiscoverer
from core.quant_engine.nervous import RegimeShiftDetector

def load_fitness_weights(filepath: str) -> dict:
    """Carga los pesos de la ecuación fitness desde el archivo CSV."""
    try:
        df = pd.read_csv(filepath)
        # Convertir a diccionario: {'feature': weight}
        return df.iloc[0].to_dict()
    except Exception as e:
        raise RuntimeError(f"No se pudieron cargar los pesos del fitness: {e}")

def compute_fitness_score(metrics: dict, weights: dict) -> float:
    """Calcula el score fitness aplicando los pesos aprendidos a las métricas OOS."""
    score = weights.get('intercept', 0.0)
    for k, v in metrics.items():
        if k in weights:
            score += v * weights[k]
    return score

def evaluate_parameter_set(df_train: pd.DataFrame, df_val: pd.DataFrame, params: dict, fitness_weights: dict = None, mode: str = 'synthetic') -> dict:
    """
    Evalúa un conjunto de hiperparámetros sobre un universo sintético (Train/Test).
    params debe contener: {'spline_tol': float, 'cusum_h': float, 'drift_k': float}
    """
    spline_tol = params['spline_tol']
    cusum_h = params['cusum_h']
    drift_k = params['drift_k']
    
    try:
        # 1. Preparación In-Sample
        log_r, vol_z, p_raw, dt_val = MarketLoader.prepare_quant_input(df_train)
        t = np.arange(len(log_r), dtype=np.float64)
        
        blender = ContinuousBlender(tolerance=spline_tol)
        blender.fit(t, log_r, dominant_periods=np.array([]), feature_idx=0)
        blender.fit(t, vol_z, dominant_periods=np.array([]), feature_idx=1)
        smooth_r, r_dot, _ = blender.compute_continuous(0, t)
        v_smooth, v_dot, _ = blender.compute_continuous(1, t)
        
        # Pre-cálculo OOS
        _, _, p_val_raw, _ = MarketLoader.prepare_quant_input(df_val)
        horizon = len(df_val)
        
        # 2. Detección de Régimen
        detector = RegimeShiftDetector(threshold=cusum_h, drift=drift_k)
        cusum_report = detector.detect(log_r, smooth_r)
        shift_idx = cusum_report['shift_indices']
        num_regimes = len(shift_idx)
        
        start_idx = shift_idx[-1] if num_regimes > 0 else 0
        if len(t) - start_idx < 10:
            start_idx = 0
            
        # 3. Descubrimiento SINDy (para métricas OOS)
        x_matrix = np.column_stack((smooth_r, v_smooth))
        x_dot_matrix = np.column_stack((r_dot, v_dot))
        discoverer = PhysicsDiscoverer(poly_degree=1)
        
        p_rep = discoverer.extract_equations(
            t=t[start_idx:], x=x_matrix[start_idx:], x_dot=x_dot_matrix[start_idx:], 
            dt=dt_val, horizon_steps=horizon, sigma_res_r=0, sigma_res_v=0, 
            last_price=p_raw[-1], disable_norm=False, disable_returns=False
        )
        
        r2_train = p_rep.get('score', 0)
        complexity = p_rep.get('complexity', 0)
        p_pred = np.array(p_rep['prediction'].get('det_price_path', []))
        
        if len(p_pred) == 0 or len(p_pred) != len(p_val_raw):
            return None # Integración fallida o divergente
            
        # 4. Evaluación OOS
        pred_dir = np.sign(np.diff(p_pred, prepend=p_raw[-1]))
        true_dir = np.sign(np.diff(p_val_raw, prepend=p_raw[-1]))
        true_returns = np.diff(p_val_raw, prepend=p_raw[-1]) / np.insert(p_val_raw[:-1], 0, p_raw[-1])
        strat_returns = pred_dir * true_returns
        
        aciertos_mask = (pred_dir == true_dir)
        hit_ratio = np.mean(aciertos_mask) if len(aciertos_mask) > 0 else 0.0
        
        real_returns = np.abs(true_returns)
        W = np.mean(real_returns[aciertos_mask]) if np.any(aciertos_mask) else 0.0
        fallos_mask = (pred_dir != true_dir)
        L = np.mean(real_returns[fallos_mask]) if np.any(fallos_mask) else 1e-9
        L = L if L > 0 else 1e-9
        b = W / L
        kelly_f = hit_ratio - ((1.0 - hit_ratio) / b) if b > 0 else 0.0
        
        gains = np.sum(strat_returns[strat_returns > 0])
        losses = np.sum(np.abs(strat_returns[strat_returns < 0]))
        profit_factor = gains / losses if losses > 0 else 10.0
        
        equity = np.cumprod(1.0 + strat_returns)
        peaks = np.maximum.accumulate(equity)
        drawdowns = (peaks - equity) / peaks
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
        
        mape = np.mean(np.abs((p_val_raw - p_pred) / p_val_raw)) * 100.0
        
        residuos = p_val_raw - p_pred
        if len(residuos) > 1 and np.var(residuos) > 0:
            rho_res = np.corrcoef(residuos[:-1], residuos[1:])[0, 1]
            if np.isnan(rho_res): rho_res = 1.0
        else:
            rho_res = 1.0
            
        metrics = {
            'r2_train': float(r2_train),
            'complexity': int(complexity),
            'regimes': int(num_regimes),
            'mcc_test': float(kelly_f),
            'hit_ratio': float(hit_ratio),
            'profit_factor': float(profit_factor),
            'max_drawdown': float(max_drawdown),
            'mape_test': float(mape),
            'rho_residuos': float(rho_res)
        }
        
        # 5. Cálculo del Fitness Score según el modo
        if mode == 'classic':
            # Fórmula clásica: R2_ponderado_IS / (1.0 + 0.1 * num_quiebres)
            boundaries = [0] + shift_idx + [len(t)]
            weighted_r2_sum = 0.0
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i+1]
                length = end - start
                if length < 15:
                    continue
                # Ejecutar física en cada bloque para obtener R2
                p_rep_regime = discoverer.extract_equations(
                    t=t[start:end], x=x_matrix[start:end], x_dot=x_dot_matrix[start:end], 
                    dt=dt_val, horizon_steps=1, sigma_res_r=0, sigma_res_v=0, 
                    last_price=p_raw[end-1], disable_norm=False, disable_returns=False
                )
                peso = length / float(len(t))
                r2 = max(0, p_rep_regime.get('score', 0.0))
                weighted_r2_sum += r2 * peso
                
            score = weighted_r2_sum / (1.0 + 0.1 * num_regimes)
        else:
            score = compute_fitness_score(metrics, fitness_weights)
        
        return {
            'params': params,
            'metrics': metrics,
            'fitness_score': score
        }
        
    except Exception as e:
        return None
