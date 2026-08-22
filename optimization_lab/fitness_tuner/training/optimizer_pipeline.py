import os
import sys
import numpy as np
import pandas as pd
import logging

# Permitir importar modulos desde la raiz de optimization_lab
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.ui.market_loader import MarketLoader
from core.quant_engine.blender import ContinuousBlender
from core.quant_engine.physics import PhysicsDiscoverer
from core.quant_engine.nervous import RegimeShiftDetector

from telemetry import setup_telemetry
from core.ui.synthetic_generator import SyntheticMarketGenerator
from core.quant_engine.meta_optimizer import FitnessLearner

logger = setup_telemetry()

# Silenciar logs internos
logging.getLogger('core.quant_engine.blender').setLevel(logging.ERROR)
logging.getLogger('core.quant_engine.physics').setLevel(logging.ERROR)

class GlobalOptimizer:
    """
    Motor de optimización diseñado para aprender empíricamente los pesos de la 
    Ecuación Fitness a través de universos sintéticos paralelos.
    Utiliza un enfoque One-Shot Massive Window para entrenamiento y prueba.
    """
    def __init__(self):
        pass

    def collect_matrix_universe(self, df_universe: pd.DataFrame, grid_k: np.ndarray) -> pd.DataFrame:
        """
        Divide el universo sintético en un corte estático único:
        - In-Sample (Train): Primeras 1850 velas (~5 años).
        - Out-of-Sample (Test): Últimas 150 velas.
        Evalúa la malla de drifts y registra las métricas.
        """
        # Split estático acordado con el Jefe
        train_len = 1850
        df_train = df_universe.iloc[:train_len]
        df_val = df_universe.iloc[train_len:]
        
        all_records = []
        
        try:
            # 1. Preparación In-Sample (común para todo el universo)
            log_r, vol_z, p_raw, dt_val = MarketLoader.prepare_quant_input(df_train)
            t = np.arange(len(log_r), dtype=np.float64)
            
            blender = ContinuousBlender(tolerance=0.005)
            blender.fit(t, log_r, dominant_periods=np.array([]), feature_idx=0)
            blender.fit(t, vol_z, dominant_periods=np.array([]), feature_idx=1)
            smooth_r, r_dot, _ = blender.compute_continuous(0, t)
            v_smooth, v_dot, _ = blender.compute_continuous(1, t)
            
            # Pre-cálculo OOS
            _, _, p_val_raw, _ = MarketLoader.prepare_quant_input(df_val)
            horizon = len(df_val)
            
            for k_drift in grid_k:
                try:
                    # 2. Detección de Régimen Variable
                    detector = RegimeShiftDetector(threshold=5.0, drift=k_drift)
                    cusum_report = detector.detect(log_r, smooth_r)
                    shift_idx = cusum_report['shift_indices']
                    num_regimes = len(shift_idx)
                    
                    start_idx = shift_idx[-1] if num_regimes > 0 else 0
                    if len(t) - start_idx < 10:
                        start_idx = 0
                        
                    # 3. Descubrimiento SINDy
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

                    # 4. Evaluación Out-of-Sample
                    pred_dir = np.sign(np.diff(p_pred, prepend=p_raw[-1]))
                    true_dir = np.sign(np.diff(p_val_raw, prepend=p_raw[-1]))
                    
                    true_returns = np.diff(p_val_raw, prepend=p_raw[-1]) / np.insert(p_val_raw[:-1], 0, p_raw[-1])
                    
                    # Backtest simulado de la estrategia
                    strat_returns = pred_dir * true_returns
                    
                    # Hit Ratio
                    aciertos_mask = (pred_dir == true_dir)
                    hit_ratio = np.mean(aciertos_mask) if len(aciertos_mask) > 0 else 0.0
                    
                    # Kelly empírico
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
                    profit_factor = gains / losses if losses > 0 else 10.0
                    
                    # Max Drawdown (MDD)
                    equity = np.cumprod(1.0 + strat_returns)
                    peaks = np.maximum.accumulate(equity)
                    drawdowns = (peaks - equity) / peaks
                    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0
                    
                    # MAPE
                    mape = np.mean(np.abs((p_val_raw - p_pred) / p_val_raw)) * 100.0
                    
                    # Autocorrelación de Residuos OOS
                    residuos = p_val_raw - p_pred
                    if len(residuos) > 1 and np.var(residuos) > 0:
                        rho_res = np.corrcoef(residuos[:-1], residuos[1:])[0, 1]
                        if np.isnan(rho_res): rho_res = 1.0
                    else:
                        rho_res = 1.0
                    
                    all_records.append({
                        'drift_k': k_drift,
                        'r2_train': float(r2_train),
                        'complexity': int(complexity),
                        'regimes': int(num_regimes),
                        'mcc_test': float(kelly_f), # Mapeado conceptualmente
                        'hit_ratio': float(hit_ratio),
                        'profit_factor': float(profit_factor),
                        'max_drawdown': float(max_drawdown),
                        'mape_test': float(mape),
                        'rho_residuos': float(rho_res)
                    })
                    
                except Exception as inner_e:
                    pass
        except Exception as e:
            logger.error(f"Error procesando universo en el split único: {e}")
            
        return pd.DataFrame(all_records)

    def run_optimization_loop(self, num_universes: int = 1, candles_per_universe: int = 2000):
        """
        Bucle de Meta-Optimización: Aprender los pesos de la función Fitness 
        en múltiples universos paralelos usando One-Shot Massive Window.
        Consolida los resultados de todos los universos antes de ajustar la regresión.
        """
        logger.info("Iniciando Pipeline de Meta-Optimización de Fitness en Multiversos (One-Shot)...")
        
        generator = SyntheticMarketGenerator(base_ticker='BTC-USD')
        logging.getLogger('core.ui.synthetic_generator').setLevel(logging.WARNING)
        
        # Malla expandida de drift-k para mayor resolución en regímenes bajos (50 niveles a petición del Jefe)
        grid_k = np.linspace(0.1, 5.0, num=50)
        
        multiverse_results = []
        
        for u in range(num_universes):
            logger.info(f"[Universo {u+1}/{num_universes}] Generando datos (300 trayectorias Monte Carlo) y recolectando matriz...")
            try:
                # Generación a 300 trayectorias
                df_synth = generator.generate_multiverse(num_candles=candles_per_universe, mc_trajectories=300)
                
                # Recolección masiva de métricas OOS e IS
                df_results = self.collect_matrix_universe(df_synth, grid_k)
                
                logger.info(f"   [+] Matriz de universo {u+1} recolectada: {len(df_results)} iteraciones.")
                multiverse_results.append(df_results)
                    
            except Exception as e:
                logger.error(f"Error procesando universo {u+1}: {e}")
                
        if not multiverse_results:
            logger.error("No se recolectaron datos válidos. Abortando.")
            return
            
        # Consolidar todos los universos en un único dataset para evitar degeneración por varianza cero
        df_all_results = pd.concat(multiverse_results, ignore_index=True)
        
        # Guardar para auditoría cuantitativa externa
        output_matrix_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'synthetic', 'all_universes_matrix.csv')
        df_all_results.to_csv(output_matrix_path, index=False)
        logger.info(f"Matriz consolidada exportada a: {output_matrix_path} ({len(df_all_results)} filas)")
        
        # Ajustar la regresión globalmente
        learner = FitnessLearner(alpha=1.0)
        final_weights = learner.fit(df_all_results)
        
        if not final_weights:
            logger.error("No se pudieron aprender los pesos consolidados.")
            return
            
        logger.info("==================================================")
        logger.info("RESULTADO FINAL: PESOS DE LA ECUACION FITNESS (CONSENSO)")
        logger.info("==================================================")
        for feature, weight in final_weights.items():
            if feature not in ['intercept', 'meta_r2']:
                logger.info(f" -> {feature}: {weight:+.4f}")
        logger.info(f" -> Intercept: {final_weights.get('intercept', 0.0):+.4f}")
        logger.info(f" -> Meta R2 (Varianza Explicada OOS): {final_weights.get('meta_r2', 0.0):.4f}")
            
        # Exportar pesos para uso futuro en producción
        df_final = pd.DataFrame([final_weights])
        output_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'synthetic', 'fitness_equation_weights.csv')
        df_final.to_csv(output_path, index=False)
        logger.info(f"Ecuación Suprema exportada a: {output_path}")

if __name__ == "__main__":
    optimizer = GlobalOptimizer()
    # Ejecución masiva de 50 universos para asegurar consenso óptimo
    optimizer.run_optimization_loop(num_universes=50, candles_per_universe=2000)
