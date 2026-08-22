import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from typing import Dict, Any

from core.ui.market_loader import MarketLoader
from core.quant_engine.sensor import SpectralAnalyzer
from core.quant_engine.blender import ContinuousBlender
from core.quant_engine.nervous import RegimeShiftDetector
from core.quant_engine.physics import PhysicsDiscoverer
from core.quant_engine.auto_tuner import CUSUMAutoTuner
from core.quant_engine.auto_tuner_predictive import PredictiveAutoTuner

logger = logging.getLogger(__name__)

class WalkForwardEvaluator:
    """
    Motor de Backtesting Purista (Out-of-Sample) para las Ecuaciones de SINDy.
    Usa una Arquitectura Expanding Window con Auto-Tuning Direccional incorporado.
    """
    
    def __init__(self, df: pd.DataFrame, disable_norm: bool = False, disable_returns: bool = False, tuner_type: str = 'cusum', context_window: int = 0):
        self.df = df
        self.disable_norm = disable_norm
        self.disable_returns = disable_returns
        self.tuner_type = tuner_type
        self.context_window = context_window

    def run(self, initial_window: int = 150, stride: int = 20, horizon: int = 150, blocks: int = 15) -> pd.DataFrame:
        total_len = len(self.df)
        results = []
        
        # El límite de parada asegura que el modelo no asuma el futuro que no existe (total_len - horizon)
        max_valid_start = total_len - horizon
        
        print(f"Iniciando Walk-Forward (Ventana:{initial_window}, Salto:{stride}, Horizonte:{horizon} velas en {blocks} bloques, Contexto:{self.context_window})")
        
        end_idx = initial_window
        while end_idx <= max_valid_start:
            df_slice = self.df.iloc[:end_idx].copy() # Expanding Window Estricta (El Pasado)
            
            # Truncamiento por ventana de contexto (Moving Window)
            if self.context_window > 0 and len(df_slice) > self.context_window:
                df_slice = df_slice.tail(self.context_window)
                
            actual_prices = self.df.iloc[end_idx:end_idx+horizon]['Close'].values # El Futuro Oculto
            
            try:
                # 1. Ejecutar Auto-Tuner Oculto solo con datos del pasado
                if self.tuner_type == 'predictive':
                    tuner = PredictiveAutoTuner(df_slice, disable_norm=self.disable_norm, disable_returns=self.disable_returns, test_blocks=3)
                else:
                    tuner = CUSUMAutoTuner(df_slice, disable_norm=self.disable_norm, disable_returns=self.disable_returns)
                best_drift, tuner_report = tuner.run_search()
                
                # 2. Reconstruir el estado físico del pasado con el mejor parámetro
                log_returns, volumen_z, precio_raw, dt_val = MarketLoader.prepare_quant_input(
                    df_slice, disable_norm=self.disable_norm, disable_returns=self.disable_returns
                )
                t = np.arange(len(log_returns), dtype=np.float64) * dt_val
                
                # Moldeado topológico (Tolerancia estricta del auto-tuner)
                sensor = SpectralAnalyzer()
                fft_results = sensor.analyze(np.column_stack((log_returns, volumen_z)), dt=dt_val)
                b = ContinuousBlender(tolerance=0.0050)
                b.fit(t, log_returns, fft_results[0]['periods'], 0)
                b.fit(t, volumen_z, fft_results[1]['periods'], 1)
                
                r_smooth, r_dot, _ = b.compute_continuous(0, t)
                v_smooth, v_dot, _ = b.compute_continuous(1, t)
                
                # 3. Detectar Quiebres del pasado
                detector = RegimeShiftDetector(threshold=5.0, drift=best_drift)
                rep = detector.detect(log_returns, r_smooth)
                shift_idx = rep['shift_indices']
                
                # 4. Localizar el Régimen Vigente (Último bloque)
                start_regime = shift_idx[-1] if len(shift_idx) > 0 else 0
                end_regime = len(t)
                
                if end_regime - start_regime < 15:
                    # Si el régimen es absurdo por exceso de ruido, tomamos desde el penúltimo.
                    start_regime = shift_idx[-2] if len(shift_idx) > 1 else 0
                    
                # 5. Extracción Física (Grado 1 para evitar sobreajuste destructivo)
                disc = PhysicsDiscoverer(poly_degree=1)
                x_matrix = np.column_stack((r_smooth, v_smooth))
                x_dot_matrix = np.column_stack((r_dot, v_dot))
                
                physics_rep = disc.extract_equations(
                    t=t[start_regime:end_regime], x=x_matrix[start_regime:end_regime], 
                    x_dot=x_dot_matrix[start_regime:end_regime], dt=dt_val, 
                    horizon_steps=horizon, sigma_res_r=0, sigma_res_v=0, 
                    last_price=precio_raw[-1], disable_norm=self.disable_norm, 
                    disable_returns=self.disable_returns
                )
                
                pred_prices = physics_rep.get('prediction', {}).get('det_price_path', [])
                
            except Exception as e:
                logger.warning(f"Error en Iteración {end_idx}: {e}")
                pred_prices = []

            # 6. Evaluación Vectorial vs Futuro Real
            row_metrics = {
                'Iteracion (Velas Vistas)': end_idx,
                'Drift (k)': best_drift,
                'SINDy R2': physics_rep.get('score', 0) if 'physics_rep' in locals() else 0,
                'Validez': 'OK'
            }
            
            block_size = horizon // blocks
            
            # Verificación de fallo matemático (e.g. inf o Nans o R2 colapsado)
            if len(pred_prices) != horizon or physics_rep.get('empty_r_eq', False):
                row_metrics['Validez'] = 'FALLO MATEMÁTICO'
            else:
                for b in range(blocks):
                    p_pred = np.array(pred_prices[b*block_size : (b+1)*block_size])
                    p_act = np.array(actual_prices[b*block_size : (b+1)*block_size])
                    
                    # A. Error Porcentual (MAPE)
                    mape = np.mean(np.abs((p_act - p_pred) / p_act)) * 100
                    
                    # Naive Forecast (Línea Plana desde el último precio conocido)
                    last_known_price = precio_raw[-1]
                    p_naive = np.full_like(p_act, fill_value=last_known_price)
                    naive_mape = np.mean(np.abs((p_act - p_naive) / p_act)) * 100
                    
                    # B. Hit Ratio Vectorial Local (Pendiente dentro del Bloque)
                    delta_pred_local = np.sign(p_pred[-1] - p_pred[0])
                    delta_act_local = np.sign(p_act[-1] - p_act[0])
                    hit_local = 1.0 if delta_pred_local == delta_act_local else 0.0
                    
                    # C. Hit Ratio Acumulado Macro (Destino final vs Origen T=0)
                    delta_pred_cum = np.sign(p_pred[-1] - last_known_price)
                    delta_act_cum = np.sign(p_act[-1] - last_known_price)
                    hit_cum = 1.0 if delta_pred_cum == delta_act_cum else 0.0
                    
                    row_metrics[f'MAPE_B{b+1}'] = round(mape, 2)
                    row_metrics[f'Naive_MAPE_B{b+1}'] = round(naive_mape, 2)
                    row_metrics[f'Hit_B{b+1}'] = hit_local
                    row_metrics[f'CumHit_B{b+1}'] = hit_cum
                    
            results.append(row_metrics)
            end_idx += stride
            
        df_results = pd.DataFrame(results)
        return df_results
