import numpy as np
import pandas as pd
import logging
from src.ui.market_loader import MarketLoader
from src.quant_engine.sensor import SpectralAnalyzer
from src.quant_engine.blender import ContinuousBlender
from src.quant_engine.nervous import RegimeShiftDetector
from src.quant_engine.physics import PhysicsDiscoverer

logger = logging.getLogger(__name__)

class PredictiveAutoTuner:
    """
    Sintonizador "In-Sample Walk-Forward".
    Evita el Data Leakage evaluando el Drift k contra un "Futuro Interno"
    y escoge el parámetro que logre el mayor Alpha vs el Naive Forecast.
    """
    
    def __init__(self, df: pd.DataFrame, disable_norm: bool = False, disable_returns: bool = False, test_blocks: int = 3):
        self.df = df
        self.disable_norm = disable_norm
        self.disable_returns = disable_returns
        self.test_blocks = test_blocks # Por defecto 3 tramos (30 velas)
        self.block_size = 10
        
    def run_search(self, min_drift: float = 0.1, max_drift: float = 5.0, steps: float = 0.1):
        test_size = self.test_blocks * self.block_size
        
        # Si el dataframe es muy pequeño, fallback de seguridad
        if len(self.df) <= test_size + 20: 
            return min_drift, {'status': 'Fallback: Insufficient data', 'alpha_edge': 0}
            
        # 1. División Interna estricta para evitar Fuga de Datos
        df_train = self.df.iloc[:-test_size]
        df_test = self.df.iloc[-test_size:]
        
        # Obtenemos los precios reales del futuro interno
        _, _, p_test_raw, _ = MarketLoader.prepare_quant_input(df_test, disable_norm=self.disable_norm, disable_returns=self.disable_returns)
        
        # 2. Preparación y Cacheado del Spline (Sólo sobre la data de Entrenamiento)
        log_returns, volumen_z, precio_raw, dt_val = MarketLoader.prepare_quant_input(
            df_train, disable_norm=self.disable_norm, disable_returns=self.disable_returns
        )
        t = np.arange(len(log_returns), dtype=np.float64) * dt_val
        total_velas = len(t)
        
        sensor = SpectralAnalyzer(top_k=2)
        data_matrix = np.column_stack((log_returns, volumen_z))
        fft_results = sensor.analyze(data_matrix, dt=dt_val)
        periodos_retornos = fft_results[0]['periods']
        periodos_volumen = fft_results[1]['periods']
        
        blender = ContinuousBlender(tolerance=0.0050)
        blender.fit(t, log_returns, periodos_retornos, feature_idx=0)
        blender.fit(t, volumen_z, periodos_volumen, feature_idx=1)
        
        r_smooth, r_dot, _ = blender.compute_continuous(0, t)
        v_smooth, v_dot, _ = blender.compute_continuous(1, t)
        
        x_matrix = np.column_stack((r_smooth, v_smooth))
        x_dot_matrix = np.column_stack((r_dot, v_dot))
        discoverer = PhysicsDiscoverer(poly_degree=1)
        
        best_drift = min_drift
        best_alpha = -float('inf')
        best_report = {'alpha_edge': 0, 'sindy_r2': 0, 'num_quiebres': 0}
        
        drift_values = np.arange(min_drift, max_drift + steps, steps)
        
        # Base origin point for Naive Forecast
        last_known_price = precio_raw[-1]
        
        for k in drift_values:
            detector = RegimeShiftDetector(threshold=5.0, drift=k)
            cusum_report = detector.detect(log_returns, r_smooth)
            shift_idx = cusum_report['shift_indices']
            
            # Aislar el último régimen válido para proyectar
            if len(shift_idx) > 0:
                start = shift_idx[-1]
            else:
                start = 0
            end = total_velas
            
            length = end - start
            
            # Si el último régimen es muy corto, esta k no sirve
            if length < 10:
                continue
                
            # Extraer física y predecir el futuro interno
            try:
                p_rep = discoverer.extract_equations(
                    t=t[start:end], x=x_matrix[start:end], x_dot=x_dot_matrix[start:end], 
                    dt=dt_val, horizon_steps=test_size, sigma_res_r=0, sigma_res_v=0, 
                    last_price=last_known_price, disable_norm=self.disable_norm, disable_returns=self.disable_returns
                )
                p_pred = np.array(p_rep['predicted_prices'])
            except Exception:
                continue
                
            # Calcular Alpha y Hit Direccional por bloques
            alphas_bloques = []
            hits_bloques = []
            for b in range(self.test_blocks):
                bloque_pred = p_pred[b*self.block_size : (b+1)*self.block_size]
                bloque_act = p_test_raw[b*self.block_size : (b+1)*self.block_size]
                
                if len(bloque_act) == 0 or len(bloque_pred) == 0:
                    continue
                    
                sindy_mape = np.mean(np.abs((bloque_act - bloque_pred) / bloque_act)) * 100
                naive_array = np.full_like(bloque_act, fill_value=last_known_price)
                naive_mape = np.mean(np.abs((bloque_act - naive_array) / bloque_act)) * 100
                
                alphas_bloques.append(naive_mape - sindy_mape)
                
                # Evaluación Vectorial de Dirección y Tendencia (Hit Ratio)
                delta_pred = np.sign(bloque_pred[-1] - bloque_pred[0])
                delta_act = np.sign(bloque_act[-1] - bloque_act[0])
                hits_bloques.append(1.0 if delta_pred == delta_act else 0.0)
                
            if len(alphas_bloques) == 0:
                continue
                
            mediana_alpha = np.median(alphas_bloques)
            tasa_acierto = np.mean(hits_bloques) # Porcentaje de veces que atinó la dirección (0.0 a 1.0)
            
            # SCORE COMPUESTO: Alpha_Edge + Recompensa Direccional
            # Le sumamos 0.5 puntos de Alpha ficticio por cada 100% de acierto direccional.
            # Así, si empata en error absoluto, el tuner siempre elegirá el que acertó la tendencia.
            composite_score = mediana_alpha + (tasa_acierto * 0.5)
            
            # Guardar el mejor basado en el Score Compuesto Direccional
            if composite_score > best_alpha:
                best_alpha = composite_score
                best_drift = k
                best_report = {
                    'alpha_edge': mediana_alpha,
                    'hit_ratio_interno': tasa_acierto,
                    'num_quiebres': len(shift_idx),
                    'sindy_r2': p_rep['score']
                }
                
        return round(best_drift, 2), best_report
