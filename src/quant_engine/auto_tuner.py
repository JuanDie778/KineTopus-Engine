import numpy as np
import pandas as pd
import logging
from src.ui.market_loader import MarketLoader
from src.quant_engine.sensor import SpectralAnalyzer
from src.quant_engine.blender import ContinuousBlender
from src.quant_engine.nervous import RegimeShiftDetector
from src.quant_engine.physics import PhysicsDiscoverer

logger = logging.getLogger(__name__)

class CUSUMAutoTuner:
    """
    Optimizador Grid-Search rápido para encontrar el Drift k óptimo de CUSUM
    basado en la métrica "R2 Ponderado por Dimensión Temporal".
    """
    
    def __init__(self, df: pd.DataFrame, disable_norm: bool = False, disable_returns: bool = False):
        self.df = df
        self.disable_norm = disable_norm
        self.disable_returns = disable_returns
        
    def run_search(self, min_drift: float = 0.1, max_drift: float = 5.0, steps: float = 0.1):
        # 1. Preparación y Cacheado del Spline (Sólo se calcula una vez)
        log_returns, volumen_z, precio_raw, dt_val = MarketLoader.prepare_quant_input(
            self.df, disable_norm=self.disable_norm, disable_returns=self.disable_returns
        )
        t = np.arange(len(log_returns), dtype=np.float64) * dt_val
        total_velas = len(t)
        
        sensor = SpectralAnalyzer(top_k=2)
        data_matrix = np.column_stack((log_returns, volumen_z))
        fft_results = sensor.analyze(data_matrix, dt=dt_val)
        periodos_retornos = fft_results[0]['periods']
        periodos_volumen = fft_results[1]['periods']
        
        # Tolerancia fija y relajada al ruido (0.0050)
        blender = ContinuousBlender(tolerance=0.0050)
        blender.fit(t, log_returns, periodos_retornos, feature_idx=0)
        blender.fit(t, volumen_z, periodos_volumen, feature_idx=1)
        
        r_smooth, r_dot, _ = blender.compute_continuous(0, t)
        v_smooth, v_dot, _ = blender.compute_continuous(1, t)
        
        # SINDy Preparación
        x_matrix = np.column_stack((r_smooth, v_smooth))
        x_dot_matrix = np.column_stack((r_dot, v_dot))
        # Complejidad fija en 1 (Lineal)
        discoverer = PhysicsDiscoverer(poly_degree=1)
        
        best_drift = min_drift
        best_score = -float('inf')
        best_report = {}
        
        drift_values = np.arange(min_drift, max_drift + steps, steps)
        
        for k in drift_values:
            detector = RegimeShiftDetector(threshold=5.0, drift=k) # Umbral H fijo a 5.0
            cusum_report = detector.detect(log_returns, r_smooth)
            shift_idx = cusum_report['shift_indices']
            
            boundaries = [0] + shift_idx + [total_velas]
            weighted_r2_sum = 0.0
            
            # Ejecutar Física en cada bloque descubierto
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i+1]
                length = end - start
                
                # Ignorar puramente regímenes matemáticamente inservibles (< 15 velas) 
                if length < 15:
                    continue
                    
                p_rep = discoverer.extract_equations(
                    t=t[start:end], x=x_matrix[start:end], x_dot=x_dot_matrix[start:end], 
                    dt=dt_val, horizon_steps=1, sigma_res_r=0, sigma_res_v=0, 
                    last_price=precio_raw[end-1], disable_norm=self.disable_norm, disable_returns=self.disable_returns
                )
                
                # Ponderación respecto al peso de todo el dataset
                peso = length / float(total_velas)
                r2 = max(0, p_rep['score']) # No restamos R2 negativos, los limitamos a 0
                weighted_r2_sum += r2 * peso

            # Fórmula Híbrida de Fitness Institucional
            num_quiebres = len(shift_idx)
            fitness = weighted_r2_sum / (1.0 + 0.1 * num_quiebres)
            
            if fitness > best_score:
                best_score = fitness
                best_drift = k
                best_report = {
                    'fitness': fitness,
                    'weighted_r2': weighted_r2_sum,
                    'num_quiebres': num_quiebres
                }
                
        return round(best_drift, 2), best_report
