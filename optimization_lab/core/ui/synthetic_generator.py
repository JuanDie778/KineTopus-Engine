import numpy as np
import pandas as pd
import logging
from typing import Tuple

from core.ui.market_loader import MarketLoader
from core.quant_engine.blender import ContinuousBlender
from core.quant_engine.physics import PhysicsDiscoverer
from core.quant_engine.nervous import RegimeShiftDetector

logger = logging.getLogger(__name__)

class SyntheticMarketGenerator:
    """
    Generador estocástico de datos de mercado.
    Usa la inercia real (SINDy) de un activo como "semilla" y genera
    multiversos independientes proyectados mediante Euler-Maruyama.
    La aleatoriedad garantiza que surjan nuevos regímenes topológicos para el optimizador.
    """
    def __init__(self, base_ticker: str = 'BTC-USD'):
        self.base_ticker = base_ticker
        self._df_real = None
        
    def generate_multiverse(self, num_candles: int = 1500, mc_trajectories: int = 2000) -> pd.DataFrame:
        """
        Genera un DataFrame continuo ('Date', 'Open', 'High', 'Low', 'Close', 'Volume')
        basado en la física real extraída del ticker base.
        
        Args:
            num_candles (int): Tamaño del universo en velas.
            mc_trajectories (int): Densidad de caminos Monte Carlo para estabilizar la mediana (Consenso).
            
        Returns:
            pd.DataFrame: DataFrame sintético con OHLCV.
            """
        logger.info(f"[SyntheticGenerator] Iniciando extracción de inercia real para {self.base_ticker}...")
        
        # 1. Obtener datos reales de referencia
        if self._df_real is None:
            df_real = MarketLoader.load_ticker_data(self.base_ticker, period='1y', interval='1d')
            if isinstance(df_real.columns, pd.MultiIndex):
                df_real.columns = df_real.columns.droplevel(1)
            self._df_real = df_real.copy()
        else:
            df_real = self._df_real.copy()
            
        t_hist = np.arange(len(df_real), dtype=np.float64)
        log_returns, vol, raw_price, dt_val = MarketLoader.prepare_quant_input(df_real)
        
        # Normalización manual de volumen para el Blender (Capa 2)
        mu_v = np.mean(vol)
        sigma_v = np.std(vol) if np.std(vol) > 1e-8 else 1.0
        vol_z = (vol - mu_v) / sigma_v
        
        # 2. Suavizado Topológico
        blender = ContinuousBlender(tolerance=0.0025)
        blender.fit(t_hist, log_returns, dominant_periods=np.array([]), feature_idx=0)
        blender.fit(t_hist, vol_z, dominant_periods=np.array([]), feature_idx=1)
        smooth_r, r_dot, _ = blender.compute_continuous(0, t_hist)
        v_smooth, v_dot, _ = blender.compute_continuous(1, t_hist)
        
        # 3. Detectar Régimen Activo
        detector = RegimeShiftDetector(threshold=5.0, drift=1.0)
        cusum_report = detector.detect(log_returns, smooth_r)
        shift_idx = cusum_report['shift_indices']
        
        boundaries = [0] + shift_idx + [len(t_hist)]
        start_idx = 0
        for i_b in range(len(boundaries) - 1):
            if boundaries[i_b+1] - boundaries[i_b] >= 15:
                start_idx = boundaries[i_b]
                
        # 4. Descubrimiento SINDy sobre el Régimen Activo
        t_slice = t_hist[start_idx:]
        x_slice = np.column_stack((smooth_r, v_smooth))[start_idx:]
        x_dot_slice = np.column_stack((r_dot, v_dot))[start_idx:]
        
        sigma_res_r = float(np.std(log_returns[start_idx:] - smooth_r[start_idx:]))
        sigma_res_v = float(np.std(vol_z[start_idx:] - v_smooth[start_idx:]))
        
        discoverer = PhysicsDiscoverer(poly_degree=1)
        logger.debug(f"[SyntheticGenerator] Computando {mc_trajectories} caminos Monte Carlo ({num_candles} pasos)...")
        report = discoverer.extract_equations(
            t=t_slice, x=x_slice, x_dot=x_dot_slice, dt=dt_val, 
            horizon_steps=num_candles,
            sigma_res_r=sigma_res_r, sigma_res_v=sigma_res_v,
            last_price=raw_price[-1]
        )
        
        if report['prediction']['unstable'] or not report['prediction']['price_percentiles']:
            logger.warning("[SyntheticGenerator] Física inestable. Usando extrapolación determinística.")
            p50 = report['prediction']['det_price_path']
        else:
            p_price = report['prediction']['price_percentiles']
            p50 = p_price[2] # Mediana (Consenso)
            
        if len(p50) < num_candles:
            raise ValueError(f"Fallo matemático SINDy. Se generaron {len(p50)} velas, se requerían {num_candles}.")
            
        p50 = np.array(p50[:num_candles])
        
        # 5. Modelado Estocástico de OHLCV (Generación del Caos Sintético)
        volatilidad_velas = 0.008
        open_sintetico = p50 * (1.0 + np.random.normal(0, volatilidad_velas * 0.3, size=num_candles))
        high_sintetico = np.maximum(open_sintetico, p50) * (1.0 + np.abs(np.random.normal(0, volatilidad_velas * 0.5, size=num_candles)))
        low_sintetico = np.minimum(open_sintetico, p50) * (1.0 - np.abs(np.random.normal(0, volatilidad_velas * 0.5, size=num_candles)))
        
        mean_v = df_real['Volume'].mean() if 'Volume' in df_real.columns else 1000.0
        std_v = df_real['Volume'].std() if 'Volume' in df_real.columns else 150.0
        volume_sintetico = np.maximum(100.0, np.random.normal(mean_v, std_v * 0.20, size=num_candles))
        
        # Índices de fechas (continuación del real)
        fechas = pd.date_range(start=df_real.index[-1] + pd.Timedelta(days=1), periods=num_candles, freq="D")
        
        df_sintetico = pd.DataFrame({
            'Open': open_sintetico,
            'High': high_sintetico,
            'Low': low_sintetico,
            'Close': p50,
            'Volume': volume_sintetico
        }, index=fechas)
        df_sintetico.index.name = 'Date'
        
        logger.info(f"[SyntheticGenerator] Universo generado exitosamente. Forma: {df_sintetico.shape}")
        return df_sintetico
