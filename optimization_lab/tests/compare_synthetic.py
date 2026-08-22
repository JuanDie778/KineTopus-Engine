import sys
import os
import logging
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

# Resolver el path para importar desde core/
current_dir = os.path.abspath(os.getcwd())
if current_dir not in sys.path:
    sys.path.append(current_dir)

from core.ui.market_loader import MarketLoader
from core.ui.synthetic_generator import SyntheticMarketGenerator
from core.quant_engine.blender import ContinuousBlender

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def calculate_metrics(df: pd.DataFrame, name: str) -> dict:
    """Calcula los hechos estilizados y métricas TDA de una serie de precios."""
    # 1. Preparar log returns (Capa 1)
    if 'Close' not in df.columns:
        # yfinance usa MultiIndex o 'Adj Close' a veces, asumimos limpieza básica
        close_series = df.iloc[:, 3] if len(df.columns) > 3 else df.iloc[:, 0]
    else:
        close_series = df['Close']
        
    log_returns = np.diff(np.log(close_series.values))
    # Padding para alinear índices
    log_returns = np.insert(log_returns, 0, 0)
    
    # Momentos estadísticos
    kurt = kurtosis(log_returns, fisher=True) # Exceso de curtosis (>0 es Fat Tail)
    skw = skew(log_returns)
    volatility = np.std(log_returns) * np.sqrt(365) # Volatilidad Anualizada
    
    # Autocorrelación (Lag 1)
    # R: Eficiencia de mercado (cercana a 0)
    # R^2: Volatility Clustering (positiva)
    if len(log_returns) > 1:
        lag1_r = np.corrcoef(log_returns[:-1], log_returns[1:])[0, 1]
        log_ret_sq = log_returns**2
        lag1_r2 = np.corrcoef(log_ret_sq[:-1], log_ret_sq[1:])[0, 1]
    else:
        lag1_r, lag1_r2 = 0.0, 0.0
        
    # Complejidad TDA (Continuous Blender)
    t_arr = np.arange(len(log_returns), dtype=np.float64)
    blender = ContinuousBlender(tolerance=0.0025)
    # Desactivamos el logger interno para no ensuciar la consola
    blender_logger = logging.getLogger('core.quant_engine.blender')
    blender_logger.setLevel(logging.WARNING)
    
    telemetry = blender.fit(t_arr, log_returns, dominant_periods=np.array([]), feature_idx=0)
    mse = telemetry.get('mse', 0.0)
    smoothness = telemetry.get('smoothness_penalty', 0.0)
    
    return {
        'name': name,
        'kurtosis': kurt,
        'skewness': skw,
        'volatility_ann': volatility,
        'autocorr_r1': lag1_r,
        'autocorr_rsq1': lag1_r2,
        'tda_mse': mse,
        'tda_smoothness': smoothness
    }

def print_comparison(metrics_real: dict, metrics_synth: dict):
    logger.info("=" * 80)
    logger.info("🔬 REPORTE DE AUDITORÍA QUANT: DATOS REALES VS SINTÉTICOS")
    logger.info("=" * 80)
    logger.info(f"{'Métrica':<25} | {'Real (BTC-USD)':<20} | {'Sintético (Multiverso)':<20}")
    logger.info("-" * 80)
    
    keys = [
        ('Fat Tails (Kurtosis)', 'kurtosis', '>0 ideal'),
        ('Asimetría (Skewness)', 'skewness', '~0 ideal'),
        ('Volatilidad Anualizada', 'volatility_ann', '%'),
        ('Eficiencia (ACF Ret Lag1)', 'autocorr_r1', '~0 ideal'),
        ('Vol Clustering (ACF R2)', 'autocorr_rsq1', '>0 ideal'),
        ('Topología: Ajuste MSE', 'tda_mse', 'Bajo ideal'),
        ('Topología: Suavidad', 'tda_smoothness', 'Bajo ideal')
    ]
    
    for label, key, context in keys:
        vr = metrics_real[key]
        vs = metrics_synth[key]
        
        # Formateo
        str_r = f"{vr:.4f}"
        str_s = f"{vs:.4f}"
        if key == 'volatility_ann':
            str_r = f"{vr*100:.2f}%"
            str_s = f"{vs*100:.2f}%"
            
        logger.info(f"{label:<25} | {str_r:<20} | {str_s:<20} | {context}")
        
    logger.info("=" * 80)
    logger.info("CONCLUSIONES DEL ABOGADO DEL DIABLO (@AUDITOR):")
    if abs(metrics_synth['autocorr_r1']) > 0.3:
        logger.warning("-> [ALERTA] Alta autocorrelación lineal detectada. El generador introdujo memoria en el precio (Inevitable si SINDy es puramente determinista sin suficiente ruido).")
    else:
        logger.info("-> [OK] Autocorrelación lineal baja. El Multiverso mantiene la Hipótesis de Mercados Eficientes.")
        
    if metrics_synth['kurtosis'] < 0:
        logger.warning("-> [ALERTA] Kurtosis negativa (Platicúrtica). Faltan colas gruesas y cisnes negros en el Multiverso.")
    else:
        logger.info("-> [OK] Kurtosis positiva detectada. Presencia de Fat Tails confirmada.")
        
    mse_ratio = metrics_synth['tda_mse'] / (metrics_real['tda_mse'] + 1e-8)
    if mse_ratio > 10:
        logger.warning(f"-> [ALERTA] El MSE TDA del sintético es {mse_ratio:.1f}x peor que el real. Demasiado ruido artificial.")
    else:
        logger.info("-> [OK] La suavidad topológica del sintético es asimilable por el Motor Diferencial (Capa 2).")

if __name__ == "__main__":
    logger.info("Descargando 1500 velas reales de BTC-USD...")
    # Fix para limpiar warnings the yfinance
    import warnings
    warnings.filterwarnings('ignore')
    
    df_real = MarketLoader.load_ticker_data('BTC-USD', period='max', interval='1d')
    if isinstance(df_real.columns, pd.MultiIndex):
        df_real.columns = df_real.columns.droplevel(1)
    df_real = df_real.tail(1500).copy()
    
    logger.info("Generando 1500 velas del Multiverso Sintético...")
    # Apagar el logger del generador para evitar spam
    gen_logger = logging.getLogger('core.ui.synthetic_generator')
    gen_logger.setLevel(logging.WARNING)
    
    generator = SyntheticMarketGenerator(base_ticker='BTC-USD')
    df_synth = generator.generate_multiverse(num_candles=1500, mc_trajectories=100)
    
    logger.info("Calculando Huellas Físicas y Estadísticas...")
    metrics_r = calculate_metrics(df_real, "BTC Real")
    metrics_s = calculate_metrics(df_synth, "BTC Sintético")
    
    print_comparison(metrics_r, metrics_s)
