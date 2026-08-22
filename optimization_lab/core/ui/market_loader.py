import yfinance as yf
import pandas as pd
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class MarketLoader:
    """
    Carga de datos dinámicos desde Yahoo Finance y sanitización rígida
    para el Motor Quant Físico.
    """
    
    @staticmethod
    def load_ticker_data(symbol: str, period: str = "1mo", interval: str = "1h") -> pd.DataFrame:
        """
        Descarga el histórico del ticker usando yfinance.
        
        Args:
            symbol (str): Ticker del activo (ej. 'AAPL', 'BTC-USD').
            period (str): Periodo a descargar (ej. '1mo', '1y').
            interval (str): Frecuencia de velas (ej. '1h', '1d').
            
        Returns:
            pd.DataFrame: DataFrame crudo de yfinance.
        """
        logger.info(f"Descargando datos de {symbol} (Periodo: {period}, Intervalo: {interval})")
        
        # Intercept synthetic data requests for in-memory parallel universes
        if "Synthetic" in symbol or period == "Synthetic" or interval == "Simulated":
            import os
            import pickle
            cache_path = os.path.join(os.path.dirname(__file__), "..", "..", "notebooks_val", ".synthetic_prices.pkl")
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "rb") as f:
                        cache = pickle.load(f)
                    if symbol in cache:
                        logger.info(f"Cargados datos sintéticos desde caché local para {symbol}")
                        return cache[symbol]
                except Exception as e:
                    logger.warning(f"Error cargando caché sintética: {e}")
            raise ValueError(f"No se encontró el activo sintético {symbol} en caché local (.synthetic_prices.pkl).")

        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            raise ValueError(f"No se pudieron cargar datos para el ticker: {symbol}")
            
        return df

    @staticmethod
    def prepare_quant_input(df: pd.DataFrame, disable_norm: bool = False, disable_returns: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Sanitización Rígida: Extrae Precio (Close) y Volumen.
        Calcula Retornos Logarítmicos vectorizados y Normaliza (Z-Score) el volumen.
        Si disable_norm=True, pasa los datos crudos directos.
        
        Args:
            df (pd.DataFrame): DataFrame con la data (al menos 'Close').
            disable_norm (bool): Si es True (Modo Calibración), omite logs y Z-Score.
            disable_returns (bool): Si es True, usa precio absoluto en vez de retornos logarítmicos.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, float]: 
                - Arreglo float64 contiguo de Retornos Logarítmicos
                - Arreglo float64 contiguo de Volúmenes (Normalizados)
                - Arreglo float64 de Precio Absoluto original
                - dt temporal estimado
        """
        # Tolerancia y Reparación de Estructura CSV para Modelos Sintéticos
        df.columns = [str(c).title() for c in df.columns]  # Normalizar por si es "close"
        if 'Close' not in df.columns:
             if 'Price' in df.columns: df['Close'] = df['Price']
             elif 'V' in df.columns: df['Close'] = df['V'] # Generic Variable fallback
             else: raise KeyError("El DataFrame debe contener una columna 'Close' para proyectarse.")
             
        if 'Volume' not in df.columns:
             df['Volume'] = np.ones(len(df)) # Vector neutro si no hay volumen
             
        # Sanitización de NaNs
        df = df.ffill().bfill()
        
        # Extracción y Contigüidad (Regla 16GB)
        # Se requiere np.ascontiguousarray() para evitar fragmentación en memoria tras slices de pandas
        precio_raw = np.ascontiguousarray(df['Close'].values, dtype=np.float64)
        volumen = np.ascontiguousarray(df['Volume'].values, dtype=np.float64)
        
        # Evitar ceros en volumen (causas exógenas como premarket inactivo)
        EPSILON = 1e-8
        volumen[volumen == 0] = EPSILON
        
        if disable_returns or disable_norm:
            # Bypass de Retornos: Queremos medir la inercia P(t) y no su variación temporal r(t)
            log_returns = precio_raw.copy()
        else:
            # Hito 1: Cálculo de Retornos Logarítmicos (Vectorizado, Numpy Puro)
            # Prevención contra precios <= 0
            precio_seguro = np.maximum(precio_raw, EPSILON)
            log_returns = np.zeros_like(precio_seguro)
            log_returns[1:] = np.log(precio_seguro[1:] / precio_seguro[:-1])
            
        if disable_norm:
            # Modo Calibración: Tomamos el volumen tal cual
            volumen_z = volumen.copy()
        else:
            # 2. Normalización Volumen V (Z-Score Global)
            v_mean = np.mean(volumen)
            v_std = np.std(volumen)
            if v_std > 1e-8:
                volumen_z = (volumen - v_mean) / v_std
            else:
                volumen_z = volumen - v_mean
            
        # Estimación dt (Técnica de Frecuencia de Muestreo Constante)
        dt = 1.0 # Por defecto asuminos dt = 1 vela
        
        # El Motor Quant ahora analiza los retornos logarítmicos
        return log_returns, volumen_z, precio_raw, dt
