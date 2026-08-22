import requests
import pandas as pd
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class ClimateLoader:
    """
    Carga de datos climáticos históricos desde Open-Meteo API (Gratuita, sin API Key).
    Devuelve un DataFrame compatible con el Motor Quant (columnas 'Close' y 'Volume')
    donde 'Close' = Temperatura (°C) y 'Volume' = Presión Atmosférica (hPa).
    """
    
    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
    
    # Coordenadas de Valencia, España
    VALENCIA_LAT = 39.4698
    VALENCIA_LON = -0.3774
    
    @staticmethod
    def load_climate_data(
        start_date: str,
        end_date: str,
        lat: float = None,
        lon: float = None,
        timezone: str = "Europe/Madrid"
    ) -> pd.DataFrame:
        """
        Descarga datos climáticos horarios desde Open-Meteo.
        
        Args:
            start_date (str): Fecha inicio en formato 'YYYY-MM-DD'.
            end_date (str): Fecha fin en formato 'YYYY-MM-DD'.
            lat (float): Latitud. Por defecto Valencia.
            lon (float): Longitud. Por defecto Valencia.
            timezone (str): Zona horaria para los datos.
            
        Returns:
            pd.DataFrame: DataFrame con columnas 'Close' (Temperatura °C) 
                          y 'Volume' (Presión Atmosférica hPa).
        """
        if lat is None:
            lat = ClimateLoader.VALENCIA_LAT
        if lon is None:
            lon = ClimateLoader.VALENCIA_LON
            
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': start_date,
            'end_date': end_date,
            'hourly': 'temperature_2m,surface_pressure',
            'timezone': timezone
        }
        
        logger.info(f"Descargando datos climáticos ({start_date} → {end_date}) desde Open-Meteo...")
        
        response = requests.get(ClimateLoader.BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        hourly = data.get('hourly', {})
        
        if not hourly or 'time' not in hourly:
            raise ValueError("Open-Meteo devolvió una respuesta vacía o malformada.")
        
        df = pd.DataFrame({
            'Datetime': pd.to_datetime(hourly['time']),
            'Close': hourly['temperature_2m'],         # Temperatura → Señal Primaria
            'Volume': hourly['surface_pressure']        # Presión Atmosférica → Señal Secundaria
        })
        
        df = df.set_index('Datetime')
        
        # Sanitización de NaNs (por huecos meteorológicos)
        df = df.ffill().bfill()
        
        # Validación de integridad
        n_nans = df.isnull().sum().sum()
        if n_nans > 0:
            logger.warning(f"Se encontraron {n_nans} valores nulos tras sanitización. Rellenando con interpolación.")
            df = df.interpolate(method='linear')
        
        logger.info(f"✅ Datos climáticos cargados: {len(df)} registros horarios ({start_date} → {end_date})")
        
        return df
