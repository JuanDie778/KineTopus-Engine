import numpy as np
import pandas as pd
import logging
import warnings
from typing import Dict, Any

from src.ui.market_loader import MarketLoader
import pmdarima as pm

logger = logging.getLogger(__name__)

class AutoARIMAWalkForwardEvaluator:
    """
    Motor de Backtesting para Benchmark Clásico (Auto-ARIMA).
    Usa la misma arquitectura Expanding Window que Kinetopus para asegurar igualdad de condiciones.
    """
    
    def __init__(self, df: pd.DataFrame, disable_norm: bool = False, disable_returns: bool = False):
        self.df = df
        self.disable_norm = disable_norm
        self.disable_returns = disable_returns

    def run(self, initial_window: int = 150, stride: int = 20, horizon: int = 150, blocks: int = 15) -> pd.DataFrame:
        total_len = len(self.df)
        results = []
        
        # El límite de parada asegura que el modelo no asuma el futuro
        max_valid_start = total_len - horizon
        
        print(f"Iniciando Auto-ARIMA Walk-Forward (Ventana:{initial_window}, Salto:{stride}, Horizonte:{horizon} velas en {blocks} bloques)")
        
        end_idx = initial_window
        while end_idx <= max_valid_start:
            df_slice = self.df.iloc[:end_idx].copy() # Pasado
            actual_prices = self.df.iloc[end_idx:end_idx+horizon]['Close'].values # Futuro
            
            last_known_price = df_slice['Close'].iloc[-1]
            train_data = df_slice['Close'].values
            
            # Auto-ARIMA fitting
            try:
                # Omitimos warnings para no ensuciar el log
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Limitamos p y q para que la simulación masiva no sea inmensamente lenta
                    model = pm.auto_arima(train_data, start_p=1, start_q=1,
                                          max_p=3, max_q=3, m=1,
                                          start_P=0, seasonal=False,
                                          d=None, D=0, trace=False,
                                          error_action='ignore',  
                                          suppress_warnings=True, 
                                          stepwise=True)
                
                pred_prices = model.predict(n_periods=horizon)
                if isinstance(pred_prices, pd.Series):
                    pred_prices = pred_prices.values
                    
                validez = 'OK'
                r2_fake = 1.0 # Solo por compatibilidad de columnas
            except Exception as e:
                logger.warning(f"Error ARIMA en Iteración {end_idx}: {e}")
                pred_prices = []
                validez = 'FALLO MATEMÁTICO'
                r2_fake = 0.0

            # Evaluación Vectorial vs Futuro Real (Exactamente igual que Kinetopus)
            row_metrics = {
                'Iteracion (Velas Vistas)': end_idx,
                'Drift (k)': 0.0, # ARIMA no usa drift
                'SINDy R2': r2_fake, 
                'Validez': validez
            }
            
            block_size = horizon // blocks
            
            if len(pred_prices) != horizon or validez != 'OK':
                row_metrics['Validez'] = 'FALLO MATEMÁTICO'
            else:
                for b in range(blocks):
                    p_pred = np.array(pred_prices[b*block_size : (b+1)*block_size])
                    p_act = np.array(actual_prices[b*block_size : (b+1)*block_size])
                    
                    # A. Error Porcentual (MAPE)
                    mape = np.mean(np.abs((p_act - p_pred) / p_act)) * 100
                    
                    # Naive Forecast (Línea plana en el último precio)
                    p_naive = np.full_like(p_act, fill_value=last_known_price)
                    naive_mape = np.mean(np.abs((p_act - p_naive) / p_act)) * 100
                    
                    # B. Hit Ratio Vectorial Local
                    delta_pred_local = np.sign(p_pred[-1] - p_pred[0])
                    delta_act_local = np.sign(p_act[-1] - p_act[0])
                    hit_local = 1.0 if delta_pred_local == delta_act_local else 0.0
                    
                    # C. Hit Ratio Acumulado Macro
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
