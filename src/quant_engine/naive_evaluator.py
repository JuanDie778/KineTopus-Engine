import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class NaiveWalkForwardEvaluator:
    """
    Motor de Backtesting Basado en Persistencia Naive (Línea Plana P(t+k) = P(t)).
    Sirve como baseline de inteligencia cero para el Comparative Benchmark Environment (CBE).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        disable_norm: bool = False,
        disable_returns: bool = False,
        context_window: int = 0
    ):
        self.df = df
        self.disable_norm = disable_norm
        self.disable_returns = disable_returns
        self.context_window = context_window

        if 'Ticker' in self.df.columns and len(self.df) > 0:
            self.ticker = str(self.df['Ticker'].iloc[0])
        elif hasattr(self.df, 'attrs') and 'ticker' in self.df.attrs:
            self.ticker = str(self.df.attrs['ticker'])
        else:
            self.ticker = 'UNKNOWN'

    def _get_ticker(self) -> str:
        if 'Ticker' in self.df.columns and len(self.df) > 0:
            return str(self.df['Ticker'].iloc[0])
        elif hasattr(self.df, 'attrs') and 'ticker' in self.df.attrs:
            return str(self.df.attrs['ticker'])
        return getattr(self, 'ticker', 'UNKNOWN')

    def run(
        self,
        initial_window: int = 150,
        stride: int = 20,
        horizon: int = 300,
        blocks: int = 60
    ) -> pd.DataFrame:
        if horizon % blocks != 0:
            raise ValueError(f"El horizonte ({horizon}) debe ser divisible entre el número de bloques ({blocks}).")

        # Construir orden de columnas exacto de 247 columnas (7 metadata + 60*4 bloques)
        expected_cols: List[str] = [
            'Model', 'Ticker', 'Iteracion (Velas Vistas)', 'Drift (k)', 'SINDy R2', 'Validez', 'Latencia_ms'
        ]
        for b in range(1, blocks + 1):
            expected_cols.extend([f'MAPE_B{b}', f'Naive_MAPE_B{b}', f'RMSE_B{b}', f'Hit_B{b}', f'CumHit_B{b}', f'Profit_B{b}'])

        total_len = len(self.df)
        max_valid_start = total_len - horizon
        if max_valid_start < initial_window or total_len == 0:
            return pd.DataFrame(columns=expected_cols)

        # Extraer array NumPy float64 fuera del bucle hot path para eliminar sobrecosto de pandas
        if 'Close' in self.df.columns:
            close_col = 'Close'
        elif 'close' in self.df.columns:
            close_col = 'close'
        else:
            raise KeyError("El DataFrame debe contener una columna 'Close' o 'close'.")

        close_prices = np.ascontiguousarray(self.df[close_col].values, dtype=np.float64)
        ticker_val = self._get_ticker()
        block_size = horizon // blocks

        results: List[Dict[str, Any]] = []
        end_idx = initial_window

        while end_idx <= max_valid_start:
            t0 = time.perf_counter()

            # Futuro real out-of-sample y último precio conocido in-sample
            actual_prices = close_prices[end_idx : end_idx + horizon]
            last_known_price = close_prices[end_idx - 1]

            # Reestructurar precios reales en forma 2D (bloques, block_size)
            actual_blocks = actual_prices.reshape(blocks, block_size)

            # A. Error Porcentual (MAPE) y RMSE
            denom = np.where(np.abs(actual_blocks) < 1e-12, 1e-12, actual_blocks)
            mape_per_block = np.mean(np.abs((actual_blocks - last_known_price) / denom), axis=1) * 100.0
            mape_per_block = np.round(mape_per_block, 2)

            rmse_per_block = np.sqrt(np.mean((actual_blocks - last_known_price)**2, axis=1))
            rmse_per_block = np.round(rmse_per_block, 4)

            # B. Hit Ratio Vectorial Local
            # delta_pred_local es sign(P_last - P_last) = 0.0 constante
            delta_act_local = np.sign(actual_blocks[:, -1] - actual_blocks[:, 0])
            hit_local_per_block = (delta_act_local == 0.0).astype(np.float64)

            # C. Hit Ratio Acumulado Macro
            # delta_pred_cum es sign(P_last - P_last) = 0.0 constante
            delta_act_cum = np.sign(actual_blocks[:, -1] - last_known_price)
            hit_cum_per_block = (delta_act_cum == 0.0).astype(np.float64)

            # D. Profit % (Para Naive es 0.0 constante)
            profit_per_block = np.zeros(blocks, dtype=np.float64)

            latency_ms = (time.perf_counter() - t0) * 1000.0

            row_metrics: Dict[str, Any] = {
                'Model': 'Naive',
                'Ticker': ticker_val,
                'Iteracion (Velas Vistas)': end_idx,
                'Drift (k)': 0.0,
                'SINDy R2': 0.0,
                'Validez': 'OK',
                'Latencia_ms': round(latency_ms, 4)
            }

            for b in range(blocks):
                b_num = b + 1
                mape_val = float(mape_per_block[b])
                row_metrics[f'MAPE_B{b_num}'] = mape_val
                row_metrics[f'Naive_MAPE_B{b_num}'] = mape_val
                row_metrics[f'RMSE_B{b_num}'] = float(rmse_per_block[b])
                row_metrics[f'Hit_B{b_num}'] = float(hit_local_per_block[b])
                row_metrics[f'CumHit_B{b_num}'] = float(hit_cum_per_block[b])
                row_metrics[f'Profit_B{b_num}'] = float(profit_per_block[b])

            results.append(row_metrics)
            end_idx += stride

        df_results = pd.DataFrame(results)
        return df_results[expected_cols]
