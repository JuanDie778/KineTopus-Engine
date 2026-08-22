import copy
import logging
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class PyTorchLSTM(nn.Module):
    """
    Module PyTorch LSTM de 1 capa para series temporales cuantitativas.
    """
    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 1, output_size: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class LSTMWalkForwardEvaluator:
    """
    Evaluador Walk-Forward para Benchmark Red Neuronal LSTM.
    Usa PyTorch en CPU y cumple estrictamente con el contrato CBE de 247 columnas.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        disable_norm: bool = False,
        disable_returns: bool = False,
        ticker: str = 'UNKNOWN',
        context_window: int = 0
    ):
        self.df = df.copy()
        self.disable_norm = disable_norm
        self.disable_returns = disable_returns
        self.ticker = ticker
        self.context_window = context_window
        self.device = torch.device('cpu')

    def _resolve_ticker(self) -> str:
        if self.ticker != 'UNKNOWN':
            return str(self.ticker)
        if 'Ticker' in self.df.columns and len(self.df) > 0:
            return str(self.df['Ticker'].iloc[0])
        if hasattr(self.df, 'attrs') and 'ticker' in self.df.attrs:
            return str(self.df.attrs['ticker'])
        return 'UNKNOWN'

    def run(
        self,
        initial_window: int = 150,
        stride: int = 20,
        horizon: int = 300,
        blocks: int = 60
    ) -> pd.DataFrame:
        if horizon % blocks != 0:
            raise ValueError(f"El horizonte ({horizon}) debe ser divisible entre el número de bloques ({blocks}).")

        expected_cols: List[str] = [
            'Model', 'Ticker', 'Iteracion (Velas Vistas)', 'Drift (k)', 'SINDy R2', 'Validez', 'Latencia_ms'
        ]
        for b in range(1, blocks + 1):
            expected_cols.extend([f'MAPE_B{b}', f'Naive_MAPE_B{b}', f'RMSE_B{b}', f'Hit_B{b}', f'CumHit_B{b}', f'Profit_B{b}'])

        total_len = len(self.df)
        max_valid_start = total_len - horizon
        if max_valid_start < initial_window or total_len == 0:
            return pd.DataFrame(columns=expected_cols)

        if 'Close' in self.df.columns:
            close_col = 'Close'
        elif 'close' in self.df.columns:
            close_col = 'close'
        else:
            raise KeyError("El DataFrame debe contener una columna 'Close' o 'close'.")

        close_prices = np.ascontiguousarray(self.df[close_col].values, dtype=np.float64)
        ticker_val = self._resolve_ticker()
        block_size = horizon // blocks
        seq_length = 30

        results: List[Dict[str, Any]] = []
        end_idx = initial_window

        while end_idx <= max_valid_start:
            t0 = time.perf_counter()
            validez = 'OK'
            r2_val = 1.0

            try:
                # Slicing in-sample data
                if self.context_window > 0 and end_idx > self.context_window:
                    train_prices = close_prices[end_idx - self.context_window : end_idx]
                else:
                    train_prices = close_prices[:end_idx]

                actual_prices = close_prices[end_idx : end_idx + horizon]
                last_known_price = close_prices[end_idx - 1]

                # Transformación a Retornos Logarítmicos
                r_train = np.log(np.maximum(train_prices[1:], 1e-12) / np.maximum(train_prices[:-1], 1e-12))

                # Normalización Z-score estricta in-sample de retornos
                if self.disable_norm:
                    mean_in = 0.0
                    std_in = 1.0
                    Z_in = r_train.copy()
                else:
                    mean_in = float(np.mean(r_train))
                    std_in = float(np.std(r_train))
                    if std_in < 1e-8:
                        std_in = 1.0
                    Z_in = (r_train - mean_in) / std_in

                N_train = len(Z_in)
                if N_train <= seq_length:
                    raise ValueError(f"Insuficientes datos ({N_train}) para formar secuencias L={seq_length}.")

                # Generación de secuencias L=30
                X_list = []
                y_list = []
                for i in range(N_train - seq_length):
                    X_list.append(Z_in[i : i + seq_length])
                    y_list.append(Z_in[i + seq_length])

                X_arr = np.array(X_list, dtype=np.float32)[..., np.newaxis]
                y_arr = np.array(y_list, dtype=np.float32)[..., np.newaxis]

                X_tensor = torch.tensor(X_arr, dtype=torch.float32, device=self.device)
                y_tensor = torch.tensor(y_arr, dtype=torch.float32, device=self.device)

                # Instanciación del modelo y entrenamiento con Early Stopping
                model = PyTorchLSTM(input_size=1, hidden_size=32, num_layers=1, output_size=1).to(self.device)
                optimizer = optim.Adam(model.parameters(), lr=0.01)
                criterion = nn.MSELoss()

                best_loss = float('inf')
                patience = 5
                patience_cnt = 0
                best_weights = copy.deepcopy(model.state_dict())

                model.train()
                for epoch in range(50):
                    optimizer.zero_grad()
                    out = model(X_tensor)
                    loss = criterion(out, y_tensor)
                    loss_val = loss.item()

                    if np.isnan(loss_val) or np.isinf(loss_val):
                        raise ValueError("Divergencia en la pérdida de entrenamiento (NaN/Inf).")

                    loss.backward()
                    optimizer.step()

                    if loss_val < best_loss - 1e-5:
                        best_loss = loss_val
                        best_weights = copy.deepcopy(model.state_dict())
                        patience_cnt = 0
                    else:
                        patience_cnt += 1
                        if patience_cnt >= patience:
                            break

                model.load_state_dict(best_weights)

                # Predicción recursiva multipaso de RETORNOS (k=1..horizon)
                model.eval()
                input_window = Z_in[-seq_length:].copy()
                preds_scaled = []

                with torch.no_grad():
                    for step in range(horizon):
                        inp_tensor = torch.tensor(
                            input_window.reshape(1, seq_length, 1),
                            dtype=torch.float32,
                            device=self.device
                        )
                        pred_z = model(inp_tensor).item()

                        if np.isnan(pred_z) or np.isinf(pred_z):
                            raise ValueError(f"Predicción NaN/Inf detectada en paso {step+1}.")

                        preds_scaled.append(pred_z)

                        input_window = np.roll(input_window, -1)
                        input_window[-1] = pred_z

                preds_scaled_arr = np.array(preds_scaled, dtype=np.float64)
                
                # Des-normalizar retornos
                r_pred = preds_scaled_arr * std_in + mean_in
                
                # Reconstrucción exponencial de precios
                cum_ret = np.cumsum(r_pred)
                pred_prices = last_known_price * np.exp(cum_ret)

            except Exception as e:
                logger.warning(f"Error LSTM en Iteración {end_idx}: {e}")
                pred_prices = None
                validez = 'FALLO MATEMÁTICO'
                r2_val = 0.0

            latency_ms = (time.perf_counter() - t0) * 1000.0

            row_metrics: Dict[str, Any] = {
                'Model': 'LSTM',
                'Ticker': ticker_val,
                'Iteracion (Velas Vistas)': end_idx,
                'Drift (k)': 0.0,
                'SINDy R2': r2_val,
                'Validez': validez,
                'Latencia_ms': round(latency_ms, 4)
            }

            if validez != 'OK' or pred_prices is None or len(pred_prices) != horizon:
                row_metrics['Validez'] = 'FALLO MATEMÁTICO'
                row_metrics['SINDy R2'] = 0.0
                for b in range(1, blocks + 1):
                    row_metrics[f'MAPE_B{b}'] = np.nan
                    row_metrics[f'Naive_MAPE_B{b}'] = np.nan
                    row_metrics[f'RMSE_B{b}'] = np.nan
                    row_metrics[f'Hit_B{b}'] = np.nan
                    row_metrics[f'CumHit_B{b}'] = np.nan
                    row_metrics[f'Profit_B{b}'] = np.nan
            else:
                for b in range(blocks):
                    b_num = b + 1
                    p_pred = pred_prices[b * block_size : (b + 1) * block_size]
                    p_act = actual_prices[b * block_size : (b + 1) * block_size]

                    denom = np.where(np.abs(p_act) < 1e-12, 1e-12, p_act)
                    mape = np.mean(np.abs((p_act - p_pred) / denom)) * 100.0
                    rmse = np.sqrt(np.mean((p_act - p_pred)**2))

                    p_naive = np.full_like(p_act, fill_value=last_known_price)
                    naive_mape = np.mean(np.abs((p_act - p_naive) / denom)) * 100.0

                    delta_pred_local = np.sign(p_pred[-1] - p_pred[0])
                    delta_act_local = np.sign(p_act[-1] - p_act[0])
                    hit_local = 1.0 if delta_pred_local == delta_act_local else 0.0

                    delta_pred_cum = np.sign(p_pred[-1] - last_known_price)
                    delta_act_cum = np.sign(p_act[-1] - last_known_price)
                    hit_cum = 1.0 if delta_pred_cum == delta_act_cum else 0.0

                    act_ret_pct = ((p_act[-1] - last_known_price) / last_known_price) * 100.0
                    profit = delta_pred_cum * act_ret_pct

                    row_metrics[f'MAPE_B{b_num}'] = round(float(mape), 2)
                    row_metrics[f'Naive_MAPE_B{b_num}'] = round(float(naive_mape), 2)
                    row_metrics[f'RMSE_B{b_num}'] = round(float(rmse), 4)
                    row_metrics[f'Hit_B{b_num}'] = float(hit_local)
                    row_metrics[f'CumHit_B{b_num}'] = float(hit_cum)
                    row_metrics[f'Profit_B{b_num}'] = round(float(profit), 2)

            results.append(row_metrics)
            end_idx += stride

        df_results = pd.DataFrame(results)
        return df_results[expected_cols]
