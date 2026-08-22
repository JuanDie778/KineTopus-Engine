import numpy as np
import pandas as pd
import logging
from typing import Iterator, Tuple

logger = logging.getLogger(__name__)

class WalkForwardSplitter:
    """
    Divide un DataFrame temporal en pares estructurados de (Entrenamiento, Validación)
    para prevenir estrictamente el Data Leakage durante la optimización.
    """
    def __init__(self, initial_window: int = 150, horizon_steps: int = 150, stride: int = 20):
        """
        Args:
            initial_window (int): Tamaño del contexto In-Sample inicial.
            horizon_steps (int): Cuántas velas a futuro debe predecir el modelo (Out-of-Sample).
            stride (int): Cuántas velas avanzar en cada iteración del bucle temporal.
        """
        self.initial_window = initial_window
        self.horizon_steps = horizon_steps
        self.stride = stride

    def yield_splits(self, df: pd.DataFrame) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generador Python que avanza el tiempo de forma segura.
        
        Yields:
            Tuple[pd.DataFrame, pd.DataFrame]: (df_train, df_val)
        """
        n_total = len(df)
        if n_total < self.initial_window + self.horizon_steps:
            logger.warning(f"[WalkForwardSplitter] Datos insuficientes ({n_total}) para la ventana requerida.")
            return

        start_idx = 0
        end_idx = self.initial_window
        
        while end_idx + self.horizon_steps <= n_total:
            df_train = df.iloc[start_idx:end_idx]
            df_val = df.iloc[end_idx : end_idx + self.horizon_steps]
            
            yield df_train, df_val
            
            # Expandir o Deslizar la ventana
            # Usaremos "Expanding Window" por defecto (start_idx = 0)
            # start_idx += self.stride (si quisieramos Rolling Window)
            end_idx += self.stride
