import numpy as np
import logging
import time

logger = logging.getLogger(__name__)

class RegimeShiftDetector:
    """
    Capa 3: Sistema Nervioso del Motor Quant (Gatillo CUSUM).
    Supervisa rigurosamente las anomalías entre las predicciones continuas del Moldeador
    y la realidad cruda del mercado. Evita falsos positivos estocásticos permitiendo una "deriva" (drift),
    y dispara una alerta matemática cuando el umbral (H) de error sistemático se quiebra.
    """

    def __init__(self, threshold: float = 4.0, drift: float = 0.5):
        """
        Inicializa el detector CUSUM.
        
        Args:
            threshold (float): Límite crítico (H). Si se supera, el régimen antiguo está "roto".
            drift (float): Holgura (k). Ruido esperado. Las desviaciones por debajo no se acumulan.
        """
        self.H = threshold
        self.k = drift

    def detect(self, real_values: np.ndarray, model_values: np.ndarray) -> dict:
        """
        Computa el algoritmo CUSUM vectorizado (Z-Score residual continuo).
        
        Args:
            real_values (np.ndarray): Matriz 1D de datos crudos, np.float64.
            model_values (np.ndarray): Matriz 1D de la curva spline topológica, np.float64.
            
        Returns:
            dict: {
                'shift_detected': bool,
                'shift_indices': np.ndarray,
                's_pos': np.ndarray,
                's_neg': np.ndarray
            }
        """
        # Validación de Entradas en Memoria (Contigüidad RAM requerida)
        if not (isinstance(real_values, np.ndarray) and isinstance(model_values, np.ndarray)):
            raise TypeError("Regla Estricta: Entradas CUSUM no son arrays numpy")
            
        if real_values.dtype != np.float64 or model_values.dtype != np.float64:
             logger.warning("Forzando casting a float64 para el CUSUM")
             real_values = real_values.astype(np.float64)
             model_values = model_values.astype(np.float64)

        if len(real_values) != len(model_values):
            raise ValueError("Incongruencia dimensional entre lo crudo y el modelo topológico")

        n = len(real_values)
        if n == 0:
            return {'shift_detected': False, 'shift_indices': np.array([], dtype=np.int32), 
                    's_pos': np.array([], dtype=np.float64), 's_neg': np.array([], dtype=np.float64),
                    'runtime_ms': 0.0}

        start_t = time.perf_counter()

        # 1. Calcular el Error Físico
        residuo = real_values - model_values
        
        # 2. Normalización Fundamental Z-Score 
        # (Corrección del @BUILDER: Media global causa fugas del futuro al pasado.
        # Asumimos que el modelo físico tiene media de error 0.
        # Estimamos la desviación estándar usando diferencias para ser robustos a los quiebres estructurales).
        diff_res = np.diff(residuo)
        if len(diff_res) == 0:
            sigma = 1e-8
        else:
            # np.std(diff) / sqrt(2) estima la std del ruido original ignorando step-shifts
            sigma = np.std(diff_res) / np.sqrt(2) 
            
        if sigma < 1e-8:
             z_scores = np.zeros_like(residuo) 
        else:
             # El modelo debería estar centrado, el residuo puro es la señal
             z_scores = residuo / sigma

        # 3. Reservar Memoria CUSUM Contigua
        S_pos = np.zeros(n, dtype=np.float64)
        S_neg = np.zeros(n, dtype=np.float64)

        # 4. Acumulación Dinámica (Loop secuencial CUSUM con Reseteo)
        # Numba/Cython no se usan para mantener minimalismo de dependencias (solo requerimos numpy base).
        # Para N=4000 (ventanas estándar), el loop for de Python+NumPy es aceptablemente sub-milisegundo.
        triggers_pos_list = []
        triggers_neg_list = []
        
        for i in range(1, n):
            S_pos[i] = max(0.0, S_pos[i-1] + z_scores[i] - self.k)
            S_neg[i] = max(0.0, S_neg[i-1] - z_scores[i] - self.k)
            
            # Reseteo del Sistema Nervioso (Fase 8)
            # Evita saturación en el límite y permite atrapar múltiples quiebres
            if S_pos[i] > self.H:
                triggers_pos_list.append(i)
                S_pos[i] = 0.0
                
            if S_neg[i] > self.H:
                triggers_neg_list.append(i)
                S_neg[i] = 0.0

        # 5. Detección de Gatillos
        triggers_pos = np.array(triggers_pos_list, dtype=np.int32)
        triggers_neg = np.array(triggers_neg_list, dtype=np.int32)
        
        raw_shifts = np.unique(np.concatenate([triggers_pos, triggers_neg]))
        raw_shifts = np.sort(raw_shifts).astype(np.int32)
        
        # 6. Agrupación (Clustering) de Macro-Quiebres
        # CUSUM puede disparar múltiples "flags" amarillas seguidas durante un shock turbulento.
        # Agrupamos esas banderas para crear un único "Quiebre Estructural", requiriendo un silencio
        # de al menos 30 velas (zona de paz) para considerar que entramos a otro régimen separado.
        macro_shifts = []
        if len(raw_shifts) > 0:
            macro_shifts.append(raw_shifts[0])
            for s in raw_shifts[1:]:
                # Si han pasado más de 30 velas sin quiebres, es un régimen nuevo y sólido
                if s - macro_shifts[-1] > 30:
                    macro_shifts.append(s)
                    
        macro_shifts = np.array(macro_shifts, dtype=np.int32)
        
        runtime = (time.perf_counter() - start_t) * 1000.0

        return {
            'shift_detected': len(macro_shifts) > 0,
            'shift_indices': macro_shifts.tolist(), # Convertimos a lista nativa para compatibilidad con app.py
            'raw_triggers': raw_shifts.tolist(), # Exportamos los gatillos brutos para el gráfico UI amarillo
            's_pos': S_pos,
            's_neg': S_neg,
            'runtime_ms': float(runtime)
        }
