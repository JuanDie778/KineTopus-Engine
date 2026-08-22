import numpy as np
import logging
import time
from scipy.interpolate import UnivariateSpline

logger = logging.getLogger(__name__)

class ContinuousBlender:
    """
    Capa 2: Suavizado Topológico (Motor Quant - Hito 2).
    Transforma datos discretos ruidosos en variedades topológicas continuas usando
    splines parametrizados por las frecuencias dominantes descubiertas por el Sensor (Capa 1).
    Provee derivadas analíticas estables en lugar de diferencias finitas (ruidosas).
    Asegura máxima contigüidad en RAM (np.float64 estricto).
    """

    def __init__(self, tolerance: float = 0.5):
        """
        Inicializa el ContinuousBlender.
        
        Args:
            tolerance (float): Factor base paramétrico para dictar la agresividad del suavizado.
        """
        self.tol = tolerance
        # Diccionario para persistir en memoria los polinomios analíticos por cada feature (Precio, Volumen)
        self._models = {} 

    def fit(self, t: np.ndarray, y: np.ndarray, dominant_periods: np.ndarray, feature_idx: int = 0) -> dict:
        """
        Ajusta una variedad continua a los datos (t, y) usando las ventanas temporales.
        
        Args:
            t (np.ndarray): Vector de tiempo contiguo (N,)
            y (np.ndarray): Array 1D de datos estocásticos (N,), np.float64.
            dominant_periods (np.ndarray): Matriz 1D de periodos de Capa 1.
            feature_idx (int): Identificador interno numérico.
            
        Returns:
            dict: Telemetría de ajuste {'mse': float, 'runtime_ms': float}
        """
        start_t = time.perf_counter()
        if not isinstance(y, np.ndarray) or y.dtype != np.float64:
            raise TypeError(f"Regla de RAM (16GB): El input `y` debe ser np.float64. Recibido {type(y)}")
        if not isinstance(t, np.ndarray) or t.dtype != np.float64:
            raise TypeError(f"Regla de RAM (16GB): El input `t` debe ser np.float64.")

        # Heurística de Suavizado Topológico (Fase 7 - Precisión Z-Score):
        # Como y ahora tiene Varianza 1.0 (Normalizado), `s` representa la suma de los errores 
        # cuadrados permitidos frente a la varianza unitaria.
        if len(dominant_periods) > 0:
            primary_period = np.max(dominant_periods)
            s = primary_period * len(y) * self.tol * 0.1 
        else:
            s = len(y) * self.tol

        # === MEMORIA LÍQUIDA (Control Dinámico de Entropía) ===
        # @BUILDER: En lugar de confiar ciegamente en todos los puntos por igual,
        # detectamos la volatilidad local instantánea. Los puntos con alta turbulencia (ruido extremo)
        # reciben un peso 'w' menor, obligando al Spline a ignorar el caos reciente y aferrarse a la inercia estructural.
        diff_y = np.diff(y)
        w = np.ones(len(y), dtype=np.float64)
        
        if len(diff_y) > 30:
            alpha = 2.0 / (30.0 + 1.0)
            var_t = np.var(diff_y[:30]) / 2.0
            if var_t < 1e-10: var_t = 1e-10
            
            # La primera vela hereda el peso de la varianza inicial
            w[0] = 1.0 / (np.sqrt(var_t) + 1e-6)
            
            for i in range(1, len(y)):
                inst_var = (diff_y[i-1]**2) / 2.0
                var_t = alpha * inst_var + (1.0 - alpha) * var_t
                sigma_local = np.sqrt(var_t)
                w[i] = 1.0 / (sigma_local + 1e-6)
                
            # Normalizar pesos para no distorsionar el factor 's' global asimétricamente
            w = w / np.mean(w)

        try:
            # Se inyecta la memoria líquida a través del vector de pesos 'w'
            spline = UnivariateSpline(t, y, w=w, s=s, k=3)
            self._models[feature_idx] = spline
            
            # Calcular Telemetría MSE
            y_smooth = spline(t)
            mse = np.mean((y - y_smooth)**2)
            
            runtime = (time.perf_counter() - start_t) * 1000.0
            logger.info(f"Topología ajustada para feature {feature_idx}. Factor de inercia *s*={s:.2f} | MSE={mse:.5f}")
            
            return {
                'mse': float(mse),
                'runtime_ms': float(runtime)
            }
        except Exception as e:
            logger.error(f"Fallo matemático al asimilar spline en feature {feature_idx}: {e}")
            raise

    def compute_continuous(self, feature_idx: int, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Deriva vectorialmente sobre la función matemática (No sobre datos numéricos pasados).
        Extrae la posición, inercia y aceleración perfectas.
        
        Args:
            feature_idx (int): Identificador del modelo físico.
            t (np.ndarray): Vector de tiempo a interpolar.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Matrices 1D (y_smooth, dy_dt, d2y_dt2) 
        """
        if feature_idx not in self._models:
            raise ValueError(f"Feature {feature_idx} no ha recibido fit().")
            
        spline = self._models[feature_idx]
        
        # Evaluar la función suavizada en t
        y_smooth = spline(t)
        
        # Evaluar la primera derivada analítica
        dy_dt = spline.derivative(n=1)(t)
        
        # Evaluar la segunda derivada analítica
        d2y_dt2 = spline.derivative(n=2)(t)
        
        # Forzar tipos puros por especificación paramétrica
        return (
            y_smooth.astype(np.float64), 
            dy_dt.astype(np.float64), 
            d2y_dt2.astype(np.float64)
        )
