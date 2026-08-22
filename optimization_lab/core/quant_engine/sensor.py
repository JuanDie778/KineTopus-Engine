import numpy as np
import concurrent.futures
import logging
import time

logger = logging.getLogger(__name__)

class SpectralAnalyzer:
    """
    Sensor Layer (Capa 1) del Motor Quant.
    Responsable de ingerir series temporales (Precio, Volumen) y aplicar FFT
    para descubrir las frecuencias dominantes (el "ritmo" del mercado).
    Garantiza el uso estricto de numpy float64 y procesamiento paralelo.
    """

    def __init__(self, top_k: int = 3, n_workers: int = None):
        """
        Inicializa el analizador espectral.

        Args:
            top_k (int): Número de frecuencias dominantes a extraer por serie.
            n_workers (int, opcional): Número de procesos paralelos. Por defecto, usa todos los cores.
        """
        self.top_k = top_k
        self.n_workers = n_workers

    @staticmethod
    def _compute_fft(series: np.ndarray, dt: float, top_k: int) -> dict:
        """
        Calcula la FFT de una serie 1D individual, extrayendo los periodos dominantes.
        Este método es estático para ser serializable por ProcessPoolExecutor.

        Args:
            series (np.ndarray): Array 1D de datos (float64).
            dt (float): Intervalo de tiempo entre muestras.
            top_k (int): Número de top frecuencias a extraer.

        Returns:
            dict: Diccionario con 'periods' y 'amplitudes' (np.ndarray float64).
        """
        if not isinstance(series, np.ndarray) or series.dtype != np.float64:
             raise TypeError("La serie debe ser un numpy array de tipo float64 (Golden Rule de Memoria).")
             
        start_t = time.perf_counter()
        n = len(series)
        
        # Eliminar la media (componente DC) para centrar la señal en cero
        series_centered = series - np.mean(series)
        
        # Calcular FFT Real (más eficiente para señales reales que np.fft.fft)
        fft_result = np.fft.rfft(series_centered)
        frequencies = np.fft.rfftfreq(n, d=dt)
        
        # Densidad Espectral de Potencia (Magnitud al cuadrado)
        power = np.abs(fft_result) ** 2
        
        # Ignorar la frecuencia 0 (ya eliminamos la media de todas formas)
        power[0] = 0.0

        # Encontrar los índices de las 'top_k' frecuencias de mayor potencia
        # argpartition es O(N) lo cual es más eficiente que ordenar todo el array
        if len(power) < top_k:
            top_k = len(power)
            
        top_indices = np.argpartition(power, -top_k)[-top_k:]
        # Ordenar esos top_k por potencia descendente
        top_indices = top_indices[np.argsort(-power[top_indices])]
        
        top_freqs = frequencies[top_indices]
        top_amplitudes = np.sqrt(power[top_indices]) # Volver a amplitud lineal

        # Convertir frecuencias a periodos (Ventanas de tiempo). 
        # Si la frecuencia es 0, el periodo tiende a infinito, lo limitamos.
        top_periods = np.zeros_like(top_freqs)
        valid_freqs = top_freqs > 0
        top_periods[valid_freqs] = 1.0 / top_freqs[valid_freqs]
        
        runtime = (time.perf_counter() - start_t) * 1000.0

        return {
            'periods': top_periods.astype(np.float64),
            'amplitudes': top_amplitudes.astype(np.float64),
            'runtime_ms': float(runtime)
        }

    def analyze(self, data: np.ndarray, dt: float = 1.0) -> dict:
        """
        Analiza un array multidimensional distribuyendo la carga en paralelo.

        Args:
            data (np.ndarray): Matriz 2D de forma (N_muestras, N_features). Debe ser float64.
            dt (float): Paso de tiempo.

        Returns:
            dict: Resultados por cada 'feature' (columna). Claves son índices de columna 0..M.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("El input principal debe ser un numpy array (Prohibido pandas en hot loops).")
            
        if data.dtype != np.float64:
            logger.warning(f"Forzando cast a float64. Tipo detectado: {data.dtype}")
            data = data.astype(np.float64)

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_features = data.shape[1]
        results = {}

        # Ejecución en Paralelo (Hito 1: ProcessPoolExecutor)
        # Empaquetar argumentos
        tasks_args = [(data[:, i], dt, self.top_k) for i in range(n_features)]
        
        # Si solo hay una feature, evitar el overhead del executor
        if n_features == 1:
            results[0] = self._compute_fft(*tasks_args[0])
            return results

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Enviar tareas
            futures = {executor.submit(self._compute_fft, *args): i for i, args in enumerate(tasks_args)}
            
            # Recibir resultados a medida que terminan
            for future in concurrent.futures.as_completed(futures):
                feature_idx = futures[future]
                try:
                    results[feature_idx] = future.result()
                except Exception as exc:
                    logger.error(f"Fallo en FFT para feature {feature_idx}: {exc}")
                    # Retornar arrays vacíos como fallback de seguridad
                    results[feature_idx] = {
                        'periods': np.array([], dtype=np.float64),
                        'amplitudes': np.array([], dtype=np.float64)
                    }

        return results
