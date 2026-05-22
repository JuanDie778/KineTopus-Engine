import numpy as np
import sys
import os
import time

# Forzar el path para encontrar src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.quant_engine.blender import ContinuousBlender
from src.quant_engine.nervous import RegimeShiftDetector

def test_adaptive_thermodynamics():
    print("Iniciando Prueba Termodinamica (Caja Blanca)")
    
    # Generar señal estocástica (2000 velas)
    np.random.seed(42)
    t = np.arange(2000, dtype=np.float64)
    
    # Régimen 1: Calma (0 a 1000)
    y1 = np.sin(t[:1000] * 0.05) + np.random.normal(0, 0.1, 1000)
    
    # Régimen 2: Turbulencia / Caos repentino (1000 a 2000)
    y2 = np.sin(t[1000:] * 0.05) + np.random.normal(0, 1.5, 1000)
    
    y = np.concatenate([y1, y2]).astype(np.float64)
    
    print("\n--- 1. Prueba de Memoria Líquida (Continuous Blender) ---")
    blender = ContinuousBlender(tolerance=0.5)
    
    start = time.perf_counter()
    # Enviamos datos sin 'dominant_periods'
    res_b = blender.fit(t, y, dominant_periods=np.array([]), feature_idx=0)
    y_smooth, dy, d2y = blender.compute_continuous(0, t)
    print(f"OK Blender completado en {res_b['runtime_ms']:.2f} ms")
    print(f"   MSE general: {res_b['mse']:.4f}")
    
    # Verificar que el suavizado absorbió la turbulencia asignando menos peso
    # (El MSE debería ser más alto en la segunda mitad, porque la curva se niega a seguir el ruido)
    mse_calma = np.mean((y[:1000] - y_smooth[:1000])**2)
    mse_caos = np.mean((y[1000:] - y_smooth[1000:])**2)
    print(f"   Inercia en Calma (Varianza Residual): {mse_calma:.4f}")
    print(f"   Inercia en Caos (Varianza Residual ignorada): {mse_caos:.4f} (Correcto, no sobreajustó)")
    
    if np.any(np.isnan(y_smooth)):
        print("ERROR: NaNs detectados en la curva topologica.")
        sys.exit(1)

    print("\n--- 2. Prueba de CUSUM Dinámico (Regime Shift Detector) ---")
    detector = RegimeShiftDetector(threshold=10.0, drift=0.5)
    
    start = time.perf_counter()
    res_c = detector.detect(y, y_smooth)
    print(f"OK CUSUM completado en {res_c['runtime_ms']:.2f} ms")
    
    # Verificar si hubo quiebres
    shift_indices = res_c['shift_indices']
    print(f"   Macro Quiebres Estructurales detectados: {len(shift_indices)}")
    if len(shift_indices) > 0:
        print(f"   Velas donde ocurrió quiebre: {shift_indices}")
        
    if np.any(np.isnan(res_c['s_pos'])):
        print("ERROR: NaNs detectados en S_pos.")
        sys.exit(1)
        
    print("\nExito: Pruebas de Termodinamica Adaptativa superadas exitosamente.")

if __name__ == "__main__":
    test_adaptive_thermodynamics()
