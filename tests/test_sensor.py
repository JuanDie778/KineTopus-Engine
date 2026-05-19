import sys
import os
import time
import numpy as np

# Add the project root to sys.path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.quant_engine.sensor import SpectralAnalyzer

def test_spectral_analyzer():
    print("Iniciando pruebas de @BUILDER para SpectralAnalyzer (Capa 1)...")
    
    # 1. Configuración de prueba
    dt = 1.0  # Muestras cada 1 min
    n_samples = 4000
    t = np.arange(n_samples) * dt
    
    # Inyectar frecuencias conocidas (Periodos de 100, 50, y 25)
    # y1 = Precio simulado
    period_1 = 100.0
    period_2 = 25.0
    y1 = 5.0 * np.sin(2 * np.pi * t / period_1) + 2.0 * np.sin(2 * np.pi * t / period_2)
    # Añadir un poco de ruido estocástico
    y1 += np.random.normal(0, 0.5, n_samples)
    
    # y2 = Volumen simulado (Periodos de 50 y 200)
    period_3 = 50.0
    y2 = 10.0 * np.sin(2 * np.pi * t / period_3) + 3.0 * np.sin(2 * np.pi * t / 200.0)
    y2 += np.random.normal(0, 1.0, n_samples)
    
    # Construir matriz de datos (P, V)
    data = np.column_stack((y1, y2))
    print(f"[*] Tipo de dato generado: {type(data)} | Dtype: {data.dtype} | Forma: {data.shape}")
    
    assert data.dtype == np.float64, "Error crítico: El input no es float64"
    
    # 2. Inicializar analizador
    analyzer = SpectralAnalyzer(top_k=2) # Buscamos las 2 frecuencias principales de cada uno
    
    # 3. Ejecución
    start_time = time.perf_counter()
    results = analyzer.analyze(data, dt=dt)
    end_time = time.perf_counter()
    
    print(f"[*] Tiempo de ejecución FFT Paralela: {end_time - start_time:.4f} segundos")
    
    # 4. Validar resultados
    print("\n--- Resultados ---")
    for feature_idx in results:
        res = results[feature_idx]
        periods = res['periods']
        amplitudes = res['amplitudes']
        
        name = "Precio (Feature 0)" if feature_idx == 0 else "Volumen (Feature 1)"
        print(f"\n[{name}]")
        print(f"  Tipo de salida de periodos: {type(periods)} | Dtype: {periods.dtype}")
        
        assert periods.dtype == np.float64, "Violación de regla: Salida no es float64"
        
        for p, a in zip(periods, amplitudes):
            print(f"  -> Periodo dominante detectado: {p:.2f} (Amplitud: {a:.2f})")
            
    # Asertos heurísticos (dado que hay ruido, permitimos un pequeño margen de error iterativo)
    # Feature 0 (Precio) debería tener p~100 y p~25
    assert np.any(np.isclose(results[0]['periods'], 100.0, atol=2.0)), "Fallo detectando periodo de 100.0 en Precio"
    assert np.any(np.isclose(results[0]['periods'], 25.0, atol=2.0)), "Fallo detectando periodo de 25.0 en Precio"
    
    # Feature 1 (Volumen) debería tener p~50
    assert np.any(np.isclose(results[1]['periods'], 50.0, atol=2.0)), "Fallo detectando periodo de 50.0 en Volumen"
    
    print("\n[OK] Pruebas estructurales de @BUILDER pasadas. Tipos float64 garantizados.")

if __name__ == '__main__':
    test_spectral_analyzer()
