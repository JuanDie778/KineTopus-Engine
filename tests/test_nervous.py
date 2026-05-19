import sys
import os
import time
import numpy as np

# Permitir cargar módulos desde la raíz del proyecto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.quant_engine.nervous import RegimeShiftDetector

def test_cusum_regime_shift():
    print("Iniciando pruebas de @BUILDER para RegimeShiftDetector (Capa 3 - CUSUM)...")
    
    # 1. Simulación del Mercado y del Modelo Topológico
    n_samples = 1000
    t = np.arange(n_samples)
    
    # Modelo predictivo/físico esperado (una onda estacionaria)
    model_values = np.sin(0.05 * t) * 10
    
    # La realidad inicial del mercado: el modelo más un poco de ruido estocástico
    np.random.seed(99)
    real_values = model_values + np.random.normal(0, 1.0, n_samples)
    
    # EVENTO CATASTRÓFICO: En la vela 600, el régimen cambia 
    # (El banco central imprimió dinero, o hubo liquidaciones masivas).
    shift_index = 600
    real_values[shift_index:] += 6.0 # Desviación constante y persistente
    
    # Conformidad RAM 16GB
    real_values = real_values.astype(np.float64)
    model_values = model_values.astype(np.float64)
    
    # 2. Inicialización del Detector
    # threshold H=15.0 (ignorar ruido acumulado puro)
    # drift k=1.5 (ignorar ruido menor a 1.5 sigmas)
    detector = RegimeShiftDetector(threshold=15.0, drift=1.5)
    
    # 3. Ejecución Vectorial CUSUM
    start_time = time.perf_counter()
    report = detector.detect(real_values, model_values)
    end_time = time.perf_counter()
    
    shift_detected = report['shift_detected']
    triggers = report['shift_indices']
    
    print(f"[*] Tiempo de cálculo de CUSUM ({n_samples} velas): {(end_time - start_time) * 1000:.2f} ms")
    
    # 4. Validaciones Unitarias Matemáticas
    # La estructura temporal inicial no debe detectar ruptura
    # El algoritmo debería detectar el cambio *después* de la vela 600
    
    print("\n--- Resultados CUSUM del Sistema Nervioso ---")
    print(f"  -> Ruptura Estructural Detectada: {shift_detected}")
    
    assert shift_detected is True, "Fallo Crítico: El sistema nervioso ignoró la bomba (Shift no detectado)"
    
    # Ver que clase de datos arrojó 
    assert triggers.dtype == np.int32, "Violación Tipo Índice: Esperado Int32 para índices vectoriales"
    
    # Obtener el primer instante en que el Algoritmo gritó
    first_trigger = triggers[0]
    print(f"  -> Hito Catastrófico Insertado en [Índice {shift_index}]")
    print(f"  -> CUSUM gatilló la primera alerta en [Índice {first_trigger}] (Latencia de Detección: {first_trigger - shift_index} velas)")
    
    # El trigger siempre requiere acumulación, por su naturaleza, no dispara en la 600, esperará ~5-15 velas 
    # comprobando que fue persistente y no un error temporal.
    assert first_trigger > shift_index, "Detección Imposible: Detectó el shift antes de que sucediera o en el mismo instante sin acumular error, violando el mecanismo CUSUM continuo"
    assert first_trigger < shift_index + 30, "Lentitud Extrema: El umbral de reacción de la Capa 3 es demasiado alto (Apatía Matemática)"

    print("\n[OK] Tests de CUSUM de @BUILDER pasados. El Sistema Nervioso reaccionó asertivamente en milisegundos evitando falsos positivos.")

if __name__ == '__main__':
    test_cusum_regime_shift()
