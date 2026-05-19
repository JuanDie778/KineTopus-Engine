import sys
import os
import time
import numpy as np

# Permitir cargar módulos desde la raíz del proyecto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.quant_engine.physics import PhysicsDiscoverer

def test_sindy_oscillator():
    print("Iniciando pruebas de @BUILDER para Extracción Física SINDy (Capa 4)...")
    
    # 1. Sistema Dinámico Sintético (Oscilador Armónico Simple)
    # Ecuaciones reales:
    # d(Precio)/dt = -Volumen
    # d(Volumen)/dt = Precio
    
    t = np.linspace(0, 10, 500)
    
    # Resolver analíticamente (Sin(t) y Cos(t))
    precio = np.cos(t)     # P(t) = cos(t)
    volumen = np.sin(t)    # V(t) = sin(t)
    
    # Derivadas analíticas verdaderas (Capa 2 ya debería habernos dado esto sin ruido)
    dp_dt = -np.sin(t)     # dP/dt = -V
    dv_dt = np.cos(t)      # dV/dt = P
    
    # Agrupar las matrices vectoriales (N, 2) garantizando float64
    x = np.column_stack((precio, volumen)).astype(np.float64)
    x_dot = np.column_stack((dp_dt, dv_dt)).astype(np.float64)
    
    # 2. Instanciar Motor de Ecuaciones
    # Poly degree 2 para permitir ruido cuadrático, threshold de 0.1 para forzar esparcir.
    discoverer = PhysicsDiscoverer(threshold=0.1, poly_degree=2)
    
    # 3. Regresión 
    start_time = time.perf_counter()
    report = discoverer.extract_equations(t=t, x=x, x_dot=x_dot, last_price=1.0)
    end_time = time.perf_counter()
    
    equations = report['equations']
    score = report['score']
    complexity = report['complexity']
    prediction = report['prediction']
    
    print(f"[*] Tiempo de cálculo STLSQ (500 velas, {complexity} params activos): {(end_time - start_time) * 1000:.2f} ms")
    
    # 4. Resultados Analíticos vs Algebraicos (Caja Blanca Mínima)
    print("\n--- Ecuaciones Detectadas (Caja Blanca) ---")
    print(f"  -> R2 Score Explicativo: {score:.5f}")
    
    # Deberíamos tener 2 ecuaciones
    assert len(equations) == 2, "Sistema de Ecuaciones incompleto"
    assert isinstance(equations, list), "Las ecuaciones deben ser una lista de strings legibles"
    
    # Valiando la Física Encontrada (usando 'r' en inercia de precio y 'V' en inercia de volumen)
    print(f"  -> dr/dt = {equations[0]}")
    print(f"  -> dV/dt = {equations[1]}")
    
    print("\n--- Telemetría Predictiva (t+1) ---")
    print(f"  -> Precio / Volumen Estimado (Integración ODE Euler): {prediction['x_next']}")
    assert 'x_next' in prediction and 'x_dot_next' in prediction, "Falta la extrapolación numérica en el reporte."
    
    # Si STLSQ funcionó y evitamos fuga numérica, el score será 1.000 y solo sobrevivirán los coeficientes inerciales principales + sesgo Z-Score.
    assert score > 0.99, "Error Crítico: STLSQ falló en reconocer un oscilador armónico perfecto (R2 pobre)."
    assert complexity in [2, 3], "Esparcidad fallida: STLSQ mantuvo residuos inútiles no nulos (Sobre-ajuste / Bloat)."
    
    # Chequear el parseo de strings de Pysindy (que incluye 'r' y 'V')
    assert "V" in equations[0], "No se detectó la variable Volumen (V) en la inercia del Retorno."
    assert "r" in equations[1], "No se detectó la variable Retorno (r) en la inercia del Volumen."
    
    print("\n[OK] Extracción de ODE confirmada. PySINDy respetó el Contrato Continuista. No hay Pandas en memoria.")

def test_sindy_stochastic_monte_carlo():
    print("\nIniciando Pruebas de Estrés (@AUDITOR) para Euler-Maruyama (Fase 9)...")
    
    t = np.linspace(0, 10, 500)
    precio = np.cos(t)     
    volumen = np.sin(t)    
    dp_dt = -np.sin(t)     
    dv_dt = np.cos(t)      
    
    x = np.column_stack((precio, volumen)).astype(np.float64)
    x_dot = np.column_stack((dp_dt, dv_dt)).astype(np.float64)
    
    discoverer = PhysicsDiscoverer(threshold=0.1, poly_degree=2)
    
    start_time = time.perf_counter()
    # dt=0.02 para evitar explosión de Euler explícito en oscilador armónico
    # Se introduce el testeo de volatilidades asimétricas duales (Fase 12)
    report = discoverer.extract_equations(t=t, x=x, x_dot=x_dot, dt=0.02, horizon_steps=150, sigma_res_r=0.05, sigma_res_v=0.08, last_price=1.0)
    end_time = time.perf_counter()
    
    runtime_ms = (end_time - start_time) * 1000
    print(f"[*] Tiempo de cálculo Monte Carlo (150 pasos x 1000 trayectorias): {runtime_ms:.2f} ms")
    
    # 500 ms limit check for responsiveness
    assert runtime_ms < 500.0, f"Error: Renderizado demasiado lento ({runtime_ms:.2f}ms). Podría bloquear Streamlit."
    
    prediction = report['prediction']
    
    # Percentiles Check
    percentiles = prediction.get('price_percentiles', [])
    assert len(percentiles) == 5, "Debe generar exactamente 5 percentiles (P5, P25, P50, P75, P95)"
    assert len(percentiles[0]) == 150, "Los percentiles deben cubrir el horizonte temporal completo (150 pasos)"
    
    # Check dispersion (Cone Shape)
    p5_end = percentiles[0][-1]
    p50_end = percentiles[2][-1]
    p95_end = percentiles[4][-1]
    
    assert p5_end < p50_end < p95_end, "La dispersión estocástica falló. Los percentiles no se distribuyen coherentemente."
    
    print("[OK] Prueba de estrés Monte Carlo superada. Integración vectorizada ultra-rápida (16GB RAM Ready).")

if __name__ == '__main__':
    test_sindy_oscillator()
    test_sindy_stochastic_monte_carlo()
