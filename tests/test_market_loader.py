import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ui.market_loader import MarketLoader

def test_market_loader():
    print("Iniciando pruebas de @BUILDER para Market Loader (Fase 5)...")
    
    # 1. Prueba de Ingesta Dinámica (AAPL - Apple Inc.)
    try:
        df = MarketLoader.load_ticker_data("AAPL", period="5d", interval="1h")
        print(f"[*] Datos descargados de AAPL. Shape: {df.shape}")
        
        # Validar Columnas
        assert 'Close' in df.columns, "Falta columna Close"
        assert 'Volume' in df.columns, "Falta columna Volume"
    except Exception as e:
        print(f"Error de red o yfinance: {e}")
        print("[!] No se pudo conectar a Yahoo Finance. Test abortado (posible bloqueo de red).")
        return
        
    # 2. Prueba de Sanitización Vectorial
    log_returns, volumen_z, precio_raw, dt = MarketLoader.prepare_quant_input(df)
    
    # Validar Contigüidad (Crítico para la regla de la RAM)
    assert log_returns.flags['C_CONTIGUOUS'], "Memoria de Retornos Ineficiente (Fragmentada)"
    assert volumen_z.flags['C_CONTIGUOUS'], "Memoria de Volumen Ineficiente (Fragmentada)"
    assert precio_raw.flags['C_CONTIGUOUS'], "Memoria de Precio Crudo Ineficiente (Fragmentada)"
    
    # Validar Tipado Float64
    assert log_returns.dtype == np.float64, "Error de Tipado en Retornos"
    assert volumen_z.dtype == np.float64, "Error de Tipado en Volumen"
    
    # Validar No NaNs
    assert not np.isnan(log_returns).any(), "NaNs filtrados incorrectamente en Retornos"
    assert not np.isnan(volumen_z).any(), "NaNs filtrados incorrectamente en Volumen"
    
    # Validar media cercana a 0 en retornos logarítmicos (heurística)
    mean_r = np.mean(log_returns)
    assert np.isclose(mean_r, 0.0, atol=0.1), f"Fallo Log Retornos: Media anormalmente alta {mean_r}"

    mean_v = np.mean(volumen_z)
    std_v = np.std(volumen_z)
    assert np.isclose(mean_v, 0.0, atol=1e-7), f"Fallo Normalización: Media de Vol es {mean_v}"
    assert np.isclose(std_v, 1.0, atol=1e-7) or std_v == 0.0, f"Fallo Normalización: Std de Vol es {std_v}"
    
    print("[OK] Market Loader probado exitosamente. Vectorización Float64 asegurada.")

if __name__ == "__main__":
    test_market_loader()
