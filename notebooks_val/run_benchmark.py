"""
Script ejecutor para 7_Comparative_Benchmark.ipynb.
Ejecuta los 4 evaluadores (Kinetopus, ARIMA, LSTM, Naive) en Walk-Forward
sobre los activos ['BTC-USD', 'MSFT', 'XLF'] y consolida los resultados en unified_benchmark_db.csv.
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd

# Ignorar advertencias no críticas
warnings.filterwarnings('ignore')

# Configurar stdout a UTF-8 si es posible
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Asegurar import desde el root del proyecto
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.ui.market_loader import MarketLoader


def main():
    print("=" * 70)
    print("  COMPARATIVE BENCHMARK ENVIRONMENT (CBE) - RUNNER")
    print("=" * 70)

    VENTANA_INICIAL = 150      # Mínimo de velas para entrenar
    SALTO = 20                 # Stride entre iteraciones
    HORIZONTE = 300            # Velas futuras a predecir
    BLOQUES = 60               # Granularidad (cada bloque = 5 velas)
    CONTEXT_WINDOW = 1825      # Máximo lookback (~5 años diarios)

    TICKERS = ['BTC-USD', 'ETH-USD', 'MSFT', 'QQQ', 'XLF', 'GLD']
    MODELS_TO_RUN = ['Kinetopus', 'Kinetopus_Original', 'ARIMA', 'LSTM', 'Naive']

    DB_PATH = os.path.join(project_root, 'notebooks_val', 'unified_benchmark_db.csv')
    KINETOPUS_CSV = os.path.join(project_root, 'notebooks_val', 'backtest_kinetopus.csv')
    ARIMA_CSV = os.path.join(project_root, 'notebooks_val', 'backtest_arima_.csv')

    REUSE_EXISTING_DATA = True

    if not REUSE_EXISTING_DATA and os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
            print(f"   [CLEAN] Eliminado dataset previo {DB_PATH} para regeneración limpia.")
        except Exception as e:
            print(f"   [WARN] No se pudo eliminar dataset previo: {e}")

    print(f"Configuracion del Benchmark:")
    print(f"   Ventana Inicial: {VENTANA_INICIAL}")
    print(f"   Salto: {SALTO}")
    print(f"   Horizonte: {HORIZONTE} velas ({BLOQUES} bloques)")
    print(f"   Context Window: {CONTEXT_WINDOW}")
    print(f"   Tickers: {TICKERS}")
    print(f"   Modelos: {MODELS_TO_RUN}")
    print(f"   Reutilizar existentes: {REUSE_EXISTING_DATA}")

    # --- Cargar pares procesados ---
    def load_processed_pairs(csv_path: str) -> set:
        if not os.path.isfile(csv_path):
            return set()
        try:
            df = pd.read_csv(csv_path, usecols=['Ticker', 'Model'])
            return set(zip(df['Ticker'], df['Model']))
        except Exception:
            return set()

    expected_cols = [
        'Model', 'Ticker', 'Iteracion (Velas Vistas)', 'Drift (k)', 'SINDy R2', 'Validez', 'Latencia_ms'
    ]
    for b in range(1, BLOQUES + 1):
        expected_cols.extend([f'MAPE_B{b}', f'Naive_MAPE_B{b}', f'RMSE_B{b}', f'Hit_B{b}', f'CumHit_B{b}', f'Profit_B{b}'])

    def save_results(df_results: pd.DataFrame, csv_path: str, ticker: str, model_name: str):
        df_save = df_results.copy()
        df_save['Model'] = model_name
        df_save['Ticker'] = ticker
        if 'Latencia_ms' not in df_save.columns:
            df_save['Latencia_ms'] = 0.0

        for col in expected_cols:
            if col not in df_save.columns:
                df_save[col] = np.nan

        df_save = df_save[expected_cols]

        file_exists = os.path.isfile(csv_path)
        df_save.to_csv(csv_path, mode='a', header=not file_exists, index=False)
        print(f"   [OK] Guardadas {len(df_save)} filas para ({ticker}, {model_name})")

    processed_pairs = load_processed_pairs(DB_PATH)
    print(f"\nPares ya procesados en DB: {len(processed_pairs)}")
    for pair in sorted(processed_pairs):
        print(f"   - {pair}")

    # --- Reutilización de Datos Existentes ---
    if REUSE_EXISTING_DATA:
        print("\nImportando datos existentes...")

        # Kinetopus
        if os.path.isfile(KINETOPUS_CSV) and 'Kinetopus' in MODELS_TO_RUN:
            df_kt = pd.read_csv(KINETOPUS_CSV)
            if 'Ticker' in df_kt.columns:
                for ticker in TICKERS:
                    pair = (ticker, 'Kinetopus')
                    if pair not in processed_pairs:
                        df_ticker = df_kt[df_kt['Ticker'] == ticker].copy()
                        if len(df_ticker) > 0:
                            df_ticker['Model'] = 'Kinetopus'
                            save_results(df_ticker, DB_PATH, ticker, 'Kinetopus')
                            processed_pairs.add(pair)
                            print(f"   + Importado Kinetopus/{ticker}: {len(df_ticker)} filas")
                        else:
                            print(f"   ! No hay datos de Kinetopus para {ticker} en {KINETOPUS_CSV}")

        # ARIMA
        if os.path.isfile(ARIMA_CSV) and 'ARIMA' in MODELS_TO_RUN:
            df_ar = pd.read_csv(ARIMA_CSV)
            if 'Ticker' in df_ar.columns:
                for ticker in TICKERS:
                    pair = (ticker, 'ARIMA')
                    if pair not in processed_pairs:
                        df_ticker = df_ar[df_ar['Ticker'] == ticker].copy()
                        if len(df_ticker) > 0:
                            df_ticker['Model'] = 'ARIMA'
                            save_results(df_ticker, DB_PATH, ticker, 'ARIMA')
                            processed_pairs.add(pair)
                            print(f"   + Importado ARIMA/{ticker}: {len(df_ticker)} filas")
                        else:
                            print(f"   ! No hay datos de ARIMA para {ticker} en {ARIMA_CSV}")

    # --- Factory de Evaluadores ---
    def create_evaluator(model_name: str, df_market: pd.DataFrame, ticker: str):
        try:
            if model_name == 'Kinetopus':
                from src.quant_engine.evaluator import WalkForwardEvaluator
                return WalkForwardEvaluator(df_market, disable_norm=False, disable_returns=False, context_window=CONTEXT_WINDOW), True

            elif model_name == 'Kinetopus_Original':
                from src.quant_engine.evaluator_original import OriginalWalkForwardEvaluator
                return OriginalWalkForwardEvaluator(df_market, disable_norm=False, disable_returns=False, context_window=CONTEXT_WINDOW), True

            elif model_name == 'ARIMA':
                from src.quant_engine.arima_evaluator import AutoARIMAWalkForwardEvaluator
                return AutoARIMAWalkForwardEvaluator(df_market, disable_norm=False, disable_returns=False), True

            elif model_name == 'LSTM':
                from src.quant_engine.lstm_evaluator import LSTMWalkForwardEvaluator
                return LSTMWalkForwardEvaluator(df_market, disable_norm=False, disable_returns=False, ticker=ticker, context_window=CONTEXT_WINDOW), True

            elif model_name == 'Naive':
                from src.quant_engine.naive_evaluator import NaiveWalkForwardEvaluator
                return NaiveWalkForwardEvaluator(df_market, disable_norm=False, disable_returns=False, context_window=CONTEXT_WINDOW), True

            else:
                return None, False
        except ImportError as e:
            print(f"   [ERROR] No se pudo importar evaluador para {model_name}: {e}")
            return None, False

    # --- Loop Principal de Ejecución ---
    print("\n" + "=" * 70)
    print("  EJECUCION DE BENCHMARK WALK-FORWARD")
    print("=" * 70)

    total_new = 0
    skipped = 0
    failed = 0

    for ticker in TICKERS:
        print(f"\n--------------------------------------------------")
        print(f"Activo: {ticker}")
        print(f"--------------------------------------------------")

        try:
            df_market = MarketLoader.load_ticker_data(ticker, period='10y', interval='1d')
            print(f"   Descargadas {len(df_market)} velas para {ticker}")
        except Exception as e:
            print(f"   [ERROR] Descargando datos para {ticker}: {e}")
            continue

        if len(df_market) < VENTANA_INICIAL + HORIZONTE:
            print(f"   [SKIP] Historial demasiado corto ({len(df_market)} velas).")
            continue

        for model_name in MODELS_TO_RUN:
            pair = (ticker, model_name)

            if pair in processed_pairs:
                print(f"   [SKIP] {model_name}: Ya procesado.")
                skipped += 1
                continue

            print(f"   [RUN] {model_name}: Ejecutando Walk-Forward...")

            evaluator, ok = create_evaluator(model_name, df_market, ticker)
            if not ok:
                failed += 1
                continue

            t0 = time.time()
            try:
                df_results = evaluator.run(
                    initial_window=VENTANA_INICIAL,
                    stride=SALTO,
                    horizon=HORIZONTE,
                    blocks=BLOQUES
                )
                elapsed = time.time() - t0

                if 'Model' not in df_results.columns:
                    df_results.insert(0, 'Model', model_name)
                if 'Ticker' not in df_results.columns:
                    df_results.insert(1, 'Ticker', ticker)

                save_results(df_results, DB_PATH, ticker, model_name)
                processed_pairs.add(pair)
                total_new += len(df_results)

                valid_pct = (df_results['Validez'] == 'OK').mean() * 100 if 'Validez' in df_results.columns else 100
                print(f"   [DONE] {model_name}: {len(df_results)} iteraciones en {elapsed:.1f}s (Validez OK: {valid_pct:.0f}%)")

            except Exception as e:
                elapsed = time.time() - t0
                print(f"   [FAIL] {model_name}: Error tras {elapsed:.1f}s -> {e}")
                failed += 1
                continue

            time.sleep(0.5)

    print("\n" + "=" * 70)
    print(f"  RESUMEN: {total_new} filas nuevas | {skipped} omitidas | {failed} fallidas")
    print("=" * 70)

    # --- Generación de Reporte de Métricas Final ---
    if os.path.isfile(DB_PATH):
        print("\n" + "=" * 70)
        print("  REPORTE COMPLETO DE METRICAS DEL BENCHMARK UNIFICADO")
        print("=" * 70)

        df_db = pd.read_csv(DB_PATH)
        print(f"\nDataset Final: {len(df_db)} filas total en {DB_PATH}\n")

        print("Distribucion de Filas por Modelo x Ticker:")
        print(pd.crosstab(df_db['Model'], df_db['Ticker'], margins=True))

        print("\nTasa de Validez Matematica por Modelo:")
        if 'Validez' in df_db.columns:
            val_df = df_db.groupby('Model')['Validez'].apply(lambda x: (x == 'OK').mean() * 100)
            for m, pct in val_df.items():
                print(f"   - {m:10s}: {pct:.1f}% OK")

        # Filtrar solo filas con Validez OK para comparar métricas
        df_valid = df_db[df_db['Validez'] == 'OK'].copy() if 'Validez' in df_db.columns else df_db.copy()

        # Metricas a horizontes clave: Bloque 1 (velas 1-5), Bloque 10 (velas 46-50), Bloque 30 (velas 146-150), Bloque 60 (velas 296-300)
        key_blocks = [1, 10, 30, 60]

        print("\n" + "-" * 70)
        print("1. PROMEDIO DE HIT RATIO ACUMULADO (CumHit) POR HORIZONTE")
        print("-" * 70)

        header = f"{'Modelo':12s} | " + " | ".join([f"B{b} (v.{b*5:3d})" for b in key_blocks])
        print(header)
        print("-" * len(header))

        for model_name in sorted(df_valid['Model'].unique()):
            sub = df_valid[df_valid['Model'] == model_name]
            vals = []
            for b in key_blocks:
                col = f'CumHit_B{b}'
                if col in sub.columns:
                    v = sub[col].mean() * 100
                    vals.append(f"{v:6.1f}%")
                else:
                    vals.append("  N/A  ")
            print(f"{model_name:12s} | " + " | ".join(vals))

        print("\n" + "-" * 70)
        print("2. MEDIANA DE MAPE (Error Porcentual %) POR HORIZONTE")
        print("-" * 70)

        header_mape = f"{'Modelo':12s} | " + " | ".join([f"B{b} (v.{b*5:3d})" for b in key_blocks])
        print(header_mape)
        print("-" * len(header_mape))

        for model_name in sorted(df_valid['Model'].unique()):
            sub = df_valid[df_valid['Model'] == model_name]
            vals = []
            for b in key_blocks:
                col = f'MAPE_B{b}'
                if col in sub.columns:
                    v = sub[col].dropna().median()
                    if np.isfinite(v):
                        vals.append(f"{v:6.2f}%")
                    else:
                        vals.append("  Inf %")
                else:
                    vals.append("  N/A  ")
            print(f"{model_name:12s} | " + " | ".join(vals))

        print("\n" + "-" * 70)
        print("3. RATIO DE VENTAJA MAPE vs NAIVE (Mediana MAPE / Mediana Naive_MAPE) [< 1.0 es mejor]")
        print("-" * 70)

        print(header_mape)
        print("-" * len(header_mape))

        for model_name in sorted(df_valid['Model'].unique()):
            sub = df_valid[df_valid['Model'] == model_name]
            vals = []
            for b in key_blocks:
                col_m = f'MAPE_B{b}'
                col_n = f'Naive_MAPE_B{b}'
                if col_m in sub.columns and col_n in sub.columns:
                    m_val = sub[col_m].dropna().median()
                    n_val = sub[col_n].dropna().median()
                    if np.isfinite(m_val) and np.isfinite(n_val) and n_val > 1e-8:
                        ratio = m_val / n_val
                        vals.append(f"{ratio:6.3f} ")
                    else:
                        vals.append("  N/A  ")
                else:
                    vals.append("  N/A  ")
            print(f"{model_name:12s} | " + " | ".join(vals))

        print("\n" + "-" * 70)
        print("4. PROMEDIO DE PROFIT DE ESTRATEGIA LONG/SHORT (%) POR HORIZONTE")
        print("-" * 70)

        print(header)
        print("-" * len(header))

        for model_name in sorted(df_valid['Model'].unique()):
            sub = df_valid[df_valid['Model'] == model_name]
            vals = []
            for b in key_blocks:
                col = f'Profit_B{b}'
                if col in sub.columns:
                    v = sub[col].dropna().mean()
                    if np.isfinite(v):
                        vals.append(f"{v:+6.2f}%")
                    else:
                        vals.append("  N/A  ")
                else:
                    vals.append("  N/A  ")
            print(f"{model_name:12s} | " + " | ".join(vals))

        print("\n" + "-" * 70)
        print("5. MEDIANA DE RMSE (ERROR CUADRATICO MEDIO) POR HORIZONTE")
        print("-" * 70)

        print(header)
        print("-" * len(header))

        for model_name in sorted(df_valid['Model'].unique()):
            sub = df_valid[df_valid['Model'] == model_name]
            vals = []
            for b in key_blocks:
                col = f'RMSE_B{b}'
                if col in sub.columns:
                    v = sub[col].dropna().median()
                    if np.isfinite(v):
                        vals.append(f"{v:6.2f}")
                    else:
                        vals.append("  Inf  ")
                else:
                    vals.append("  N/A  ")
            print(f"{model_name:12s} | " + " | ".join(vals))

        print("\n" + "=" * 70)
        print("  BENCHMARK COMPLETADO EXITOSAMENTE")
        print("=" * 70)


if __name__ == '__main__':
    main()
