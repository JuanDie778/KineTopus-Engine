"""
Test de Sanidad del Entorno Comparativo de Benchmark (CBE).
Ejecuta un mini Walk-Forward con parámetros reducidos sobre MSFT sintético
y valida integridad de esquema, valores y consistencia cruzada entre los 4 evaluadores.

Uso: python -m notebooks_val.test_benchmark_sanity
"""

import sys
import os
import time
import numpy as np
import pandas as pd

# Asegurar que el proyecto raíz esté en el path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_synthetic_market(n: int = 800, seed: int = 42) -> pd.DataFrame:
    """Genera un DataFrame sintético con estructura OHLCV realista."""
    np.random.seed(seed)
    dates = pd.date_range('2020-01-01', periods=n, freq='1D')
    log_ret = np.random.normal(loc=0.0003, scale=0.015, size=n).astype(np.float64)
    close = 100.0 * np.exp(np.cumsum(log_ret))
    volume = np.random.uniform(1e5, 1e6, size=n).astype(np.float64)
    
    df = pd.DataFrame({
        'Open': close * (1.0 + np.random.normal(0, 0.002, n)),
        'High': close * 1.01,
        'Low': close * 0.99,
        'Close': close,
        'Volume': volume,
    }, index=dates)
    return df


def run_check(name: str, condition: bool, detail: str = ""):
    """Imprime PASS/FAIL para un check individual."""
    status = "✅ PASS" if condition else "❌ FAIL"
    msg = f"  {status} | {name}"
    if detail and not condition:
        msg += f" → {detail}"
    print(msg)
    return condition


def main():
    print("=" * 70)
    print("  CBE SANITY TEST — Mini Walk-Forward con datos sintéticos")
    print("=" * 70)
    
    # Parámetros reducidos para test rápido
    INITIAL_WINDOW = 150
    STRIDE = 100       # Stride grande para pocas iteraciones (~5)
    HORIZON = 300
    BLOCKS = 60
    
    df_market = generate_synthetic_market(n=800)
    total_checks = 0
    passed_checks = 0
    
    # =========================================================================
    # 1. IMPORTAR LOS 4 EVALUADORES
    # =========================================================================
    print("\n📦 Fase 1: Importación de Evaluadores")
    
    evaluators = {}
    
    try:
        from src.quant_engine.naive_evaluator import NaiveWalkForwardEvaluator
        evaluators['Naive'] = NaiveWalkForwardEvaluator(df_market)
        ok = True
    except ImportError as e:
        ok = False
    total_checks += 1
    if run_check("Import NaiveWalkForwardEvaluator", ok):
        passed_checks += 1
    
    try:
        from src.quant_engine.arima_evaluator import AutoARIMAWalkForwardEvaluator
        evaluators['ARIMA'] = AutoARIMAWalkForwardEvaluator(df_market)
        ok = True
    except ImportError as e:
        ok = False
    total_checks += 1
    if run_check("Import AutoARIMAWalkForwardEvaluator", ok):
        passed_checks += 1
    
    try:
        from src.quant_engine.lstm_evaluator import LSTMWalkForwardEvaluator
        evaluators['LSTM'] = LSTMWalkForwardEvaluator(df_market)
        ok = True
    except ImportError as e:
        ok = False
    total_checks += 1
    if run_check("Import LSTMWalkForwardEvaluator", ok):
        passed_checks += 1
    
    try:
        from src.quant_engine.evaluator import WalkForwardEvaluator
        evaluators['Kinetopus'] = WalkForwardEvaluator(df_market)
        ok = True
    except ImportError as e:
        ok = False
    total_checks += 1
    if run_check("Import WalkForwardEvaluator (Kinetopus)", ok):
        passed_checks += 1
    
    # =========================================================================
    # 2. EJECUTAR CADA EVALUADOR
    # =========================================================================
    print(f"\n🏃 Fase 2: Ejecución Walk-Forward (stride={STRIDE}, ~5 iteraciones)")
    
    results = {}
    for model_name, evaluator in evaluators.items():
        t0 = time.time()
        try:
            df_res = evaluator.run(
                initial_window=INITIAL_WINDOW,
                stride=STRIDE,
                horizon=HORIZON,
                blocks=BLOCKS
            )
            elapsed = time.time() - t0
            results[model_name] = df_res
            ok = True
            detail = f"{len(df_res)} filas en {elapsed:.1f}s"
        except Exception as e:
            ok = False
            detail = str(e)
        
        total_checks += 1
        if run_check(f"Ejecución {model_name}", ok, detail):
            passed_checks += 1
            if ok:
                print(f"           ↳ {detail}")
    
    if not results:
        print("\n❌ FALLO CRÍTICO: Ningún evaluador produjo resultados. Abortando.")
        sys.exit(1)
    
    # =========================================================================
    # 3. VALIDAR ESQUEMA DE COLUMNAS
    # =========================================================================
    print("\n📐 Fase 3: Validación de Esquema de Columnas")
    
    expected_block_cols = []
    for b in range(1, BLOCKS + 1):
        expected_block_cols.extend([f'MAPE_B{b}', f'Naive_MAPE_B{b}', f'RMSE_B{b}', f'Hit_B{b}', f'CumHit_B{b}', f'Profit_B{b}'])
    
    required_meta_cols = ['Iteracion (Velas Vistas)', 'Drift (k)', 'SINDy R2', 'Validez']
    
    for model_name, df_res in results.items():
        # Check columnas de métricas por bloque
        has_all_blocks = all(col in df_res.columns for col in expected_block_cols)
        total_checks += 1
        if run_check(f"{model_name}: Tiene 360 columnas de bloques (60×6)", has_all_blocks):
            passed_checks += 1
        
        # Check columnas de metadatos mínimos
        has_meta = all(col in df_res.columns for col in required_meta_cols)
        total_checks += 1
        if run_check(f"{model_name}: Tiene columnas de metadatos requeridas", has_meta):
            passed_checks += 1
    
    # =========================================================================
    # 4. VALIDAR VALORES (No NaN/Inf en filas válidas)
    # =========================================================================
    print("\n🔍 Fase 4: Integridad de Valores")
    
    for model_name, df_res in results.items():
        if 'Validez' not in df_res.columns:
            continue
        
        valid_rows = df_res[df_res['Validez'] == 'OK']
        if len(valid_rows) == 0:
            total_checks += 1
            run_check(f"{model_name}: Al menos 1 fila con Validez='OK'", False, "0 filas válidas")
            continue
        
        # Check: no NaN en MAPE de filas válidas
        mape_cols = [f'MAPE_B{b}' for b in range(1, BLOCKS + 1)]
        existing_mape_cols = [c for c in mape_cols if c in valid_rows.columns]
        has_nan = valid_rows[existing_mape_cols].isna().any().any()
        total_checks += 1
        if run_check(f"{model_name}: Sin NaN en MAPE (filas válidas)", not has_nan):
            passed_checks += 1
        
        # Check: no Inf en MAPE de filas válidas  
        has_inf = np.isinf(valid_rows[existing_mape_cols].values.astype(float)).any()
        total_checks += 1
        if run_check(f"{model_name}: Sin Inf en MAPE (filas válidas)", not has_inf):
            passed_checks += 1
        
        # Check: Hit y CumHit son binarios {0.0, 1.0}
        hit_cols = [f'Hit_B{b}' for b in range(1, BLOCKS + 1)]
        cumhit_cols = [f'CumHit_B{b}' for b in range(1, BLOCKS + 1)]
        existing_hit = [c for c in hit_cols + cumhit_cols if c in valid_rows.columns]
        hit_vals = valid_rows[existing_hit].values.flatten()
        hit_vals_clean = hit_vals[~np.isnan(hit_vals)]
        all_binary = set(np.unique(hit_vals_clean)).issubset({0.0, 1.0})
        total_checks += 1
        if run_check(f"{model_name}: Hit/CumHit son binarios (0.0 o 1.0)", all_binary):
            passed_checks += 1
    
    # =========================================================================
    # 5. VALIDACIÓN CRUZADA: Naive MAPE
    # =========================================================================
    print("\n🔄 Fase 5: Validación Cruzada (Naive MAPE)")
    
    if 'Naive' in results:
        df_naive = results['Naive']
        valid_naive = df_naive[df_naive['Validez'] == 'OK']
        
        if len(valid_naive) > 0:
            # Check: MAPE_B{b} == Naive_MAPE_B{b} para el evaluador Naive
            all_equal = True
            for b in range(1, BLOCKS + 1):
                mape_col = f'MAPE_B{b}'
                naive_col = f'Naive_MAPE_B{b}'
                if mape_col in valid_naive.columns and naive_col in valid_naive.columns:
                    if not np.allclose(valid_naive[mape_col].values, valid_naive[naive_col].values, atol=1e-6):
                        all_equal = False
                        break
            
            total_checks += 1
            if run_check("Naive: MAPE_B{b} == Naive_MAPE_B{b} (identidad)", all_equal):
                passed_checks += 1
    
    # Cross-check Naive_MAPE entre evaluadores
    naive_mape_by_model = {}
    for model_name, df_res in results.items():
        valid = df_res[df_res['Validez'] == 'OK'] if 'Validez' in df_res.columns else df_res
        if len(valid) > 0 and 'Naive_MAPE_B1' in valid.columns:
            iters = valid['Iteracion (Velas Vistas)'].values
            naive_b1 = valid['Naive_MAPE_B1'].values
            naive_mape_by_model[model_name] = dict(zip(iters, naive_b1))
    
    if len(naive_mape_by_model) >= 2:
        model_names = list(naive_mape_by_model.keys())
        ref_model = model_names[0]
        ref_data = naive_mape_by_model[ref_model]
        
        for other_model in model_names[1:]:
            other_data = naive_mape_by_model[other_model]
            common_iters = set(ref_data.keys()) & set(other_data.keys())
            
            if common_iters:
                matches = all(
                    abs(ref_data[it] - other_data[it]) < 0.5
                    for it in common_iters
                )
                total_checks += 1
                if run_check(
                    f"Naive_MAPE_B1 consistente: {ref_model} vs {other_model} ({len(common_iters)} iteraciones comunes)",
                    matches
                ):
                    passed_checks += 1
    
    # =========================================================================
    # 6. CONTEO DE FILAS CONSISTENTE
    # =========================================================================
    print("\n📊 Fase 6: Consistencia de Conteo de Filas")
    
    # Los evaluadores con la misma interfaz deberían producir el mismo número de iteraciones
    row_counts = {name: len(df) for name, df in results.items()}
    if row_counts:
        # Naive y LSTM deberían tener el mismo conteo (ambos no fallan internamente de formas distintas)
        if 'Naive' in row_counts and 'LSTM' in row_counts:
            total_checks += 1
            if run_check(
                f"Conteo Naive ({row_counts['Naive']}) == LSTM ({row_counts['LSTM']})",
                row_counts['Naive'] == row_counts['LSTM']
            ):
                passed_checks += 1
        
        print(f"  ℹ️  Conteos: {row_counts}")
    
    # =========================================================================
    # RESUMEN FINAL
    # =========================================================================
    print("\n" + "=" * 70)
    rate = (passed_checks / total_checks * 100) if total_checks > 0 else 0
    status = "✅ ALL PASSED" if passed_checks == total_checks else "⚠️ SOME FAILED"
    print(f"  {status}: {passed_checks}/{total_checks} checks ({rate:.0f}%)")
    print("=" * 70)
    
    sys.exit(0 if passed_checks == total_checks else 1)


if __name__ == '__main__':
    main()
