# Original User Request

## 2026-08-11T13:44:27Z

Build a Comparative Benchmark Environment (CBE) for the **KineTopus Engine** — a Scientific Machine Learning (SciML) quant motor that discovers differential equations from financial time series. The CBE will compare KineTopus against 3 baselines (Auto-ARIMA, LSTM, Naive Persistence) using identical Walk-Forward evaluation protocols, producing a unified CSV database of per-block metrics for academic comparison.

Working directory: c:\Users\ussaa\Documents\KineTopus Engine
Integrity mode: development

## Codebase Context (CRITICAL — Read Before Implementing)

The project already has a working Walk-Forward evaluation infrastructure. **You MUST study these files before writing any code:**

### Existing Evaluators (Your Templates)
- `src/quant_engine/evaluator.py` — **KineTopus Walk-Forward evaluator** (the gold standard implementation). Contains `WalkForwardEvaluator` class with `.run(initial_window, stride, horizon, blocks)` interface and per-block metric computation (MAPE, Naive_MAPE, Hit, CumHit for each of 60 blocks).
- `src/quant_engine/arima_evaluator.py` — **Auto-ARIMA Walk-Forward evaluator**. Contains `AutoARIMAWalkForwardEvaluator` with identical `.run()` interface and identical metric schema. Uses `pmdarima`.

### Engine Components (DO NOT MODIFY)
- `src/quant_engine/sensor.py` — `SpectralAnalyzer` (FFT layer)
- `src/quant_engine/blender.py` — `ContinuousBlender` (Spline layer)
- `src/quant_engine/nervous.py` — `RegimeShiftDetector` (CUSUM layer)
- `src/quant_engine/physics.py` — `PhysicsDiscoverer` (SINDy + Euler-Maruyama)
- `src/quant_engine/auto_tuner.py` — `CUSUMAutoTuner`
- `src/quant_engine/auto_tuner_predictive.py` — `PredictiveAutoTuner`
- `src/ui/market_loader.py` — `MarketLoader` with `load_ticker_data()` and `prepare_quant_input()`

### Existing Data (Reusable)
- `notebooks_val/backtest_kinetopus.csv` — Existing KineTopus Walk-Forward results (~1234 rows, tickers: MSFT, XLF, C, BTC-USD, SPY, AAPL, ETH-USD, JPM, BAC, MA, AMZN)
- `notebooks_val/backtest_arima_.csv` — Existing ARIMA Walk-Forward results
- These were generated with parameters: Ventana=150, Salto=20, Horizonte=300, Bloques=60, Contexto=1825

### Walk-Forward Protocol (ALL evaluators MUST follow this exactly)
```
SHARED_CONFIG = {
    'VENTANA_INICIAL': 150,      # Initial training window
    'SALTO': 20,                 # Stride between iterations
    'HORIZONTE': 300,            # Future candles to predict
    'BLOQUES': 60,               # Metric granularity (each block = 5 candles)
    'CONTEXT_WINDOW': 1825,      # Max lookback (~5 years daily)
    'PERIODO': '10y',            # Downloaded history
    'INTERVALO': '1d',           # Candle timeframe
}
```

### Per-Block Metric Schema (ALL evaluators MUST produce these columns)
For each block b in {1, ..., 60}:
- `MAPE_B{b}`: mean(|actual - pred| / actual) * 100
- `Naive_MAPE_B{b}`: mean(|actual - P_last| / actual) * 100
- `Hit_B{b}`: 1 if sign(pred[-1]-pred[0]) == sign(act[-1]-act[0]) else 0
- `CumHit_B{b}`: 1 if sign(pred[-1]-P_last) == sign(act[-1]-P_last) else 0

Plus metadata columns: `Model`, `Ticker`, `Iteracion (Velas Vistas)`, `Drift (k)`, `SINDy R2`, `Validez`

### Hardware Constraints
- 16GB RAM, CPU-only (no GPU)
- All NumPy operations MUST use float64
- LSTM must be lightweight: 1 layer, 32-64 units, <=50 epochs per iteration with early stopping

## Requirements

### R1. Naive Persistence Evaluator
Create `src/quant_engine/naive_evaluator.py` with a `NaiveWalkForwardEvaluator` class that:
- Implements the same `.run(initial_window, stride, horizon, blocks)` interface as the existing evaluators
- Predicts P(t+k) = P(t) for all k (flat line from last known price)
- Produces the same per-block metric columns (MAPE_B{b}, Naive_MAPE_B{b}, Hit_B{b}, CumHit_B{b}) for b=1..60
- Records `Validez` = 'OK' always (naive never fails mathematically)
- Records `Latencia_ms` for each iteration

### R2. LSTM Baseline Evaluator
Create `src/quant_engine/lstm_evaluator.py` with a `LSTMWalkForwardEvaluator` class that:
- Implements the same `.run()` interface as the existing evaluators
- Uses a lightweight LSTM model (PyTorch or Keras) with 1 recurrent layer, 32-64 hidden units
- At each Walk-Forward iteration, trains on the available past data (expanding or context-windowed) and predicts `horizon` future price steps
- Uses recursive multi-step prediction (predict 1 step -> feed back -> predict next)
- Produces the same per-block metric schema as all other evaluators
- Includes proper normalization (Min-Max or Z-score) fitted ONLY on training data (no look-ahead bias)
- Uses early stopping (patience=5, max_epochs=50) to prevent excessive training time
- Handles failures gracefully: if training diverges or produces NaN, mark `Validez` = 'FALLO MATEMATICO'
- Records `Latencia_ms` for each iteration

### R3. Unified Benchmark Notebook with Checkpoint/Resume
Create `notebooks_val/7_Comparative_Benchmark.ipynb` that:
- Orchestrates running all 4 evaluators (KineTopus, ARIMA, LSTM, Naive) across a configurable list of tickers
- Default tickers: `['BTC-USD', 'MSFT', 'XLF']`
- For each (ticker, model) combination, checks if results already exist in the output CSV before re-running (checkpoint/resume capability, same pattern as `1_Walk_Forward_Validation.ipynb`)
- Saves all results to `notebooks_val/unified_benchmark_db.csv` with a `Model` column identifying the source
- Can optionally reuse data from existing CSVs (`backtest_kinetopus.csv`, `backtest_arima_.csv`) by loading, filtering for target tickers, adding a `Model` column, and merging — to avoid redundant computation
- Shows progress and prints clear status messages per model/ticker

### R4. Automated Validation Test
Create `notebooks_val/test_benchmark_sanity.py` (a standalone Python script, NOT a notebook) that:
- Runs a mini Walk-Forward test with reduced parameters (initial_window=150, stride=50, horizon=300, blocks=60, max ~5 iterations) on a single ticker (MSFT)
- Executes all 4 evaluators on the same data
- Validates that:
  - Each evaluator returns a DataFrame with the correct column schema
  - No NaN values in metric columns for rows with Validez='OK'
  - The Naive evaluator's MAPE values are identical to the Naive_MAPE values from other evaluators (cross-validation check)
  - All evaluators produce the same number of rows for the same data
  - LSTM predictions are finite (no Inf or NaN in predicted prices)
- Prints PASS/FAIL for each check
- Can be run with: `python -m notebooks_val.test_benchmark_sanity` from the project root

## Acceptance Criteria

### Evaluator Correctness
- [ ] `NaiveWalkForwardEvaluator.run()` executes without errors and produces a DataFrame with columns matching the schema: `Iteracion (Velas Vistas)`, `Drift (k)`, `SINDy R2`, `Validez`, `MAPE_B1`..`MAPE_B60`, `Naive_MAPE_B1`..`Naive_MAPE_B60`, `Hit_B1`..`Hit_B60`, `CumHit_B1`..`CumHit_B60`
- [ ] `LSTMWalkForwardEvaluator.run()` executes without errors on CPU with <=16GB RAM usage
- [ ] LSTM produces valid price predictions (no NaN/Inf) for at least 80% of Walk-Forward iterations
- [ ] Both new evaluators follow the same expanding-window protocol as the existing evaluators (no look-ahead bias)

### Benchmark Notebook
- [ ] `7_Comparative_Benchmark.ipynb` can be run start-to-finish without errors on the 3 target tickers
- [ ] Checkpoint/resume works: re-running the notebook skips already-computed (ticker, model) pairs
- [ ] The output `unified_benchmark_db.csv` contains data from all 4 models and all 3 tickers
- [ ] Each row in the unified CSV includes a `Model` column with one of: `Kinetopus`, `ARIMA`, `LSTM`, `Naive`

### Validation
- [ ] `test_benchmark_sanity.py` runs all checks and prints PASS for all of them
- [ ] The sanity test completes in under 10 minutes on a 16GB RAM CPU-only machine
