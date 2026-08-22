# Comparative Benchmark Environment (CBE) — Test Infrastructure (`TEST_INFRA.md`)

## 1. Overview & Test Architecture

The Comparative Benchmark Environment (CBE) End-to-End test suite (`tests/test_cbe_e2e.py`) validates time-series forecasting evaluators across four model families:
1. **KineTopus Engine (SINDy)**: Physics-informed discovery engine (`WalkForwardEvaluator` in `src/quant_engine/evaluator.py`).
2. **AutoARIMA**: Statistical benchmark (`AutoARIMAWalkForwardEvaluator` in `src/quant_engine/arima_evaluator.py`).
3. **Naive Persistence**: Zero-intelligence baseline predicting $P(t+k)=P(t)$ (`NaiveWalkForwardEvaluator` in `src/quant_engine/naive_evaluator.py`).
4. **LSTM Baseline**: Deep learning sequence baseline (`LSTMWalkForwardEvaluator` in `src/quant_engine/lstm_evaluator.py`).

### Test Principles
- **Opaque-Box Requirement-Driven Testing**: Tests validate interface contracts, metrics, and application workflows derived directly from `ORIGINAL_REQUEST.md` and `PROJECT.md` without relying on private implementation details.
- **Progressive Testability**: Evaluator loading is dynamic and non-blocking. Evaluators not yet implemented on disk (such as M2 LSTM) trigger clean pytest skips rather than breaking test execution.
- **Hardware & Precision Bounds**: All synthetic fixtures enforce `numpy.float64` precision, fixed random seeds (`np.random.seed(42)`), and execute within 16GB RAM limits on CPU.

---

## 2. 4-Tier Test Taxonomy

### Tier 1: Feature Coverage (Unit & Functional Interface Checks)
Verifies duck-typed initialization `__init__(df, disable_norm=False, disable_returns=False)` and method execution `.run(initial_window, stride, horizon, blocks)` for all four evaluators, as well as flexibility under non-default window parameters.

### Tier 2: Boundary & Corner Cases
Verifies engine resilience under adverse input conditions:
- **Missing Required Columns**: Missing `'Close'` column raises explicit `KeyError` or returns clean mathematical failure status.
- **Short Time-Series**: Series shorter than `initial_window + horizon` return clean 0-row DataFrames with preserved column schemas.
- **Zero & Negative Prices**: Anomalous zero/negative price values avoid `ZeroDivisionError` and `Inf`/`NaN` in metrics via epsilon clamping or fallback.
- **NaN & Inf Input Values**: Corrupted price inputs are sanitized or handled gracefully.
- **Mathematical Failure Fallback**: Degenerate signals record `Validez='FALLO MATEMÁTICO'` without uncaught exceptions breaking execution loops.

### Tier 3: Cross-Feature Combinations & Schema Constraints
Verifies 247-column unified result schema integrity across all evaluators:
- **Column Count & Ordering**: Exactly 247 columns (7 metadata columns + 240 block metrics for $b \in [1, 60]$).
- **Block Metric Naming**: Standardized column naming: `MAPE_B{b}`, `Naive_MAPE_B{b}`, `Hit_B{b}`, `CumHit_B{b}` for $b = 1 \dots 60$.
- **Naive Cross-Validation Invariant**: Assert `MAPE_B{b} == Naive_MAPE_B{b}` for `NaiveWalkForwardEvaluator`. Assert `Naive_MAPE_B{b}` equivalence across all evaluators on identical price slices.
- **Metric Boundaries**: Verify directional hit ratios (`Hit_B{b}`, `CumHit_B{b}`) remain strictly within $\{0.0, 1.0\}$.

### Tier 4: Real-World Application Scenarios & System Integration
Verifies multi-iteration walk-forward evaluation workflows and system integration:
- **Walk-Forward Step Execution**: Validates iteration count matches exact step formula: $\text{iterations} = \lfloor \frac{\text{len}(df) - \text{horizon} - \text{initial\_window}}{\text{stride}} \rfloor + 1$.
- **Checkpoint CSV Resumption & Deduplication**: Simulates `7_Comparative_Benchmark.ipynb` checkpoint persistence in `unified_benchmark_db.csv`, verifying loading, deduplication of `(Ticker, Model)` pairs, appending new models, and roundtrip read/write accuracy.
- **Sanity Script Verification**: Integrates standalone validation runner `notebooks_val/test_benchmark_sanity.py` when available on disk.

---

## 3. Feature Checklist Matrix

| Requirement ID | Feature Description | Test Function in `tests/test_cbe_e2e.py` | Tier | Target Component | Status |
|----------------|---------------------|------------------------------------------|------|------------------|--------|
| R1 / R2 / Contract | Duck-typed interface check (Naive) | `test_tier1_evaluator_duck_typed_interface_naive` | Tier 1 | Naive Evaluator | PASS |
| R1 / R2 / Contract | Duck-typed interface check (ARIMA) | `test_tier1_evaluator_duck_typed_interface_arima` | Tier 1 | AutoARIMA Evaluator | PASS |
| Contract | Duck-typed interface check (SINDy) | `test_tier1_evaluator_duck_typed_interface_sindy` | Tier 1 | KineTopus SINDy Evaluator | PASS |
| R2 | Duck-typed interface check (LSTM) | `test_tier1_evaluator_duck_typed_interface_lstm` | Tier 1 | LSTM Evaluator | SKIP / READY |
| Contract | Window Parameter Flexibility | `test_tier1_evaluator_parameter_flexibility` | Tier 1 | Evaluators | PASS |
| Corner Case | Missing 'Close' Column Handling | `test_tier2_missing_required_columns` | Tier 2 | Evaluator Input Sanitization | PASS |
| Corner Case | Short Series Boundary | `test_tier2_short_time_series` | Tier 2 | Walk-Forward Loop Bounds | PASS |
| Corner Case | Zero & Negative Price Safety | `test_tier2_zero_and_negative_prices` | Tier 2 | Numerical Stability | PASS |
| Corner Case | NaN & Inf Input Resilience | `test_tier2_nan_and_inf_inputs` | Tier 2 | Input Sanitization | PASS |
| R2 Fallback | Mathematical Failure Fallback | `test_tier2_fallback_behavior_on_mathematical_failure` | Tier 2 | Fallback (`FALLO MATEMÁTICO`) | PASS |
| Schema Contract | 247-Column Unified Schema | `test_tier3_metric_schema_247_columns` | Tier 3 | Benchmark Output Schema | PASS |
| Schema Contract | Column Naming & Ordering | `test_tier3_column_naming_and_ordering` | Tier 3 | Metric Column Names | PASS |
| R1 Invariant | Naive Cross-Validation Invariant | `test_tier3_naive_cross_validation_equivalence` | Tier 3 | Naive Metric Identity | PASS |
| Schema Contract | Hit Ratio Range Bounds | `test_tier3_hit_ratio_boundary_values` | Tier 3 | Directional Metric Bounds | PASS |
| R3 / E2E Workflow | Full Walk-Forward Execution | `test_tier4_full_walk_forward_evaluation_workflow` | Tier 4 | Walk-Forward Engine | PASS |
| R3 Checkpoint | CSV Resume & Deduplication | `test_tier4_checkpoint_csv_integrity_and_resume` | Tier 4 | Checkpoint Database (`unified_benchmark_db.csv`) | PASS |
| R4 Integration | Automated Sanity Test Integration | `test_tier4_sanity_script_integration` | Tier 4 | Sanity Test Script (`test_benchmark_sanity.py`) | SKIP / READY |

---

## 4. Test Execution & Verification

### Standard Test Command
```bash
python -m pytest tests/test_cbe_e2e.py -v
```

### Coverage Verification Command
```bash
python -m pytest tests/test_cbe_e2e.py --tb=short
```
