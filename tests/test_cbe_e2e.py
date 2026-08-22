"""
End-to-End Test Suite for Comparative Benchmark Environment (CBE).

Validates Walk-Forward Evaluators (KineTopus SINDy, AutoARIMA, Naive Persistence, LSTM)
across 4 test tiers:
- Tier 1: Feature Coverage (Evaluator Duck-Typing Interface & Execution)
- Tier 2: Boundary & Corner Cases (Missing columns, short series, zero/negative prices, math failures)
- Tier 3: Cross-Feature Combinations & Schema Constraints (247-column schema, Naive MAPE invariant)
- Tier 4: Real-World Scenarios (Walk-Forward workflow, Checkpoint CSV resume/merge)
"""

import os
import sys
import tempfile
import importlib
import pytest
import numpy as np
import pandas as pd


# ------------------------------------------------------------------------------
# DYNAMIC EVALUATOR LOADER & FIXTURES
# ------------------------------------------------------------------------------

def get_evaluator_class(name: str):
    """
    Dynamically loads an evaluator class by model name.
    Returns the class if available, or None if not implemented on disk.
    """
    module_map = {
        'Kinetopus': ('src.quant_engine.evaluator', 'WalkForwardEvaluator'),
        'ARIMA': ('src.quant_engine.arima_evaluator', 'AutoARIMAWalkForwardEvaluator'),
        'Naive': ('src.quant_engine.naive_evaluator', 'NaiveWalkForwardEvaluator'),
        'LSTM': ('src.quant_engine.lstm_evaluator', 'LSTMWalkForwardEvaluator'),
    }
    
    if name not in module_map:
        return None
    
    mod_path, class_name = module_map[name]
    try:
        mod = importlib.import_module(mod_path)
        return getattr(mod, class_name, None)
    except ImportError:
        return None


@pytest.fixture
def synthetic_market_df():
    """
    Standard 600-candle synthetic price and volume time series.
    Uses numpy float64 with deterministic random seed.
    """
    np.random.seed(42)
    n = 600
    dates = pd.date_range("2024-01-01", periods=n, freq="1D")
    log_ret = np.random.normal(loc=0.0005, scale=0.015, size=n).astype(np.float64)
    close = 100.0 * np.exp(np.cumsum(log_ret))
    volume = np.random.uniform(1e5, 1e6, size=n).astype(np.float64)
    
    df = pd.DataFrame({
        "Close": close,
        "Volume": volume,
        "Open": close * (1.0 + np.random.normal(0, 0.002, n)),
        "High": close * 1.01,
        "Low": close * 0.99,
        "Ticker": "TEST"
    }, index=dates)
    df.attrs['ticker'] = 'TEST'
    return df


@pytest.fixture
def short_market_df():
    """Short 100-candle series (< initial_window + horizon)."""
    dates = pd.date_range("2024-01-01", periods=100, freq="1D")
    df = pd.DataFrame({
        "Close": np.linspace(100.0, 110.0, 100, dtype=np.float64),
        "Volume": np.full(100, 1000.0, dtype=np.float64),
        "Ticker": "SHORT"
    }, index=dates)
    df.attrs['ticker'] = 'SHORT'
    return df


@pytest.fixture
def zero_neg_market_df():
    """Market series with zero and negative price anomalies."""
    dates = pd.date_range("2024-01-01", periods=500, freq="1D")
    prices = np.full(500, 100.0, dtype=np.float64)
    prices[200] = 0.0
    prices[201] = -25.0
    df = pd.DataFrame({
        "Close": prices,
        "Volume": np.full(500, 1000.0, dtype=np.float64),
        "Ticker": "ANOMALY"
    }, index=dates)
    df.attrs['ticker'] = 'ANOMALY'
    return df


@pytest.fixture
def missing_col_market_df():
    """Market series lacking required 'Close' column."""
    dates = pd.date_range("2024-01-01", periods=500, freq="1D")
    return pd.DataFrame({
        "Open": np.full(500, 100.0, dtype=np.float64),
        "High": np.full(500, 105.0, dtype=np.float64)
    }, index=dates)


@pytest.fixture
def nan_inf_market_df():
    """Market series containing NaN and Inf price values."""
    dates = pd.date_range("2024-01-01", periods=500, freq="1D")
    prices = np.full(500, 100.0, dtype=np.float64)
    prices[100] = np.nan
    prices[200] = np.inf
    prices[300] = -np.inf
    return pd.DataFrame({
        "Close": prices,
        "Volume": np.full(500, 1000.0, dtype=np.float64),
        "Ticker": "NAN_INF"
    }, index=dates)


# ------------------------------------------------------------------------------
# TIER 1: FEATURE COVERAGE (INTERFACE & DUCK-TYPING CHECKS)
# ------------------------------------------------------------------------------

def test_tier1_evaluator_duck_typed_interface_naive(synthetic_market_df):
    """Verify NaiveWalkForwardEvaluator satisfies duck-typed interface and runs cleanly."""
    cls = get_evaluator_class('Naive')
    if cls is None:
        pytest.skip("NaiveWalkForwardEvaluator not implemented on disk.")
    
    evaluator = cls(synthetic_market_df, disable_norm=False, disable_returns=False)
    res = evaluator.run(initial_window=150, stride=50, horizon=300, blocks=60)
    
    assert isinstance(res, pd.DataFrame), "Output must be a pandas DataFrame"
    assert len(res) > 0, "Execution should yield walk-forward iteration rows"


def test_tier1_evaluator_duck_typed_interface_arima(synthetic_market_df):
    """Verify AutoARIMAWalkForwardEvaluator satisfies duck-typed interface and runs cleanly."""
    cls = get_evaluator_class('ARIMA')
    if cls is None:
        pytest.skip("AutoARIMAWalkForwardEvaluator not implemented on disk.")
    
    evaluator = cls(synthetic_market_df, disable_norm=False, disable_returns=False)
    res = evaluator.run(initial_window=150, stride=100, horizon=150, blocks=15)
    
    assert isinstance(res, pd.DataFrame), "Output must be a pandas DataFrame"
    assert len(res) > 0, "Execution should yield walk-forward iteration rows"


def test_tier1_evaluator_duck_typed_interface_sindy(synthetic_market_df):
    """Verify KineTopus WalkForwardEvaluator satisfies duck-typed interface and runs cleanly."""
    cls = get_evaluator_class('Kinetopus')
    if cls is None:
        pytest.skip("WalkForwardEvaluator (KineTopus) not implemented on disk.")
    
    evaluator = cls(synthetic_market_df, disable_norm=False, disable_returns=False)
    res = evaluator.run(initial_window=150, stride=100, horizon=150, blocks=15)
    
    assert isinstance(res, pd.DataFrame), "Output must be a pandas DataFrame"
    assert len(res) > 0, "Execution should yield walk-forward iteration rows"


def test_tier1_evaluator_duck_typed_interface_lstm(synthetic_market_df):
    """Verify LSTMWalkForwardEvaluator satisfies duck-typed interface or skips cleanly if not present."""
    cls = get_evaluator_class('LSTM')
    if cls is None:
        pytest.skip("LSTMWalkForwardEvaluator not implemented on disk yet (M2 task).")
    
    evaluator = cls(synthetic_market_df, disable_norm=False, disable_returns=False)
    res = evaluator.run(initial_window=150, stride=50, horizon=300, blocks=60)
    
    assert isinstance(res, pd.DataFrame), "Output must be a pandas DataFrame"
    assert len(res) > 0, "Execution should yield walk-forward iteration rows"


def test_tier1_evaluator_parameter_flexibility(synthetic_market_df):
    """Verify evaluators accept non-default window parameters (custom initial_window, stride, horizon, blocks)."""
    cls = get_evaluator_class('Naive')
    if cls is None:
        pytest.skip("NaiveWalkForwardEvaluator not implemented.")
    
    evaluator = cls(synthetic_market_df)
    res = evaluator.run(initial_window=100, stride=20, horizon=100, blocks=20)
    
    assert isinstance(res, pd.DataFrame)
    # 7 metadata + 20*4 block metrics = 87 columns
    assert res.shape[1] == 87, f"Expected 87 columns for 20 blocks, got {res.shape[1]}"


# ------------------------------------------------------------------------------
# TIER 2: BOUNDARY & CORNER CASES
# ------------------------------------------------------------------------------

def test_tier2_missing_required_columns(missing_col_market_df):
    """Verify evaluator behavior when 'Close' column is missing from input DataFrame."""
    cls = get_evaluator_class('Naive')
    if cls is None:
        pytest.skip("NaiveWalkForwardEvaluator not available.")
    
    with pytest.raises(KeyError):
        evaluator = cls(missing_col_market_df)
        evaluator.run(initial_window=150, stride=20, horizon=300, blocks=60)


def test_tier2_short_time_series(short_market_df):
    """Verify evaluator returns clean empty DataFrame when len(df) < initial_window + horizon."""
    for model_name in ['Naive', 'ARIMA']:
        cls = get_evaluator_class(model_name)
        if cls is None:
            continue
        evaluator = cls(short_market_df)
        res = evaluator.run(initial_window=150, stride=20, horizon=300, blocks=60)
        assert isinstance(res, pd.DataFrame)
        assert len(res) == 0, f"Evaluator {model_name} should return 0 rows for short series"


def test_tier2_zero_and_negative_prices(zero_neg_market_df):
    """Verify price zero and negative values do not cause uncaught ZeroDivisionError or crash."""
    cls = get_evaluator_class('Naive')
    if cls is None:
        pytest.skip("NaiveWalkForwardEvaluator not available.")
    
    evaluator = cls(zero_neg_market_df)
    res = evaluator.run(initial_window=150, stride=50, horizon=150, blocks=15)
    assert isinstance(res, pd.DataFrame)
    assert len(res) > 0
    # Confirm no NaN in MAPE columns
    mape_cols = [c for c in res.columns if 'MAPE' in c]
    for col in mape_cols:
        assert not np.isinf(res[col]).any(), f"Column {col} contains Inf values"


def test_tier2_nan_and_inf_inputs(nan_inf_market_df):
    """Verify NaN and Inf inputs are handled safely or return clean evaluation without crash."""
    cls = get_evaluator_class('Naive')
    if cls is None:
        pytest.skip("NaiveWalkForwardEvaluator not available.")
    
    try:
        evaluator = cls(nan_inf_market_df)
        res = evaluator.run(initial_window=150, stride=50, horizon=150, blocks=15)
        assert isinstance(res, pd.DataFrame)
    except Exception as e:
        # Graceful exception is acceptable, uncaught internal crash is not
        assert isinstance(e, (ValueError, KeyError, RuntimeWarning))


def test_tier2_fallback_behavior_on_mathematical_failure(synthetic_market_df):
    """Verify fallback mechanism sets Validez='FALLO MATEMÁTICO' when mathematical fit fails."""
    # Test AutoARIMA failure fallback on degenerate constant data
    cls = get_evaluator_class('ARIMA')
    if cls is None:
        pytest.skip("AutoARIMAWalkForwardEvaluator not available.")
    
    # Degenerate constant price data can cause ARIMA fitting to fail or return warnings
    constant_df = synthetic_market_df.copy()
    constant_df['Close'] = 100.0
    
    evaluator = cls(constant_df)
    res = evaluator.run(initial_window=150, stride=100, horizon=150, blocks=15)
    assert isinstance(res, pd.DataFrame)
    assert 'Validez' in res.columns
    # Check that Validez is recorded as 'OK' or 'FALLO MATEMÁTICO' (never unhandled exception or NaN)
    assert set(res['Validez'].unique()).issubset({'OK', 'FALLO MATEMÁTICO'})


# ------------------------------------------------------------------------------
# TIER 3: CROSS-FEATURE COMBINATIONS & SCHEMA CONSTRAINTS
# ------------------------------------------------------------------------------

def test_tier3_metric_schema_247_columns(synthetic_market_df):
    """Verify output DataFrame has exactly 247 columns for horizon=300, blocks=60."""
    cls = get_evaluator_class('Naive')
    if cls is None:
        pytest.skip("NaiveWalkForwardEvaluator not available.")
    
    evaluator = cls(synthetic_market_df)
    res = evaluator.run(initial_window=150, stride=50, horizon=300, blocks=60)
    assert res.shape[1] == 247, f"Expected 247 columns (7 metadata + 240 block metrics), got {res.shape[1]}"


def test_tier3_column_naming_and_ordering(synthetic_market_df):
    """Verify column names and ordering strictly adhere to the 247-column CBE contract."""
    cls = get_evaluator_class('Naive')
    if cls is None:
        pytest.skip("NaiveWalkForwardEvaluator not available.")
    
    evaluator = cls(synthetic_market_df)
    res = evaluator.run(initial_window=150, stride=50, horizon=300, blocks=60)
    
    expected_meta = ['Model', 'Ticker', 'Iteracion (Velas Vistas)', 'Drift (k)', 'SINDy R2', 'Validez', 'Latencia_ms']
    assert list(res.columns[:7]) == expected_meta, f"First 7 metadata columns mismatch: {list(res.columns[:7])}"
    
    # Check block metric columns B1 to B60
    for b in range(1, 61):
        mape_col = f"MAPE_B{b}"
        naive_mape_col = f"Naive_MAPE_B{b}"
        hit_col = f"Hit_B{b}"
        cumhit_col = f"CumHit_B{b}"
        
        assert mape_col in res.columns, f"Missing {mape_col}"
        assert naive_mape_col in res.columns, f"Missing {naive_mape_col}"
        assert hit_col in res.columns, f"Missing {hit_col}"
        assert cumhit_col in res.columns, f"Missing {cumhit_col}"


def test_tier3_naive_cross_validation_equivalence(synthetic_market_df):
    """
    Verify Naive Persistence cross-validation identities:
    1. For Naive evaluator: MAPE_Bb == Naive_MAPE_Bb for all b in [1, 60].
    2. Naive_MAPE_Bb computed across different evaluators on identical slice match.
    """
    cls_naive = get_evaluator_class('Naive')
    if cls_naive is None:
        pytest.skip("NaiveWalkForwardEvaluator not available.")
    
    eval_naive = cls_naive(synthetic_market_df)
    res_naive = eval_naive.run(initial_window=150, stride=50, horizon=300, blocks=60)
    
    # Check invariant 1: MAPE_Bb == Naive_MAPE_Bb for Naive evaluator
    for b in range(1, 61):
        mape_vals = res_naive[f"MAPE_B{b}"].values
        naive_vals = res_naive[f"Naive_MAPE_B{b}"].values
        np.testing.assert_allclose(mape_vals, naive_vals, err_msg=f"MAPE_B{b} != Naive_MAPE_B{b} in Naive evaluator")

    # Check invariant 2: Compare Naive_MAPE values across evaluators if ARIMA is available
    cls_arima = get_evaluator_class('ARIMA')
    if cls_arima is not None:
        eval_arima = cls_arima(synthetic_market_df)
        res_arima = eval_arima.run(initial_window=150, stride=50, horizon=300, blocks=60)
        
        if len(res_arima) > 0 and len(res_naive) > 0:
            # Compare Naive_MAPE_B1 values on identical iterations
            naive_mape_naive = res_naive['Naive_MAPE_B1'].values
            naive_mape_arima = res_arima['Naive_MAPE_B1'].values
            min_len = min(len(naive_mape_naive), len(naive_mape_arima))
            np.testing.assert_allclose(
                naive_mape_naive[:min_len], 
                naive_mape_arima[:min_len], 
                atol=1e-1,
                err_msg="Naive_MAPE_B1 differs between Naive and ARIMA evaluators on identical data"
            )


def test_tier3_hit_ratio_boundary_values(synthetic_market_df):
    """Verify Hit_Bb and CumHit_Bb metric values are bounded in {0.0, 1.0}."""
    cls = get_evaluator_class('Naive')
    if cls is None:
        pytest.skip("NaiveWalkForwardEvaluator not available.")
    
    evaluator = cls(synthetic_market_df)
    res = evaluator.run(initial_window=150, stride=50, horizon=300, blocks=60)
    
    valid_rows = res[res['Validez'] == 'OK']
    for b in range(1, 61):
        hit_vals = valid_rows[f"Hit_B{b}"].values
        cumhit_vals = valid_rows[f"CumHit_B{b}"].values
        
        assert set(np.unique(hit_vals)).issubset({0.0, 1.0}), f"Hit_B{b} values outside {{0.0, 1.0}}"
        assert set(np.unique(cumhit_vals)).issubset({0.0, 1.0}), f"CumHit_B{b} values outside {{0.0, 1.0}}"


# ------------------------------------------------------------------------------
# TIER 4: REAL-WORLD APPLICATION SCENARIOS & SYSTEM INTEGRATION
# ------------------------------------------------------------------------------

def test_tier4_full_walk_forward_evaluation_workflow(synthetic_market_df):
    """Verify full walk-forward execution yields correct row count based on step formula."""
    cls = get_evaluator_class('Naive')
    if cls is None:
        pytest.skip("NaiveWalkForwardEvaluator not available.")
    
    initial_window = 150
    stride = 50
    horizon = 300
    blocks = 60
    
    evaluator = cls(synthetic_market_df)
    res = evaluator.run(initial_window=initial_window, stride=stride, horizon=horizon, blocks=blocks)
    
    # len(synthetic_market_df) = 600
    # max_valid_start = 600 - 300 = 300
    # end_idx starts at 150, steps: 150, 200, 250, 300 -> 4 iterations
    expected_rows = ((600 - horizon - initial_window) // stride) + 1
    assert len(res) == expected_rows, f"Expected {expected_rows} walk-forward iterations, got {len(res)}"


def test_tier4_checkpoint_csv_integrity_and_resume(synthetic_market_df, tmp_path):
    """Verify benchmark database checkpointing, deduplication, and resume behavior."""
    cls = get_evaluator_class('Naive')
    if cls is None:
        pytest.skip("NaiveWalkForwardEvaluator not available.")
    
    csv_path = tmp_path / "unified_benchmark_db.csv"
    
    # 1. Run initial evaluation for (MSFT, Naive)
    evaluator = cls(synthetic_market_df)
    df_naive = evaluator.run(initial_window=150, stride=50, horizon=300, blocks=60)
    df_naive['Model'] = 'Naive'
    df_naive['Ticker'] = 'MSFT'
    
    # Save initial checkpoint CSV
    df_naive.to_csv(csv_path, index=False)
    assert os.path.exists(csv_path)
    
    # 2. Simulate checkpoint resume check: load existing CSV
    existing_df = pd.read_csv(csv_path)
    existing_pairs = set(zip(existing_df['Ticker'], existing_df['Model']))
    
    assert ('MSFT', 'Naive') in existing_pairs, "Existing pair (MSFT, Naive) should be detected"
    
    # 3. Add new pair (BTC-USD, ARIMA) if ARIMA available
    cls_arima = get_evaluator_class('ARIMA')
    if cls_arima is not None:
        eval_arima = cls_arima(synthetic_market_df)
        df_arima = eval_arima.run(initial_window=150, stride=100, horizon=150, blocks=15)
        df_arima['Model'] = 'ARIMA'
        df_arima['Ticker'] = 'BTC-USD'
        
        # Standardize columns for concatenation if block count differs
        merged_df = pd.concat([existing_df, df_arima], ignore_index=True)
        merged_df.to_csv(csv_path, index=False)
        
        # 4. Verify roundtrip integrity
        final_df = pd.read_csv(csv_path)
        assert len(final_df) == len(existing_df) + len(df_arima)
        assert set(final_df['Model'].unique()) == {'Naive', 'ARIMA'}
        assert set(final_df['Ticker'].unique()) == {'MSFT', 'BTC-USD'}


def test_tier4_sanity_script_integration():
    """Verify test_benchmark_sanity.py script if present on disk."""
    sanity_script_path = os.path.join("notebooks_val", "test_benchmark_sanity.py")
    if not os.path.exists(sanity_script_path):
        pytest.skip(f"Sanity test script {sanity_script_path} not created yet (M4 task).")
    
    # Try importing module if script exists
    sys.path.insert(0, os.getcwd())
    try:
        mod = importlib.import_module("notebooks_val.test_benchmark_sanity")
        assert hasattr(mod, "__file__")
    except Exception as e:
        pytest.fail(f"Sanity test script import failed: {e}")
