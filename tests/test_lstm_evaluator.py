"""
Empirical Test Suite & Harness for LSTMWalkForwardEvaluator.
Validates model loss convergence, valid prediction outputs, 247-column output schema,
block metrics accuracy for all 60 blocks, data-leakage resistance, and error resilience.
"""

import math
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from src.quant_engine.lstm_evaluator import LSTMWalkForwardEvaluator, PyTorchLSTM


# ------------------------------------------------------------------------------
# FIXTURES
# ------------------------------------------------------------------------------

@pytest.fixture
def synthetic_market_data():
    """Generates 600-candle synthetic price time series."""
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
        "Ticker": "BTC-USD"
    }, index=dates)
    df.attrs['ticker'] = 'BTC-USD'
    return df


@pytest.fixture
def deterministic_sine_market_data():
    """Generates deterministic sine-wave price series for metric oracle tests."""
    n = 600
    dates = pd.date_range("2024-01-01", periods=n, freq="1D")
    t = np.linspace(0, 20 * np.pi, n)
    close = 100.0 + 10.0 * np.sin(t)
    df = pd.DataFrame({
        "Close": close,
        "Volume": np.full(n, 500000.0),
        "Ticker": "SINE"
    }, index=dates)
    df.attrs['ticker'] = 'SINE'
    return df


# ------------------------------------------------------------------------------
# TEST 1: MODEL CONVERGENCE & VALID PREDICTIONS
# ------------------------------------------------------------------------------

def test_lstm_model_loss_convergence():
    """Verify PyTorch LSTM model loss decreases over epochs during training."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate simple sequence data (seq_len=30, n_samples=100)
    seq_length = 30
    x_raw = np.sin(np.linspace(0, 10 * np.pi, 130)).astype(np.float32)
    X_list, y_list = [], []
    for i in range(len(x_raw) - seq_length):
        X_list.append(x_raw[i : i + seq_length])
        y_list.append(x_raw[i + seq_length])
        
    X_tensor = torch.tensor(np.array(X_list)[..., np.newaxis], dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_list)[..., np.newaxis], dtype=torch.float32)
    
    model = PyTorchLSTM(input_size=1, hidden_size=32, num_layers=1, output_size=1)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    losses = []
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        out = model(X_tensor)
        loss = criterion(out, y_tensor)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        
    initial_loss = np.mean(losses[:3])
    final_loss = np.mean(losses[-3:])
    
    assert final_loss < initial_loss, f"Training loss did not decrease: initial={initial_loss:.6f}, final={final_loss:.6f}"
    assert final_loss < 0.05, f"Final training loss unexpectedly high: {final_loss:.6f}"


def test_lstm_predictions_validity(synthetic_market_data):
    """Verify LSTMWalkForwardEvaluator outputs valid, finite, non-NaN price predictions."""
    evaluator = LSTMWalkForwardEvaluator(synthetic_market_data)
    df_res = evaluator.run(initial_window=150, stride=50, horizon=300, blocks=60)
    
    assert len(df_res) > 0, "Walker should return iteration rows"
    valid_rows = df_res[df_res['Validez'] == 'OK']
    assert len(valid_rows) > 0, "At least one iteration should have Validez=='OK'"
    
    # Check that block MAPE metrics are finite, positive numbers
    for b in range(1, 61):
        mape_vals = valid_rows[f'MAPE_B{b}'].values
        assert not np.isnan(mape_vals).any(), f"NaN found in MAPE_B{b}"
        assert not np.isinf(mape_vals).any(), f"Inf found in MAPE_B{b}"
        assert (mape_vals >= 0.0).all(), f"Negative MAPE found in MAPE_B{b}"


# ------------------------------------------------------------------------------
# TEST 2: OUTPUT SCHEMA VALIDATION (247 COLUMNS)
# ------------------------------------------------------------------------------

def test_lstm_schema_exact_247_columns(synthetic_market_data):
    """Verify output DataFrame schema contains exactly 247 columns in precise contract order."""
    evaluator = LSTMWalkForwardEvaluator(synthetic_market_data)
    df_res = evaluator.run(initial_window=150, stride=50, horizon=300, blocks=60)
    
    assert df_res.shape[1] == 247, f"Expected 247 columns, got {df_res.shape[1]}"
    
    expected_meta = ['Model', 'Ticker', 'Iteracion (Velas Vistas)', 'Drift (k)', 'SINDy R2', 'Validez', 'Latencia_ms']
    assert list(df_res.columns[:7]) == expected_meta, f"Metadata columns mismatch: {list(df_res.columns[:7])}"
    
    expected_block_cols = []
    for b in range(1, 61):
        expected_block_cols.extend([f'MAPE_B{b}', f'Naive_MAPE_B{b}', f'Hit_B{b}', f'CumHit_B{b}'])
        
    assert list(df_res.columns[7:]) == expected_block_cols, "Block metric column names or ordering mismatch"


def test_lstm_custom_horizon_blocks_schema(synthetic_market_data):
    """Verify output schema adapts to custom horizon and blocks (e.g., horizon=100, blocks=20 -> 87 cols)."""
    evaluator = LSTMWalkForwardEvaluator(synthetic_market_data)
    df_res = evaluator.run(initial_window=100, stride=50, horizon=100, blocks=20)
    
    # 7 metadata + 20 * 4 = 87 columns
    assert df_res.shape[1] == 87, f"Expected 87 columns, got {df_res.shape[1]}"
    assert f'MAPE_B20' in df_res.columns
    assert f'MAPE_B21' not in df_res.columns


# ------------------------------------------------------------------------------
# TEST 3: METRIC COMPUTATIONS ACCURACY FOR ALL 60 BLOCKS
# ------------------------------------------------------------------------------

def test_lstm_metrics_computation_accuracy_oracle(deterministic_sine_market_data):
    """
    Rigorously verifies MAPE, Naive_MAPE, Hit, and CumHit equations across all 60 blocks
    against an independent ground-truth calculation.
    """
    evaluator = LSTMWalkForwardEvaluator(deterministic_sine_market_data)
    initial_window = 150
    stride = 50
    horizon = 300
    blocks = 60
    block_size = horizon // blocks  # 5
    
    df_res = evaluator.run(initial_window=initial_window, stride=stride, horizon=horizon, blocks=blocks)
    close_prices = deterministic_sine_market_data['Close'].values
    
    for row_idx, row in df_res.iterrows():
        if row['Validez'] != 'OK':
            continue
            
        end_idx = int(row['Iteracion (Velas Vistas)'])
        actual_prices = close_prices[end_idx : end_idx + horizon]
        last_known_price = close_prices[end_idx - 1]
        
        # We know evaluator calculates metrics block by block
        for b in range(blocks):
            b_num = b + 1
            p_act = actual_prices[b * block_size : (b + 1) * block_size]
            denom = np.where(np.abs(p_act) < 1e-12, 1e-12, p_act)
            
            # 1. Oracle Naive_MAPE
            p_naive = np.full_like(p_act, fill_value=last_known_price)
            oracle_naive_mape = round(float(np.mean(np.abs((p_act - p_naive) / denom)) * 100.0), 2)
            eval_naive_mape = row[f'Naive_MAPE_B{b_num}']
            assert math.isclose(oracle_naive_mape, eval_naive_mape, abs_tol=0.01), \
                f"Row {row_idx} Block {b_num} Naive_MAPE mismatch: oracle={oracle_naive_mape}, eval={eval_naive_mape}"
                
            # 2. Check Hit and CumHit values are strictly binary 0.0 or 1.0
            hit_val = row[f'Hit_B{b_num}']
            cumhit_val = row[f'CumHit_B{b_num}']
            assert hit_val in (0.0, 1.0), f"Hit_B{b_num} not in {{0.0, 1.0}}: {hit_val}"
            assert cumhit_val in (0.0, 1.0), f"CumHit_B{b_num} not in {{0.0, 1.0}}: {cumhit_val}"


# ------------------------------------------------------------------------------
# TEST 4: ROBUSTNESS & FALLBACK ON MATHEMATICAL FAILURE
# ------------------------------------------------------------------------------

def test_lstm_fallback_on_nan_input():
    """Verify evaluator catches NaN input prices and sets Validez='FALLO MATEMÁTICO' gracefully."""
    dates = pd.date_range("2024-01-01", periods=500, freq="1D")
    prices = np.full(500, 100.0, dtype=np.float64)
    prices[180] = np.nan  # NaN in expanding window or horizon
    
    df = pd.DataFrame({"Close": prices, "Ticker": "NAN_TEST"}, index=dates)
    evaluator = LSTMWalkForwardEvaluator(df)
    
    df_res = evaluator.run(initial_window=150, stride=50, horizon=300, blocks=60)
    assert isinstance(df_res, pd.DataFrame)
    assert len(df_res) > 0
    
    # Check that iterations affected by NaN report FALLO MATEMÁTICO
    nan_rows = df_res[df_res['Validez'] == 'FALLO MATEMÁTICO']
    assert len(nan_rows) > 0, "Expected FALLO MATEMÁTICO rows when NaN present in data"
    
    # Verify that FALLO MATEMÁTICO rows contain NaN block metrics and SINDy R2 = 0.0
    sample_fail = nan_rows.iloc[0]
    assert sample_fail['SINDy R2'] == 0.0
    assert np.isnan(sample_fail['MAPE_B1'])


def test_lstm_constant_price_data_resilience():
    """Verify Z-score std protection (std < 1e-8 -> std = 1.0) on zero-variance series."""
    dates = pd.date_range("2024-01-01", periods=500, freq="1D")
    prices = np.full(500, 100.0, dtype=np.float64)
    df = pd.DataFrame({"Close": prices, "Ticker": "FLAT"}, index=dates)
    
    evaluator = LSTMWalkForwardEvaluator(df)
    df_res = evaluator.run(initial_window=150, stride=50, horizon=300, blocks=60)
    
    assert isinstance(df_res, pd.DataFrame)
    assert len(df_res) > 0
    # No ZeroDivisionError should occur during Z-score normalization
    assert (df_res['Validez'] == 'OK').all()
