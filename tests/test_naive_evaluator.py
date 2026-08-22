import unittest
import pandas as pd
import numpy as np
from src.quant_engine.naive_evaluator import NaiveWalkForwardEvaluator


class TestNaiveWalkForwardEvaluator(unittest.TestCase):

    def test_naive_evaluator_basic(self):
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=1000)
        prices = 100.0 + np.cumsum(np.random.randn(1000))
        df = pd.DataFrame({'Close': prices, 'Ticker': 'TEST'}, index=dates)

        evaluator = NaiveWalkForwardEvaluator(df)
        res = evaluator.run(initial_window=150, stride=20, horizon=300, blocks=60)

        # 1. Column count schema (247 columns)
        self.assertEqual(res.shape[1], 247, f"Expected 247 columns, got {res.shape[1]}")

        # 2. Metadata column names & constant values
        expected_meta = ['Model', 'Ticker', 'Iteracion (Velas Vistas)', 'Drift (k)', 'SINDy R2', 'Validez', 'Latencia_ms']
        self.assertEqual(list(res.columns[:7]), expected_meta)
        self.assertTrue((res['Model'] == 'Naive').all())
        self.assertTrue((res['Ticker'] == 'TEST').all())
        self.assertTrue((res['Drift (k)'] == 0.0).all())
        self.assertTrue((res['SINDy R2'] == 0.0).all())
        self.assertTrue((res['Validez'] == 'OK').all())
        self.assertTrue((res['Latencia_ms'] >= 0.0).all())

        # 3. Iteration counts
        # total_len = 1000, max_valid_start = 1000 - 300 = 700.
        # end_idx starts at 150, stride = 20: 150, 170, ..., 700 -> 28 iterations.
        self.assertEqual(len(res), 28)

        # 4. Cross-validation identity: MAPE_B{b} == Naive_MAPE_B{b}
        for b in range(1, 61):
            mape_col = f'MAPE_B{b}'
            naive_col = f'Naive_MAPE_B{b}'
            self.assertIn(mape_col, res.columns)
            self.assertIn(naive_col, res.columns)
            np.testing.assert_array_equal(res[mape_col].values, res[naive_col].values)

        # 5. No NaN or Inf values
        self.assertFalse(res.isna().any().any())

    def test_naive_evaluator_attrs_ticker(self):
        np.random.seed(123)
        dates = pd.date_range('2021-01-01', periods=500)
        prices = 50.0 + np.cumsum(np.random.randn(500))
        df = pd.DataFrame({'Close': prices}, index=dates)
        df.attrs['ticker'] = 'MSFT'

        evaluator = NaiveWalkForwardEvaluator(df)
        res = evaluator.run(initial_window=150, stride=20, horizon=300, blocks=60)

        self.assertTrue((res['Ticker'] == 'MSFT').all())

    def test_naive_evaluator_empty_or_short_df(self):
        dates = pd.date_range('2022-01-01', periods=100)
        prices = 10.0 + np.arange(100)
        df = pd.DataFrame({'Close': prices, 'Ticker': 'SHORT'}, index=dates)

        evaluator = NaiveWalkForwardEvaluator(df)
        res = evaluator.run(initial_window=150, stride=20, horizon=300, blocks=60)

        self.assertEqual(res.shape[0], 0)
        self.assertEqual(res.shape[1], 247)

    def test_naive_evaluator_indivisible_blocks(self):
        df = pd.DataFrame({'Close': np.arange(500), 'Ticker': 'TEST'})
        evaluator = NaiveWalkForwardEvaluator(df)
        with self.assertRaises(ValueError):
            evaluator.run(initial_window=150, stride=20, horizon=300, blocks=7)


if __name__ == '__main__':
    unittest.main()
