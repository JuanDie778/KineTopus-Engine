import sys
import pandas as pd
from src.ui.market_loader import MarketLoader
from src.quant_engine.blender import ContinuousBlender
from src.quant_engine.nervous import RegimeShiftDetector
from src.quant_engine.physics import PhysicsDiscoverer
import numpy as np

def test_gld():
    df = MarketLoader.load_ticker_data('GLD', period='1y', interval='1d')
    log_returns, volumen_z, p_raw, dt_val = MarketLoader.prepare_quant_input(df, disable_norm=False, disable_returns=False)
    t = np.arange(len(p_raw), dtype=np.float64)
    
    b = ContinuousBlender(tolerance=0.005) # Loose fitting
    b.fit(t, log_returns, [len(df)//4], 0)
    b.fit(t, volumen_z, [len(df)//4], 1)
    
    r_smooth, r_dot, _ = b.compute_continuous(0, t)
    v_smooth, v_dot, _ = b.compute_continuous(1, t)
    
    sensor = RegimeShiftDetector(threshold=2.0, drift=0.8)
    rep = sensor.detect(log_returns, r_smooth)
    shift_idx = rep['shift_indices']
    
    print(f'Quiebres CUSUM H=2.0, k=0.8: {len(shift_idx)}')
    print(f'Indices: {shift_idx}')
        
    boundaries = [0] + shift_idx + [len(t)]
    disc = PhysicsDiscoverer(poly_degree=1) 
    x_matrix = np.column_stack((r_smooth, v_smooth))
    x_dot_matrix = np.column_stack((r_dot, v_dot))
    
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i+1]
        if end - start < 15: continue
            
        p_rep = disc.extract_equations(
            t=t[start:end], x=x_matrix[start:end], x_dot=x_dot_matrix[start:end], 
            dt=dt_val, horizon_steps=1, sigma_res_r=0, sigma_res_v=0, 
            last_price=p_raw[end-1], disable_norm=False, disable_returns=False
        )
        print(f'\n[Régimen {i}] Velas {start} a {end}')
        print(f'  R2 Score: {p_rep["score"]:.4f}')
        print(f'  dr/dt = {p_rep["equations"][0]}')

if __name__ == '__main__':
    test_gld()
