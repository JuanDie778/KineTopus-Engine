import pandas as pd
from src.quant_engine.nervous import RegimeShiftDetector
from src.quant_engine.blender import ContinuousBlender
from src.quant_engine.physics import PhysicsDiscoverer
import numpy as np

df = pd.read_csv('5_resorte_roto.csv')
p = df['Close'].values.astype(np.float64)
v = df['Volume'].values if 'Volume' in df.columns else np.ones_like(p)
t = np.arange(len(p), dtype=np.float64)

b = ContinuousBlender(tolerance=0.0005)
b.fit(t, p, [100], 0)
b.fit(t, v, [100], 1)
r_smooth, r_dot, _ = b.compute_continuous(0, t)
v_smooth, v_dot, _ = b.compute_continuous(1, t)

d = RegimeShiftDetector(threshold=15.0, drift=1.5)
rep = d.detect(p, r_smooth)

boundaries = [0] + rep['shift_indices'] + [len(t)]
print('Regimes detected:', len(boundaries)-1)

disc = PhysicsDiscoverer(poly_degree=1)  # Setting exactly to what user used
disc2 = PhysicsDiscoverer(poly_degree=2)  

x_matrix = np.column_stack((r_smooth, v_smooth))
x_dot_matrix = np.column_stack((r_dot, v_dot))

for i in range(len(boundaries) - 1):
    start = boundaries[i]
    end = boundaries[i+1]
    if end - start < 15: continue
    
    rep_eq = disc.extract_equations(
        t=t[start:end], x=x_matrix[start:end], x_dot=x_dot_matrix[start:end], 
        dt=1.0, horizon_steps=1, sigma_res_r=0, sigma_res_v=0, 
        last_price=p[end-1], disable_norm=True, disable_returns=True
    )
    rep_eq2 = disc2.extract_equations(
        t=t[start:end], x=x_matrix[start:end], x_dot=x_dot_matrix[start:end], 
        dt=1.0, horizon_steps=1, sigma_res_r=0, sigma_res_v=0, 
        last_price=p[end-1], disable_norm=True, disable_returns=True
    )
    print('')
    print('--- Regime', i, 'Velas', start, end, '---')
    print('GRADO 1: R2 =', rep_eq['score'])
    print('dP/dt =', rep_eq['equations'][0])
    print('dV/dt =', rep_eq['equations'][1])
    print('GRADO 2: R2 =', rep_eq2['score'])
    print('dP/dt =', rep_eq2['equations'][0])
    print('dV/dt =', rep_eq2['equations'][1])
