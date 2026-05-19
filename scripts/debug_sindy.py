import pandas as pd
import numpy as np
import pysindy as ps
from src.ui.market_loader import MarketLoader
from src.quant_engine.blender import ContinuousBlender
from src.quant_engine.physics import PhysicsDiscoverer

df = pd.read_csv('1_bala_cinematica.csv')
p, v, raw, dt = MarketLoader.prepare_quant_input(df, disable_norm=True, disable_returns=True)
t = np.arange(len(p)) * dt

print("P range:", p.min(), p.max())
print("V range:", v.min(), v.max())

blender = ContinuousBlender(tolerance=0.0005)
blender.fit(t, p, np.array([]), 0)
blender.fit(t, v, np.array([]), 1)

p_s, p_d, _ = blender.compute_continuous(0, t)
v_s, v_d, _ = blender.compute_continuous(1, t)

print("P_dot range:", p_d.min(), p_d.max())
print("V_dot range:", v_d.min(), v_d.max())

disc = PhysicsDiscoverer(poly_degree=2)
x = np.column_stack((p_s, v_s))
x_dot = np.column_stack((p_d, v_d))
res = disc.extract_equations(t, x, x_dot, disable_norm=True, disable_returns=True)
print("SINDy output:", res['equations'])
print("R2:", res['score'])
