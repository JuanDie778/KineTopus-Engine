import pandas as pd
from src.quant_engine.nervous import RegimeShiftDetector
from src.quant_engine.blender import ContinuousBlender
from src.quant_engine.physics import PhysicsDiscoverer
import numpy as np

def run_test(poly_degree, smooth_tol, cusum_h):
    df = pd.read_csv('6_traicion_volumen.csv')
    p = df['Close'].values.astype(np.float64)
    v = df['Volume'].values if 'Volume' in df.columns else np.ones_like(p)
    t = np.arange(len(p), dtype=np.float64)

    b = ContinuousBlender(tolerance=smooth_tol)
    b.fit(t, p, [100], 0)
    b.fit(t, v, [100], 1)
    r_smooth, r_dot, _ = b.compute_continuous(0, t)
    v_smooth, v_dot, _ = b.compute_continuous(1, t)

    # Note: the engine usually checks shift on `log_returns` vs `r_smooth`.
    # Here, disable_returns=True means p vs r_smooth directly.
    d = RegimeShiftDetector(threshold=cusum_h, drift=1.5)
    rep = d.detect(p, r_smooth)
    
    boundaries = [0] + rep['shift_indices'] + [len(t)]
    
    print(f"\n=============================================")
    print(f"PARAMS: poly_degree={poly_degree}, smooth_tol={smooth_tol}, cusum_H={cusum_h}")
    print(f"Regimes detected: {len(boundaries)-1} (Shifts at: {rep['shift_indices']})")

    disc = PhysicsDiscoverer(poly_degree=poly_degree)
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
        sc = rep_eq['score']
        print(f"\n--- Regime {i} (Velas {start}-{end}) ---")
        print(f"R2 = {sc:.4f}")
        print("dP/dt =", rep_eq['equations'][0])
        print("dV/dt =", rep_eq['equations'][1])

# Experimento 1: Parámetros Base
run_test(poly_degree=1, smooth_tol=0.0005, cusum_h=15.0)

# Experimento 2: Aumentando el ruido tolerado por Spline (suavizado más rígido) y grado 2 polinómico
run_test(poly_degree=2, smooth_tol=0.005, cusum_h=10.0)

# Experimento 3: CUSUM súper sensible
run_test(poly_degree=2, smooth_tol=0.0001, cusum_h=5.0)
