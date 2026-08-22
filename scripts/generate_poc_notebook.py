import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

text_intro = """# 🧪 PoC: Kinetopus Synthetic Market Generator (Ecuaciones de BTC)
**Concepto:** Extraer la física matemática (ODEs gobernantes) de un activo real altamente técnico como **BTC-USD** con SINDy, y luego utilizar el módulo estocástico de **Euler-Maruyama** de Kinetopus para proyectar un horizonte extremadamente largo (ej. 5.000 velas) y evaluar si las trayectorias sintéticas generadas son verosímiles a "ojo" a un mercado real."""

code_imports = """import sys
import os

# Resolución dinámica de la raíz del proyecto (evita ModuleNotFoundError en Jupyter)
current_dir = os.path.abspath(os.getcwd())
while current_dir != os.path.dirname(current_dir):
    if 'src' in os.listdir(current_dir):
        if current_dir not in sys.path:
            sys.path.append(current_dir)
        break
    current_dir = os.path.dirname(current_dir)

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import UnivariateSpline

# Kinetopus Core
from src.ui.market_loader import MarketLoader
from src.quant_engine.blender import ContinuousBlender
from src.quant_engine.physics import PhysicsDiscoverer"""

text_data = """## 1. Descarga de Datos de Referencia (BTC) y Suavizado Topológico
Descargaremos las últimas 300 velas diarias de BTC para extraer la inercia física de su micro-estructura. Luego utilizaremos el `ContinuousBlender` de Kinetopus para suavizar las variedades de precio y volumen."""

code_data = """# 1. Bajar datos de BTC
df = yf.download('BTC-USD', period='1y', interval='1d')
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

# Asegurar cast a float64 en el índice temporal
t = np.arange(len(df), dtype=np.float64)

# Preparamos el tensor sanitizado (Log Returns y Volumen Z-Score)
log_returns, vol, raw_price, dt_val = MarketLoader.prepare_quant_input(df, disable_norm=False, disable_returns=False)

# Normalización Z-score de Volumen manual para alimentar las derivadas
mu_v = np.mean(vol)
sigma_v = np.std(vol) if np.std(vol) > 1e-8 else 1.0
vol_z = (vol - mu_v) / sigma_v

# 2. Suavizado Topológico continuo (Capa 2: Splines de Memoria Líquida)
blender = ContinuousBlender(tolerance=0.0025)
telemetry_r = blender.fit(t, log_returns, dominant_periods=np.array([]), feature_idx=0)
telemetry_v = blender.fit(t, vol_z, dominant_periods=np.array([]), feature_idx=1)

smooth_r, r_dot, r_dot2 = blender.compute_continuous(0, t)
v_smooth, v_dot, v_dot2 = blender.compute_continuous(1, t)

print("Preprocesamiento y variedades continuas asimiladas de forma vectorizada!")
print(f"Spline Retornos MSE: {telemetry_r['mse']:.6f} | Spline Volumen MSE: {telemetry_v['mse']:.6f}")"""

text_mc = """## 2. Descubrimiento SINDy y Simulación de Euler-Maruyama
Usamos la clase `PhysicsDiscoverer` nativa del motor de Kinetopus para calibrar y ajustar el modelo. Luego simulamos caminos sintéticos de 5.000 días inyectando shocks estocásticos (ruido de Wiener) de acuerdo con los residuos históricos reales del activo."""

code_mc = """# 3. Descubrimiento Físico con SINDy
discoverer = PhysicsDiscoverer(poly_degree=1)
x_matrix = np.column_stack((smooth_r, v_smooth))
x_dot_matrix = np.column_stack((r_dot, v_dot))

# Calcular residuos reales (volatilidad real histórica)
sigma_res_r = float(np.std(log_returns - smooth_r))
sigma_res_v = float(np.std(vol_z - v_smooth))

# Ejecutar descubrimiento físico (SINDy)
report = discoverer.extract_equations(
    t=t, x=x_matrix, x_dot=x_dot_matrix, dt=dt_val, 
    horizon_steps=60, # Solo para obtener el reporte nativo inicial
    sigma_res_r=sigma_res_r, sigma_res_v=sigma_res_v,
    last_price=raw_price[-1], disable_norm=False, disable_returns=False
)

print("Ecuaciones del Atractor BTC descubiertas:")
print(f"dr/dt = {report['equations'][0]}")
print(f"dV/dt = {report['equations'][1]}")
print(f"R2 Score: {report['score']:.4f}")

# 4. Simulador Estocástico de Euler-Maruyama a Ultra-Largo Plazo (5,000 pasos)
horizonte_sintetico = 5000
n_simulaciones = 5

paths_precio = []
paths_vol = []

# Extraer el modelo SINDy ajustado internamente
sindy_model = discoverer.model

# Generar 5 trayectorias simuladas de forma explícita para visualización
for sim in range(n_simulaciones):
    x_current = x_matrix[-1:].copy() # (1, 2)
    p_current = raw_price[-1]
    
    p_history = [p_current]
    v_history = [x_current[0, 1]]
    
    for step in range(horizonte_sintetico):
        # Predict del campo vectorial EDO: dx/dt = f(x)
        x_dot_pred = sindy_model.predict(x_current) # (1, 2)
        
        # Inyección de shocks estocásticos independientes (Ruido de Wiener)
        noise_r = np.random.normal(0, 1) * sigma_res_r * 100.0 * np.sqrt(dt_val)
        noise_v = np.random.normal(0, 1) * sigma_res_v * np.sqrt(dt_val)
        
        # Actualización Euler-Maruyama
        # Canal 0: Retorno porcentual (escalado por 100 en el motor)
        # Canal 1: Volumen normalizado
        x_current[0, 0] = x_current[0, 0] + x_dot_pred[0, 0] * dt_val + noise_r
        x_current[0, 1] = x_current[0, 1] + x_dot_pred[0, 1] * dt_val + noise_v
        
        # Limitar para evitar explosión infinita del simulador
        x_current[0, 0] = np.clip(x_current[0, 0], -10.0, 10.0)
        x_current[0, 1] = np.clip(x_current[0, 1], -10.0, 10.0)
        
        # Reconstruir precio nominal: P(t+1) = P(t) * exp(r_t / 100)
        r_nominal = x_current[0, 0] / 100.0
        p_current = p_current * np.exp(r_nominal)
        
        p_history.append(p_current)
        v_history.append(x_current[0, 1])
        
    paths_precio.append(p_history)
    paths_vol.append(v_history)

print(f"Simuladas {n_simulaciones} realidades de mercado a {horizonte_sintetico} velas con éxito!")"""

text_plot = """## 3. Inspección Visual (El Test de Verosimilitud)
Graficaremos los caminos estocásticos generados. Esto nos permite evaluar visualmente si los patrones (tendencias, micro-reversiones, autocorrelaciones) imitan el comportamiento caótico y fractal de un activo de trading real."""

code_plot = """# Generamos el eje de tiempo sintético
t_sintetico = np.arange(horizonte_sintetico + 1)

fig = go.Figure()

colores = ['#ff00ff', '#ff9900', '#00ffcc', '#ffff00', '#ff3366']

# Dibujar las realidades paralelas sintéticas
for i in range(n_simulaciones):
    fig.add_trace(go.Scatter(
        x=t_sintetico, y=paths_precio[i],
        mode='lines',
        name=f'Mercado Sintético #{i+1}',
        line=dict(width=1.5, color=colores[i])
    ))

fig.update_layout(
    title=f"Generador de Trading Sintético BTC-USD (Euler-Maruyama, {horizonte_sintetico} pasos)",
    template="plotly_dark",
    height=650,
    xaxis_title="Tiempo Sintético (t)",
    yaxis_title="Precio Nominal Proyectado",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

fig.show()"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_markdown_cell(text_data),
    nbf.v4.new_code_cell(code_data),
    nbf.v4.new_markdown_cell(text_mc),
    nbf.v4.new_code_cell(code_mc),
    nbf.v4.new_markdown_cell(text_plot),
    nbf.v4.new_code_cell(code_plot)
]

os.makedirs('notebooks_val', exist_ok=True)
with open('notebooks_val/4_Synthetic_Market_Generator_PoC.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Notebook generado exitosamente en notebooks_val/4_Synthetic_Market_Generator_PoC.ipynb")
