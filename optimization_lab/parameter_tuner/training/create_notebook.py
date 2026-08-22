import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

markdown_1 = """# Análisis Topológico de Hiperparámetros (Meta-Optimización)
Este notebook analiza la malla de optimización de parámetros (`drift_k`, `spline_tol`, `cusum_h`) evaluada mediante la **Ecuación Fitness**.
"""

code_1 = """import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# Cargar Resultados
filepath = '../best_hyperparameters.csv'
if os.path.exists(filepath):
    df = pd.read_csv(filepath)
    display(df.head(10))
else:
    print("Aún no se ha ejecutado el optimizador completo. Ejecuta: python grid_search_tuner.py")
"""

markdown_2 = """## Superficie de Rendimiento 3D"""

code_2 = """if 'df' in locals() and not df.empty:
    fig = px.scatter_3d(df, x='drift_k', y='spline_tol', z='cusum_h',
                  color='penalized_fitness', size='universes_survived',
                  hover_data=['mean_fitness', 'std_fitness'],
                  color_continuous_scale='Viridis',
                  title="Superficie 3D de Hiperparámetros vs Fitness Score")
    fig.update_layout(template='plotly_dark')
    fig.show()
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(markdown_1),
    nbf.v4.new_code_cell(code_1),
    nbf.v4.new_markdown_cell(markdown_2),
    nbf.v4.new_code_cell(code_2)
]

notebook_path = os.path.join(os.path.dirname(__file__), 'notebooks', '1_Parameter_Analysis.ipynb')
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f"Notebook created at {notebook_path}")
