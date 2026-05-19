import streamlit as st
import os
import uuid
import pandas as pd
import numpy as np
import json
import asyncio

from src.core.config import Config
from src.ui.styles import apply_custom_css

from src.ui.market_loader import MarketLoader
from src.quant_engine.sensor import SpectralAnalyzer
from src.quant_engine.blender import ContinuousBlender
from src.quant_engine.nervous import RegimeShiftDetector
from src.quant_engine.physics import PhysicsDiscoverer
from src.quant_engine.symbolic import SymbolicTranslator
from src.components.quant_widgets import QuantDashboard

st.set_page_config(page_title="Kinetopus Engine", page_icon="🔭", layout="wide")
apply_custom_css()


def run_quant_pipeline(df, smooth_tol, cusum_h, horizon_steps, cusum_k, disable_norm, disable_returns, poly_degree):
    try:
        with st.spinner("1️⃣ Profiling y Sanitización (Regla 16GB)..."):
             log_returns, volumen_z, precio_raw, dt_val = MarketLoader.prepare_quant_input(df, disable_norm=disable_norm, disable_returns=disable_returns)
             t = np.arange(len(log_returns), dtype=np.float64) * dt_val
             
        with st.spinner("2️⃣ Capa Sensor (FFT)..."):
             sensor = SpectralAnalyzer(top_k=2)
             # Empaquetamos en (N, 2)
             data_matrix = np.column_stack((log_returns, volumen_z))
             fft_results = sensor.analyze(data_matrix, dt=dt_val)
             periodos_retornos = fft_results[0]['periods']
             periodos_volumen = fft_results[1]['periods']

        with st.spinner("3️⃣ Capa Moldeador Topológico (Splines)..."):
             blender = ContinuousBlender(tolerance=smooth_tol)
             telemetry_r = blender.fit(t, log_returns, periodos_retornos, feature_idx=0)
             telemetry_v = blender.fit(t, volumen_z, periodos_volumen, feature_idx=1)
             
             r_smooth, r_dot, r_dot2 = blender.compute_continuous(0, t)
             v_smooth, v_dot, v_dot2 = blender.compute_continuous(1, t)
             
        with st.spinner("4️⃣ Sistema Nervioso (CUSUM)..."):
             detector = RegimeShiftDetector(threshold=cusum_h, drift=cusum_k)
             cusum_report = detector.detect(log_returns, r_smooth)
             
        with st.spinner(f"5️⃣ Extracción Física Estocástica (SINDy Tuning t+{horizon_steps})..."):
             discoverer = PhysicsDiscoverer(poly_degree=poly_degree)
             x_matrix = np.column_stack((r_smooth, v_smooth))
             x_dot_matrix = np.column_stack((r_dot, v_dot))
             
             # Calcular nivel de ruido para difusión Monte Carlo
             sigma_res_r = float(np.std(log_returns - r_smooth))
             sigma_res_v = float(np.std(volumen_z - v_smooth))
             
             # 1. SINDy Global (Histórico de Referencia)
             physics_report_global = discoverer.extract_equations(
                 t=t, x=x_matrix, x_dot=x_dot_matrix, dt=dt_val, 
                 horizon_steps=horizon_steps, sigma_res_r=sigma_res_r, sigma_res_v=sigma_res_v,
                 last_price=precio_raw[-1], disable_norm=disable_norm, disable_returns=disable_returns
             )
             
             # 2. SINDy por Regímenes (CUSUM Slicing)
             regime_physics_reports = []
             shift_idx = cusum_report['shift_indices']
             
             # Construir tramos (borde izquierdo, borde derecho)
             boundaries = [0] + shift_idx + [len(t)]
             
             for i in range(len(boundaries) - 1):
                 start_idx = boundaries[i]
                 end_idx = boundaries[i+1]
                 
                 # Ignorar micro-regímenes muy cortos (ruido puro) que harían crashear SINDy
                 if end_idx - start_idx < 15:
                     continue
                     
                 t_slice = t[start_idx:end_idx]
                 x_slice = x_matrix[start_idx:end_idx]
                 x_dot_slice = x_dot_matrix[start_idx:end_idx]
                 price_slice = precio_raw[start_idx:end_idx]
                 
                 # Cáculo de Ruido Estocástico (Volatilidad) Exclusivo para este Régimen
                 local_sigma_r = float(np.std(log_returns[start_idx:end_idx] - r_smooth[start_idx:end_idx]))
                 local_sigma_v = float(np.std(volumen_z[start_idx:end_idx] - v_smooth[start_idx:end_idx]))
                 
                 # El horizonte solo aplica al último tramo (para predecir el futuro real)
                 is_last_regime = (i == len(boundaries) - 2)
                 h_steps = horizon_steps if is_last_regime else 1
                 
                 rep = discoverer.extract_equations(
                     t=t_slice, x=x_slice, x_dot=x_dot_slice, dt=dt_val,
                     horizon_steps=h_steps, sigma_res_r=local_sigma_r, sigma_res_v=local_sigma_v,
                     last_price=price_slice[-1], disable_norm=disable_norm, disable_returns=disable_returns
                 )
                 rep['sigma_res_r'] = local_sigma_r
                 rep['sigma_res_v'] = local_sigma_v
                 rep['regime_name'] = f"Régimen {i+1} (Velas {start_idx}-{end_idx})"
                 rep['is_active'] = is_last_regime
                 regime_physics_reports.append(rep)
                 
             # El reporte de contingencia será el global si por algún motivo no hay regímenes válidos,
             # de lo contrario, la predicción Monte Carlo y la EDO vigente la dicta el último régimen
             if regime_physics_reports:
                 active_physics_report = regime_physics_reports[-1]
             else:
                 active_physics_report = physics_report_global
             
             # --- AUTOPILOT QUANT: Rutina de Rescate 2D (Spline Tuning Automático) ---
             # Evaluamos la estabilidad sobre el RÉGIMEN ACTUAL, no sobre la global que sabemos que está rota
             best_rescue_r2 = 0.0
             best_rescue_tol = None
             
             if active_physics_report.get('prediction', {}).get('unstable', False):
                 rescue_tols = [0.0005, 0.0001, 0.00005]
                 
                 for tol_try in rescue_tols:
                     # 1. Remoldeo Spline Rápido P/V (sin guardar telemetría principal)
                     b_rescue = ContinuousBlender(tolerance=tol_try)
                     b_rescue.fit(t, log_returns, periodos_retornos, feature_idx=0)
                     b_rescue.fit(t, volumen_z, periodos_volumen, feature_idx=1)
                     
                     r_s, r_d, _ = b_rescue.compute_continuous(0, t)
                     v_s, v_d, _ = b_rescue.compute_continuous(1, t)
                     
                     # Recortar estado al régimen vigente para hacer rescate real
                     if regime_physics_reports:
                         start_idx = boundaries[-2]
                         end_idx = boundaries[-1]
                     else:
                         start_idx, end_idx = 0, len(t)
                         
                     x_m = np.column_stack((r_s, v_s))[start_idx:end_idx]
                     x_d_m = np.column_stack((r_d, v_d))[start_idx:end_idx]
                     t_slice = t[start_idx:end_idx]
                     
                     # Ruido estocástico del rescate en el régimen actual
                     local_rescue_sigma_r = float(np.std(log_returns[start_idx:end_idx] - r_s[start_idx:end_idx]))
                     local_rescue_sigma_v = float(np.std(volumen_z[start_idx:end_idx] - v_s[start_idx:end_idx]))
                     
                     # 2. Reevaluación SINDy Estática
                     d_rescue = PhysicsDiscoverer(poly_degree=poly_degree)
                     p_rep = d_rescue.extract_equations(
                         t=t_slice, x=x_m, x_dot=x_d_m, dt=dt_val, 
                         horizon_steps=1, sigma_res_r=local_rescue_sigma_r, sigma_res_v=local_rescue_sigma_v, last_price=precio_raw[-1], disable_norm=disable_norm, disable_returns=disable_returns
                     )
                     
                     # Criterio estricto de rescate: Cero falsos positivos con inercia anulada de precio (Fase 13)
                     if p_rep['score'] > best_rescue_r2 and not p_rep.get('empty_r_eq', True):
                         best_rescue_r2 = p_rep['score']
                         best_rescue_tol = tol_try
                         
                 # Exposición en Interfaz de Usuario
                 if active_physics_report.get('empty_r_eq', False):
                      error_reason = "Falso Positivo de R2 por volatilidad de Volumen. Ecuación Inercial del Precio vacía (dr/dt = 0)."
                 else:
                      error_reason = "R2 por debajo de umbral crítico."
                      
                 if best_rescue_r2 >= 0.40:
                     st.warning(f"🚨 Física del Régimen Inestable con Tolerancia actual. 💡 Ajusta Tolerancia Spline a **{best_rescue_tol:.5f}** para R2 de **{best_rescue_r2:.2f}**.")
                 else:
                     st.error(f"🚨 Ruido Extremo en Régimen Actual: {error_reason} Proyección Monte Carlo anulada irreversiblemente.")
                 
             # Inyectar telemetría de capas previas a active_physics_report para simplificar dashboard
             active_physics_report['blender_mse'] = telemetry_r['mse']
             active_physics_report['cusum_runtime'] = cusum_report['runtime_ms']

        # 6. (Capa de IA eliminada para eficiencia local pura)

        # 7. Solución Analítica de Simetría (SymPy) del Régimen Vigente
        with st.spinner("7️⃣ Integración Simbólica y Geometría Temporal..."):
            symbolic = SymbolicTranslator(active_physics_report)
            p0 = precio_raw[-1] if disable_returns else log_returns[-1]
            # Extraer v0 correcta del régimen activo final
            v0 = v_smooth[-1]
            analytical_solution = symbolic.generate_analytical_solution(p0=p0, v0=v0, disable_returns=disable_returns)

        # Render Dashboard
        QuantDashboard.render(
             t=t, 
             raw_price=precio_raw, 
             smooth_price=r_smooth, 
             log_returns=log_returns,
             cusum_s_pos=cusum_report['s_pos'], 
             cusum_s_neg=cusum_report['s_neg'], 
             cusum_triggers=cusum_report['shift_indices'], 
             physics_report_global=physics_report_global, 
             regime_physics_reports=regime_physics_reports,
             active_physics_report=active_physics_report,
             cusum_threshold=cusum_h,
             v_smooth=v_smooth,
             v_dot=v_dot,
             r_dot=r_dot,
             r_dot2=r_dot2,
             disable_returns=disable_returns,
             analytical_solution=analytical_solution
        )

    except Exception as e:
         st.error(f"Fallo del Motor Quant: {e}")

def render_quant_mode():
    st.subheader("🔭 Motor Quant Físico (Estación Predictiva)")
    
    col_input, col_config = st.columns([2, 1])
    with col_input:
        with st.form("mercado_form"):
            ticker = st.text_input("Símbolo del Activo (Ej. AAPL, BTC-USD):", "SPY")
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                market_period = st.selectbox("Periodo Histórico:", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"], index=3)
            with col_t2:
                market_interval = st.selectbox("Resolución (Velas):", ["1h", "1d", "1wk", "1mo"], index=1)
                
            uploaded_file = st.file_uploader("📂 O sube Datos Sintéticos (Ignora el Ticker de arriba)", type=["csv", "txt"])
            
            submitted = st.form_submit_button("Descargar y Analizar", type="primary")
            if submitted:
                # Flag temporario para indicar procesamiento en el flujo exterior
                st.session_state['needs_download'] = True
        
        if uploaded_file is not None:
            st.info("📌 Tienes un archivo local cargado. El Ticker y los periodos de Yahoo Finance serán ignorados. (Cierra el archivo con la 'X' para usar datos reales).")
            
        if 'quant_df' not in st.session_state:
            st.session_state['quant_df'] = None
            
        if st.session_state.get('needs_download', False):
            with st.spinner("📥 Descargando y Computando Topología..."):
                try:
                    if uploaded_file is not None:
                        st.session_state['quant_df'] = pd.read_csv(uploaded_file)
                    else:
                        st.session_state['quant_df'] = MarketLoader.load_ticker_data(ticker, period=market_period, interval=market_interval)
                except Exception as e:
                    st.error(f"Error cargando los datos: {e}")
            st.session_state['needs_download'] = False
                         
    with col_config:
         st.markdown("⚙️ **Afinamiento Físico**")
         disable_returns = st.checkbox("Modo Física Clásica (Usar Precio Absoluto)", value=False, help="Bypass de Log Returns. Usa directamente P(t) - Ideal para Cinemática y Modelos Toy.")
         disable_norm = st.checkbox("Desactivar Normalización (Modo Calibración)", value=False, help="Procesa valores crudos para tests con modelos físicos perfectos (Toy Models).")
         horizon_steps = st.slider("Horizonte Predictivo (t+N) [Euler]", 1, 500, 60)
         poly_degree = st.slider("Complejidad SINDy (Grado Polinomial)", 1, 3, 1, help="1: Lineal (Gravedad), 2: Interactivo (Zorros/Conejos, Lotka-Volterra o Mercado Clásico), 3: Extra Complejo.")
         smooth_tol = st.slider("Tolerancia Spline (Aislamiento de Ruido)", 0.0001, 0.005, 0.005, step=0.0001, format="%.4f")
         cusum_h = st.slider("Umbral CUSUM (H - Robustez Anómala)", 5.0, 50.0, 5.0)
         cusum_k = st.slider("Drift CUSUM (k - Tolerancia al Ruido)", 0.1, 5.0, 1.0, format="%.2f")
         
         if st.button("🤖 Auto-Tune CUSUM Drift", help="Busca el Drift óptimo ponderando la pureza (R2) de SINDy a lo largo del histórico."):
             if st.session_state.get('quant_df') is None:
                 st.warning("Aún no hay datos de mercado. Presiona 'Analizar Datos' en el panel principal primero.")
             else:
                 with st.spinner("Buscando Resonancia Topológica (Testando 50 Drifts)..."):
                     from src.quant_engine.auto_tuner import CUSUMAutoTuner
                     tuner = CUSUMAutoTuner(st.session_state['quant_df'], disable_norm=disable_norm, disable_returns=disable_returns)
                     best_k, rep = tuner.run_search()
                     
                     st.success(f"🏆 **¡Drift CUSUM Óptimo Encontrado!**\n\n"
                                f"► **Desliza k a:** `{best_k}`\n\n"
                                f"*(Logró un R2 ponderado de `{rep['weighted_r2']:.4f}` aislando `{rep['num_quiebres']}` macro-quiebres)*.")

    if st.session_state['quant_df'] is not None:
         # Limpiar gráficos previos de Streamlit si lo amerita el diseño, pero rerun lo hace solo.
         st.divider()
         run_quant_pipeline(st.session_state['quant_df'], smooth_tol, cusum_h, horizon_steps, cusum_k, disable_norm, disable_returns, poly_degree)


def main():
    st.title("🔭 Kinetopus Engine")
    st.caption("Motor de Física Computacional y Topología Financiera.")

    # Inicializar Session State
    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = str(uuid.uuid4())

    # Sidebar
    with st.sidebar:
        st.header("🛠️ Configuración Global")
        st.info(f"Sesión ID: {st.session_state['session_id'][:8]}...")
        
        if st.button("Reiniciar Sesión"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    render_quant_mode()

if __name__ == "__main__":
    main()
