import streamlit as st
import os
import uuid
import pandas as pd
import numpy as np
import json

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


def run_quant_pipeline(df, smooth_tol, cusum_h, horizon_steps, cusum_k, disable_norm, disable_returns, poly_degree, context_window):
    try:
        with st.spinner("1️⃣ Profiling y Sanitización (Regla 16GB)..."):
             log_returns_full, vol_full, price_full, dt_val = MarketLoader.prepare_quant_input(df, disable_norm=disable_norm, disable_returns=disable_returns)
             global_t = np.arange(len(log_returns_full), dtype=np.float64) * dt_val
             
             if context_window > 0 and len(log_returns_full) > context_window:
                 log_returns = log_returns_full[-context_window:]
                 volumen_z = vol_full[-context_window:]
                 precio_raw = price_full[-context_window:]
                 t = global_t[-context_window:]
             else:
                 log_returns = log_returns_full
                 volumen_z = vol_full
                 precio_raw = price_full
                 t = global_t
             
        with st.spinner("2️⃣ Capa Sensor (FFT)..."):
             sensor = SpectralAnalyzer(top_k=2)
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
             boundaries = [0] + shift_idx + [len(t)]
             
             for i in range(len(boundaries) - 1):
                 start_idx = boundaries[i]
                 end_idx = boundaries[i+1]
                 
                 if end_idx - start_idx < 15:
                     continue
                     
                 t_slice = t[start_idx:end_idx]
                 x_slice = x_matrix[start_idx:end_idx]
                 x_dot_slice = x_dot_matrix[start_idx:end_idx]
                 price_slice = precio_raw[start_idx:end_idx]
                 
                 local_sigma_r = float(np.std(log_returns[start_idx:end_idx] - r_smooth[start_idx:end_idx]))
                 local_sigma_v = float(np.std(volumen_z[start_idx:end_idx] - v_smooth[start_idx:end_idx]))
                 
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
                 
             if regime_physics_reports:
                 active_physics_report = regime_physics_reports[-1]
             else:
                 active_physics_report = physics_report_global
             
             # --- AUTOPILOT QUANT: Rutina de Rescate 2D ---
             best_rescue_r2 = 0.0
             best_rescue_tol = None
             
             if active_physics_report.get('prediction', {}).get('unstable', False):
                 rescue_tols = [0.0005, 0.0001, 0.00005]
                 
                 for tol_try in rescue_tols:
                     b_rescue = ContinuousBlender(tolerance=tol_try)
                     b_rescue.fit(t, log_returns, periodos_retornos, feature_idx=0)
                     b_rescue.fit(t, volumen_z, periodos_volumen, feature_idx=1)
                     
                     r_s, r_d, _ = b_rescue.compute_continuous(0, t)
                     v_s, v_d, _ = b_rescue.compute_continuous(1, t)
                     
                     if regime_physics_reports:
                         start_idx = boundaries[-2]
                         end_idx = boundaries[-1]
                     else:
                         start_idx, end_idx = 0, len(t)
                         
                     x_m = np.column_stack((r_s, v_s))[start_idx:end_idx]
                     x_d_m = np.column_stack((r_d, v_d))[start_idx:end_idx]
                     t_slice = t[start_idx:end_idx]
                     
                     local_rescue_sigma_r = float(np.std(log_returns[start_idx:end_idx] - r_s[start_idx:end_idx]))
                     local_rescue_sigma_v = float(np.std(volumen_z[start_idx:end_idx] - v_s[start_idx:end_idx]))
                     
                     d_rescue = PhysicsDiscoverer(poly_degree=poly_degree)
                     p_rep = d_rescue.extract_equations(
                         t=t_slice, x=x_m, x_dot=x_d_m, dt=dt_val, 
                         horizon_steps=1, sigma_res_r=local_rescue_sigma_r, sigma_res_v=local_rescue_sigma_v, last_price=precio_raw[-1], disable_norm=disable_norm, disable_returns=disable_returns
                     )
                     
                     if p_rep['score'] > best_rescue_r2 and not p_rep.get('empty_r_eq', True):
                         best_rescue_r2 = p_rep['score']
                         best_rescue_tol = tol_try
                         
                 if active_physics_report.get('empty_r_eq', False):
                      error_reason = "Falso Positivo de R2 por volatilidad de Volumen. Ecuación Inercial del Precio vacía (dr/dt = 0)."
                 else:
                      error_reason = "R2 por debajo de umbral crítico."
                      
                 if best_rescue_r2 >= 0.40:
                     st.warning(f"🚨 Física del Régimen Inestable con Tolerancia actual. 💡 Ajusta Tolerancia Spline a **{best_rescue_tol:.5f}** para R2 de **{best_rescue_r2:.2f}**.")
                 else:
                     st.error(f"🚨 Ruido Extremo en Régimen Actual: {error_reason} Proyección Monte Carlo anulada irreversiblemente.")
                 
             active_physics_report['blender_mse'] = telemetry_r['mse']
             active_physics_report['cusum_runtime'] = cusum_report['runtime_ms']

        # 7. Solución Analítica de Simetría (SymPy) del Régimen Vigente
        with st.spinner("7️⃣ Integración Simbólica y Geometría Temporal..."):
            symbolic = SymbolicTranslator(active_physics_report)
            p0 = precio_raw[-1] if disable_returns else log_returns[-1]
            v0 = v_smooth[-1]
            analytical_solution = symbolic.generate_analytical_solution(p0=p0, v0=v0, disable_returns=disable_returns)

        # Guardar en Session State para la pestaña de exportación
        st.session_state['latest_telemetry'] = {
            "active_equations": active_physics_report.get('equations', []),
            "r2_score": active_physics_report.get('score', 0.0),
            "noise_sigma_r": active_physics_report.get('sigma_res_r', 0.0),
            "num_regimes": len(regime_physics_reports),
            "horizon_steps": horizon_steps,
            "analytical_solution": analytical_solution
        }

        # --- PANEL EJECUTIVO DE MÉTRICAS CUANTITATIVAS (KPIs) ---
        st.markdown("### 📊 Estado Topológico y Telemetría en Vivo")
        
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        
        # 1. Vector Direccional
        last_velocity = r_dot[-1]
        is_bullish = last_velocity > 0
        direction_label = "🟢 ALCISTA (Long)" if is_bullish else "🔴 BAJISTA (Short)"
        velocity_pct = float(last_velocity * 100) if not disable_returns else float(last_velocity)
        
        col_kpi1.metric(
            label="Inercia / Vector Direccional",
            value=direction_label,
            delta=f"{velocity_pct:+.3f}% vel.",
            delta_color="normal"
        )
        
        # 2. Duración del Régimen Activo
        current_regime_len = len(t) - boundaries[-2]
        col_kpi2.metric(
            label="Régimen SINDy Vigente",
            value=f"Régimen {len(regime_physics_reports)}",
            delta=f"{current_regime_len} velas activo",
            delta_color="off"
        )
        
        # 3. Pureza de Ecuación (R2)
        r2_val = active_physics_report.get('score', 0.0)
        r2_quality = "Excelente" if r2_val > 0.8 else ("Aceptable" if r2_val > 0.5 else "Ruidoso")
        col_kpi3.metric(
            label="Ajuste SINDy (R²)",
            value=f"{r2_val:.3f}",
            delta=r2_quality,
            delta_color="normal" if r2_val > 0.5 else "inverse"
        )
        
        # 4. Semáforo de Tensión CUSUM
        current_tension = max(cusum_report['s_pos'][-1], cusum_report['s_neg'][-1])
        tension_ratio = current_tension / cusum_h if cusum_h > 0 else 0
        if tension_ratio > 0.9:
            cusum_status = "🔴 Quiebre Inminente"
        elif tension_ratio > 0.6:
            cusum_status = "🟡 Tensión Alta"
        else:
            cusum_status = "🟢 Estable"
            
        col_kpi4.metric(
            label="Semáforo CUSUM (Régimen)",
            value=cusum_status,
            delta=f"Tensión: {current_tension:.2f} / H={cusum_h:.1f}",
            delta_color="off"
        )
        
        st.divider()

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
             analytical_solution=analytical_solution,
             full_raw_price=price_full,
             full_t=global_t
        )

    except Exception as e:
         st.error(f"Fallo del Motor Quant: {e}")


def render_predictive_station():
    # ⚡ PRESETS DE 1-CLICK
    st.markdown("#### ⚡ Escenarios de Demostración Rápida (1-Click)")
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    
    selected_preset = None
    if col_p1.button("🟠 Bitcoin (BTC-USD)", use_container_width=True):
        selected_preset = "BTC-USD"
    if col_p2.button("🏢 Microsoft (MSFT)", use_container_width=True):
        selected_preset = "MSFT"
    if col_p3.button("⚡ Solana (SOL-USD)", use_container_width=True):
        selected_preset = "SOL-USD"
    if col_p4.button("🥇 Oro Spot (GC=F)", use_container_width=True):
        selected_preset = "GC=F"
        
    if selected_preset:
        st.session_state['active_ticker'] = selected_preset
        st.session_state['needs_download'] = True
        st.rerun()

    if 'active_ticker' not in st.session_state:
        st.session_state['active_ticker'] = "BTC-USD"
        
    st.markdown("---")
    col_input, col_config = st.columns([2, 1])
    
    with col_input:
        with st.form("mercado_form"):
            ticker = st.text_input("Símbolo del Activo (Yahoo Finance):", st.session_state['active_ticker'])
            
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                market_period = st.selectbox("Periodo Histórico:", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "15y", "max"], index=5)
            with col_t2:
                market_interval = st.selectbox("Resolución (Velas):", ["1h", "1d", "1wk", "1mo"], index=1)
                
            uploaded_file = st.file_uploader("📂 O sube Datos Sintéticos (CSV)", type=["csv", "txt"])
            submitted = st.form_submit_button("🚀 Descargar y Analizar", type="primary", use_container_width=True)
            
            if submitted:
                st.session_state['active_ticker'] = ticker
                st.session_state['needs_download'] = True
        
        if uploaded_file is not None:
            st.info("📌 Tienes un archivo local cargado. El Ticker y los periodos de Yahoo Finance serán ignorados.")
            
        if 'quant_df' not in st.session_state:
            st.session_state['quant_df'] = None
            
        # AUTO-LOAD INICIAL PARA EVITAR LIENZO VACÍO
        if st.session_state['quant_df'] is None and not st.session_state.get('initial_load_done', False):
            st.session_state['needs_download'] = True
            st.session_state['initial_load_done'] = True
            
        if st.session_state.get('needs_download', False):
            with st.spinner("📥 Descargando datos y computando topología del mercado..."):
                try:
                    if uploaded_file is not None:
                        st.session_state['quant_df'] = pd.read_csv(uploaded_file)
                    else:
                        st.session_state['quant_df'] = MarketLoader.load_ticker_data(st.session_state['active_ticker'], period=market_period, interval=market_interval)
                except Exception as e:
                    st.error(f"Error cargando los datos: {e}")
            st.session_state['needs_download'] = False
                         
    with col_config:
         st.markdown("⚙️ **Configuración Principal**")
         context_window = st.slider("Ventana de Contexto (Velas)", 0, 5000, 1000, help="0 = Todo el historial. Limita las velas analizadas por los Splines para captar inercia reciente.")
         horizon_steps = st.slider("Horizonte Predictivo (t+N)", 1, 500, 150, help="Velas proyectadas hacia el futuro por Euler-Maruyama.")
         smooth_tol = st.slider("Tolerancia Spline (Filtrado de Ruido)", 0.0001, 0.005, 0.005, step=0.0001, format="%.4f")
         cusum_k = st.slider("Drift CUSUM (k - Tolerancia al Ruido)", 0.1, 5.0, 1.0, format="%.2f")
         
         # ACORDEÓN DE PARÁMETROS AVANZADOS (PROGRESSIVE DISCLOSURE)
         with st.expander("🔬 Parámetros Avanzados del Laboratorio"):
             cusum_h = st.slider("Umbral CUSUM (H - Robustez Anómala)", 5.0, 50.0, 5.0)
             poly_degree = st.slider("Complejidad SINDy (Grado Polinomial)", 1, 3, 1, help="1: Lineal (Gravedad), 2: Interactivo (Lotka-Volterra), 3: Complejo.")
             disable_returns = st.checkbox("Modo Física Clásica (Usar Precio Absoluto)", value=False, help="Bypass de Log Returns. Usa directamente P(t).")
             disable_norm = st.checkbox("Desactivar Normalización (Calibración)", value=False, help="Procesa valores crudos para tests con modelos físicos perfectos.")

         st.markdown("---")
         st.info("💡 **Calibración Automática:** Presiona el botón para buscar el Drift ($k$) óptimo que maximiza el $R^2$ de SINDy.")
         
         if st.button("🤖 🔍 Auto-Tune CUSUM Drift (Calibrar k)", type="secondary", use_container_width=True):
             if st.session_state.get('quant_df') is None:
                 st.warning("⚠️ Primero descarga o sube los datos de mercado presionando 'Descargar y Analizar'.")
             else:
                 with st.spinner("Buscando Resonancia Topológica (Testando 50 Drifts)..."):
                     from src.quant_engine.auto_tuner import CUSUMAutoTuner
                     tuner = CUSUMAutoTuner(st.session_state['quant_df'], disable_norm=disable_norm, disable_returns=disable_returns)
                     best_k, rep = tuner.run_search()
                     
                     st.success(f"🏆 **¡Drift CUSUM Óptimo Encontrado!**\n\n"
                                f"👉 **Ajusta manualmente el slider 'Drift CUSUM (k)' a:** `{best_k}`\n\n"
                                f"*(Logró un $R^2$ ponderado de `{rep['weighted_r2']:.4f}` aislando `{rep['num_quiebres']}` macro-quiebres)*.")

    if st.session_state['quant_df'] is not None:
         st.divider()
         run_quant_pipeline(st.session_state['quant_df'], smooth_tol, cusum_h, horizon_steps, cusum_k, disable_norm, disable_returns, poly_degree, context_window)


def render_paper_tab():
    st.markdown("### 📜 Abstract & Arquitectura del Paper Académico")
    
    st.markdown("""
    > **Título:** *Kinetopus Engine: A Parsimonious Physical-Mathematical Pipeline for Local-First Financial Time Series Forecasting using Sparse Identification of Nonlinear Dynamics (SINDy)*  
    > **Autores:** Juan Diego & Antigravity Quant Team  
    > **Licencia:** Apache 2.0 | **Hardware:** 100% CPU (16GB RAM Local-First)
    """)
    
    st.markdown("#### 🎯 Resumen Ejecutivo")
    st.write("""
    El análisis de series temporales financieras se ha debatido históricamente entre modelos lineales clásicos (ej. ARIMA) y 'cajas negras' de Deep Learning de alto consumo computacional (ej. Transformers, LSTMs).
    
    **Kinetopus Engine** propone un paradigma alternativo de **Caja Blanca** basado en la física computacional y la teoría de sistemas dinámicos:
    1. **Sensor Espectral (FFT):** Identifica las frecuencias dominantes del activo.
    2. **Moldeador Topológico (Splines Cúbicos $C^2$):** Filtra el ruido microestructural sin desfase de fase (*zero phase-lag*).
    3. **Sistema Nervioso (CUSUM):** Detecta quiebres estructurales en tiempo real para evitar la degradación del modelo.
    4. **Motor Físico (SINDy):** Descubre las ecuaciones diferenciales ordinarias subyacentes $\\dot{X} = f(X, V)$ de forma parsimoniosa.
    5. **Integración Estocástica (Euler-Maruyama):** Proyecta conos de incertidumbre Monte Carlo fieles a la volatilidad del régimen activo.
    """)
    
    st.markdown("#### 🏛️ Diagrama del Pipeline de 5 Capas")
    st.code("""
    [Ingesta de Mercado (Precios & Volumen)]
                      │
                      ▼
    ┌───────────────────────────────────┐
    │ 1. Capa Sensor (FFT Espectral)    │ ➔ Frecuencias Dominantes
    └─────────────────┬─────────────────┘
                      ▼
    ┌───────────────────────────────────┐
    │ 2. Moldeador (Splines Cúbicos C²) │ ➔ Curva Continua & Derivadas Puras (dr/dt, d²r/dt²)
    └─────────────────┬─────────────────┘
                      ▼
    ┌───────────────────────────────────┐
    │ 3. Sistema Nervioso (CUSUM)       │ ➔ Slicing de Regímenes & Detección de Quiebres
    └─────────────────┬─────────────────┘
                      ▼
    ┌───────────────────────────────────┐
    │ 4. Motor Físico (SINDy Parsimony) │ ➔ Descubrimiento de EDOs: dr/dt = f(r, V)
    └─────────────────┬─────────────────┘
                      ▼
    ┌───────────────────────────────────┐
    │ 5. Visualizador & Monte Carlo     │ ➔ Cono Estocástico Euler-Maruyama (t+150)
    └───────────────────────────────────┘
    """, language="text")


def render_export_tab():
    st.markdown("### 💾 Exportación de Telemetría Cuantitativa")
    
    latest = st.session_state.get('latest_telemetry')
    if latest is None:
        st.warning("⚠️ Primero ejecuta un análisis en la pestaña 'Estación Predictiva' para generar la telemetría.")
    else:
        st.success("✅ Telemetría del Régimen Vigente lista para exportar.")
        
        st.json(latest)
        
        json_str = json.dumps(latest, indent=4)
        st.download_button(
            label="📥 Descargar Telemetría en Formato JSON",
            data=json_str,
            file_name=f"kinetopus_telemetry_{st.session_state.get('active_ticker', 'asset')}.json",
            mime="application/json",
            type="primary"
        )


def main():
    # Hero Header
    st.title("🔭 Kinetopus Engine")
    st.markdown("""
    <span class="quant-badge">Soberanía Local (16GB RAM)</span>
    <span class="quant-badge">Caja Blanca (SINDy EDOs)</span>
    <span class="quant-badge">Zero Phase-Lag</span>
    <span class="quant-badge-secondary">Licencia Apache 2.0</span>
    """, unsafe_allow_html=True)
    st.caption("Motor Cuantitativo de Física Computacional y Topología Financiera Continua.")

    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = str(uuid.uuid4())

    # Pestañas de Navegación
    tab_station, tab_paper, tab_export = st.tabs([
        "🔭 Estación Predictiva", 
        "📜 Abstract & Arquitectura (Paper)", 
        "💾 Exportar Telemetría"
    ])
    
    with tab_station:
        render_predictive_station()
        
    with tab_paper:
        render_paper_tab()
        
    with tab_export:
        render_export_tab()


if __name__ == "__main__":
    main()
