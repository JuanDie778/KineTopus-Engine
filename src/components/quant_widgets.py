import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import numpy as np

class QuantDashboard:
    """
    Componente visual que orquesta los gráficos interactivos de la Caja Blanca (Motor Quant).
    """

    @staticmethod
    def render(
         t: np.ndarray, 
         raw_price: np.ndarray, 
         smooth_price: np.ndarray, 
         log_returns: np.ndarray,
         cusum_s_pos: np.ndarray, 
         cusum_s_neg: np.ndarray, 
         cusum_triggers: list, 
         physics_report_global: dict,
         regime_physics_reports: list,
         active_physics_report: dict,
         cusum_threshold: float,
         v_smooth: np.ndarray,
         v_dot: np.ndarray,
         r_dot: np.ndarray,
         r_dot2: np.ndarray,
         disable_returns: bool = False,
         analytical_solution: str = None
    ):    
        st.markdown("### 🔭 Telemetría Físico-Predictiva")

        st.markdown("### 🔭 Telemetría Físico-Predictiva")

        # 1. Crear Subplots: Panel Superior (Precio y Cono), Panel Inferior Izquierdo (Radar Topológico), Panel Inferior Derecho (CUSUM)
        fig = make_subplots(
            rows=2, cols=2, 
            specs=[[{"colspan": 2}, None], [{}, {}]],
            shared_xaxes=False, 
            vertical_spacing=0.15,
            horizontal_spacing=0.1,
            subplot_titles=("Inercia de Precio Absoluto y Cono Estocástico", 
                            "Atractor de Espacio de Fase (Radar)",
                            "Tensión CUSUM / Quiebres (Capa 3)"),
            row_heights=[0.6, 0.4]
        )

        # --- PANEL 1: PRECIO E INERCIA ---
        # Precio Crudo (Marcadores tenues)
        fig.add_trace(go.Scatter(
            x=t, y=raw_price,
            mode='markers',
            name='Precio Crudo (Ruido)',
            marker=dict(color='rgba(150, 150, 150, 0.4)', size=4)
        ), row=1, col=1)

        # Spline Continuo (La Inercia Pura) - Fragmentado por Regímenes (Fase 8)
        # Cortaremos el array 't' y 'nominal_smooth_price' en base a cusum_triggers
        
        # 🚨 HOTFIX FÍSICA: Reconstrucción de Precio Nominal para Plotting
        # En modo financiero (returns), smooth_price son Logs. Debemos recomponer a precio absoluto.
        # En modo Clásico (disable_returns), smooth_price YA ES el precio absoluto.
        if disable_returns:
            nominal_smooth_price = smooth_price
        else:
            nominal_smooth_price = raw_price[0] * np.exp(np.cumsum(smooth_price))
        
        colors = ['#00ffcc', '#ff9900', '#cc33ff', '#ff3366', '#33ccff'] 
        start_idx = 0
        
        for i, trigger_idx in enumerate(list(cusum_triggers) + [len(t)]):
            segment_t = t[start_idx:trigger_idx]
            segment_y = nominal_smooth_price[start_idx:trigger_idx]
            
            if len(segment_t) > 0:
                color = colors[i % len(colors)]
                fig.add_trace(go.Scatter(
                    x=segment_t, y=segment_y,
                    mode='lines',
                    name='Inercia Spline (Fragmentada)' if i == 0 else f'Régimen {i+1} (Quiebre)',
                    line=dict(color=color, width=2),
                    showlegend=(i == 0)
                ), row=1, col=1)
                
            start_idx = trigger_idx

        # Cono de Incertidumbre Estocástica (Monte Carlo t+N) (Fase 9)
        pred_x_path = active_physics_report.get('prediction', {}).get('t_path', [])
        pred = active_physics_report.get('prediction')
        p_price = []
        if pred and 'price_percentiles' in pred and not active_physics_report.get('empty_r_eq', False):
            p_price = pred['price_percentiles']
            
        if len(pred_x_path) > 0 and len(p_price) == 5:
            p5, p25, p50, p75, p95 = p_price
            
            # P5 a P95 (Alpha bajo)
            fig.add_trace(go.Scatter(
                x=pred_x_path + pred_x_path[::-1], 
                y=p95 + p5[::-1],
                fill='toself',
                fillcolor='rgba(255, 0, 255, 0.1)',
                line=dict(color='rgba(255, 255, 255, 0)'),
                name='Cono Estocástico (Monte Carlo)',
                showlegend=True
            ), row=1, col=1)
            
            # P25 a P75 (Alpha medio)
            fig.add_trace(go.Scatter(
                x=pred_x_path + pred_x_path[::-1], 
                y=p75 + p25[::-1],
                fill='toself',
                fillcolor='rgba(255, 0, 255, 0.2)',
                line=dict(color='rgba(255, 255, 255, 0)'),
                showlegend=False
            ), row=1, col=1)
            
            # Mediana P50
            fig.add_trace(go.Scatter(
                x=pred_x_path, y=p50,
                mode='lines',
                line=dict(color='#ff00ff', width=2),
                showlegend=False
            ), row=1, col=1)
            
            # Anclaje Final Mediana
            fig.add_trace(go.Scatter(
                x=[pred_x_path[-1]], y=[p50[-1]],
                mode='markers+text',
                name='Objetivo Estocástico P50',
                marker=dict(color='white', size=12, symbol='star'),
                text=[f"P50(t+{len(pred_x_path)}): {p50[-1]:.2f}"],
                textposition='top right',
                showlegend=False
            ), row=1, col=1)

        # Proyección Determinística Pura (La Función Matemática)
        # Se muestra SIEMPRE, incluso si Monte Carlo falla por inestabilidad de ruido
        det_price = pred.get('det_price_path', []) if pred else []
        det_t = pred.get('det_t_path', []) if pred else []
        
        if len(det_price) > 0 and len(det_t) > 0:
            fig.add_trace(go.Scatter(
                x=det_t, y=det_price,
                mode='lines',
                name='Proyección Matemática (Pura Inercia)',
                line=dict(color='#00ffcc', width=2, dash='dash'),
                showlegend=True
            ), row=1, col=1)
            
            # Etiqueta de la función pura al final
            fig.add_trace(go.Scatter(
                x=[det_t[-1]], y=[det_price[-1]],
                mode='markers+text',
                name='Target Matemático',
                marker=dict(color='#00ffcc', size=10, symbol='circle'),
                text=[f"F(t+{len(det_t)}): {det_price[-1]:.2f}"],
                textposition='bottom right',
                showlegend=False
            ), row=1, col=1)

        # --- PANEL 2: RADAR TOPOLÓGICO (Espacio de Fase) ---
        # 🚨 AUDITORIA DE FÍSICA: 
        # El Atractor debe graficar el estado dinámico (x, \dot{x}).
        # Aquí, 'smooth_price' contiene los Retornos r (Velocidad del Capital).
        # El eje Y debe ser 'smooth_price'. El eje X es 'v_smooth' (Momento/Volumen).
        if r_dot is not None and v_smooth is not None:
            # Gráfico paramétrico de Velocidad (Y) vs Volumen/Aceleración (X) con Gradiente Temporal
            # Usamos el índice de tiempo para colorear la evolución
            time_array = np.arange(len(v_smooth))
            
            fig.add_trace(go.Scatter(
                x=v_smooth, 
                y=smooth_price,
                mode='lines+markers',
                line=dict(color='rgba(0, 255, 200, 0.35)', width=1.5), # Línea cyan semitransparente para marcar bien el camino
                marker=dict(
                    size=4, # Puntos ligeramente más grandes para el gradiente
                    color=time_array,
                    colorscale='Viridis', # Va de oscuro (pasado) a brillante/amarillo (presente)
                    showscale=False,
                    opacity=0.9
                ),
                name='Órbita de Fase',
                showlegend=False
            ), row=2, col=1)
            
            # Punto final (El Presente)
            t_current = t[-1]
            fig.add_trace(go.Scatter(
                x=[v_smooth[-1]], 
                y=[smooth_price[-1]],
                mode='markers',
                marker=dict(color='yellow', size=10, symbol='circle', line=dict(color='white', width=1)),
                name='Estado Actual',
                showlegend=False
            ), row=2, col=1)
            
            # --- Radar de Espacio de Fase (Panel 2) ---
            # Mostramos el atractor del régimen vigente
            if pred and 'price_percentiles' in pred and len(pred['price_percentiles']) == 5 and 'vol_percentiles' in pred and len(pred['vol_percentiles']) == 5:
                p_pred_50 = pred['price_percentiles'][2]
                v_pred_50 = pred['vol_percentiles'][2]
                
                fig.add_trace(go.Scatter(
                    x=v_pred_50, y=p_pred_50,
                    mode='lines',
                    line=dict(color='#00ffcc', width=1),
                    name='Atractor (Fase Vigente)'
                ), row=2, col=1)
                
                fig.add_trace(go.Scatter(
                    x=[v_pred_50[0]], 
                    y=[p_pred_50[0]],
                    mode='markers',
                    marker=dict(color='yellow', size=10, symbol='circle'),
                    name='Estado Actual',
                    showlegend=False
                ), row=2, col=1)

        # --- PANEL 3: SISTEMA NERVIOSO CUSUM ---
        # Acumulador Positivo
        fig.add_trace(go.Scatter(
            x=t, y=cusum_s_pos,
            mode='lines',
            name='S+ (Sobre-Reacción)',
            line=dict(color='red', width=1, dash='dot')
        ), row=2, col=2)

        # Acumulador Negativo
        fig.add_trace(go.Scatter(
            x=t, y=cusum_s_neg,
            mode='lines',
            name='S- (Sub-Reacción)',
            line=dict(color='green', width=1, dash='dot')
        ), row=2, col=2)

        # Línea de Umbral H
        fig.add_hline(y=cusum_threshold, line_dash="dash", line_color="white", 
                      annotation_text=f"Umbral (H={cusum_threshold})", row=2, col=2)

        # Marcar los índices donde hubo quiebre estructural
        if len(cusum_triggers) > 0:
            trigger_y = [max(cusum_s_pos[idx], cusum_s_neg[idx]) for idx in cusum_triggers]
            fig.add_trace(go.Scatter(
                x=t[cusum_triggers], y=trigger_y,
                mode='markers',
                name='Quiebre CUSUM',
                marker=dict(color='yellow', size=8, symbol='x'),
                showlegend=False
            ), row=2, col=2)
            
            # Muros Topológicos (V-Lines en CUSUM y Price)
            for trigger_idx in cusum_triggers:
                fig.add_vline(x=t[trigger_idx], line_dash="dash", line_color="rgba(255, 255, 0, 0.4)", row=1, col=1)
                fig.add_vline(x=t[trigger_idx], line_dash="dash", line_color="rgba(255, 255, 0, 0.4)", row=2, col=2)

        # 🚨 HOTFIX UI: Limitar Auto-Scale Y para que divergencias exponenciales no aplanen el histórico
        hist_min = float(np.min(raw_price))
        hist_max = float(np.max(raw_price))
        hist_range = hist_max - hist_min if hist_max != hist_min else abs(hist_max) * 0.1
        
        # Definimos el límite máximo y mínimo razonable del canvas (hasta +300% de la desviación histórica)
        # Si Monte Carlo o SINDy se van a la luna o al infierno, simplemente se saldrán de la pantalla y el usuario puede usar
        # el scroll del mouse para alejarse si lo desea, protegiendo siempre la visibilidad del "Presente".
        safe_y_max = hist_max + (hist_range * 1.5)
        safe_y_min = hist_min - (hist_range * 1.5)
        
        # Queremos que abarque las proyecciones (p50 y determinística) solo SI están dentro del rango safe
        visible_y_max = hist_max
        visible_y_min = hist_min
        
        if len(det_price) > 0:
            valid_det = [v for v in det_price if not np.isnan(v) and not np.isinf(v)]
            if valid_det:
                visible_y_max = max(visible_y_max, min(safe_y_max, max(valid_det)))
                visible_y_min = min(visible_y_min, max(safe_y_min, min(valid_det)))

        if len(p_price) == 5:
            valid_p50 = [v for v in p_price[2] if not np.isnan(v) and not np.isinf(v)]
            if valid_p50:
                visible_y_max = max(visible_y_max, min(safe_y_max, max(valid_p50)))
                visible_y_min = min(visible_y_min, max(safe_y_min, min(valid_p50)))
                
        plot_buffer = hist_range * 0.15
        fig.update_yaxes(range=[visible_y_min - plot_buffer, visible_y_max + plot_buffer], row=1, col=1)

        # Configuración de Layout Oscuro "Data Sci-Fi"
        fig.update_layout(
            height=600,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- NUEVO PANEL: DINÁMICA DE FUERZAS (DERIVADAS DEL SPLINE) ---
        st.markdown("### 🔍 Dinámica Newtoniana del Capital (Física del Spline)")
        
        var_name = 'P' if disable_returns else 'r'
        
        fig_derivatives = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.10,
            subplot_titles=(f"Primera Derivada (Velocidad / Momentum) - d{var_name}/dt", 
                            f"Segunda Derivada (Aceleración / Fuerza Neta) - d²{var_name}/dt²")
        )
        
        # 1. Primera Derivada Trace (Velocidad)
        fig_derivatives.add_trace(go.Scatter(
            x=t, y=r_dot,
            mode='lines',
            name=f'Velocidad (d{var_name}/dt)',
            line=dict(color='#00ffcc', width=2)
        ), row=1, col=1)
        
        # Línea de referencia cero en velocidad
        fig_derivatives.add_hline(y=0.0, line_dash="dash", line_color="rgba(255, 255, 255, 0.3)", row=1, col=1)
        
        # 2. Segunda Derivada Trace (Aceleración)
        fig_derivatives.add_trace(go.Scatter(
            x=t, y=r_dot2,
            mode='lines',
            name=f'Aceleración (d²{var_name}/dt²)',
            line=dict(color='#ff9900', width=2)
        ), row=2, col=1)
        
        # Línea de referencia cero en aceleración
        fig_derivatives.add_hline(y=0.0, line_dash="dash", line_color="rgba(255, 255, 255, 0.3)", row=2, col=1)
        
        # Añadir Muros CUSUM para correlación de quiebres en ambos gráficos de derivadas
        for trigger_idx in cusum_triggers:
            fig_derivatives.add_vline(x=t[trigger_idx], line_dash="dash", line_color="rgba(255, 255, 0, 0.4)", row=1, col=1)
            fig_derivatives.add_vline(x=t[trigger_idx], line_dash="dash", line_color="rgba(255, 255, 0, 0.4)", row=2, col=1)
            
        fig_derivatives.update_layout(
            height=500,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=False
        )
        
        fig_derivatives.update_xaxes(title_text="Tiempo (t)", row=2, col=1)
        fig_derivatives.update_yaxes(title_text=f"d{var_name}/dt", row=1, col=1)
        fig_derivatives.update_yaxes(title_text=f"d²{var_name}/dt²", row=2, col=1)
        
        st.plotly_chart(fig_derivatives, use_container_width=True)

        # --- PANEL 3: CAJA BLANCA ---
        st.markdown("### 🧮 Ecuaciones de Momentum (SINDy)")
        var_name = 'P' if disable_returns else 'r'
        
        # Régimen Activo (Vigente)
        st.markdown("#### 🟢 Régimen Activo (Vigente)")
        st.code(f"d{var_name}/dt = {active_physics_report['equations'][0]}\ndV/dt = {active_physics_report['equations'][1]}", language="text")
        st.caption(f"🚀 Spline MSE: `{active_physics_report.get('blender_mse', 0):.6f}` | "
                  f"🧠 CUSUM Latency: `{active_physics_report.get('cusum_runtime', 0):.2f}ms` | "
                  f"⚙️ SINDy R2: `{active_physics_report.get('score', 0):.4f}` "
                  f"(Terminos: `{active_physics_report.get('complexity', 0)}`) | "
                  f"📉 Ruido σ_r: `{active_physics_report.get('sigma_res_r', 0):.5f}`")
        
        # Expanders para Regímenes Históricos y Global
        with st.expander("Ver Físicas Históricas (Regímenes Anteriores)"):
            st.markdown("**Ecuación Inercial Global (Rendimiento Promediado)**")
            st.code(f"d{var_name}/dt = {physics_report_global['equations'][0]}\ndV/dt = {physics_report_global['equations'][1]}", language="text")
            st.caption(f"R2: `{physics_report_global.get('score', 0):.4f}` | Términos: `{physics_report_global.get('complexity', 0)}`")
            
            st.divider()
            for rep in reversed(regime_physics_reports[:-1]): # Todos menos el último, invertidos (el más reciente arriba)
                st.markdown(f"**{rep['regime_name']}**")
                st.code(f"d{var_name}/dt = {rep['equations'][0]}\ndV/dt = {rep['equations'][1]}", language="text")
                st.caption(f"R2: `{rep.get('score', 0):.4f}` | Términos: `{rep.get('complexity', 0)}` | 📉 Ruido σ_r: `{rep.get('sigma_res_r', 0):.5f}`")
        
        if analytical_solution:
            st.markdown("### 📜 Solución General del Atractor (Régimen Vigente)")
            st.info(analytical_solution)
