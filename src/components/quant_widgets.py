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
         analytical_solution: str = None,
         full_raw_price: np.ndarray = None,
         full_t: np.ndarray = None
    ):    
        st.markdown("### 🔭 Telemetría Físico-Predictiva")
        var_name = 'P' if disable_returns else 'r'
        # 1. Crear Subplots de alta resolución (Lienzo Único de 1000px)
        # Fila 1: Precio crudo e inercia (Colspan 2)
        # Fila 2: Primera Derivada / Velocidad (Colspan 2)
        # Fila 3: Segunda Derivada / Aceleración (Colspan 2)
        # Fila 4: Radar Atractor de Fase (Col 1) y CUSUM / Tensión (Col 2)
        fig = make_subplots(
            rows=4, cols=2, 
            specs=[
                [{"colspan": 2}, None],
                [{"colspan": 2}, None],
                [{"colspan": 2}, None],
                [{}, {}]
            ],
            shared_xaxes=False, 
            vertical_spacing=0.05,
            horizontal_spacing=0.08,
            subplot_titles=(
                "Inercia de Precio Absoluto y Cono Estocástico", 
                f"Primera Derivada (Velocidad / Momentum) - d{var_name}/dt",
                f"Segunda Derivada (Aceleración / Fuerza Neta) - d²{var_name}/dt²",
                "Atractor de Espacio de Fase (Radar)",
                "Tensión CUSUM / Quiebres (Capa 3)"
            ),
            row_heights=[0.38, 0.17, 0.17, 0.28]
        )

        # --- PANEL 1: PRECIO E INERCIA (Fila 1, Col 1) ---
        # Historial Ignorado (Fuera de la Ventana de Contexto)
        if full_raw_price is not None and full_t is not None:
             fig.add_trace(go.Scatter(
                 x=full_t, y=full_raw_price,
                 mode='lines',
                 name='Historial Ignorado (Contexto)',
                 line=dict(color='rgba(255, 255, 255, 0.1)', width=1.5),
                 showlegend=True
             ), row=1, col=1)

        # Precio Crudo (Ventana Activa - Marcadores tenues)
        fig.add_trace(go.Scatter(
            x=t, y=raw_price,
            mode='markers',
            name='Precio Crudo (Ruido)',
            marker=dict(color='rgba(150, 150, 150, 0.4)', size=4)
        ), row=1, col=1)

        # Spline Continuo (La Inercia Pura) - Fragmentado por Regímenes (Fase 8)
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

        # --- PANEL 2: PRIMERA DERIVADA (VELOCIDAD) (Fila 2, Col 1) ---
        start_idx = 0
        for i, trigger_idx in enumerate(list(cusum_triggers) + [len(t)]):
            segment_t = t[start_idx:trigger_idx]
            segment_r_dot = r_dot[start_idx:trigger_idx]
            if len(segment_t) > 0:
                color = colors[i % len(colors)]
                fig.add_trace(go.Scatter(
                    x=segment_t, y=segment_r_dot,
                    mode='lines',
                    name=f'Velocidad (d{var_name}/dt)' if i == 0 else None,
                    line=dict(color=color, width=2),
                    showlegend=False
                ), row=2, col=1)
            start_idx = trigger_idx
        
        # Línea de referencia cero en velocidad
        fig.add_hline(y=0.0, line_dash="dash", line_color="rgba(255, 255, 255, 0.3)", row=2, col=1)

        # --- PANEL 3: SEGUNDA DERIVADA (ACELERACIÓN) (Fila 3, Col 1) ---
        start_idx = 0
        for i, trigger_idx in enumerate(list(cusum_triggers) + [len(t)]):
            segment_t = t[start_idx:trigger_idx]
            segment_r_dot2 = r_dot2[start_idx:trigger_idx]
            if len(segment_t) > 0:
                color = colors[i % len(colors)]
                fig.add_trace(go.Scatter(
                    x=segment_t, y=segment_r_dot2,
                    mode='lines',
                    name=f'Aceleración (d²{var_name}/dt²)' if i == 0 else None,
                    line=dict(color=color, width=2),
                    showlegend=False
                ), row=3, col=1)
            start_idx = trigger_idx
        
        # Línea de referencia cero en aceleración
        fig.add_hline(y=0.0, line_dash="dash", line_color="rgba(255, 255, 255, 0.3)", row=3, col=1)

        # --- PANEL 4: RADAR TOPOLÓGICO (Espacio de Fase) (Fila 4, Col 1) ---
        if r_dot is not None and v_smooth is not None:
            time_array = np.arange(len(v_smooth))
            
            fig.add_trace(go.Scatter(
                x=v_smooth, 
                y=smooth_price,
                mode='lines+markers',
                line=dict(color='rgba(0, 255, 200, 0.35)', width=1.5),
                marker=dict(
                    size=4,
                    color=time_array,
                    colorscale='Viridis',
                    showscale=False,
                    opacity=0.9
                ),
                name='Órbita de Fase',
                showlegend=False
            ), row=4, col=1)
            
            # Punto final (El Presente)
            fig.add_trace(go.Scatter(
                x=[v_smooth[-1]], 
                y=[smooth_price[-1]],
                mode='markers',
                marker=dict(color='yellow', size=10, symbol='circle', line=dict(color='white', width=1)),
                name='Estado Actual',
                showlegend=False
            ), row=4, col=1)
            
            if pred and 'price_percentiles' in pred and len(pred['price_percentiles']) == 5 and 'vol_percentiles' in pred and len(pred['vol_percentiles']) == 5:
                p_pred_50 = pred['price_percentiles'][2]
                v_pred_50 = pred['vol_percentiles'][2]
                
                fig.add_trace(go.Scatter(
                    x=v_pred_50, y=p_pred_50,
                    mode='lines',
                    line=dict(color='#00ffcc', width=1),
                    name='Atractor (Fase Vigente)',
                    showlegend=False
                ), row=4, col=1)
                
                fig.add_trace(go.Scatter(
                    x=[v_pred_50[0]], 
                    y=[p_pred_50[0]],
                    mode='markers',
                    marker=dict(color='yellow', size=10, symbol='circle'),
                    name='Estado Actual',
                    showlegend=False
                ), row=4, col=1)

        # --- PANEL 5: SISTEMA NERVIOSO CUSUM (Fila 4, Col 2) ---
        # Acumulador Positivo
        fig.add_trace(go.Scatter(
            x=t, y=cusum_s_pos,
            mode='lines',
            name='S+ (Sobre-Reacción)',
            line=dict(color='red', width=1, dash='dot')
        ), row=4, col=2)

        # Acumulador Negativo
        fig.add_trace(go.Scatter(
            x=t, y=cusum_s_neg,
            mode='lines',
            name='S- (Sub-Reacción)',
            line=dict(color='green', width=1, dash='dot')
        ), row=4, col=2)

        # Línea de Umbral H
        fig.add_hline(y=cusum_threshold, line_dash="dash", line_color="white", 
                      annotation_text=f"Umbral (H={cusum_threshold})", row=4, col=2)

        # Marcar los índices donde hubo quiebre estructural
        if len(cusum_triggers) > 0:
            trigger_y = [max(cusum_s_pos[idx], cusum_s_neg[idx]) for idx in cusum_triggers]
            fig.add_trace(go.Scatter(
                x=t[cusum_triggers], y=trigger_y,
                mode='markers',
                name='Quiebre CUSUM',
                marker=dict(color='yellow', size=8, symbol='x'),
                showlegend=False
            ), row=4, col=2)

        # Muros Topológicos (V-Lines en los 4 subplots de tiempo)
        if len(cusum_triggers) > 0:
            for trigger_idx in cusum_triggers:
                fig.add_vline(x=t[trigger_idx], line_dash="dash", line_color="rgba(255, 255, 0, 0.4)", row=1, col=1)
                fig.add_vline(x=t[trigger_idx], line_dash="dash", line_color="rgba(255, 255, 0, 0.4)", row=2, col=1)
                fig.add_vline(x=t[trigger_idx], line_dash="dash", line_color="rgba(255, 255, 0, 0.4)", row=3, col=1)
                fig.add_vline(x=t[trigger_idx], line_dash="dash", line_color="rgba(255, 255, 0, 0.4)", row=4, col=2)

        # Sincronización selectiva del eje X de tiempo para los subplots temporales
        fig.update_xaxes(matches='x', row=1, col=1)
        fig.update_xaxes(matches='x', row=2, col=1)
        fig.update_xaxes(matches='x', row=3, col=1)
        fig.update_xaxes(matches='x', row=4, col=2)

        # Ocultar ticks numéricos redundantes de los subplots superiores alineados
        fig.update_xaxes(showticklabels=False, row=1, col=1)
        fig.update_xaxes(showticklabels=False, row=2, col=1)

        # Configurar títulos de los ejes para alta fidelidad de telemetría
        fig.update_xaxes(title_text="Tiempo (t)", row=3, col=1)
        fig.update_xaxes(title_text="Tiempo (t)", row=4, col=2)
        fig.update_xaxes(title_text="Volumen / Momento (V)", row=4, col=1)
        
        fig.update_yaxes(title_text="Precio Nominal", row=1, col=1)
        fig.update_yaxes(title_text=f"d{var_name}/dt", row=2, col=1)
        fig.update_yaxes(title_text=f"d²{var_name}/dt²", row=3, col=1)
        fig.update_yaxes(title_text="Retorno / Velocidad (r)", row=4, col=1)
        fig.update_yaxes(title_text="Tensión", row=4, col=2)

        # 🚨 HOTFIX UI: Limitar Auto-Scale Y del panel de precio para evitar divergencias de Monte Carlo
        hist_min = float(np.min(raw_price))
        hist_max = float(np.max(raw_price))
        hist_range = hist_max - hist_min if hist_max != hist_min else abs(hist_max) * 0.1
        
        safe_y_max = hist_max + (hist_range * 1.5)
        safe_y_min = hist_min - (hist_range * 1.5)
        
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

        # Configuración de Layout Oscuro "Data Sci-Fi" unificado
        fig.update_layout(
            height=1000,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # --- PANEL 3: CAJA BLANCA ---
        st.markdown("### 🧮 Ecuaciones de Momentum Descubiertas (SINDy)")
        var_name = 'P' if disable_returns else 'r'
        
        def _to_latex(eq_str, v_name):
            if not eq_str or eq_str.strip() in ["0", "0.0", ""]:
                return r"0"
            s = eq_str.replace("*", " ")
            s = s.replace("x0", v_name).replace("x1", "V")
            return s
        
        eq_r_active = _to_latex(active_physics_report['equations'][0], var_name)
        eq_v_active = _to_latex(active_physics_report['equations'][1], 'V')
        
        # Régimen Activo (Vigente)
        st.markdown("#### 🟢 Régimen Activo (Vigente)")
        st.latex(rf"\begin{{aligned}} \frac{{d{var_name}}}{{dt}} &= {eq_r_active} \\ \frac{{dV}}{{dt}} &= {eq_v_active} \end{{aligned}}")
        
        st.caption(f"🚀 Spline MSE: `{active_physics_report.get('blender_mse', 0):.6f}` | "
                  f"🧠 CUSUM Latency: `{active_physics_report.get('cusum_runtime', 0):.2f}ms` | "
                  f"⚙️ SINDy R2: `{active_physics_report.get('score', 0):.4f}` "
                  f"(Términos: `{active_physics_report.get('complexity', 0)}`) | "
                  f"📉 Ruido σ_r: `{active_physics_report.get('sigma_res_r', 0):.5f}`")
        
        # Expanders para Regímenes Históricos y Global
        with st.expander("📂 Ver Físicas Históricas (Regímenes Anteriores y Global)"):
            st.markdown("**Ecuación Inercial Global (Rendimiento Promediado)**")
            eq_r_glob = _to_latex(physics_report_global['equations'][0], var_name)
            eq_v_glob = _to_latex(physics_report_global['equations'][1], 'V')
            st.latex(rf"\begin{{aligned}} \frac{{d{var_name}}}{{dt}} &= {eq_r_glob} \\ \frac{{dV}}{{dt}} &= {eq_v_glob} \end{{aligned}}")
            st.caption(f"R2: `{physics_report_global.get('score', 0):.4f}` | Términos: `{physics_report_global.get('complexity', 0)}`")
            
            st.divider()
            for rep in reversed(regime_physics_reports[:-1]): # Todos menos el último, invertidos
                st.markdown(f"**{rep['regime_name']}**")
                eq_r_reg = _to_latex(rep['equations'][0], var_name)
                eq_v_reg = _to_latex(rep['equations'][1], 'V')
                st.latex(rf"\begin{{aligned}} \frac{{d{var_name}}}{{dt}} &= {eq_r_reg} \\ \frac{{dV}}{{dt}} &= {eq_v_reg} \end{{aligned}}")
                st.caption(f"R2: `{rep.get('score', 0):.4f}` | Términos: `{rep.get('complexity', 0)}` | 📉 Ruido σ_r: `{rep.get('sigma_res_r', 0):.5f}`")
        
        if analytical_solution:
            st.markdown("### 📜 Solución General del Atractor (Régimen Vigente)")
            st.info(analytical_solution)

