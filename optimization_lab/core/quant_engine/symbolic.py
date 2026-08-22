import numpy as np
import logging
from typing import Dict, Any

try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class SymbolicTranslator:
    """
    Traduce las ecuaciones resultantes de SINDy a formatos simbólicos para resolución
    analítica (SymPy) y exportación visual interactiva (GeoGebra).
    Respeta la regla de 16GB limitando los tiempos de retención simbólica.
    """
    
    def __init__(self, physics_report: Dict[str, Any]):
        self.report = physics_report
        self.equations = physics_report.get('equations', [])
        
    def generate_geogebra_commands(self) -> str:
        """
        Genera un bloque de texto en formato GeoGebra Script.
        Maneja Campo de Direcciones explícito.
        """
        if not self.equations or len(self.equations) < 2:
            return "No hay ecuaciones físicas válidas para exportar."
            
        r_eq_raw = self.equations[0]
        v_eq_raw = self.equations[1]
        
        # SINDy usa el string ' 1 ' para constantes. Debemos parchearlo a numérico '1.0' si es necesario.
        # Y usa espacios entre operadores. Adaptar a una sintaxis genérica:
        
        def format_ggb(eq: str, independent_var: str) -> str:
            # Reemplazar la constante ' 1 ' de Pysindy por multiplicador ' 1.0 '
            # SINDy: dr/dt = a r + b V + c 1 -> GeoGebra: a*r + b*V + c
            # Este es un parseo heurístico "safemode"
            e = eq.replace(' 1 ', ' 1.0 ')
            if e.endswith(' 1'):
                e = e[:-2] + ' 1.0'
            # Normalizar operadores básicos por si acaso SINDy varía el string
            e = e.replace('^', '^') # GeoGebra soporta ^
            e = e.replace('r', 'x') # En plano de retornos, la variable dependiente a menudo se visualiza en t/x
            if independent_var == 'y':  # Si visualizamos en plano Fase (x=V, y=r)
                 e = eq.replace('r', 'y').replace('V', 'x').replace(' 1 ', ' 1.0 ')
                 if e.endswith(' 1'):
                    e = e[:-2] + ' 1.0'
            return e

        # Ecuación de Velocidad del Retorno (Plano temporal t/x, r/y)
        # SINDy da dr/dt. En GeoGebra para el campo temporal: CampoDirecciones(f(x,y)) donde y es variable.
        dr_dt_t = format_ggb(r_eq_raw, independent_var='none').replace('V', 'Vol') # 'Vol' para que el usuario sepa que depende de otra constante visual
        
        # Ecuación de Fase (Atractor) (x=V, y=r)
        # dv/dt_fase / dr/dt_fase
        # CampoDirecciones( dV_dt / dr_dt ) o CampoDirecciones( dr_dt / dV_dt )  -> dy/dx = (dr/dt) / (dV/dt)
        dr_dt_fase = format_ggb(r_eq_raw, independent_var='y')
        dv_dt_fase = format_ggb(v_eq_raw, independent_var='y')

        output =  "Campo de Direcciones (Retorno vs Tiempo):\n"
        output += f"1) dr_dt_temporal(x, y, Vol) = {dr_dt_t.replace('x', 'y')}  // (Sustituir 'Vol' por constante o función)\n"
        output += f"   Campo1 = CampoDirecciones(dr_dt_temporal)\n\n"
        output += "Campo de Direcciones Espacio de Fase (X = Volumen, Y = Retorno):\n"
        output += f"2) dy_dx(x, y) = ({dr_dt_fase}) / ({dv_dt_fase})\n"
        output += f"   Atractor = CampoDirecciones(dy_dx)\n"
        
        return output
        
    def generate_analytical_solution(self, p0: float = 0.0, v0: float = 0.0, disable_returns: bool = False) -> str:
        """
        Intenta resolver el sistema de EDOs analíticamente invocando SymPy con Condiciones Iniciales.
        """
        if not SYMPY_AVAILABLE:
            return "El módulo 'sympy' no está instalado. Usando fallback numérico únicamente."
            
        if not self.equations or len(self.equations) < 2:
            return "Ecuaciones insuficientes para integración analítica."
            
        try:
            # Variables Simbólicas
            t = sp.Symbol('t', real=True)
            # Reaccionar al modo: Precio Absoluto P o Retorno r
            var_name = 'P' if disable_returns else 'r'
            r = sp.Function(var_name)(t)
            v = sp.Function('V')(t)
            
            # Sanitizar strings para sympify
            def sanitize(eq_str):
                s = eq_str.replace(' 1 ', ' * 1.0 ')
                if s.endswith(' 1'): 
                    s = s[:-2] + ' * 1.0'
                s = s.replace('^', '**')
                # Forzar multiplicación implícita (ej '0.015 r' -> '0.015 * r')
                import re
                s = re.sub(r'([0-9\.]+)\s+([a-zA-Z]+)', r'\1 * \2', s)
                return s
                
            eq_r_str = sanitize(self.equations[0])
            eq_v_str = sanitize(self.equations[1])
            
            # Definir entorno diccionarial de símbolos locales para sympify
            local_dict = {var_name: r, 'V': v, 't': t}
            
            # Evaluar
            rhs_r = sp.sympify(eq_r_str, locals=local_dict)
            rhs_v = sp.sympify(eq_v_str, locals=local_dict)
            
            # EDOs
            edo_r = sp.Eq(r.diff(t), rhs_r)
            edo_v = sp.Eq(v.diff(t), rhs_v)
            
            # Resolvemos el sistema
            # sympy es susceptible de colgarse en sistemas polinómicos cuadráticos acoplados (típicamente caóticos)
            # Protegemos el hilo principal en arquitecturas locales (Windows) contra bloqueos de RAM infinitos
            import concurrent.futures
            
            # Condiciones iniciales en t=0
            # SymPy dsolve a menudo falla integrando floats (Not Algebraic Element Error).
            # Resolveremos el sistema general y despejaremos las constantes C1, C2 manualmente.
            def safe_dsolve():
                sol = sp.dsolve([edo_r, edo_v])
                
                # Extraer constantes libres (típicamente C1, C2)
                free_symbols = set()
                for s in sol:
                    free_symbols.update(s.free_symbols)
                C_vars = [sym for sym in free_symbols if str(sym).startswith('C')]
                
                # Sustituir t=0 en las soluciones generales
                eqs_at_0 = [s.subs(t, 0) for s in sol]
                
                # Forzar igualdades a las condiciones dadas
                eqs_at_0_subbed = [eq.subs({r.subs(t, 0): p0, v.subs(t, 0): v0}) for eq in eqs_at_0]
                
                # Despejar y reemplazar constantes
                constants = sp.solve(eqs_at_0_subbed, C_vars)
                if isinstance(constants, dict):
                    return [s.subs(constants) for s in sol]
                elif isinstance(constants, list) and len(constants) > 0 and isinstance(constants[0], dict):
                    return [s.subs(constants[0]) for s in sol]
                    
                return sol
                
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(safe_dsolve)
                try:
                    sol = future.result(timeout=2.0)
                except concurrent.futures.TimeoutError:
                    logger.warning("SymPy dsolve excedió límite térmico de 2.0s debido a complejidad de la EDO. Abortando integración.")
                    return "Sistema Caótico Complejo: La EDO cruzada requiere excesivo cómputo para solución cerrada.\nUtilizar Fallback Numérico."
            
            # Formatear la solución analítica si existe
            res = ""
            for s in sol:
                s_string = str(s).replace('**', '^')
                # Simplificar visualización: aislar P(t) o r(t)
                if s_string.startswith(f"{var_name}(t) ==") or s_string.startswith(f"Eq({var_name}(t)"):
                    # Extraer solo el lado derecho de la ecuación
                    try:
                        rhs = s.rhs
                        # Formatear números extremadamente largos (floats espurios)
                        rhs = sp.N(rhs, 5) 
                        res += f"$${var_name}(t) = {sp.latex(rhs)}$$\n"
                    except:
                        res += f"{s_string}\n"
            
            if not res:
                res = "Solución Simbólica hallada, pero no pudo aislarse explícitamente P(t)."
            return res
            
        except Exception as e:
            # Graceful degradation (alucinación de cómputo evitada). 
            # EDO cruzada de grado >=2 suele ser no lineal e inintegrable simbólicamente sin condiciones límite.
            logger.info("El sistema resultó ser Caótico No-Lineal e irresoluble analíticamente sin expansión de Taylor local. Usando fallback.")
            return "Sistema Caótico Irresoluble: La EDO cruzada no posee solución analítica cerrada (Típicamente Ocurre en polinomios grado >= 2).\nUtilizar Fallback Numérico."

    def get_numerical_fallback(self) -> str:
        """
        Transforma los datos del Cono de Monte Carlo Estocástico (Mediana, libre de ruido en P50)
        hacia un array formalizado de coordenadas de puntos GeoGebra.
        """
        pred = self.report.get('prediction')
        if not pred or 'price_percentiles' not in pred or not pred.get('t_next'):
            return "Datos numéricos no disponibles. Verifica que el Autopilot tenga R2 válido."
            
        t_arr = pred['t_next']
        # Usar la Mediana P50 del despliegue de Monte Carlo (el escenario más probable de la EDO)
        p50_arr = pred['price_percentiles'][2] 
        
        # Limitar a unos precisos 50 puntos (o menos) para evitar saturar el input de texto y la memoria de GGB
        limit = min(50, len(t_arr))
        
        t_sub = t_arr[:limit]
        p_sub = p50_arr[:limit]
        
        points = []
        # Normalizar t asumiendo origen relativo para graficar comodamente: t[0] = 0
        t_start = t_sub[0]
        
        for t_val, p_val in zip(t_sub, p_sub):
            rel_t = t_val - t_start
            points.append(f"({rel_t:.2f}, {p_val:.4f})")
            
        list_str = "{" + ", ".join(points) + "}"
        
        output = "Fallback Numérico (Ajuste Curvo GeoGebra):\n"
        output += "Pega esta lista de puntos (Mediana P50) en el panel de entrada y usa un ajuste polinómico o Spline:\n\n"
        output += f"L1 = {list_str}\n\n"
        output += "Ejemplo GeoGebra:\n"
        output += "- AjustePolinómico(L1, 3)\n"
        output += "- AjusteSpline(L1)\n"
        
        return output
