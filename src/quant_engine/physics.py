import numpy as np
import pysindy as ps
import logging
import time

logger = logging.getLogger(__name__)

class PhysicsDiscoverer:
    """
    Capa 4: El núcleo descubridor (Regresión Dispersa).
    Toma las variedades diferenciables asiladas de la Capa 2 y encuentra
    la red matemática mínima (ecuaciones) que explican el vector inercial.
    Garantiza el uso de numpy (float64) y minimiza la complejidad mediante STLSQ.
    """

    def __init__(self, threshold: float = 0.05, poly_degree: int = 2):
        """
        Inicializa el descubridor de ecuaciones Físicas.
        
        Args:
            threshold (float): Factor de dispersión. Corta los coeficientes < threshold a 0.
            poly_degree (int): Máximo grado polinomial para crear combinaciones y cruces.
        """
        self.threshold = threshold
        self.poly_degree = poly_degree
        
        # Sequentially Thresholded Least Squares: Rápido y ligero en RAM (NumPy puro debajo)
        self.optimizer = ps.STLSQ(threshold=self.threshold)
        
        # Librería de Características. Ej. poly_degree=2 generará: 1, x0, x1, x0^2, x0*x1, x1^2
        self.library = ps.PolynomialLibrary(degree=self.poly_degree)
        
        self.model = ps.SINDy(
            optimizer=self.optimizer,
            feature_library=self.library,
            differentiation_method=ps.SmoothedFiniteDifference()
        )
        self.feature_names = ['P', 'V']
        
    def extract_equations(self, t: np.ndarray, x: np.ndarray, x_dot: np.ndarray, dt: float = 1.0, horizon_steps: int = 1, sigma_res_r: float = 0.0, sigma_res_v: float = 0.0, last_price: float = 1.0, disable_norm: bool = False, disable_returns: bool = False) -> dict:
        """
        Ajusta la regresión matricial descubriendo la topología dinámica.
        
        Args:
            t (np.ndarray): Vector temporal contiguo (N,)
            x (np.ndarray): Matriz 2D combinada de componentes modeladas [Retornos r_t, Volumen V_t] o en Crudo.
            x_dot (np.ndarray): Matriz 2D de derivadas analíticas [dr_dt, dV_dt] (N, 2)
            dt (float): Diferencial de tiempo para la proyección t+N (Euler temporal).
            horizon_steps (int): Cantidad de pasos a proyectar en el futuro recursivamente.
            sigma_res_r (float): Factor de difusión estocástica (ruido de Capa 2) para Retornos Logarítmicos.
            sigma_res_v (float): Factor de difusión estocástica (ruido de Capa 2) para Volumen Z-Scored.
            last_price (float): Último precio absoluto conocido, para reconstruir el Cono de Precio en t+N.
            disable_norm (bool): Si es True bypass de todas las transformaciones de escala.
            disable_returns (bool): Si es True, bypassea logs y renombra las variables en UI a "P" directo.
            
        Returns:
            dict: {
                'equations': list[str],
                'score': float,
                'complexity': int
            }
        """
        # Validación Férrea RAM / 16GB Rule
        if not (isinstance(x, np.ndarray) and isinstance(x_dot, np.ndarray) and isinstance(t, np.ndarray)):
            raise TypeError("Entradas no vectoriales detectadas. El Motor solo acepta Numpy Arrays.")
            
        if x.dtype != np.float64 or x_dot.dtype != np.float64:
             logger.warning("Auto-casting a float64 para regresión PySINDy")
             x = x.astype(np.float64)
             x_dot = x_dot.astype(np.float64)
             
        # 🚨 HOTFIX FÍSICA: Escalado Porcentual (Capa 4)
        # SINDy elimina coeficientes bajo `threshold`. Los retornos logarítmicos son muy pequeños (1e-3).
        # Multiplicamos los retornos x 100.0 (ahora son porcentajes) antes de que el motor decida descartarlos.
        # Solo escalamos la columna index=0 (los retornos r_t) y su derivada emparejada d_r/d_t
        # Modo Calibración no aplica transformaciones
        if disable_norm:
            x_scaled = x.copy()
            x_dot_scaled = x_dot.copy()
            mu_v = 0.0
            sigma_v = 1.0
        else:
            x_scaled = x.copy()
            x_dot_scaled = x_dot.copy()
            
            # Hotfix FASE 9: Escala 100x para retornos logaritmicos microscopicos. Si bypasseamos no multiplicamos la inercia del precio por 100.
            if not disable_returns:
                x_scaled[:, 0] = x[:, 0] * 100.0
                x_dot_scaled[:, 0] = x_dot[:, 0] * 100.0
            
            # Normalización Interna Fase 11: Z-Score de la inercia cruzada (Volumen)
            mu_v = np.mean(x[:, 1])
            sigma_v = np.std(x[:, 1])
            if sigma_v < 1e-8: sigma_v = 1.0
            
            x_scaled[:, 1] = (x[:, 1] - mu_v) / sigma_v
            x_dot_scaled[:, 1] = x_dot[:, 1] / sigma_v
             
        # Ajuste de Forma por si ingresan DataFrames transformados con forma (N,) en lugar de (N,1) o (N,M)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if x_dot.ndim == 1:
            x_dot = x_dot.reshape(-1, 1)

        # Regresión. Clave: inyectar x_dot y t para saltarse el diferenciador interno.
        # Bucle de Autocalibración Grid Search (Esparcidad)
        grid_thresholds = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
        best_r2_score = -np.inf
        best_model = None
        best_threshold = 0.05

        start_t = time.perf_counter()
        
        # Desactivar Ridge regularization (L2) para Sistemas Crudos Autodirigidos
        if disable_returns or disable_norm:
            self.model.optimizer.alpha = 0.0
        else:
            self.model.optimizer.alpha = 0.05
        
        for th in grid_thresholds:
            self.model.optimizer.threshold = th
            try:
                self.model.fit(x_scaled, t=t, x_dot=x_dot_scaled)
                r2_score = self.model.score(x_scaled, t=t, x_dot=x_dot_scaled)
                
                if r2_score > best_r2_score:
                     best_r2_score = r2_score
                     # Clone internal state of model by refitting since ps.SINDy doesnt deepcopy well manually
                     best_threshold = th
                     
            except Exception as e:
                 logger.warning(f"SINDy Fallido (thresh={th}): {e}. Omitiendo este n-hyperparameter...")
                 continue
                 
         # Ajustar el SINDy de forma final usando el Atractor de Mayor Significación
        self.model.optimizer.threshold = best_threshold
        self.model.fit(x_scaled, t=t, x_dot=x_dot_scaled)
        final_r2_score = best_r2_score
        equations_str = self.model.equations(precision=5)
        
        # Override names via regex replacement since pysindy might cache 'x0', 'x1' via no constructor.
        final_eqs = []
        var_name = 'P' if disable_returns else 'r'
        for eq in equations_str:
             tmp_eq = eq.replace('x0', var_name).replace('x1', 'V')
             final_eqs.append(tmp_eq)
        
        equations_str = final_eqs
        
        # Calcular Complejidad (Cant. de parámetros que sobrevivieron el umbral y no son 0)
        # El coeficient_ matrix es de forma (n_features_out, n_features_library)
        coefs = self.model.coefficients()
        complexity = np.count_nonzero(coefs)
        
        # Extrapolación Numérica Estocástica (Monte Carlo t+N) usando Euler-Maruyama (Fase 9)
        # Tomamos el último estado conocido y lo repetimos 1000 veces (usando el escalado interno)
        x_last_single = x_scaled[-1:]  # Forma (1, 2). np.float64
        x_current = np.repeat(x_last_single, 1000, axis=0)
        t_current = t[-1]
        
        predicted_path_x = []
        predicted_path_t = []
        
        # 🚨 HOTFIX FÍSICA: Kill-Switch Anti-Alucinaciones
        unstable_physics = False
        empty_r_eq = False
        
        # 1. Chequeo Global Topológico
        if final_r2_score < 0.05:
            logger.warning(f"Física Débil/Inexistente (R2={final_r2_score:.2f}). Evitando iteración Monte Carlo por Alucinación Matemática.")
            unstable_physics = True
            
        # 2. Chequeo Estricto de Componentes (Fase 13)
        if np.count_nonzero(coefs[0]) == 0:
            logger.warning("Falso Positivo Estocástico Detectado: SINDy anuló los Retornos (dr/dt = 0). Bloqueando cono divergente.")
            unstable_physics = True
            empty_r_eq = True
            
        # --- NUEVO: Extrapolación Determinística Pura (Ecuación Desnuda) ---
        # Calculamos donde apunta la ecuación sin importar si Monte Carlo decide que es inestable
        det_x_current = x_scaled[-1:] # Forma (1, 2)
        det_t_current = t[-1]
        det_path_x = []
        det_path_t = []
        
        for _ in range(horizon_steps):
            try:
                det_x_dot = self.model.predict(det_x_current)
                det_x_current = det_x_current + det_x_dot * dt
                
                # Clip preventivo extremo para que matemáticas explosivas (e^x) no rompan Python entero
                if np.any(np.isnan(det_x_current)) or np.max(np.abs(det_x_current)) > 1e10:
                    break
                    
                det_path_x.append(det_x_current[0].copy())
                det_t_current += dt
                det_path_t.append(det_t_current)
            except Exception:
                break
                
        # Consolidar trayecto determinístico
        det_price_path = []
        if len(det_path_x) > 0:
            path_det_2d = np.array(det_path_x)
            if disable_returns:
                det_price_path = path_det_2d[:, 0].tolist()
            else:
                r_pred_det = path_det_2d[:, 0] / 100.0
                cum_ret_det = np.cumsum(r_pred_det)
                det_price_path = (last_price * np.exp(cum_ret_det)).tolist()
        
        # --- Bucle predictivo Monte Carlo Analítico (t+1 a t+N) ---
        if not unstable_physics:
            for _ in range(horizon_steps):
             x_dot_pred = self.model.predict(x_current) # (1000, 2)
             x_dot_pred_last = x_dot_pred
             
             # Shock Estocástico Dual Independiente (Fase 12)
             # Rutina 1: Ruido para el canal primario
             if disable_norm:
                 noise_r = np.random.normal(0, 1, size=(1000, 1)) * sigma_res_r * np.sqrt(dt)
             else:
                 noise_r = np.random.normal(0, 1, size=(1000, 1)) * sigma_res_r * 100.0 * np.sqrt(dt)
                 noise_r = np.clip(noise_r, -3 * sigma_res_r * 100.0 * np.sqrt(dt), 3 * sigma_res_r * 100.0 * np.sqrt(dt))

             # Rutina 2: Ruido para el canal secundario
             noise_v = np.random.normal(0, 1, size=(1000, 1)) * sigma_res_v * np.sqrt(dt)
             if not disable_norm:
                 noise_v = np.clip(noise_v, -3 * sigma_res_v * np.sqrt(dt), 3 * sigma_res_v * np.sqrt(dt))

             noise = np.column_stack((noise_r, noise_v))
             
             x_current = x_current + x_dot_pred * dt + noise
             
             # Limitaciones destructivas solo se aplican a mercados financieros relativos, no a trayectorias absolutas Físicas
             if not disable_norm:
                 x_current[:, 0] = np.clip(x_current[:, 0], -10.0, 10.0)    
                 x_current[:, 1] = np.clip(x_current[:, 1], -10.0, 10.0)    
             
             t_next = t_current + dt
             
             # Criterio de Falla "Física Inestable" - Graceful Degradation (Anti Crash UI)
             if np.any(np.isnan(x_current)) or np.any(np.isinf(x_current)) or np.max(np.abs(x_current)) > 1e6:
                 unstable_physics = True
                 break
                 
             predicted_path_x.append(x_current.copy())
             predicted_path_t.append(t_next)
             t_current = t_next
             
        # Si se activó inestabilidad durante el bucle Euler
        if unstable_physics and len(predicted_path_x) > 0:
             logger.warning("Caída de Float64 detectada. Abortando predicción t+N restante.")
             predicted_path_x = []
             predicted_path_t = []
        
        # Consolidación a Tensor 3D (horizon_steps, 1000, 2)
        if not unstable_physics and len(predicted_path_x) > 0:
            path_3d = np.array(predicted_path_x)
            
            if disable_returns:
                # El estado 0 ya es el precio escalar bruto P(t). Integrado sin logs.
                price_paths = path_3d[:, :, 0]
                p_price = np.percentile(price_paths, [5, 25, 50, 75, 95], axis=1).tolist()
                
                fallback_x_next = path_3d[0, 0, :].copy()
                fallback_x_next = fallback_x_next.tolist()
            else:
                # Reconstrucción Vectorizada de Precio Absoluto (Hito 3)
                r_pred = path_3d[:, :, 0] / 100.0
                cumulative_returns = np.cumsum(r_pred, axis=0)
                
                # Proyección del precio base (Precio = P_last * exp(sum(r)))
                price_paths = last_price * np.exp(cumulative_returns)
                p_price = np.percentile(price_paths, [5, 25, 50, 75, 95], axis=1).tolist()
                
                # 🚨 HOTFIX Monte Carlo: Des-normalización del Volumen
                fallback_x_next = path_3d[0, 0, :].copy()
                fallback_x_next[0] = fallback_x_next[0] / 100.0
                fallback_x_next[1] = (fallback_x_next[1] * sigma_v) + mu_v
                fallback_x_next = fallback_x_next.tolist()
        else:
            p_price = []
            fallback_x_next = [0.0, 0.0]
            x_dot_pred_last = np.array([[0.0, 0.0]])

        prediction_t_plus_n = {
            't_path': predicted_path_t if not unstable_physics else det_path_t,
            'x_next': fallback_x_next,
            'x_dot_next': x_dot_pred_last[0].tolist(),
            'price_percentiles': p_price,
            'unstable': unstable_physics,
            'det_price_path': det_price_path,
            'det_t_path': det_path_t
        }
        
        runtime = (time.perf_counter() - start_t) * 1000.0

        return {
            'equations': equations_str,
            'score': float(final_r2_score),
            'complexity': int(complexity),
            'prediction': prediction_t_plus_n,
            'runtime_ms': float(runtime),
            'threshold_used': float(best_threshold),
            'zscore_params': {'mu_v': float(mu_v), 'sigma_v': float(sigma_v)}, # Exportamos params para la inferencia pura externa si hace falta
            'empty_r_eq': empty_r_eq
        }
