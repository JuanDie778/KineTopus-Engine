import logging
import gc
import pandas as pd
import numpy as np
from typing import List, Callable, Any

logger = logging.getLogger(__name__)

class DriftOptimizer:
    """
    Optimizador 1D avanzado que utiliza la Búsqueda de la Sección Dorada (Golden Section Search).
    Encuentra el máximo de la función de Fitness (drift-k óptimo) de manera iterativa,
    estrechando el intervalo de búsqueda basándose en la proporción áurea, lo que
    ahorra drásticamente tiempo de cómputo frente a la Fuerza Bruta.
    """
    
    def __init__(self, min_drift: float = 0.1, max_drift: float = 5.0, tol: float = 0.2):
        self.a = min_drift
        self.b = max_drift
        self.tol = tol
        self.phi = (np.sqrt(5) - 1) / 2  # ~0.618
        self.results = []
        
    def _evaluate_k(self, k_val: float, df_universes: List[pd.DataFrame], eval_func: Callable) -> float:
        """ Evalúa un valor específico de k en todos los universos y registra los resultados. """
        k_val = round(float(k_val), 2)
        logger.info(f"   ▶ Evaluando Drift-k: {k_val:.2f}")
        
        universe_scores = []
        metrics_acc = []
        
        for u_idx, df_universe in enumerate(df_universes):
            try:
                score, metrics = eval_func(k_val, df_universe)
                universe_scores.append(score)
                metrics_acc.append(metrics)
            except Exception as e:
                logger.error(f"Fallo en Universo {u_idx} con k={k_val}: {e}")
                universe_scores.append(-999.0)
                metrics_acc.append({})
        
        # Agregación Global
        avg_score = np.mean(universe_scores)
        avg_mcc = np.mean([m.get('mcc_test', 0) for m in metrics_acc])
        avg_alpha = np.mean([m.get('alpha_edge', 0) for m in metrics_acc])
        avg_r2 = np.mean([m.get('r2_train', 0) for m in metrics_acc])
        avg_regimes = np.mean([m.get('regimes', 0) for m in metrics_acc])
        
        self.results.append({
            'drift_k': k_val,
            'fitness': avg_score,
            'MCC': avg_mcc,
            'Alpha_Edge': avg_alpha,
            'R2_SINDy': avg_r2,
            'N_Regimes': avg_regimes
        })
        
        del universe_scores, metrics_acc
        gc.collect()
        
        logger.info(f"   Score: {avg_score:.2f} | MCC: {avg_mcc:.2f}")
        return avg_score
        
    def run_search(self, df_universes: List[pd.DataFrame], eval_func: Callable) -> pd.DataFrame:
        """
        Ejecuta el Golden Section Search para maximizar la función Fitness.
        """
        logger.info(f"[DriftOptimizer] Iniciando Búsqueda de Sección Dorada en intervalo [{self.a}, {self.b}]")
        
        a, b = self.a, self.b
        
        # Puntos iniciales
        c = b - self.phi * (b - a)
        d = a + self.phi * (b - a)
        
        fc = self._evaluate_k(c, df_universes, eval_func)
        fd = self._evaluate_k(d, df_universes, eval_func)
        
        iteration = 1
        while abs(b - a) > self.tol:
            logger.info(f"   --- Iteración {iteration}: Bracket actual [{a:.2f}, {b:.2f}] ---")
            
            if fc > fd:
                # El máximo está entre a y d
                b = d
                d = c
                fd = fc
                c = b - self.phi * (b - a)
                fc = self._evaluate_k(c, df_universes, eval_func)
            else:
                # El máximo está entre c y b
                a = c
                c = d
                fc = fd
                d = a + self.phi * (b - a)
                fd = self._evaluate_k(d, df_universes, eval_func)
                
            iteration += 1
            
        optimal_k = (b + a) / 2
        logger.info(f"[DriftOptimizer] Convergencia alcanzada. Óptimo estimado en Drift-k = {optimal_k:.2f}")
        
        df_results = pd.DataFrame(self.results)
        df_results = df_results.sort_values(by='fitness', ascending=False).reset_index(drop=True)
        return df_results
