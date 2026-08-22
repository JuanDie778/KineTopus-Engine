import os
import sys
import numpy as np
import pandas as pd
import logging
from joblib import Parallel, delayed
from itertools import product
from collections import defaultdict

# Añadir root al sys.path para importar módulos core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from parameter_tuner.validation.tuner_utils import evaluate_parameter_set, load_fitness_weights
from core.ui.synthetic_generator import SyntheticMarketGenerator
from telemetry import setup_telemetry

logger = setup_telemetry()

# Silenciar logs internos
logging.getLogger('core.quant_engine.blender').setLevel(logging.ERROR)
logging.getLogger('core.quant_engine.physics').setLevel(logging.ERROR)
logging.getLogger('core.ui.synthetic_generator').setLevel(logging.WARNING)

class ParameterTuner:
    def __init__(self, mode: str = 'synthetic'):
        self.mode = mode
        self.weights = None
        
        # Cargar pesos correspondientes si no es modo clásico
        if self.mode == 'synthetic':
            weights_path = os.path.join(os.path.dirname(__file__), "..", "..", "fitness_tuner", "results", "synthetic", "fitness_equation_weights.csv")
            self.weights = load_fitness_weights(weights_path)
        elif self.mode == 'real':
            weights_path = os.path.join(os.path.dirname(__file__), "..", "..", "fitness_tuner", "results", "real", "real_fitness_equation_weights.csv")
            self.weights = load_fitness_weights(weights_path)
        
    def generate_parameter_grid(self, test_mode: bool = False):
        """Genera el espacio topológico de parámetros."""
        if test_mode:
            splines = [0.003, 0.005]
            cusums = [5.0, 15.0]
            drifts = [1.0, 3.0, 4.5]
        else:
            # Espacio Completo (~1500 comb)
            splines = np.arange(0.001, 0.007, 0.001).tolist()
            cusums = np.arange(1.0, 30.0, 3.0).tolist()
            drifts = np.arange(0.1, 5.0, 0.2).tolist()
            
        grid = list(product(splines, cusums, drifts))
        param_list = [{'spline_tol': s, 'cusum_h': c, 'drift_k': d} for s, c, d in grid]
        return param_list

    def process_universe(self, u_idx: int, df_universe: pd.DataFrame, param_grid: list):
        """Procesa todo el grid para un universo (Out-Of-Core execution)."""
        train_len = 1850
        df_train = df_universe.iloc[:train_len]
        df_val = df_universe.iloc[train_len:]
        
        logger.info(f"   [Universo {u_idx}] Iniciando Grid Search ({len(param_grid)} params) en paralelo bajo modo '{self.mode}'...")
        
        results = Parallel(n_jobs=-1, verbose=0)(
            delayed(evaluate_parameter_set)(df_train, df_val, params, self.weights, self.mode)
            for params in param_grid
        )
        
        # Filtrar fallos
        valid_results = [r for r in results if r is not None]
        logger.info(f"   [Universo {u_idx}] Completado. {len(valid_results)}/{len(param_grid)} configuraciones validas.")
        return valid_results

    def run_meta_optimization(self, num_universes: int = 10, test_mode: bool = False):
        """Bucle principal de la meta-optimización."""
        logger.info("==================================================")
        logger.info(f"INICIANDO META-OPTIMIZACIÓN DE PARÁMETROS: MODO '{self.mode.upper()}' {'(TEST MODE)' if test_mode else ''}")
        logger.info("==================================================")
        
        param_grid = self.generate_parameter_grid(test_mode)
        generator = SyntheticMarketGenerator(base_ticker='BTC-USD')
        
        # Almacenamiento maestro de scores: clave(tupla_params) -> lista de scores en cada universo
        score_tracker = defaultdict(list)
        
        for u in range(num_universes):
            logger.info(f"[+] Generando Universo Sintético {u+1}/{num_universes}...")
            df_synth = generator.generate_multiverse(num_candles=2000, mc_trajectories=300)
            
            universe_results = self.process_universe(u+1, df_synth, param_grid)
            
            for res in universe_results:
                p = res['params']
                key = (p['spline_tol'], p['cusum_h'], p['drift_k'])
                score_tracker[key].append(res['fitness_score'])
                
        # Consolidación final: Penalized Mean
        logger.info("Consolidando topología de resultados...")
        final_rankings = []
        
        for key, scores in score_tracker.items():
            if len(scores) > 0:
                mean_score = np.mean(scores)
                std_score = np.std(scores)
                penalized_mean = mean_score - std_score
                
                final_rankings.append({
                    'spline_tol': key[0],
                    'cusum_h': key[1],
                    'drift_k': key[2],
                    'mean_fitness': mean_score,
                    'std_fitness': std_score,
                    'penalized_fitness': penalized_mean,
                    'universes_survived': len(scores)
                })
                
        df_rank = pd.DataFrame(final_rankings)
        if df_rank.empty:
            logger.error("No hay resultados validos. Revisa los parametros o errores SINDy.")
            return
            
        df_rank = df_rank.sort_values(by='penalized_fitness', ascending=False)
        
        # Crear la carpeta de resultados si no existe
        output_dir = os.path.join(os.path.dirname(__file__), "..", "results", self.mode)
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, f"best_hyperparameters{'_test' if test_mode else ''}.csv")
        df_rank.to_csv(output_file, index=False)
        
        logger.info(f"Meta-Optimización Completada. Top Resultados exportados a {output_file}")
        logger.info("================TOP 3 CONFIGURACIONES================")
        for i, row in df_rank.head(3).iterrows():
            logger.info(f" -> Spline: {row['spline_tol']:.4f} | CUSUM H: {row['cusum_h']:.1f} | Drift K: {row['drift_k']:.1f} => Score: {row['penalized_fitness']:.4f}")

if __name__ == "__main__":
    import sys
    
    # Leer modo de los argumentos (--mode synthetic / real / classic)
    mode = 'synthetic'
    if '--mode' in sys.argv:
        mode_idx = sys.argv.index('--mode')
        if mode_idx + 1 < len(sys.argv):
            mode = sys.argv[mode_idx + 1]
            
    if mode not in ['synthetic', 'real', 'classic']:
        print(f"Modo '{mode}' no soportado. Usando 'synthetic'.")
        mode = 'synthetic'
        
    tuner = ParameterTuner(mode=mode)
    
    test_mode = "--test" in sys.argv
    universes = 2 if test_mode else 100
    if '--universes' in sys.argv:
        u_idx = sys.argv.index('--universes')
        if u_idx + 1 < len(sys.argv):
            try:
                universes = int(sys.argv[u_idx + 1])
            except ValueError:
                print(f"Invalid value for --universes, using default: {universes}")
    
    tuner.run_meta_optimization(num_universes=universes, test_mode=test_mode)

