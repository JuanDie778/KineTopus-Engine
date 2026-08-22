import numpy as np
import pandas as pd
import logging
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

class FitnessLearner:
    """
    Aprendiz de Ecuación Fitness (Meta-Optimizador).
    Utiliza Regresión Ridge (L2 penalizada) para mapear las características 
    In-Sample a los resultados Out-of-Sample.
    """
    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha, fit_intercept=True)
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.weights = None
        self.intercept = None
        self.feature_names = ['r2_train', 'complexity', 'regimes', 'regimes_sq']

    def compute_composite_target(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calcula la métrica combinada Y (Target) basada en el consenso.
        Y = (MCC * 0.4) + (Profit_Factor * 0.4) - (Max_Drawdown * 0.2)
        """
        mcc = df.get('mcc_test', pd.Series(np.zeros(len(df)))).fillna(0.0).values
        # Profit Factor: recortado defensivamente para evitar outliers extremos (ej: infinito)
        pf = df.get('profit_factor', pd.Series(np.ones(len(df)))).fillna(1.0).values
        pf = np.clip(pf, 0.0, 10.0)
        
        # Max Drawdown: esperado en decimales (ej: 0.15 para 15%). 
        # Si viene en escala porcentual (ej: 15.0), lo dividimos por 100 si excede 1.0
        mdd = df.get('max_drawdown', pd.Series(np.zeros(len(df)))).fillna(0.0).values
        if np.max(mdd) > 1.0:
            mdd = mdd / 100.0
        mdd = np.clip(mdd, 0.0, 1.0)

        # Target Compuesto (Consenso)
        # Escalamos el Profit Factor restando 1.0 (para que un PF neutro de 1.0 valga 0)
        y_target = (mcc * 0.4) + ((pf - 1.0) * 0.4) - (mdd * 0.2)
        return y_target

    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extrae y construye las características X In-Sample.
        """
        r2 = df.get('r2_train', pd.Series(np.zeros(len(df)))).fillna(0.0).values
        comp = df.get('complexity', pd.Series(np.zeros(len(df)))).fillna(0.0).values
        reg = df.get('regimes', pd.Series(np.zeros(len(df)))).fillna(0.0).values
        
        # Característica Polinómica: Castigar extremos de regímenes
        reg_sq = reg ** 2
        
        X = np.column_stack([r2, comp, reg, reg_sq])
        return X

    def fit(self, df_results: pd.DataFrame) -> dict:
        """
        Entrena el modelo Ridge con la matriz de resultados de un universo.
        Devuelve el diccionario de pesos.
        """
        if df_results.empty or len(df_results) < 5:
            logger.warning("Pocos datos para entrenar el Meta-Optimizador.")
            return {}

        # 1. Preparar datos
        X = self.extract_features(df_results)
        Y = self.compute_composite_target(df_results)

        # 2. Escalar características (Crucial para regresiones lineales y comparación justa de pesos)
        X_scaled = self.scaler_x.fit_transform(X)
        Y_scaled = self.scaler_y.fit_transform(Y.reshape(-1, 1)).flatten()

        # 3. Entrenar
        self.model.fit(X_scaled, Y_scaled)

        self.weights = self.model.coef_
        self.intercept = self.model.intercept_
        
        # Evaluar qué tan bien las métricas IS predicen las OOS
        r2_fit = self.model.score(X_scaled, Y_scaled)

        # Empaquetar pesos
        weights_dict = {name: float(w) for name, w in zip(self.feature_names, self.weights)}
        weights_dict['intercept'] = float(self.intercept)
        weights_dict['meta_r2'] = float(r2_fit)
        
        logger.info(f"[FitnessLearner] Pesos aprendidos: {weights_dict} | R2_Meta: {r2_fit:.3f}")
        return weights_dict

    @staticmethod
    def aggregate_multiverse_weights(weights_list: list) -> dict:
        """
        Promedia los pesos aprendidos de todos los multiversos.
        """
        if not weights_list:
            return {}
            
        aggregated = {}
        keys = weights_list[0].keys()
        for k in keys:
             valid_vals = [w[k] for w in weights_list if w.get(k) is not None]
             aggregated[k] = np.mean(valid_vals) if valid_vals else 0.0
             
        return aggregated
