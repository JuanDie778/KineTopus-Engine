# KINETOPUS ENGINE - ROADMAP

**VISIÓN ACTUAL:** Motor Cuantitativo de "Caja Blanca" ejecutado 100% en local (16GB RAM / Ryzen 7). El algoritmo extrae topologías ocultas en series de tiempo financieras combinando Procesamiento de Señales (FFT), Sistemas Control (CUSUM) y Descubrimiento Físico Dinámico (SINDy) mapeado sobre simulaciones Estocásticas de Monte Carlo (Euler-Maruyama).

---

## 🚀 FASE 9: REFACTORIZACIÓN A FÍSICA DE MOMENTUM Y RADAR TOPOLÓGICO
**ESTADO:** 🛠️ EN DESARROLLO
**META:** Evolucionar el motor determinista de Precio Absoluto hacia un modelo de Fuerzas y Aceleración basado en Retornos Logarítmicos y Espacio de Fase.

### Hitos Estratégicos (Para @BUILDER):
- [ ] **1. Ingesta y Cálculo de Momento:** Modificar el preprocesamiento para que calcule la serie de Retornos Logarítmicos $r_t = \ln(P_t / P_{t-1})$. 
   * 🚨 **REGLA INQUEBRANTABLE DE RAM (16GB):** El cálculo debe hacerse estrictamente de forma vectorizada usando `np.log()` puro de **NumPy**. Queda terminantemente prohibido usar `pandas` dentro del hot loop.
- [ ] **2. Ajuste SINDy:** La capa de descubrimiento debe cambiar su input. El optimizador SR/SINDy ahora buscará la ecuación del momentum: $\dot{r} = f(r, V)$, capturando la interacción $r \cdot V$.
- [ ] **3. Reconstrucción del Precio:** El predictor Monte Carlo resolverá la integración sobre los retornos simulados. Al terminar, de forma vectorizada, deberá reconstruir el Precio Absoluto mediante composición exponencial para renderizar el "Cono Estructural" final.
- [ ] **4. Radar Topológico:** Implementar en Plotly (Capa 6) el nuevo "Atractor de Espacio de Fase", ploteando Velocidad ($Y$) vs. Aceleración/Volumen ($X$) identificando los bucles de equilibrio visualmente.

---

## 🚀 FASE 10: PRODUCTIZACIÓN Y EMPAQUETADO LIGERO (MODO CASCADA)
**ESTADO:** 🛠️ EN DESARROLLO (Limpieza de Arquitectura Antigravedad)
**META:** Depurar todo el código del MVP 1.0 (Sandbox y explorador de Datasets estilo Chatbot genérico) para focalizar el sistema completamente como una Estación Financiera de Ingeniería Inversa.

### HITO 10: Limpieza de Deuda Técnica (Option A Excluyente)
- [x] Agrupar documentación pasada en `docs/archive/HISTORIC_PLAN.md`.
- [x] Eliminar dependencias/paquetes sobrantes estilo Data Profiling o Chat Sandbox en `app.py`.
- [x] Eliminar carpetas innecesarias: `src/data_engine` y `src/utils`.
- [x] Refactorizar `app.py` para levantar directamente el modo Quant Ticker como predeterminado (single-purpose app) e integrar la carga CSR directamente al Motor Estocástico sin bifurcación condicional antigua.

## 🚀 FASE 11: MULTITENSOR QUANTITATIVO (FUTURO)
**ESTADO:** 📋 EN IDEACIÓN
**META:** Permitir que el motor ingiera no solo precio y volumen, sino un tensor N-Dimensional (ej: agregando RSI, MACD o Volatilidad de Opciones).
- Explorar cómo SINDy maneja matrices tridimensionales $(N, 3)$ para encontrar relaciones del tipo $\dot{P} = P - MACD^2$.

---

## 🚀 FASE 12: LABORATORIO DE OPTIMIZACIÓN Y AUTOTUNING GLOBAL (MODO ACTIVE)
**ESTADO:** 🛠️ EN DESARROLLO (Optimization Lab)
**META:** Implementar un pipeline de entrenamiento hiperparamétrico para KineTopus que maximice la métrica True Directional Alpha (TDA) y el Matthews Correlation Coefficient (MCC) mitigando la paradoja de la tasa base (Base Rate Fallacy).

- [x] **1. Corrección del Bug en PredictiveAutoTuner:** Resolver el KeyError crítico en `src/quant_engine/auto_tuner_predictive.py` (`predicted_prices` vs `det_price_path`).
- [x] **2. Formulación de la Función de Fitness TDA:** Diseñar e implementar `define_tda_objective_function` penalizando over-fitting, complejidad no-lineal y mortalidad matemática en `optimizer_pipeline.py`.
- [x] **3. Implementación del Optimizador (Search Space Explorer):** Crear un algoritmo de búsqueda robusto y eficiente en memoria (Random/Grid Search o Algoritmo Genético) optimizado para 16GB RAM.
- [x] **4. Validación Cruzada Out-of-Sample:** Diseñar la estrategia de separación In-Sample (Entrenamiento) y Out-of-Sample (Validación/Test) para batir las métricas de `3_Analytics_and_Discovery.ipynb` sin Data Leakage.
- [x] **5. Laboratorio de Análisis de Validación:** Crear el notebook interactivo `6_AutoTuner_Validation_Analysis.ipynb` para comparar el Auto-Tuner Clásico con el Meta-Optimizer entrenado sintética y realmente.