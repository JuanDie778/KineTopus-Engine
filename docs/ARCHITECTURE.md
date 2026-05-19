# ARQUITECTURA TÉCNICA v2.0 (Motor Quant Estocástico)

## 🏗️ STACK TECNOLÓGICO
* **Core:** Python 3.10+
* **Interfaz (Frontend):** Streamlit + Plotly (Dashboards Cuantitativos Dinámicos).
* **Motor Quant / Física Computacional:**
    * *Cálculo Vectorial:* `numpy` (Obligatorio float64 contiguo en RAM, prohibido Pandas en hot loops) y `scipy` (FFT y AAA).
    * *Descubrimiento Físico:* `pysindy` (Sparse Identification of Nonlinear Dynamics).
    * *Simulación Estocástica:* Euler-Maruyama (Integración Numérica).
    * *Asincronía y Paralelismo:* `asyncio` y `concurrent.futures` (ProcessPoolExecutor).
* **Integración de Mercados:** `yfinance` para ingesta de datos bajo demanda desde Yahoo Finance.
* **Validación Estadística:** `statsmodels` (Test de Ljung-Box para confirmación de Ruido Blanco y evaluación residual).

## 🔄 PIPELINE DE DATOS (El Flujo Dinámico)

El Motor Cuantitativo actúa como un "Telescopio de Caja Blanca" sobre los mercados financieros. 

### CAPA 1: INGESTA Y NORMALIZACIÓN
1.  **Market Loader:** `yfinance` descarga datos históricos (`P`, `V`).
2.  **Z-Score:** El volumen es normalizado a desviaciones estándar globales y el Precio es centrado a $0.0$ relativo a su ventana para evitar colapso matemático en escalares.

### CAPA 2: SENSOR ESPECTRAL (Transformada de Fourier)
*   **Componente:** `SpectralAnalyzer`.
*   **Misión:** Descubrir las frecuencias de oscilación dominante del mercado (Ej. Ciclo de 14 días en el SPY).
*   **Mecánica:** Scipy FFT encuentra las 2 frecuencias top (Top-K) para informar a la capa topológica sobre cómo de ancha debe ser su ventana de suavizado inercial.

### CAPA 3: MOLDEADOR TOPOLÓGICO (Splines Robustos)
*   **Componente:** `ContinuousBlender`.
*   **Misión:** Crear una versión continua infinita diferencial del Mercado discreto.
*   **Mecánica:** Utiliza UnivariateSpline para construir una sábana $C^2$ matemática. La Tolerancia de Aislamiento define cuánto ruido errático desechamos vs cuánta fidelidad a la vela pura mantenemos. Se extraen $\dot{P}$ y $\dot{V}$ de manera puramente analítica.

### CAPA 4: SISTEMA NERVIOSO (CUSUM)
*   **Componente:** `RegimeShiftDetector`.
*   **Misión:** Alertar de roturas estructurales inesperadas.
*   **Mecánica:** Vigila la diferencia entre el Precio Crudo y la Inercia del Spline. Si el error estandarizado rompe un umbral $H$, emite un gatillo (línea vertical en el Dashboard), indicando que el mercado cambió su naturaleza fundamental en ese instante.

### CAPA 5: EXTRACCIÓN FÍSICA Y SIMULACIÓN MONTE CARLO (SINDy + EM)
*   **Componente:** `PhysicsDiscoverer`.
*   **Misión:** Descubrir las Leyes Ecuacionales y proyectarlas al futuro con incertidumbre.
*   **Mecánica:**
    1.  **Regresión Rala:** SINDy con STLSQ encuentra ecuaciones simplificadas de la forma $\dot{P} = f(P, V)$.
    2.  **Euler-Maruyama:** Se inyecta la volatilidad residual ($\sigma_{res}$) del Moldeador Topológico.
    3.  **Monte Carlo Vectorizado:** Se lanzan 1,000 caminos al futuro resolviendo la Ecuación Diferencial en paralelo.
    4.  **Agregación Percentil:** Para no colapsar la memoria, la salida bruta (1,000 caminos) se comprime en percentiles $P_5, P_{25}, P_{50}, P_{75}, P_{95}$, formando el Cono Estocástico visualizado en Plotly.
    5.  **Graceful Degradation:** Cortafuegos matemático que detecta derivadas caóticas superiores a $1e6$ abortando la integración para limpiar la UI y evitar Crashes en la memoria local estricta.

### CAPA 6: DASHBOARD Y TOPOLOGÍA VISUAL (Radar Quant)
*   **Componente:** `StreamlitDashboard` / `PlotlyView`.
*   **Misión:** Renderizar la física del mercado en tiempo real y exponer la Venta Asimétrica.
*   **Mecánica:** Muestra el Precio Real con su Cono de Incertidumbre reconstruido, además de un componente interactivo de "Atractor de Espacio de Fase" (Velocidad vs Aceleración) para detectar asimetrías y rupturas topológicas visualmente.
