# SPEC_APP: Kinetopus Engine (Motor Quant de Descubrimiento Dinámico)

Este documento es el **Plano de Ingeniería** del proyecto. Está diseñado para que cualquier mantenedor futuro comprenda la arquitectura de software, el flujo de ejecución (pipeline) y las restricciones estrictas del hardware.

## 1. Objetivo Técnico
El Kinetopus Engine no es una "Caja Negra" estadística. Es un pipeline de física computacional estructurado para transformar datos discretos y ruidosos (velas) en variedades topológicas continuas, con el fin de extraer **Ecuaciones Diferenciales explícitas (EDO)** mediante SINDy.
Todo el sistema está diseñado bajo el principio **Local-First**, priorizando la eficiencia de la memoria (tope de 16GB RAM) y usando cálculos matriciales puros.

---

## 2. Arquitectura de Directorios (El Mapa del Código)

El proyecto separa de forma estricta la **Interfaz de Usuario (UI)** de la **Lógica Matemática (Engine)**.

*   `app.py`: **Punto de Entrada y Orquestador.** Recoge los parámetros del usuario vía Streamlit y ejecuta secuencialmente las capas matemáticas del motor. Contiene el bucle principal ("Hot Path") y el sistema de rescate automático (Autopilot Quant).
*   `src/ui/`: **Gestión de Interfaz e Ingesta.**
    *   `market_loader.py`: Sanitiza los datos crudos (`yfinance`), maneja NaNs y convierte los Precios Absolutos en **Retornos Logarítmicos** centrándolos alrededor de cero (vital para la estabilidad matricial). Aplica Z-Score al volumen.
    *   `styles.py`: Contiene el CSS personalizado de la aplicación.
*   `src/quant_engine/`: **El Cerebro Matemático (Core).**
    *   `sensor.py` (Capa 1): Utiliza `np.fft.fft` (Scipy/Numpy) para encontrar las frecuencias respiratorias dominantes del activo.
    *   `blender.py` (Capa 2): Utiliza Splines Univariados para convertir datos discretos en funciones $C^2$ continuas y extraer derivadas analíticas ($\dot{r}, \dot{V}$).
    *   `nervous.py` (Capa 4): Filtro CUSUM. Compara la inercia calculada con la realidad para detectar rupturas estructurales en la serie.
    *   `physics.py` (Capa 3 y 5): Ejecuta **PySINDy** para encontrar la EDO ($\dot{r} = f(r, V)$). Además, contiene el proyector probabilístico (Integrador de Euler-Maruyama) para el Monte Carlo.
    *   `symbolic.py`: (Opcional) Traduce las ecuaciones ralas a lenguaje natural.
*   `src/components/`: **Visualización.**
    *   `quant_widgets.py`: Genera los gráficos interactivos mediante `Plotly` (Radar de Espacio de Fase, Cono Estocástico).

---

## 3. Pipeline de Ejecución (El Flujo en `app.py`)

Cuando el usuario pulsa "Calcular", el código en `app.py` sigue este flujo determinista:

1.  **Ingesta (Market Loader):** Los datos (Precio y Volumen) se convierten en matrices continuas de numpy (`np.float64`).
2.  **Sensor (FFT):** Se detectan los 2 periodos principales (Top-K=2) para informar la rigidez del moldeado posterior.
3.  **Moldeador (Splines):** Se ajusta la tolerancia (ej. $0.001$) para planchar la curva sin matar el alpha del activo.
4.  **Sistema Nervioso (CUSUM Slicing):** El detector corta la línea de tiempo total en distintos **"Regímenes"**. Si el mercado cambió bruscamente hace 20 días, CUSUM crea una frontera virtual ahí.
5.  **Descubrimiento (SINDy Local y Global):** 
    *   El motor no saca una sola ecuación para toda la historia.
    *   Itera tramo por tramo (basado en CUSUM) calculando la física local de cada régimen.
    *   **Proyección:** Únicamente el último régimen activo se proyecta al futuro usando el motor estocástico (Monte Carlo).
6.  **Autopilot Quant (Rescate 2D):** Si SINDy devuelve una ecuación caótica (inestable), `app.py` aborta la renderización, baja drásticamente la tolerancia del Spline y recalcula la EDO solo para el régimen actual intentando salvar la predicción (Graceful Degradation).

---

## 4. Lógica Matemática Detallada

### 4.1. Algoritmo de Suavizado Topológico (AAA/Splines)
Convierte la serie discreta en una función continua para evitar derivadas infinitas entre ticks.
*Restricción de software:* La Tolerancia de error debe estar entre $10^{-3}$ y $10^{-5}$. Un spline muy rígido mata la señal; uno muy suelto rompe la matriz SINDy.

### 4.2. Detección de Ruptura (CUSUM)
Supervisa el residuo fuera de muestra normalizado ($z_t$). Las sumas acumuladas detectan el cambio de régimen:
$$S^+_t=\max(0,S^+_{t-1}+z_t-k)$$
$$S^-_t=\max(0,S^-_{t-1}-z_t-k)$$
Si $S > H$, se activa el recalculo matemático creando un nuevo segmento temporal.

### 4.3. SINDy (Regresión Rala - STLSQ)
Plantea la regresión matricial para hallar los coeficientes activos $\Xi$:
$$\dot{X}=\Theta(X)\Xi$$
Donde $X=[r,V]^T$ y $r$ es el **Retorno Logarítmico**. Buscamos la ecuación diferencial del momentum: $\dot{r} = f(r, V)$.
*Restricción de software:* Uso de `STLSQ` y `PolynomialLibrary(degree=2)` para capturar la fricción cruzada $r \cdot V$ sin colapsar la RAM con matrices multidimensionales hiperbólicas.

### 4.4. Proyector Estocástico (Euler-Maruyama)
Para transicionar de una predicción determinista a una probabilística, simulamos 1,000 caminos independientes vectorizados:
$$r_{t+1} = r_t + f_{SINDy}(r_t, V_t) \cdot \Delta t + \sigma_{res} \cdot \sqrt{\Delta t} \cdot \mathcal{N}(0,1)$$
*   **$f_{SINDy}$**: Ecuación extraída (El Drift).
*   **$\sigma_{res}$**: Volatilidad local del residuo del régimen.
*   **Reconstrucción Absoluta:** $P_{t+k} = P_{t} \cdot \exp(\sum r)$, esto dibuja el **Cono de Incertidumbre**.

---

## 5. Reglas de Contribución y Restricciones de Hardware

Cualquier mantenedor que amplíe este proyecto debe cumplir las siguientes normas obligatorias:

1.  **Límite de 16GB de RAM:** Este proyecto no está en un clúster AWS. Los bucles masivos (Hot Loops) como la proyección Monte Carlo **no pueden** usar DataFrames de Pandas.
2.  **NumPy es Rey:** Toda la memoria en tránsito entre la Capa 2 y la Capa 5 debe ser obligatoriamente `np.array` de tipo `np.float64` asegurando que sean "C-Contiguous" para que la caché L3 del procesador (ej. Ryzen 7) los procese sin cuellos de botella.
3.  **Procesamiento Asíncrono / Multiparámetro:** Siempre que sea posible, utilizar `ProcessPoolExecutor` o paralelismo si se evalúan varias tolerancias a la vez, aunque respetando no sobreescribir la memoria compartida.
4.  **No alterar la Normalización del Retorno:** Nunca enviar el Precio Absoluto ($30,000 en BTC) directamente a SINDy. Las matrices grandes colapsan el optimizador STLSQ. SINDy solo mastica **Retornos (pequeñas oscilaciones)** y Volúmenes Z-Score.
5.  **Manejo de Gaps:** Cualquier dato de mercado entrante debe pasar por un `ffill()` (forward fill) estricto. Los ceros absolutos o NaNs rompen las derivadas del spline instantáneamente.

---

## 6. Entorno de Desarrollo y QA (Testing)

### Configuración del Entorno (Local-First)
Dado el límite estricto de recursos y las dependencias matemáticas pesadas, el entorno de ejecución debe ser inmutable:
*   **Python Version:** `3.10+` (Requerido para soporte completo asíncrono y compatibilidad con Type Hints avanzados).
*   **Gestión de Dependencias:** El archivo `requirements.txt` es la fuente de verdad. Actualizar librerías matemáticas (`scipy`, `numpy`, `pysindy`) de forma imprudente puede alterar los resultados de precisión de punto flotante en la matriz STLSQ. Fija las versiones para producción.

### Arquitectura de Tests (`tests/`)
Cualquier nueva implementación en las capas del motor (`src/quant_engine/`) debe estar cubierta por la suite de pruebas (e.g., `tests/test_nervous.py`). Las directrices de QA para el Motor Cuantitativo son:
1.  **Estabilidad Numérica:** Los tests no solo prueban que el código no dé error (Crash), sino que prueban casos límite. Ej: ¿Qué pasa si el volumen es 0 durante 50 velas? El test debe asegurar que el Spline no divida por cero y SINDy lo ignore graciosamente.
2.  **Aislamiento del Rescate (Autopilot):** Se debe testear inyectando ruido blanco hiper-volátil a la Capa 3 y verificando que el motor dispara la "Degradación Elegante" en lugar de colapsar la UI de Streamlit.
3.  **Agnosticismo del Activo:** Las lógicas matemáticas deben testearse tanto con datos de cripto (alta volatilidad, gaps cortos) como de Forex/Acciones (menor volatilidad, ruido continuo) para validar la parametrización del Z-Score y el filtro CUSUM.