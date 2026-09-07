---
title: KineTopus Engine
emoji: 🔭
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.34.0
app_file: app.py
python_version: "3.10"
pinned: false
---

# 🔭 Kinetopus Engine: Motor de Física Computacional Financiera

> **¿Qué es?** Es una arquitecutra matematica que transforma el ruido caótico de las series temporales en leyes físicas continuas, descubriendo la ecuación que dicta su inercia

[![Status: Research Preprint](https://img.shields.io/badge/Status-Research%20Preprint-blue)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7431018)
[![SSRN Preprint](https://img.shields.io/badge/SSRN-Preprint%207431018-002D62.svg?logo=elsevier&logoColor=white)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7431018)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0008--9155--4653-A6CE39.svg?logo=orcid&logoColor=white)](https://orcid.org/0009-0008-9155-4653)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
[![Interactive Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-FFD21E)](https://huggingface.co/spaces/Juan778/KineTopus_Engine)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

📄 **Artículo Científico Formal:** Consulta el paper en [Kinetopus_Engine_Paper.pdf](Kinetopus_Engine_Paper.pdf) o en [SSRN (Preprint ID 7431018)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7431018).

---

## 🧭 Contexto y descripción de la problemática a superar

Al adentrarme en el mundo del trading, en el intento de darle sentido a lo que estaba viendo de una manera intuitiva y lógica, me surgio una duda que resulto imposible de ignorar: si se mira una gráfica de precios desde la escala correcta, ciertos tramos se parecían curiosamente a funciones matemáticas que ya habia visto en la escuela: una parábola, un decaimiento exponencial, una oscilación amortiguada... Era una asociación que el ojo reconocía con facilidad, pero que el mundo financiero convencional descartaba como ilusión o como ruido.

![alt text](docs/assets/image-1.png)

Eso me generó una pregunta que resultó imposible de ignorar: **¿y si no era una ilusión?**

El problema era que convertir esa intuición en algo tangible y riguroso era enormemente difícil. Los mercados son ruidosos por naturaleza, los datos llegan discretos y fragmentados como velas en un gráfico, y las herramientas estadísticas tradicionales (regresiones lineales, ARIMA, redes neuronales) están diseñadas para *describir* el comportamiento pasado o *ajustar* patrones, no para *descubrir* la ley física que lo genera. Son cajas negras que predicen sin explicar.

A esto se suma un problema estructural del ecosistema: la mayoría de las soluciones de análisis cuantitativo avanzado dependen de infraestructura en la nube, APIs de pago o hardware de alto rendimiento —una barrera real para cualquier investigador individual que quiera experimentar con soberanía total sobre sus datos.

**Kinetopus nació como un experimento personal para validar esta intuición.** La idea era construir, desde cero, un pipeline matemático que fuera capaz de ver lo que el ojo ya sospechaba: que debajo del ruido, los mercados financieros —y otras series temporales complejas— obedecen a dinámicas gobernadas por ecuaciones diferenciales locales. Y una vez construida la infraestructura, quedó claro que la misma lógica era aplicable mucho más allá del precio de un activo: datos climáticos, señales fisiológicas, cualquier sistema dinámico que evolucione en el tiempo.

> *No se trata de predecir el futuro con certeza. Se trata de descubrir la física que lo está construyendo en este momento.*

---

## 🌌 La Tesis del Proyecto

los datos aparentemente discretos, ruidosos y aleatorios de los mercados financieros (y otras series temporales complejas) ocultan un comportamiento matemático subyacente, estructurado como funciones deterministas gobernantes. Es decir, pasar de datos discretos temporales a ecuaciones capaces de hacer una aproximacion del futuro.

Mi propósito es demostrar y operativizar esta tesis mediante los siguientes pasos:

1. **Transición Discreto a Continuo:** Transformar la nube de puntos inconexos en funciones matemáticas continuas y derivables mediante suavizado topológico.
2. **Descubrimiento Físico:** Aplicar lógicas matemáticas y algoritmos de la física del caos para desentrañar las Ecuaciones Diferenciales Ordinarias (ODEs) que rigen la inercia de los datos.
3. **Validación Determinista:** Demostrar que, observados a la escala correcta y bajo el marco adecuado, los movimientos del algunas series temporales como el trading, pueden seguir lógicas matemáticas demostrables y predecibles a corto plazo.

---

## ⚙️ Arquitectura Matemática (Pipeline de Transmutación)

A continuacion, resumiré las 3 principales capas lógicas con las que funciona la logica del motor:

1. **Sensor Espectral (FFT):** Se detectan los 2 periodos principales (Top-K=2) para informar la rigidez del moldeado posterior.
2. **Moldeador Topológico (Splines $C^2$):** Transforma la nube de puntos inconexos en una sábana matemática continua e infinitamente derivable, eliminando el ruido numérico. Para esto, me apoye del algoritmo AAA.
3. **Descubrimiento Físico:** Aplica *Sparse Identification of Nonlinear Dynamics* lógica matemáticas y algoritmos de la física del caos para desentrañar las Ecuaciones Diferenciales Ordinarias (ODEs) que rigen la inercia de los datos. Debido a que los datos pueden cambiar su naturaleza matematica a lo largo del tiempo, se mide mediante los errores acumulados (CUMSUM) si hay un cambio de régimen o de inercia para recalcular la nueva ecuacion diferencial que se va formando.


---

## 🧪 Metodología de Validación

Construir el núcleo del motor fue solo la primera mitad del problema. La segunda, quizás más crítica, era saber si realmente funcionaba —y bajo qué condiciones.

### Paso 1 · Optimización de Parámetros

El motor expone varios parámetros que controlan la sensibilidad del sistema: la tolerancia del suavizado Spline, los umbrales del detector CUSUM, la complejidad polinomial de SINDy, entre otros. Algunos de estos parámetros tienen un impacto tan global que fijarlos manualmente con criterio físico era suficiente. Otros, en cambio, dependen fuertemente del activo y del horizonte temporal analizado.

Para estos últimos, diseñé un **score compuesto** —una función de puntuación que pondera la calidad del ajuste físico (R²), la pureza de los regímenes detectados y la coherencia de la proyección estocástica— y apliqué un algoritmo de búsqueda que maximiza esa puntuación automáticamente. El resultado: en lugar de "adivinar" los parámetros ideales, el sistema los *descubre* para cada caso.

### Paso 2 · Walk-Forward Validation

Con los parámetros calibrados, el siguiente reto era validar el rendimiento *predictivo* de manera honesta, sin contaminar el futuro con información del pasado (un vicio conocido como **look-ahead bias**).

Para eso diseñé un protocolo de **Walk-Forward Validation**, que es el estándar más riguroso en validación de modelos de series temporales:

```
[─────── Historia ────────][─ Test ─]   → Iteración 1
         [─────── Historia ────────][─ Test ─]   → Iteración 2
                  [─────── Historia ────────][─ Test ─]   → Iteración N
```

En cada iteración, el motor entrena únicamente con el bloque de historia pasada, descubre su ecuación física, proyecta las siguientes N velas y compara contra el futuro real —que nunca vio. La ventana avanza y el proceso se repite. El resultado final es una **base de datos** con cientos de iteraciones por activo y por combinación de periodo-intervalo, que sirve como materia prima para el análisis estadístico.

### Paso 3 · Análisis de Resultados

Con la base de datos generada, apliqué un análisis exploratorio (`notebooks_val/3_Analytics_and_Discovery.ipynb`) para separar el ruido de la señal: identificar en qué combinaciones de activo y temporalidad el motor demuestra una **ventaja estadística real**, cuantificada mediante Hit Ratio (precisión direccional), Alpha frente a un benchmark estadístico (ARIMA) y tasa de mortalidad matemática del modelo.




## 📊 Resultados y Demostraciones

---

> **Nota:** La optimización de parámetros aplicada en estas pruebas se basó en criterios empíricos diseñados para agilizar la experimentación. Es decir, el motor no está operando con su máxima capacidad predictiva; una calibración más exhaustiva probablemente mejoraría estos resultados. El objetivo aquí fue demostrar que el sistema es capaz de descubrir relaciones matemáticas latentes y proyectarlas de manera coherente hacia el futuro.

A continuación, comparto capturas de pruebas manuales realizadas sobre **BTC-USD** como comprobación visual:

![alt text](<docs/assets/Captura de pantalla 2026-04-21 215517.png>) ![alt text](<docs/assets/Captura de pantalla 2026-05-15 142259.png>)
![alt text](<docs/assets/Captura de pantalla 2026-04-21 230356.png>) ![alt text](<docs/assets/Captura de pantalla 2026-04-20 162859.png>)

#### ¿Por qué BTC-USD?

La elección de Bitcoin como activo principal de prueba no es casual. BTC-USD opera en un mercado 24/7 sin interrupciones de sesión, lo que genera series temporales continuas con mayor densidad de datos. Además, al ser un activo dominado por participantes que operan con análisis técnico, los patrones de comportamiento colectivo tienden a formar estructuras matemáticas más pronunciadas (soportes, resistencias, ciclos), creando un terreno fértil para que un motor de descubrimiento físico encuentre dinámicas gobernantes con mayor claridad.

#### Interpretación: Kinetopus vs. el Estado del Arte

Lo que muestran las gráficas es una correspondencia notable entre la proyección del motor y la realidad observada, manteniendo coherencia estructural durante horizontes de **50 a 100 velas** —un resultado que merece ponerse en perspectiva frente a los enfoques convencionales:

| Enfoque | Horizonte Útil Típico | Limitación Principal |
|---|---|---|
| **ARIMA / SARIMA** | ~3-5 pasos antes de colapsar a la media | Modelo lineal autorregresivo: la predicción converge rápidamente a una línea plana (la media histórica). Incapaz de capturar dinámicas no lineales. |
| **Machine Learning** (XGBoost, Random Forest) | ~5-15 pasos (con feature engineering intensivo) | Requieren ingeniería manual de features. Predicen punto a punto sin modelar la dinámica subyacente. Propensos a sobreajuste en datos financieros ruidosos. |
| **Deep Learning** (LSTM, Transformer) | ~10-30 pasos (con datasets masivos) | Cajas negras opacas que exigen grandes volúmenes de datos y hardware costoso (GPUs). Capturan correlaciones temporales, pero no descubren la *ley física* que genera el movimiento. Altamente sensibles a cambios de régimen. |
| **Kinetopus** | **~20- 100+ velas** (dependiendo del régimen) | Caja blanca: la ecuación descubierta es legible e interpretable. Opera en una laptop de 16GB sin GPU. Su limitación natural es el cambio de régimen —que el propio CUSUM detecta. |

La diferencia fundamental no es solo cuantitativa (más velas), sino **cualitativa**: mientras que los modelos estadísticos y de ML *ajustan curvas* al pasado esperando que el patrón se repita, Kinetopus *descubre la ecuación diferencial* que está generando el movimiento y la integra hacia adelante. Es la diferencia entre memorizar la respuesta y entender la física del problema.




## 🛠️ Instalación y Uso

Kinetopus Engine está diseñado bajo el principio de **Soberanía y Privacidad Total (Local-First)**. Todo el cálculo matricial y vectorial ocurre estrictamente en el hardware del usuario. Está optimizado para entornos con recursos restringidos (16GB RAM).

### Requisitos Previos
* Python 3.10 o superior.

### Configuración del Entorno
```bash
# 1. Clonar el repositorio
git clone https://github.com/tu-usuario/kinetopus-engine.git
cd kinetopus-engine

# 2. Instalar las dependencias estrictas
pip install -r requirements.txt
```

### Ejecutar la Estación de Trabajo
El motor cuenta con una interfaz construida en **Streamlit** y **Plotly** que actúa como el Dashboard Principal.

```bash
# En Windows (usando el script provisto)
./run_app.ps1

# O directamente vía Streamlit
streamlit run app.py
```


---

## 🔬 Descubrimientos y Laboratorio Quant

El repositorio incluye un entorno de validación rigurosa en la carpeta `notebooks_val/`. Algunos hallazgos clave del último reporte analítico (`3_Analytics_and_Discovery.ipynb`):

* **Resiliencia Matemática (0% Mortalidad):** A lo largo de cientos de iteraciones *Walk-Forward* en diversas configuraciones de tiempo y activos (C, MSFT, SOL-USD, XLF), el motor demostró una tasa de fallo matemático del 0.0%, comprobando la estabilidad absoluta del sistema de extracción SINDy.
* **Venciendo al Azar (Hit Ratio > 53%):** En simulaciones predictivas direccionales (El Tramo 1 del futuro), las Leyes Físicas descubiertas por Kinetopus logran identificar **"nichos explotables"**, superando consistentemente la barrera estocástica ciega (50%) y derrotando a modelos puramente estadísticos (como ARIMA) en la comparativa de supervivencia direccional.

---

## 📚 Profundiza en la Arquitectura

Si quieres entender el núcleo teórico, la topología del código o el manifiesto de diseño, te invitamos a explorar la carpeta `docs/`:

* 📜 [`PROJECT_MANIFESTO.md`](docs/PROJECT_MANIFESTO.md): Nuestra filosofía *Local-First* y los principios irrenunciables de privacidad matemática.
* 🧠 [`LOGICA_DEL_SISTEMA.md`](docs/LOGICA_DEL_SISTEMA.md) / [`SPEC_QUANT_ENGINE.md`](docs/SPEC_QUANT_ENGINE.md): Las matemáticas al desnudo. Cómo convertimos ruido de mercado en ecuaciones estocásticas puras.
* 🏗️ [`ARCHITECTURE.md`](docs/ARCHITECTURE.md): El diseño estructural y el flujo de la inercia a través de las 5 capas del motor.


---

## 📄 Licencia y Créditos

### Licencia

Este proyecto se distribuye bajo la **[Apache License 2.0](LICENSE)**.

Puedes usar, modificar y distribuir este software libremente, incluso con fines comerciales, siempre que se mantenga el aviso de copyright y la atribución al autor original. Consulta el archivo [`LICENSE`](LICENSE) para los términos completos.

---

### Créditos Académicos

Kinetopus Engine no nació en el vacío. Está construido sobre los hombros de investigadores que formalizaron las herramientas matemáticas que lo hacen posible. A todos ellos, reconocimiento explícito:

| Algoritmo / Concepto | Rol en el Motor | Referencia |
|---|---|---|
| **SINDy** *(Sparse Identification of Nonlinear Dynamics)* | Descubrimiento de la ecuación diferencial gobernante a partir de datos | Brunton, S. L., Proctor, J. L., & Kutz, J. N. (2016). *Discovering governing equations from data by sparse identification of nonlinear dynamical systems.* PNAS, 113(15), 3932–3937. [DOI](https://doi.org/10.1073/pnas.1517384113) |
| **Algoritmo AAA** *(Adaptive Antoulas–Anderson)* | Aproximación racional para el moldeado topológico continuo (Splines) | Nakatsukasa, Y., Sète, O., & Trefethen, L. N. (2018). *The AAA algorithm for rational approximation.* SIAM Journal on Scientific Computing, 40(3), A1494–A1522. [DOI](https://doi.org/10.1137/16M1106122) |
| **FFT** *(Fast Fourier Transform)* | Detección de frecuencias dominantes en Retornos y Volumen (Capa Sensor) | Cooley, J. W., & Tukey, J. W. (1965). *An algorithm for the machine calculation of complex Fourier series.* Mathematics of Computation, 19(90), 297–301. [DOI](https://doi.org/10.1090/S0025-5718-1965-0178586-1) |
| **CUSUM** *(Cumulative Sum Control Chart)* | Detección de cambios de régimen (quiebres estructurales) en la inercia del mercado | Page, E. S. (1954). *Continuous Inspection Schemes.* Biometrika, 41(1/2), 100–115. [DOI](https://doi.org/10.1093/biomet/41.1-2.100) |
| **Método de Euler–Maruyama** | Integración numérica de las SDEs para la proyección estocástica (Monte Carlo) | Maruyama, G. (1955). *Continuous Markov processes and stochastic equations.* Rendiconti del Circolo Matematico di Palermo, 4, 48–90. |

---

### 📖 Cita Académica (BibTeX)

Si utilizas este motor, su marco metodológico o sus benchmarks en tu investigación, por favor cita el preprint formal:

```bibtex
@article{ussa2026kinetopus,
  title={Kinetopus Engine: A Parsimonious Physical-Mathematical Pipeline for Local-First Financial Time Series Forecasting using Sparse Identification of Dynamical Systems (SINDy)},
  author={Ussa Aponte, Juan Diego},
  journal={SSRN Electronic Journal},
  year={2026},
  note={Preprint ID 7431018},
  url={https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7431018}
}
```

**Investigador Principal:** Juan Diego Ussa Aponte ([ORCID: 0009-0008-9155-4653](https://orcid.org/0009-0008-9155-4653))

---

### Dependencias Open-Source

El stack computacional se apoya en el ecosistema científico de Python:

- **[NumPy](https://numpy.org/)** — Cálculo matricial vectorizado (núcleo de todos los hot loops).
- **[SciPy](https://scipy.org/)** — Suavizado Spline, optimización y álgebra lineal.
- **[Streamlit](https://streamlit.io/)** — Interfaz de usuario interactiva (Dashboard).
- **[Plotly](https://plotly.com/)** — Visualizaciones dinámicas del espacio de fases y proyecciones.

