# Laboratorio de Optimización Cuántica - KineTopus Engine

**¡ATENCIÓN AGENTES! Lee esto obligatoriamente antes de proceder.**

Estás operando dentro de `optimization_lab/`, un entorno aislado ("Clean Room") diseñado exclusivamente para el entrenamiento de fitness, la meta-optimización de hiperparámetros y la validación masiva del motor.

> [!WARNING]
> **NO REVISES ni alteres código fuera de `optimization_lab/`.** Tu foco está estrictamente encapsulado en este laboratorio de optimización. No intentes analizar ni modificar el resto del proyecto general de 'KineTopus Engine'.

---

## 🎯 Estado Actual del Proyecto y Estructura

El laboratorio se ha estructurado en dos grandes módulos independientes:

### 1. `fitness_tuner/` (Optimización y descubrimiento de la Ecuación Fitness)
*   **Misión:** Aprender los pesos de las características físicas Out-of-Sample para estimar la ecuación fitness óptima.
*   **Resultados guardados:** Los pesos entrenados con datos sintéticos y reales ya están guardados en `results/synthetic/` y `results/real/`.
*   **Estado:** Completado y validado.

### 2. `parameter_tuner/` (Meta-optimización de Hiperparámetros)
*   **Misión:** Buscar la calibración estática óptima de `spline_tol`, `cusum_h` y `drift_k` usando la ecuación fitness entrenada anteriormente como función objetivo.
*   **Modo de Ejecución:** El script `training/grid_search_tuner.py` acepta un argumento `--mode` (`synthetic`, `real`, `classic`). 
    *   Si es `classic`, utiliza la fórmula clásica del auto-tuner como score.
    *   Si es `synthetic` o `real`, carga automáticamente sus respectivos coeficientes del fitness tuner.
*   **Estado:** Estructura de código, robustez frente a fallos y validaciones rápidas completadas y testeadas (`--test` exitoso). Listo para el entrenamiento masivo.

---

## 🗂️ Árbol de Directorios y Responsabilidades

*   `core/` y `telemetry.py` (en la raíz de `optimization_lab/`): **Recursos compartidos** y únicos que contienen las lógicas físicas de la señal.
*   `fitness_tuner/` y `parameter_tuner/`: Cada uno contiene:
    *   `training/`: Scripts de optimización masiva (`optimizer_pipeline.py`, `grid_search_tuner.py`).
    *   `validation/`: Suites de validación paralela (`validation_suite.py`, `synthetic_validation_suite.py`).
    *   `analysis/`: Jupyter Notebooks y reportes de análisis visual interactivo.
    *   `results/`: Subcarpetas `synthetic/`, `real/` y `classic/` que almacenan CSVs/JSONs de forma aislada.

---

## 🚀 Tu Próxima Misión (Qué debes hacer en este chat)

Tu objetivo en esta fase es **ejecutar las simulaciones y entrenamientos definitivos a gran escala** y reportar/analizar el comportamiento resultante en los notebooks de análisis:

1.  **Ejecutar los entrenamientos de Grid Search completos (1500 combinaciones × 100 universos):**
    ```bash
    # Modo Sintético
    python optimization_lab/parameter_tuner/training/grid_search_tuner.py --mode synthetic
    # Modo Real
    python optimization_lab/parameter_tuner/training/grid_search_tuner.py --mode real
    # Modo Clásico
    python optimization_lab/parameter_tuner/training/grid_search_tuner.py --mode classic
    ```
    *Nota: Estos entrenamientos son sumamente pesados (150,000 modelos de SINDy). Se recomienda ejecutarlos por separado o de fondo.*

2.  **Correr las validaciones completas sobre los conjuntos de parámetros resultantes:**
    ```bash
    # Validación sintética masiva (100 universos)
    python optimization_lab/parameter_tuner/validation/synthetic_validation_suite.py
    # Validación real masiva (50 activos reales de mercado)
    python optimization_lab/parameter_tuner/validation/validation_suite.py
    ```

3.  **Analizar las superficies de rendimiento en el notebook de visualización 3D:**
    *   Abre y asiste al usuario en la revisión del cuaderno: `parameter_tuner/analysis/1_Parameter_Analysis.ipynb`.

¡Mantén el orden, vigila la memoria L3/RAM y haz que la física cuantitativa resuelva el caos!
