# ARCHIVO HISTÓRICO DE DESARROLLO (FASES 1 a 8)

Este documento contiene el plan original y ejecutado del MVP 1.0 (Chat de Análisis de Datos) y la transición fundacional al MVP 2.0 (Motor Quant Físico). Estas fases ya se encuentran completadas al 100%.

## FASE 1 - INGESTA Y DIAGNÓSTICO
**ESTADO:** ✅ FINALIZADA
**META:** Lograr que el usuario suba un archivo y el sistema entienda automáticamente qué contiene.
### 1. ESTRUCTURA BASE (SCAFFOLDING)
Setup Inicial de directorios y dependencias de Streamlit.
### 2. GESTIÓN DE ARCHIVOS (LOADER)
Streamlit File Uploader persistiendo en el servidor.
### 3. MOTOR DE PERFILADO (PROFILER)
Clase `DataProfiler` como alternativa híbrida DuckDB/Pandas.
### 4. ESQUEMA DE METADATOS (SCHEMA)
Definición de Modelos Pydantic para los Datasets.

## FASE 2: MOTOR DE RAZONAMIENTO (PLANIFICACIÓN)
**ESTADO:** ✅ FINALIZADA
### 1. CONFIGURACIÓN LLM (GEMINI)
Cliente base de comunicación con la API.
### 2. AGENTE DE NARRATIVA (BUSINESS CONTEXT)
Prompt Engineering para agente de Consultoría de Negocios.
### 3. SANDBOX DE EJECUCIÓN (MVP)
Entorno Controlado `src/data_engine/sandbox.py` con regex de seguridad.

## FASE 3: ANÁLISIS PROFUNDO & VISUALIZACIÓN
**ESTADO:** ✅ FINALIZADA (OBSOLESCENCIA DADA EN MVP 2.0)
Interfaz de chat básica y Agente Analista generador de código Pandas/Plotly para ejecución en el Sandbox.

## FASE 4: MOTOR QUANT (MVP 2.0)
**ESTADO:** ✅ FINALIZADA
**META:** Implementar un motor de análisis de series temporales basado en física cuantitativa y detección de regímenes, garantizando el minimalismo eficiente en RAM (16GB).
### HITO 1: Ingesta y Espectro (El Sensor)
Clase `SpectralAnalyzer` y FFT.
### HITO 2: Suavizado Topológico (El Moldeador)
Clase `ContinuousBlender` y Splines.
### HITO 3: Detección y Orquestación (El Sistema Nervioso)
Clase `RegimeShiftDetector` (Algoritmo CUSUM).
### HITO 4: Extracción Física (El Motor Quant)
Integración de `pysindy` y optimizador STLSQ.
### HITO 5: El Puente de IA (El Analista)
Conector asyncio Ollama para interpretar ecuaciones dinámicas.

## FASE 5: PREDICTOR E INTEGRACIÓN DE MERCADO
**ESTADO:** ✅ FINALIZADA
**META:** Conectar APIs de mercado en vivo (`yfinance`).
### HITO 6: Conector de Mercado y Telemetría
Extracción de tickers directo de Yahoo Finance y validación de predicción ODE $t+1$.

## FASE 6: ORQUESTACIÓN Y DASHBOARD (UI)
**ESTADO:** ✅ FINALIZADA
**META:** Crear Dashboard Plotly de 3 Paneles en Streamlit integrando SINDy y CUSUM de manera interactiva.

## FASE 7: REFACTORIZACIÓN DE PRECISIÓN Y ESCALA
**ESTADO:** ✅ FINALIZADA
**META:** Eliminar el Sobreajuste del Spline y el Colapso SINDy (Normalización Z-Score, limpieza paramétrica `['P', 'V']`).

## FASE 8: DINÁMICA ESTOCÁSTICA (CONO DE INCERTIDUMBRE)
**ESTADO:** ✅ FINALIZADA
**META:** Sustituir la línea predictiva determinista $t+N$ por un mapa de densidad probabilística usando integrador Euler-Maruyama y visualizado en percentiles topológicos `go.Scatter(fill='tonexty')` para 1,000 caminos de Monte Carlo.
