import logging
import sys
import os

def setup_telemetry(log_file="optimization_lab.log", level=logging.INFO):
    """
    Configura la infraestructura de telemetría para el Laboratorio de Optimización.
    Usa un formato jerárquico diseñado para ser parseable por Agentes IA (como el @AUDITOR)
    y legible por humanos.
    """
    # Crear el logger raíz
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Limpiar handlers previos para evitar duplicados si se llama múltiples veces (ej. Jupyter)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Formato Quant/Agente: [TIEMPO] [NIVEL] [MODULO] - MENSAJE
    formatter = logging.Formatter(
        '%(asctime)s | [%(levelname)s] | %(name)s : %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 1. Handler para Consola (Stream)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 2. Handler para Archivo (File) - Crucial para la auditoría de la IA
    # Guardamos en la raíz del optimization_lab
    log_path = os.path.join(os.path.dirname(__file__), log_file)
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # El archivo guarda TODO, incluso si la consola está en INFO
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logging.info(f"Telemetría iniciada. Archivo de log: {log_path}")
    return root_logger
