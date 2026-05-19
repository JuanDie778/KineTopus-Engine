import os
import json

def strip_notebook(filepath):
    print(f"[*] Limpiando salidas de: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Iterar sobre las celdas y limpiar salidas e historial de ejecución
        for cell in data.get('cells', []):
            if cell.get('cell_type') == 'code':
                cell['outputs'] = []
                cell['execution_count'] = None
                
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
            
        new_size = os.path.getsize(filepath) / (1024 * 1024)
        print(f"[+] ¡Limpieza completada! Nuevo tamaño: {new_size:.4f} MB")
    except Exception as e:
        print(f"[!] Error al limpiar {filepath}: {e}")

def main():
    notebooks_dir = "notebooks_val"
    if not os.path.exists(notebooks_dir):
        print(f"[!] No se encontró el directorio {notebooks_dir}")
        return
        
    for filename in os.listdir(notebooks_dir):
        if filename.endswith(".ipynb"):
            filepath = os.path.join(notebooks_dir, filename)
            strip_notebook(filepath)

if __name__ == "__main__":
    main()
