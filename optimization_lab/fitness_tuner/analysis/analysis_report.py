import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

def load_data():
    df = pd.read_csv("validation_results.csv")
    with open("validation_curves.json", "r") as f:
        curves = json.load(f)
    return df, curves

def compute_statistics(df):
    print("==================================================")
    print("ESTADÍSTICAS GLOBALES POR MODELO (OOS - 50 ACTIVOS)")
    print("==================================================")
    
    metrics = ['hit_ratio', 'mcc_test', 'profit_factor', 'max_drawdown', 'mape_test']
    summary = df.groupby('model')[metrics].agg(['mean', 'median'])
    print(summary.to_string())
    
    # Tasa de victorias vs Classic
    print("\n--- TASA DE VICTORIAS VS CLASSIC ---")
    df_classic = df[df['model'] == 'Classic'].set_index('ticker')
    df_syn = df[df['model'] == 'Synthetic'].set_index('ticker')
    df_real = df[df['model'] == 'Real'].set_index('ticker')
    
    for model_name, df_comp in [('Synthetic', df_syn), ('Real', df_real)]:
        win_hit = (df_comp['hit_ratio'] > df_classic['hit_ratio']).mean() * 100
        win_pf = (df_comp['profit_factor'] > df_classic['profit_factor']).mean() * 100
        win_mcc = (df_comp['mcc_test'] > df_classic['mcc_test']).mean() * 100
        print(f"{model_name} supera a Classic en:")
        print(f" - Hit Ratio: {win_hit:.1f}% de los activos")
        print(f" - Profit Factor: {win_pf:.1f}% de los activos")
        print(f" - MCC: {win_mcc:.1f}% de los activos")
        
    print("==================================================")

def plot_boxplots(df):
    plt.figure(figsize=(12, 5))
    
    # Hit Ratio Boxplot
    plt.subplot(1, 2, 1)
    models = ['Classic', 'Synthetic', 'Real']
    data_hit = [df[df['model'] == m]['hit_ratio'].values for m in models]
    plt.boxplot(data_hit, labels=models)
    plt.title('Distribución de Hit Ratio OOS')
    plt.ylabel('Hit Ratio')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Profit Factor Boxplot (Clipping a 5 para visualización)
    plt.subplot(1, 2, 2)
    data_pf = [np.clip(df[df['model'] == m]['profit_factor'].values, 0, 5) for m in models]
    plt.boxplot(data_pf, labels=models)
    plt.title('Distribución de Profit Factor (Capped at 5.0)')
    plt.ylabel('Profit Factor')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('analysis_boxplots.png')
    print("Guardada imagen: analysis_boxplots.png")

def plot_drift_distribution(df):
    plt.figure(figsize=(10, 6))
    
    models = ['Classic', 'Synthetic', 'Real']
    colors = ['blue', 'green', 'orange']
    
    for i, model in enumerate(models):
        data = df[df['model'] == model]['drift_k']
        plt.hist(data, bins=20, alpha=0.5, label=model, color=colors[i])
        
    plt.title('Distribución de Selección de Parámetro drift-k')
    plt.xlabel('drift-k')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('analysis_drifts.png')
    print("Guardada imagen: analysis_drifts.png")

def plot_cumulative_curves(curves):
    # Inicializar arreglos para la suma global (horizonte 150 velas)
    horizon = 150
    models = ['Classic', 'Synthetic', 'Real']
    
    global_hits = {m: np.zeros(horizon) for m in models}
    
    valid_assets = 0
    for asset_data in curves:
        models_data = asset_data.get('models', {})
        if not models_data:
            continue
            
        # Comprobar si todos los modelos tienen longitud 150
        valid = True
        for m in models:
            if m not in models_data or len(models_data[m]['hit_acumulado_curve']) != horizon:
                valid = False
                break
                
        if not valid:
            continue
            
        valid_assets += 1
        for m in models:
            global_hits[m] += np.array(models_data[m]['hit_acumulado_curve'])
            
    if valid_assets == 0:
        print("No se encontraron curvas de 150 velas válidas para graficar.")
        return
        
    plt.figure(figsize=(12, 6))
    colors = {'Classic': 'blue', 'Synthetic': 'green', 'Real': 'red'}
    
    for m in models:
        plt.plot(global_hits[m], label=f'{m} (Final: {global_hits[m][-1]:.0f})', color=colors[m], linewidth=2)
        
    plt.title(f'Curva Global de Hit Acumulado ({valid_assets} Activos)')
    plt.xlabel('Velas Out-of-Sample')
    plt.ylabel('Señales de Acierto Neto (Aciertos - Fallos)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Añadir línea base 0
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('analysis_hit_curve.png')
    print("Guardada imagen: analysis_hit_curve.png")

def main():
    if not os.path.exists("validation_results.csv") or not os.path.exists("validation_curves.json"):
        print("Archivos de resultados no encontrados.")
        return
        
    df, curves = load_data()
    
    compute_statistics(df)
    plot_boxplots(df)
    plot_drift_distribution(df)
    plot_cumulative_curves(curves)
    
    print("\nAnálisis completado exitosamente.")

if __name__ == "__main__":
    main()
