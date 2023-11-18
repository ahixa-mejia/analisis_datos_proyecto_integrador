import pandas as pd
import matplotlib.pyplot as plt

try:
    archivo_csv = "datos_procesados.csv"
    dataframe = pd.read_csv(archivo_csv)

    men_data = dataframe[dataframe['sex'] == 1]
    women_data = dataframe[dataframe['sex'] == 0]

    plt.figure(figsize=(12, 6))

    # Primer subplot con histogramas
    plt.subplot(1, 2, 1)
    plt.hist(men_data['age'], bins=20, color='blue', alpha=0.5, label='Hombres', edgecolor='black')
    plt.hist(women_data['age'], bins=20, color='red', alpha=0.5, label='Mujeres', edgecolor='black')
    plt.title('Distribucion de Edades por sexo')
    plt.xlabel('Edad')
    plt.ylabel('Frecuencia')
    plt.legend()

    # Segundo subplot con gráficos de barras
    categories = ['anaemia', 'diabetes', 'smoking', 'DEATH_EVENT']
    men_counts = [men_data[category].sum() for category in categories]
    women_counts = [women_data[category].sum() for category in categories]

    x = range(len(categories))

    # Anchos positivos y negativos para barras lado a lado (ajustados para que las barras de hombres estén a la izquierda)
    bar_width = 0.4
    plt.subplot(1, 2, 2)
    plt.bar(x, men_counts, color='blue', width=-bar_width, label='Hombres', align='edge', alpha=0.6)
    plt.bar(x, women_counts, color='red', width=bar_width, label='Mujeres', align='edge', alpha=0.6)

    plt.xticks(x, categories)
    plt.xlabel('Categorías')
    plt.ylabel('Cantidad')
    plt.title('Distribucion de Categorías por Sexo')
    plt.legend()

    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"El archivo csv en la ubicación '{archivo_csv}' no se encontró.")
except Exception as e:
    print(f"Error inesperado: {e}")

