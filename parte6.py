import pandas as pd
from datasets import load_dataset
import requests
import io

def descargar_datos(url):
    url = "https://huggingface.co/datasets/mstz/heart_failure/raw/main/heart_failure_clinical_records_dataset.csv"

    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Error al descargar los datos. Código de estado: {response.status_code}")
        return None

def convertir_dataframe(archivo):
    df = pd.read_csv(io.StringIO(archivo))
    return df

def categorizar_en_grupos(df):
    df["categoria_por_edad"] = pd.cut(df["age"],
            bins=[0, 12, 20, 40, 60, 100],
            labels=["niño", "adolescente", "adulto joven", "adulto", "adulto mayor"])
    return df

def exportar_a_csv(df, filename):
    df.to_csv("datos_categorizados.csv", index=False)
    print(f"Datos exportados correctamente a '{filename}'.")

def main(url, filename):
    data = descargar_datos(url)

    if data is not None:
        df = convertir_dataframe(data)
        df = categorizar_en_grupos(df)
        exportar_a_csv(df, filename)

# Uso del script con argumentos
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Por favor, proporciona la URL y el nombre del archivo CSV como argumentos.")
        sys.exit(1)

    url = sys.argv[1]
    filename = sys.argv[2]

    main(url, filename)
