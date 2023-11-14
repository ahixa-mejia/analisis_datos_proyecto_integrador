import pandas as pd

def procesar_datos(dataframe):
    # Verificar que no existan valores faltantes
    dataframe = dataframe.dropna()
    
    # Verificar que no existan filas repetidas
    dataframe = dataframe.drop_duplicates()
    
    # Verificar si existen valores atípicos y eliminarlos
    dataframe = dataframe[dataframe["age"] >= 0]
    
    # Crear una columna para categorizar por edades
    def categorizar_edad(edad):
        if edad <= 12:
            return "Niño"
        elif 13 <= edad <= 19:
            return "Adolescente"
        elif 20 <= edad <= 39:
            return "Joven adulto"
        elif 40 <= edad <= 59:
            return "Adulto"
        else:
            return "Adulto mayor"
    
    dataframe["categoria_edad"] = dataframe ["age"].apply(categorizar_edad)
    
    # Guardar el resultado como csv
    dataframe.to_csv("datos_procesados.csv", index=False)
    
    return dataframe

# Cargar el CSV mediante la URL proporcionada
url = "https://huggingface.co/datasets/mstz/heart_failure/raw/main/heart_failure_clinical_records_dataset.csv"
dataframe = pd.read_csv(url)

# Llamar a la función para procesar los datos
dataframe_procesado = procesar_datos(dataframe)

# Encapsula toda la lógica anterior en una función que reciba un dataframe como entrada.
print(dataframe_procesado.head())






