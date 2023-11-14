import requests

def descargar_csv_desde_url(url, nombre_archivo):
    try:
        respuesta = requests.get(url)

        if respuesta.status_code == 200:
            with open(nombre_archivo, 'wb' ) as archivo:
                archivo.write(respuesta.content)
            print(f"los datos han sido descargados y guardados en {nombre_archivo} correctamente.")
        else:
            print(f"la solicitud GET no fue exitosa. codigo de estado: {respuesta.status_code}")
    except Exception as e:
        print(f"ocurrio un error: {str(e)}")
    
url = "https://huggingface.co/datasets/mstz/heart_failure/raw/main/heart_failure_clinical_records_dataset.csv"
nombre_archivo = "heart_failure_clinical_records_dataset.csv"
descargar_csv_desde_url(url, nombre_archivo)
