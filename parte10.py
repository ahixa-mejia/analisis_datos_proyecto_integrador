import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Leer el DataFrame
archivo_csv = "datos_procesados.csv"
dataframe = pd.read_csv(archivo_csv)

# Eliminar las columnas 'DEATH_EVENT', 'age' y 'categoria_edad'
X = dataframe.drop(columns=['DEATH_EVENT', 'age', 'categoria_edad'])
y = dataframe['age']

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar y ajustar el modelo de regresión lineal
modelo_regresion = LinearRegression()
modelo_regresion.fit(X_train, y_train)

# Predecir las edades en el conjunto de prueba
y_pred = modelo_regresion.predict(X_test)

# Calcular el error cuadrático medio
error_cuadratico_medio = mean_squared_error(y_test, y_pred)

# Imprimir el error cuadrático medio
print(f"Error Cuadrático Medio: {error_cuadratico_medio}")
