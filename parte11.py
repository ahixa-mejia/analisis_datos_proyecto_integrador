import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Leer el DataFrame
archivo_csv = "datos_procesados.csv"
dataframe = pd.read_csv(archivo_csv)

# Eliminar la columna 'categoria_edad'
dataframe = dataframe.drop(columns=['categoria_edad'])

# Visualizar la distribución de clases
plt.figure(figsize=(6, 4))
dataframe['DEATH_EVENT'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Distribución de Clases')
plt.xlabel('Clase')
plt.ylabel('Frecuencia')
plt.xticks(rotation=0)
plt.show()

# Separar las características (X) y la variable objetivo (y)
X = dataframe.drop(columns=['DEATH_EVENT'])
y = dataframe['DEATH_EVENT']

# Dividir el conjunto de datos en entrenamiento y prueba de manera estratificada
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Inicializar y ajustar el modelo de árbol de decisión
modelo_arbol = DecisionTreeClassifier(random_state=42)
modelo_arbol.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = modelo_arbol.predict(X_test)

# Calcular la precisión (accuracy)
precision = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {precision:.2f}")
