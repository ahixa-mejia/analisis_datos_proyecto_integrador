import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Leer el DataFrame
archivo_csv = "datos_procesados.csv"
dataframe = pd.read_csv(archivo_csv)

# Eliminar la columna 'categoria_edad'
dataframe = dataframe.drop(columns=['categoria_edad'])

# Separar las características (X) y la variable objetivo (y)
X = dataframe.drop(columns=['DEATH_EVENT'])
y = dataframe['DEATH_EVENT']

# Dividir el conjunto de datos en entrenamiento y prueba de manera estratificada
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Inicializar y ajustar el modelo de Random Forest
modelo_rf = RandomForestClassifier(random_state=42)
modelo_rf.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = modelo_rf.predict(X_test)

# Calcular la matriz de confusión
matriz_confusion = confusion_matrix(y_test, y_pred)

# Visualizar la matriz de confusión con seaborn
plt.figure(figsize=(4, 4))
sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matriz de Confusión')
plt.xlabel('Predicciones')
plt.ylabel('Valores Reales')
plt.show()

# Calcular la precisión (accuracy)
precision = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {precision:.2f}")

# Calcular el F1-Score
f1 = f1_score(y_test, y_pred)
print(f"F1-Score del modelo: {f1:.2f}")

# Comparar accuracy y F1-Score
print("¿El accuracy captura completamente el rendimiento del modelo en este caso?")
if precision == f1:
    print("Sí, ambos son iguales.")
else:
    print("No, hay diferencias entre accuracy y F1-Score.")
