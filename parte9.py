import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px

# Leer el DataFrame
archivo_csv = "datos_procesados.csv"
dataframe = pd.read_csv(archivo_csv)

# Eliminar la columna 'DEATH_EVENT' y 'categoria_edad'
X = dataframe.drop(columns=['DEATH_EVENT', 'categoria_edad']).values

# Crear el array unidimensional para 'DEATH_EVENT'
y = dataframe['DEATH_EVENT'].values

# Aplicar t-SNE para reducción de dimensionalidad
X_embedded = TSNE(n_components=3, learning_rate='auto', init='random', perplexity=3).fit_transform(X)

# Crear DataFrame para Plotly
df_embedded = pd.DataFrame({'Dimension 1': X_embedded[:, 0], 'Dimension 2': X_embedded[:, 1], 'Dimension 3': X_embedded[:, 2], 'DEATH_EVENT': y})

# Crear gráfico de dispersión 3D con Plotly
fig = px.scatter_3d(df_embedded, x='Dimension 1', y='Dimension 2', z='Dimension 3', color='DEATH_EVENT',
                    title='Visualización 3D con t-SNE', labels={'DEATH_EVENT': 'Estado de Fallecimiento'},
                    color_discrete_map={0: 'green', 1: 'red'})
fig.show()
